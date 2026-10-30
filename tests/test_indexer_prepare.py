# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Indexer preparation: numerical reference, API contracts and serving regressions."""

from dataclasses import dataclass

import pytest
import torch

import halops
from halops import (
    indexer_prepare,
    indexer_prepare_workspace,
    indexer_prepare_workspace_bytes,
)

FP8 = torch.float8_e4m3fn
HIDDEN, HEADS, HEAD_DIM, RANK, ROPE_DIM = 4096, 64, 128, 1024, 64
DEFAULT_WEIGHT_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
FP8_RMS_TOL, SCALE_RMS_TOL = 5e-3, 4e-3
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or torch.cuda.get_device_properties(0).multi_processor_count < 128,
    reason="indexer_prepare requires SM100 with at least 128 SMs",
)


def quantize_blocks(x, group_size):
    groups = x.float().reshape(x.shape[0], -1, group_size)
    # MXFP8 input quantization has a different floor from FP8 output rows.
    power = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-10) / 448))
    scale = torch.exp2(power)
    codes = (groups / scale[..., None]).clamp(-448, 448).to(FP8)
    dequantized = (codes.float() * scale[..., None]).reshape(x.shape)
    return codes.reshape(x.shape), (power + 127).to(torch.uint8), dequantized


def sfatom_scales(exponents):
    rows, groups = exponents.shape
    return (
        exponents.reshape(rows // 128, 4, 32, groups // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(rows, groups)
    )


def hadamard(x):
    result = x.float()
    width = 1
    while width < result.shape[-1]:
        groups = result.reshape(*result.shape[:-1], -1, 2, width)
        left, right = groups.unbind(-2)
        result = torch.stack((left + right, left - right), -2).reshape(x.shape)
        width *= 2
    return result * x.shape[-1] ** -0.5


def rope(x, table, positions, is_neox_style=True):
    cs = table[positions.long()]
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (ROPE_DIM // 2,)
    cos, sin = cs[:, : ROPE_DIM // 2].reshape(shape), cs[:, ROPE_DIM // 2 :].reshape(
        shape
    )
    result = x.clone()
    if is_neox_style:
        first = slice(0, ROPE_DIM // 2)
        second = slice(ROPE_DIM // 2, ROPE_DIM)
    else:
        first, second = slice(0, ROPE_DIM, 2), slice(1, ROPE_DIM, 2)
    left, right = x[..., first], x[..., second]
    result[..., first] = left * cos - right * sin
    result[..., second] = left * sin + right * cos
    return result


def quantize_rows(x):
    rounded = x.to(torch.bfloat16).float()
    scale = rounded.abs().amax(-1).clamp_min(1e-4) / 448
    return (rounded / scale[..., None]).clamp(-448, 448).to(FP8), scale


def quantize_mxfp4(x):
    rounded = x.to(torch.bfloat16).float().reshape(*x.shape[:-1], -1, 32)
    power = torch.ceil(torch.log2(rounded.abs().amax(-1).clamp_min(1e-10) / 6))
    exponents = (power + 127).clamp(0, 254).to(torch.uint8)
    scale = torch.exp2(exponents.float() - 127)
    normalized = (rounded / scale[..., None]).reshape(x.shape)
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    # HALOps quantization resolves a midpoint toward the larger magnitude.
    magnitudes = torch.bucketize(normalized.abs(), midpoints, right=True).to(
        torch.uint8
    )
    codes = magnitudes | ((normalized < 0).to(torch.uint8) << 3)
    return codes[..., ::2] | (codes[..., 1::2] << 4), exponents


def dequantize_mxfp4(packed, scales):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    decoded = values[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return decoded * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)


@dataclass
class Case:
    kwargs: dict
    wq_dequantized: torch.Tensor
    page_size: int

    @property
    def cache(self):
        return self.kwargs["paged_kv_cache_with_scale"]

    def read_cache(self):
        loc = self.kwargs["out_cache_loc"].long()
        valid = loc >= 0
        loc = loc[valid]
        page, slot = loc // self.page_size, loc % self.page_size
        output_fp4 = self.kwargs.get("output_dtype_str", "fp8") == "mxfp4"
        data_bytes = HEAD_DIM // 2 if output_fp4 else HEAD_DIM
        offsets = slot[:, None] * data_bytes + torch.arange(
            data_bytes, device=self.cache.device
        )
        codes = self.cache[page[:, None], offsets].contiguous()
        if not output_fp4:
            codes = codes.view(FP8)
        offsets = (
            self.page_size * data_bytes
            + slot[:, None] * 4
            + torch.arange(4, device=self.cache.device)
        )
        scales = self.cache[page[:, None], offsets].contiguous()
        if not output_fp4:
            scales = scales.view(torch.float32).flatten()
        return codes, scales, valid

    def written_bytes(self):
        loc = self.kwargs["out_cache_loc"].long()
        loc = loc[loc >= 0]
        page, slot = loc // self.page_size, loc % self.page_size
        mask = torch.zeros_like(self.cache, dtype=torch.bool)
        data_bytes = (
            HEAD_DIM // 2
            if self.kwargs.get("output_dtype_str", "fp8") == "mxfp4"
            else HEAD_DIM
        )
        columns = slot[:, None] * data_bytes + torch.arange(
            data_bytes, device=self.cache.device
        )
        mask[page[:, None], columns] = True
        columns = (
            self.page_size * data_bytes
            + slot[:, None] * 4
            + torch.arange(4, device=self.cache.device)
        )
        mask[page[:, None], columns] = True
        return mask


def make_case(
    tokens,
    *,
    page_size=64,
    bias=True,
    apply_hadamard=True,
    sentinels=False,
    shuffle=False,
    index_dtype=torch.int64,
    varied_groups=False,
    device="cuda",
    is_neox_style=True,
    output_dtype_str="fp8",
):
    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, dtype=dtype, device=device)

    x = randn(tokens, HIDDEN)
    q_lora = randn(tokens, RANK)
    wq = randn(HEADS * HEAD_DIM, RANK)
    if varied_groups:
        scales = torch.logspace(-2, 2, RANK // 32, device=device).repeat_interleave(32)
        q_lora = (q_lora.float() * scales).to(torch.bfloat16)
        wq = (wq.float() * scales.flip(0)).to(torch.bfloat16)
    wq, exponents, wq_dequantized = quantize_blocks(wq, 32)
    scale_input = sfatom_scales(exponents)
    pages = max(2, (tokens + page_size - 1) // page_size + 1)
    data_bytes = HEAD_DIM // 2 if output_dtype_str == "mxfp4" else HEAD_DIM
    cache = torch.full(
        (pages, page_size * (data_bytes + 4)), 0xA5, dtype=torch.uint8, device=device
    )
    loc = (
        torch.randperm(pages * page_size, device=device)[:tokens]
        if shuffle
        else torch.arange(tokens, device=device)
    ).to(index_dtype)
    if sentinels:
        loc[::3] = -7
    angles = randn(tokens + 17, ROPE_DIM // 2, dtype=torch.float32)
    kwargs = dict(
        x=x,
        q_lora=q_lora,
        wk_w=randn(HEAD_DIM + HEADS, HIDDEN),
        wq_b=wq,
        wq_b_sf=scale_input,
        k_norm_weight=randn(HEAD_DIM, dtype=torch.float32),
        k_norm_bias=randn(HEAD_DIM, dtype=torch.float32) if bias else None,
        k_norm_eps=1e-6,
        rope_cos_sin_cache=torch.cat((angles.cos(), angles.sin()), -1),
        positions=torch.randperm(tokens + 17, device=device)[:tokens].to(index_dtype),
        paged_kv_cache_with_scale=cache,
        out_cache_loc=loc,
        apply_hadamard=apply_hadamard,
        is_neox_style=is_neox_style,
        output_dtype_str=output_dtype_str,
    )
    return Case(kwargs, wq_dequantized, page_size)


def reference(case):
    inp = case.kwargs
    kw = inp["x"].float() @ inp["wk_w"].float().t()
    key, raw_weights = kw[:, :HEAD_DIM], kw[:, HEAD_DIM:]
    if inp["k_norm_bias"] is not None:
        key = key - key.mean(-1, keepdim=True)
    key = key * (key.square().mean(-1, keepdim=True) + inp["k_norm_eps"]).rsqrt()
    key = key * inp["k_norm_weight"]
    if inp["k_norm_bias"] is not None:
        key = key + inp["k_norm_bias"]
    _, _, q_lora = quantize_blocks(inp["q_lora"], 32)
    q = (q_lora @ case.wq_dequantized.t()).reshape(-1, HEADS, HEAD_DIM)
    key = rope(
        key, inp["rope_cos_sin_cache"], inp["positions"], inp.get("is_neox_style", True)
    )
    q = rope(
        q, inp["rope_cos_sin_cache"], inp["positions"], inp.get("is_neox_style", True)
    )
    if inp["apply_hadamard"]:
        key, q = hadamard(key), hadamard(q)
    weight_scale = inp.get("weight_scale", 0) or DEFAULT_WEIGHT_SCALE
    if inp.get("output_dtype_str", "fp8") == "mxfp4":
        key_codes, key_scales = quantize_mxfp4(key)
        q_codes, q_scales = quantize_mxfp4(q)
        weights = raw_weights * weight_scale
    else:
        key_codes, key_scales = quantize_rows(key)
        q_codes, q_scales = quantize_rows(q)
        weights = raw_weights * weight_scale * q_scales
    return dict(
        q=q_codes,
        q_scales=q_scales,
        weights=weights,
        key=key_codes,
        key_scales=key_scales,
        q_unquantized=q.to(torch.bfloat16),
        key_unquantized=key.to(torch.bfloat16),
    )


def relative_rms(actual, expected):
    diff = (actual.float() - expected.float()).square().mean().sqrt()
    norm = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return (diff / norm).item()


def check_rms(actual, expected, tolerance):
    torch.testing.assert_close(
        relative_rms(actual, expected), 0.0, rtol=0, atol=tolerance
    )


def prepare_case(case):
    if "workspace" not in case.kwargs:
        case.kwargs["workspace"] = indexer_prepare_workspace(
            case.kwargs["x"].shape[0], case.kwargs["x"].device
        )
    return case


def output_tensors(result):
    """Flatten the public result for byte comparisons, omitting FP8's None."""
    (query, scales), weights = result
    return (query, weights) if scales is None else (query, scales, weights)


def check_scale_exponents(actual, expected, unquantized):
    mismatch = actual != expected
    if not mismatch.any():
        return
    # Only BF16 neighbors straddling 6 * 2**exponent may select different scales.
    difference = (actual.to(torch.int16) - expected.to(torch.int16)).abs()
    assert (difference[mismatch] == 1).all()
    maxima = unquantized.float().reshape(*actual.shape, 32).abs().amax(-1)
    smaller_exponent = torch.minimum(actual, expected).float() - 127
    boundary = 6 * torch.exp2(smaller_exponent)
    bf16_step = torch.exp2(torch.floor(torch.log2(maxima.clamp_min(1e-10))) - 7)
    assert ((maxima - boundary).abs()[mismatch] <= bf16_step[mismatch]).all()


def native_call(case, query, scales, weights, **options):
    inp = case.kwargs
    halops.halops_kernel.indexer_prepare(
        inp["x"],
        inp["q_lora"],
        inp["wk_w"],
        inp["wq_b"],
        inp["wq_b_sf"],
        inp["k_norm_weight"],
        inp["k_norm_bias"],
        inp["k_norm_eps"],
        inp["rope_cos_sin_cache"],
        inp["positions"],
        case.cache,
        inp["out_cache_loc"],
        query,
        scales,
        weights,
        inp["workspace"],
        HEAD_DIM**-0.5 * HEADS**-0.5,
        HEAD_DIM,
        HEADS,
        inp["apply_hadamard"],
        False,
        **options,
    )


def check(case, result=None, *, skip_q=False):
    prepare_case(case)
    expected = reference(case)
    (query, scales), weights = (
        indexer_prepare(**case.kwargs) if result is None else result
    )
    key, key_scales, valid = case.read_cache()
    key_ref, key_scales_ref = expected["key"][valid], expected["key_scales"][valid]
    fp4 = case.kwargs.get("output_dtype_str", "fp8") == "mxfp4"
    tolerance = 2e-2 if fp4 else FP8_RMS_TOL
    if fp4:
        assert query.dtype == scales.dtype == torch.uint8
        assert query.shape == (case.kwargs["x"].shape[0], HEADS, HEAD_DIM // 2)
        assert scales.shape == (*query.shape[:-1], HEAD_DIM // 32)
        assert (key_scales != 255).all()
        check_scale_exponents(
            key_scales, key_scales_ref, expected["key_unquantized"][valid]
        )
        key = dequantize_mxfp4(key, key_scales)
        key_ref = dequantize_mxfp4(key_ref, key_scales_ref)
        if not skip_q:
            assert (scales != 255).all()
            check_scale_exponents(
                scales, expected["q_scales"], expected["q_unquantized"]
            )
            query = dequantize_mxfp4(query, scales)
            query_ref = dequantize_mxfp4(expected["q"], expected["q_scales"])
    else:
        assert scales is None and query.dtype == FP8
        assert query.shape == (case.kwargs["x"].shape[0], HEADS, HEAD_DIM)
        if key.numel():
            check_rms(key_scales, key_scales_ref, SCALE_RMS_TOL)
        key, key_ref = key.float(), key_ref.float()
        query, query_ref = query.float(), expected["q"].float()
    if key.numel():
        check_rms(key, key_ref, tolerance)
    if not skip_q:
        assert torch.isfinite(query).all() and torch.isfinite(weights).all()
        check_rms(query, query_ref, tolerance)
        check_rms(weights, expected["weights"], SCALE_RMS_TOL)
        if key.numel():
            if not fp4:
                key, key_ref = (
                    key * key_scales[:, None],
                    key_ref * key_scales_ref[:, None],
                )
            scores = torch.einsum("thd,kd->thk", query, key[-8:]).relu()
            scores_ref = torch.einsum("thd,kd->thk", query_ref, key_ref[-8:]).relu()
            check_rms(
                (scores * weights[..., None]).sum(1),
                (scores_ref * expected["weights"][..., None]).sum(1),
                tolerance,
            )
    assert (case.cache[~case.written_bytes()] == 0xA5).all()


def assert_same_outputs(actual, expected):
    for value, target in zip(
        output_tensors(actual), output_tensors(expected), strict=True
    ):
        assert torch.equal(value.view(torch.uint8), target.view(torch.uint8))


@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4"])
@pytest.mark.parametrize(
    "tokens",
    [
        *range(1, 18),
        31,
        32,
        33,
        64,
        65,
        96,
        97,
        128,
        129,
        256,
        257,
        300,
        400,
        401,
        425,
        512,
        513,
        1024,
        2048,
    ],
)
def test_dispatch_and_mutable_graph(tokens, output_dtype):
    case = prepare_case(
        make_case(
            tokens,
            output_dtype_str=output_dtype,
            is_neox_style=tokens % 2 == 0,
            shuffle=True,
            sentinels=tokens > 1,
        )
    )
    check(case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = indexer_prepare(**case.kwargs)
    for _ in range(2):
        for name in ("x", "q_lora", "k_norm_weight", "k_norm_bias"):
            case.kwargs[name].normal_()
        case.kwargs["positions"].copy_(case.kwargs["positions"].roll(1))
        case.cache.fill_(0xA5)
        graph.replay()
        check(case, actual)


@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4"])
@pytest.mark.parametrize(
    "tokens,options",
    [
        (1, {"is_neox_style": True}),
        (2, {"is_neox_style": False}),
        (5, {"bias": False}),
        (17, {"bias": False}),
        (129, {"bias": False}),
        (1, {"apply_hadamard": False}),
        (9, {"apply_hadamard": False}),
        (129, {"apply_hadamard": False}),
        (5, {"page_size": 1, "shuffle": True, "sentinels": True}),
        (17, {"page_size": 32, "shuffle": True, "index_dtype": torch.int32}),
        (129, {"page_size": 128, "shuffle": True}),
        (1, {"varied_groups": True}),
        (17, {"varied_groups": True}),
    ],
)
def test_options(output_dtype, tokens, options):
    check(make_case(tokens, output_dtype_str=output_dtype, **options))


@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4"])
@pytest.mark.parametrize("tokens", [1, 5, 17, 129])
def test_preallocated_skip_query(tokens, output_dtype):
    case = prepare_case(make_case(tokens, output_dtype_str=output_dtype, shuffle=True))
    case.kwargs["weight_scale"] = 0.125
    (query, scales), weights = indexer_prepare(**case.kwargs)
    buffers = {"q": query, "sf_q": scales, "weights": weights}
    query.view(torch.uint8).fill_(0)
    if scales is not None:
        scales.fill_(127)
    weights.fill_(19)
    saved = tuple(value.clone() for value in output_tensors(((query, scales), weights)))
    case.cache.fill_(0xA5)
    actual = indexer_prepare(**case.kwargs, **buffers, skip_q=True)
    for value, target in zip(output_tensors(actual), saved, strict=True):
        assert torch.equal(value.view(torch.uint8), target.view(torch.uint8))
    check(case, actual, skip_q=True)
    actual = indexer_prepare(**case.kwargs, **buffers)
    assert actual[0][0] is query and actual[0][1] is scales and actual[1] is weights
    check(case, actual)


@pytest.mark.parametrize("tokens", [1, 5, 17, 129])
def test_aligned_strided_activations(tokens):
    case = make_case(tokens)
    for name, extra in [("x", 8), ("q_lora", 4)]:
        original = case.kwargs[name]
        storage = torch.empty(
            (original.shape[0], original.shape[1] + extra),
            dtype=original.dtype,
            device=original.device,
        )
        sliced = storage[:, : original.shape[1]]
        sliced.copy_(original)
        case.kwargs[name] = sliced
    check(case)


@pytest.mark.parametrize("output_dtype_str", ["fp8", "mxfp4"])
def test_neox_default_and_explicit_false(output_dtype_str):
    case = prepare_case(make_case(8, output_dtype_str=output_dtype_str))
    del case.kwargs["is_neox_style"]
    default = indexer_prepare(**case.kwargs)
    default_cache = case.cache.clone()
    explicit = indexer_prepare(**case.kwargs, is_neox_style=True)
    assert_same_outputs(default, explicit)
    assert torch.equal(default_cache, case.cache)
    interleaved = indexer_prepare(**case.kwargs, is_neox_style=False)
    assert not torch.equal(
        default[0][0].view(torch.uint8), interleaved[0][0].view(torch.uint8)
    )
    assert not torch.equal(default_cache, case.cache)


def test_fp8_default_output_contract():
    case = prepare_case(make_case(8))
    del case.kwargs["output_dtype_str"]
    (query, scales), weights = indexer_prepare(**case.kwargs)
    assert scales is None
    check(case, ((query, scales), weights))
    explicit = indexer_prepare(**case.kwargs, output_dtype_str="fp8")
    native_query = torch.empty_like(query)
    native_weights = torch.empty_like(weights)
    # Both Python and the native binding default to FP8 and NeoX.
    native_call(case, native_query, None, native_weights)
    assert torch.equal(native_query.view(torch.uint8), query.view(torch.uint8))
    assert torch.equal(native_weights, weights)
    assert_same_outputs(explicit, ((query, scales), weights))


@pytest.mark.parametrize("output_dtype_str", ["fp8", "mxfp4"])
@pytest.mark.parametrize("neox", [None, True, False])
def test_native_rope_flag(output_dtype_str, neox):
    case = prepare_case(
        make_case(8, output_dtype_str=output_dtype_str, is_neox_style=neox is not False)
    )
    expected = indexer_prepare(**case.kwargs)
    query = torch.empty_like(expected[0][0])
    scales = torch.empty_like(expected[0][1]) if output_dtype_str == "mxfp4" else None
    weights = torch.empty_like(expected[-1])
    options = dict(output_dtype_str=output_dtype_str)
    if neox is not None:
        options["is_neox_style"] = neox
    native_call(case, query, scales, weights, **options)
    actual = ((query, scales), weights)
    assert_same_outputs(actual, expected)


@pytest.mark.parametrize(
    "output_dtype_str,buffer_mask",
    [("fp8", mask) for mask in (0, 1, 4, 5)] + [("mxfp4", mask) for mask in range(8)],
)
def test_independently_preallocated_outputs(output_dtype_str, buffer_mask):
    case = prepare_case(make_case(4, output_dtype_str=output_dtype_str))
    expected = indexer_prepare(**case.kwargs)
    expected_cache = case.cache.clone()
    (query, scales), weights = expected
    buffers = dict(
        q=torch.empty_like(query) if buffer_mask & 1 else None,
        sf_q=(
            torch.empty_like(scales) if buffer_mask & 2 and scales is not None else None
        ),
        weights=torch.empty_like(weights) if buffer_mask & 4 else None,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = indexer_prepare(**case.kwargs, **buffers)
    graph.replay()
    (actual_query, actual_scales), actual_weights = result
    for name, actual in zip(
        ("q", "sf_q", "weights"), (actual_query, actual_scales, actual_weights)
    ):
        if buffers[name] is not None:
            assert actual is buffers[name]
    if output_dtype_str == "fp8":
        assert actual_scales is None
    assert_same_outputs(result, expected)
    assert torch.equal(case.cache, expected_cache)
    case.kwargs["q_lora"].mul_(-0.5)
    expected = indexer_prepare(**case.kwargs)
    graph.replay()
    assert_same_outputs(result, expected)


@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4"])
@pytest.mark.parametrize("tokens", [1, 3, 5, 9, 17, 33, 97, 129, 257, 300, 512])
def test_low_precision_weights(tokens, output_dtype, weights_dtype):
    # The token counts reach every weights store of both formats.
    case = prepare_case(make_case(tokens, output_dtype_str=output_dtype))
    (expected_query, expected_scales), expected = indexer_prepare(**case.kwargs)
    assert expected.dtype == torch.float32
    expected = expected.to(weights_dtype)
    allocated = indexer_prepare(**case.kwargs, output_weights_dtype=weights_dtype)
    supplied = torch.full_like(expected, float("nan"))
    preallocated = indexer_prepare(**case.kwargs, weights=supplied)
    assert preallocated[1] is supplied
    for (query, scales), weights in (allocated, preallocated):
        assert weights.dtype == weights_dtype
        assert torch.equal(weights, expected)
        assert torch.equal(query.view(torch.uint8), expected_query.view(torch.uint8))
        if scales is not None:
            assert torch.equal(scales, expected_scales)


def test_output_weights_dtype_validation():
    case = prepare_case(make_case(4))
    weights = torch.empty((4, HEADS), dtype=torch.bfloat16, device="cuda")
    result = indexer_prepare(
        **case.kwargs, weights=weights, output_weights_dtype=torch.bfloat16
    )
    assert result[1] is weights
    allocated = indexer_prepare(**case.kwargs, output_weights_dtype=torch.float32)
    assert allocated[1].dtype == torch.float32
    with pytest.raises(ValueError, match="output_weights_dtype"):
        indexer_prepare(**case.kwargs, weights=weights, output_weights_dtype=torch.half)
    with pytest.raises(ValueError, match="output_weights_dtype"):
        indexer_prepare(**case.kwargs, output_weights_dtype=torch.float64)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "cpu", "strided", "missing"])
def test_native_rejects_invalid_fp4_scales(invalid):
    case = prepare_case(make_case(4, output_dtype_str="mxfp4"))
    query = torch.empty((4, HEADS, HEAD_DIM // 2), dtype=torch.uint8, device="cuda")
    scales = torch.empty((4, HEADS, 4), dtype=torch.uint8, device="cuda")
    if invalid == "shape":
        scales = scales[:, :, :2].contiguous()
    elif invalid == "dtype":
        scales = scales.float()
    elif invalid == "cpu":
        scales = scales.cpu()
    elif invalid == "missing":
        scales = None
    else:
        scales = torch.empty((4, HEADS, 8), dtype=torch.uint8, device="cuda")[:, :, ::2]
    with pytest.raises(RuntimeError, match="sf_q"):
        native_call(
            case,
            query,
            scales,
            torch.empty((4, HEADS), device="cuda"),
            output_dtype_str="mxfp4",
        )


@pytest.mark.parametrize("output_dtype_str", ["fp8", "mxfp4"])
@pytest.mark.parametrize("name", ["q", "weights"])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "cpu", "strided"])
def test_rejects_invalid_output_buffers(output_dtype_str, name, invalid):
    case = prepare_case(make_case(4, output_dtype_str=output_dtype_str))
    query_size = HEAD_DIM // 2 if output_dtype_str == "mxfp4" else HEAD_DIM
    shape = (4, HEADS, query_size) if name == "q" else (4, HEADS)
    dtype = (
        (torch.uint8 if output_dtype_str == "mxfp4" else torch.float8_e4m3fn)
        if name == "q"
        else torch.float32
    )
    if invalid == "shape":
        output = torch.empty((3, *shape[1:]), dtype=dtype, device="cuda")
    elif invalid == "dtype":
        output = torch.empty(shape, dtype=torch.float64, device="cuda")
    elif invalid == "cpu":
        output = torch.empty(shape, dtype=dtype, device="cpu")
    else:
        output = torch.empty((*shape[:-1], shape[-1] * 2), dtype=dtype, device="cuda")[
            ..., ::2
        ]
    with pytest.raises(RuntimeError, match=name):
        indexer_prepare(**case.kwargs, **{name: output})


def test_fp8_rejects_separate_scales():
    case = prepare_case(make_case(4))
    query = torch.empty((4, HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty((4, HEADS, 4), dtype=torch.uint8, device="cuda")
    weights = torch.empty((4, HEADS), device="cuda")
    with pytest.raises(ValueError, match="sf_q"):
        indexer_prepare(**case.kwargs, q=query, sf_q=scales, weights=weights)
    with pytest.raises(RuntimeError, match="sf_q"):
        native_call(case, query, scales, weights)


@pytest.mark.parametrize("output_dtype_str", ["fp8", "mxfp4"])
def test_rejects_cache_for_wrong_format(output_dtype_str):
    case = prepare_case(make_case(4, output_dtype_str=output_dtype_str))
    wrong_bytes = 132 if output_dtype_str == "mxfp4" else 68
    case.kwargs["paged_kv_cache_with_scale"] = torch.empty(
        (2, 64 * wrong_bytes), dtype=torch.uint8, device="cuda"
    )
    with pytest.raises(RuntimeError, match="page_size"):
        indexer_prepare(**case.kwargs)


@pytest.mark.parametrize("invalid", ["fp4", "fp16", "MXFP4", ""])
def test_rejects_output_format_arguments(invalid):
    case = prepare_case(make_case(4))
    case.kwargs["output_dtype_str"] = invalid
    with pytest.raises(ValueError, match="output_dtype_str"):
        indexer_prepare(**case.kwargs)
    with pytest.raises(RuntimeError, match="output_dtype_str"):
        native_call(
            case,
            torch.empty((4, HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device="cuda"),
            None,
            torch.empty((4, HEADS), device="cuda"),
            output_dtype_str=invalid,
        )


@pytest.mark.parametrize("name", ["x", "q_lora", "wk_w", "positions", "wq_b_sf"])
def test_rejects_cpu_inputs(name):
    case = prepare_case(make_case(4))
    case.kwargs[name] = case.kwargs[name].cpu()
    with pytest.raises((RuntimeError, AssertionError), match="CUDA"):
        indexer_prepare(**case.kwargs)


@pytest.mark.parametrize(
    "name",
    ["k_norm_weight", "k_norm_bias", "wq_b_sf", "paged_kv_cache_with_scale"],
)
def test_rejects_strided_inputs(name):
    case = prepare_case(make_case(4))
    original = case.kwargs[name]
    storage = torch.empty(
        (*original.shape[:-1], original.shape[-1] * 2),
        dtype=original.dtype,
        device=original.device,
    )
    sliced = storage[..., ::2]
    sliced.copy_(original)
    case.kwargs[name] = sliced
    with pytest.raises(RuntimeError, match="contiguous"):
        indexer_prepare(**case.kwargs)


@pytest.mark.parametrize("name,extra", [("x", 1), ("q_lora", 1)])
def test_rejects_unaligned_row_stride(name, extra):
    case = prepare_case(make_case(4))
    original = case.kwargs[name]
    case.kwargs[name] = torch.empty(
        (4, original.shape[1] + extra), dtype=torch.bfloat16, device="cuda"
    )[:, : original.shape[1]]
    with pytest.raises(RuntimeError, match="row stride"):
        indexer_prepare(**case.kwargs)


@pytest.mark.parametrize("eps", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_epsilon(eps):
    case = prepare_case(make_case(4))
    case.kwargs["k_norm_eps"] = eps
    with pytest.raises(RuntimeError, match="k_norm_eps"):
        indexer_prepare(**case.kwargs)


@pytest.mark.parametrize("tokens", [0, -1, 2**31, 2**32 + 1])
def test_workspace_query_rejects_overflow(tokens):
    with pytest.raises(RuntimeError, match="num_tokens"):
        indexer_prepare_workspace_bytes(tokens)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"head_dim": 64}, "head_dim=128"),
        ({"n_heads": 32}, "n_heads=64"),
    ],
)
def test_rejects_unsupported_options(options, message):
    case = prepare_case(make_case(4))
    with pytest.raises(AssertionError, match=message):
        indexer_prepare(**case.kwargs, **options)


@pytest.mark.parametrize("row", [96 * 132, 64 * 132 + 1])
def test_rejects_invalid_cache_layout(row):
    case = prepare_case(make_case(4))
    case.kwargs["paged_kv_cache_with_scale"] = torch.empty(
        (4, row), dtype=torch.uint8, device="cuda"
    )
    with pytest.raises(RuntimeError, match="power of two|page_size"):
        indexer_prepare(**case.kwargs)


@pytest.mark.parametrize(
    "name,dtype,message",
    [
        ("x", torch.float32, "bfloat16"),
        ("q_lora", torch.float32, "bfloat16"),
        ("wk_w", FP8, "bf16"),
        ("wq_b", torch.bfloat16, "float8_e4m3fn"),
        ("positions", torch.float32, "int32 or int64"),
    ],
)
def test_rejects_wrong_dtype(name, dtype, message):
    case = prepare_case(make_case(4))
    case.kwargs[name] = case.kwargs[name].to(dtype)
    with pytest.raises((RuntimeError, AssertionError), match=message):
        indexer_prepare(**case.kwargs)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_rejects_mixed_cuda_devices():
    case = prepare_case(make_case(4))
    case.kwargs["positions"] = case.kwargs["positions"].to("cuda:1")
    with pytest.raises(RuntimeError, match="same CUDA device"):
        indexer_prepare(**case.kwargs)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_noncurrent_device():
    case = prepare_case(make_case(4, device="cuda:1"))
    previous = torch.cuda.current_device()
    result = indexer_prepare(**case.kwargs)
    assert torch.cuda.current_device() == previous
    assert result[0][0].device == torch.device("cuda:1")
    assert result[0][1] is None
    check(case, result)


@pytest.mark.parametrize("invalid", ["small", "strided", "unaligned", "cpu"])
def test_native_workspace_validation(invalid):
    case = make_case(4)
    inp = case.kwargs
    size, prefix = halops.halops_kernel.indexer_prepare_workspace_bytes(4)
    workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
    workspace[:prefix].zero_()
    if invalid == "small":
        workspace = workspace[:128]
    elif invalid == "strided":
        workspace = workspace[::2]
    elif invalid == "unaligned":
        workspace = workspace[1:]
    else:
        workspace = workspace.cpu()
    with pytest.raises(RuntimeError, match="workspace"):
        halops.halops_kernel.indexer_prepare(
            inp["x"],
            inp["q_lora"],
            inp["wk_w"],
            inp["wq_b"],
            inp["wq_b_sf"],
            inp["k_norm_weight"],
            inp["k_norm_bias"],
            inp["k_norm_eps"],
            inp["rope_cos_sin_cache"],
            inp["positions"],
            case.cache,
            inp["out_cache_loc"],
            torch.empty((4, HEADS, HEAD_DIM), dtype=FP8, device="cuda"),
            None,
            torch.empty((4, HEADS), device="cuda"),
            workspace,
            1.0,
            HEAD_DIM,
            HEADS,
            True,
            False,
        )


def test_halops_quantizer_compatibility():
    case = make_case(64)
    weight = torch.randn((HEADS * HEAD_DIM, 1024), dtype=torch.bfloat16, device="cuda")

    _, _, case.wq_dequantized = quantize_blocks(weight, 32)
    case.kwargs["wq_b"], case.kwargs["wq_b_sf"] = halops.group_wise_quant(
        weight, mode="w8a8-mxfp8", use_swizzle=True
    )
    check(case)


def test_neox_reference_known_pairs():
    # A quarter turn: (x[i], x[i+32]) -> (-x[i+32], x[i]).
    values = torch.arange(128, dtype=torch.float32).reshape(1, 128)
    table = torch.cat((torch.zeros(1, 32), torch.ones(1, 32)), dim=-1)
    actual = rope(values, table, torch.zeros(1, dtype=torch.long))
    torch.testing.assert_close(actual[:, :32], -values[:, 32:64])
    torch.testing.assert_close(actual[:, 32:64], values[:, :32])
    torch.testing.assert_close(actual[:, 64:], values[:, 64:])


def test_fp4_exact_quantization_and_zero_groups():
    # Zero Q projection makes every FP4 byte and scale deterministic on-device.
    # Constant K bias exercises signs, midpoint rounding and distinct block scales.
    case = prepare_case(make_case(8, output_dtype_str="mxfp4", apply_hadamard=False))
    case.kwargs["wq_b"].zero_()
    case.wq_dequantized.zero_()
    case.kwargs["wk_w"][:HEAD_DIM].zero_()
    pattern = torch.tensor(
        [0, 0.25, -0.25, 0.75, -0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -6, 1, 2, 3, 4],
        device="cuda",
    ).repeat(2)
    bias = torch.cat((pattern, pattern * 2, pattern / 4, torch.zeros_like(pattern)))
    case.kwargs["k_norm_bias"].copy_(bias)
    case.kwargs["rope_cos_sin_cache"][:, :32] = 1
    case.kwargs["rope_cos_sin_cache"][:, 32:] = 0
    (query, scales), weights = indexer_prepare(**case.kwargs)
    expected = reference(case)
    assert torch.equal(query, expected["q"])
    assert torch.equal(scales, expected["q_scales"])
    key, key_scales, valid = case.read_cache()
    assert torch.equal(key, expected["key"][valid])
    assert torch.equal(key_scales, expected["key_scales"][valid])
    # Scale=1 isolates E2M1 encoding from approximate-log2 boundary errors.
    unit_scale_bias = torch.cat((pattern, pattern, pattern, torch.zeros_like(pattern)))
    case.kwargs["k_norm_bias"].copy_(unit_scale_bias)
    result = indexer_prepare(**case.kwargs)
    key, key_scales, _ = case.read_cache()
    quantized, quant_scales = halops.group_wise_quant(
        unit_scale_bias.to(torch.bfloat16).expand(8, -1).contiguous(),
        mode="w4a4-mxfp4",
        use_swizzle=False,
    )
    assert torch.equal(key, quantized)
    assert torch.equal(key_scales, quant_scales[:8])
    check(case, result)


@pytest.mark.parametrize(
    "maxima",
    [
        [1.5, 3.0, 6.0, 12.0],
        [3.0 * 2**-20, 3.0 * 2**-10, 3.0 * 2**10, 3.0 * 2**20],
        [2.984375, 3.0, 3.015625, 0.0],
    ],
)
def test_fp4_scale_power_of_two_boundaries(maxima):
    case = prepare_case(make_case(8, output_dtype_str="mxfp4", apply_hadamard=False))
    case.kwargs["wk_w"][:HEAD_DIM].zero_()
    case.kwargs["k_norm_bias"].copy_(
        torch.tensor(maxima, device="cuda").repeat_interleave(32)
    )
    case.kwargs["rope_cos_sin_cache"][:, :32] = 1
    case.kwargs["rope_cos_sin_cache"][:, 32:] = 0
    indexer_prepare(**case.kwargs, skip_q=True)
    key, scales, valid = case.read_cache()
    expected = reference(case)
    assert torch.equal(scales, expected["key_scales"][valid])
    assert torch.equal(key, expected["key"][valid])


@pytest.mark.parametrize("tokens", [5, 33, 425])
@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4"])
def test_eight_byte_aligned_rope_table(tokens, output_dtype):
    case = make_case(tokens, output_dtype_str=output_dtype)
    original = case.kwargs["rope_cos_sin_cache"]
    storage = torch.empty(
        original.numel() + 2, dtype=original.dtype, device=original.device
    )
    table = storage[2:].view_as(original)
    table.copy_(original)
    assert table.data_ptr() % 16 == 8
    case.kwargs["rope_cos_sin_cache"] = table
    check(case)


@pytest.mark.parametrize("tokens", [144, 512])
@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4"])
def test_minimum_output_alignment_graph(tokens, output_dtype):
    case = prepare_case(make_case(tokens, output_dtype_str=output_dtype))
    (query, scales), _ = indexer_prepare(**case.kwargs)
    storage = torch.empty(query.numel() + 16, dtype=torch.uint8, device=query.device)
    query_output = storage[16:].view(query.dtype).view_as(query)
    assert query_output.data_ptr() % 32 == 16
    case.kwargs["q"] = query_output
    if scales is not None:
        scale_storage = torch.empty(
            scales.numel() + 1, dtype=torch.uint8, device=scales.device
        )
        case.kwargs["sf_q"] = scale_storage[1:].view_as(scales)
        assert case.kwargs["sf_q"].data_ptr() % 4 == 1
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = indexer_prepare(**case.kwargs)
    for _ in range(2):
        case.kwargs["q_lora"].normal_()
        graph.replay()
        torch.cuda.synchronize()
        assert actual[0][0] is query_output
        check(case, actual)


@pytest.mark.parametrize("tokens", [5, 32, 128, 512])
@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4"])
def test_pdl_quantized_weight_producer_graph(tokens, output_dtype):
    case = prepare_case(make_case(tokens, output_dtype_str=output_dtype))
    weight_input = torch.randn(8192, 1024, dtype=torch.bfloat16, device="cuda")

    def chain(use_pdl):
        weights, scales = halops.group_wise_quant(weight_input, use_pdl=use_pdl)
        arguments = dict(case.kwargs, wq_b=weights, wq_b_sf=scales, use_pdl=use_pdl)
        return indexer_prepare(**arguments)

    chain(True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = chain(True)
    for _ in range(3):
        # Changed producer data exposes stale reads after capture or replay.
        weight_input.normal_()
        expected = chain(False)
        expected_cache = case.cache.clone()
        graph.replay()
        torch.cuda.synchronize()
        assert_same_outputs(actual, expected)
        assert torch.equal(case.cache, expected_cache)


@pytest.mark.parametrize("tokens", [5, 16])
@pytest.mark.parametrize("split", [4, 16])
@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4"])
def test_small_consumers_cover_multiple_tokens(
    monkeypatch, tokens, split, output_dtype
):
    # Multiple rows per consumer must finish before its readiness counter resets.
    monkeypatch.setenv("HALOPS_INDEXER_PREPARE_GRID", str(64 + 2 * split))
    monkeypatch.setenv("HALOPS_INDEXER_PREPARE_KW_KSPLIT", str(split))
    case = prepare_case(make_case(tokens, output_dtype_str=output_dtype))
    check(case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = indexer_prepare(**case.kwargs)
    for _ in range(3):
        case.kwargs["x"].normal_()
        graph.replay()
        torch.cuda.synchronize()
        check(case, result)


@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4", "mixed"])
@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
def test_workspace_direct_shape_transitions(output_dtype, use_pdl, use_graph):
    # No intervening generic/skip-Q launch may mask small -> large transitions.
    shapes = [16, 129, 5, 257, 8, 425, 9, 1024, 1, 2, 300, 512]
    workspace = indexer_prepare_workspace(max(shapes), "cuda")
    _, prefix = indexer_prepare_workspace_bytes(max(shapes))
    cases = []
    for index, tokens in enumerate(shapes):
        mode = (
            ("fp8" if index % 2 == 0 else "mxfp4")
            if output_dtype == "mixed"
            else output_dtype
        )
        case = make_case(
            tokens,
            output_dtype_str=mode,
            is_neox_style=index % 2 == 0,
            bias=index % 3 != 0,
        )
        case.kwargs.update(workspace=workspace, use_pdl=use_pdl)
        # Warm allocation/capture without touching the shared buffer.
        indexer_prepare(
            **{**case.kwargs, "workspace": indexer_prepare_workspace(tokens, "cuda")}
        )
        cases.append(case)

    if use_graph:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = [indexer_prepare(**case.kwargs) for case in cases]

    for _ in range(3):
        for case in cases:
            for name in ("x", "q_lora", "wk_w", "k_norm_weight"):
                case.kwargs[name].normal_()
            if case.kwargs["k_norm_bias"] is not None:
                case.kwargs["k_norm_bias"].normal_()
            case.kwargs["wq_b"].copy_(-case.kwargs["wq_b"].float())
            case.kwargs["wq_b_sf"].add_(1)
            case.wq_dequantized.mul_(-2)
            case.cache.fill_(0xA5)
        if use_graph:
            graph.replay()
        else:
            results = [indexer_prepare(**case.kwargs) for case in cases]
        torch.cuda.synchronize()
        # Active and inactive counters must all remain zero after each call.
        assert torch.count_nonzero(workspace[:prefix]).item() == 0
        for case, result in zip(cases, results):
            check(case, result)


@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4"])
@pytest.mark.parametrize("tokens", [1, 16, 129])
@pytest.mark.parametrize("amplitude", [0.0, 1e-8, 1e-6])
def test_mxfp8_input_floor(output_dtype, tokens, amplitude):
    case = prepare_case(make_case(tokens, output_dtype_str=output_dtype))
    case.kwargs["q_lora"].mul_(amplitude)
    check(case)


def test_mxfp8_group_scale_boundaries():
    # Independent exponent expectations at exact powers of two and BF16
    # neighbors; group magnitudes vary across the row, including zero/tiny.
    boundary = torch.tensor(
        [0.0, 1e-11, 1e-8, 448 * 2.0**-12, 448 * 2.0**-4],
        device="cuda",
        dtype=torch.bfloat16,
    )
    values = torch.stack(
        (
            torch.nextafter(boundary, torch.zeros_like(boundary)),
            boundary,
            torch.nextafter(boundary, torch.full_like(boundary, float("inf"))),
        )
    )
    x = values.reshape(1, -1, 1).expand(1, -1, 32).reshape(1, -1)
    _, exponents, decoded = quantize_blocks(x, 32)
    expected = torch.tensor(
        [[85, 85, 92, 115, 123], [85, 85, 92, 115, 123], [85, 85, 92, 116, 124]],
        dtype=torch.uint8,
        device="cuda",
    ).reshape(1, -1)
    assert torch.equal(exponents, expected)
    assert torch.isfinite(decoded).all()


@pytest.mark.parametrize(
    "setting,value",
    [
        ("GRID", "-1"),
        ("GRID", "128junk"),
        ("GRID", ""),
        ("GRID", "9999999999999999999999"),
        ("KW_KSPLIT", "3"),
        ("KW_KSPLIT", "32"),
        ("PHASES", "2"),
        ("WQB_ROT", "2"),
        ("ACT_BOUND", "true"),
        ("STORE256", "2"),
        ("PACKED_SCALES", "-1"),
        ("OUTPUT_NO_ALLOCATE", "2"),
        ("SMALL_ROPE", "-1"),
        ("SMALL_OPT", "2"),
        ("RAW_WEIGHT_SMEM", "3"),
        ("KW_CTAS", "-1"),
        ("TWO_PASS_QUERY", "2"),
    ],
)
def test_invalid_diagnostic_settings(monkeypatch, setting, value):
    name = f"HALOPS_INDEXER_PREPARE_{setting}"
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        indexer_prepare_workspace_bytes(16)


@pytest.mark.parametrize("split", [1, 16])
def test_workspace_capacity_with_split_override(monkeypatch, split):
    monkeypatch.setenv("HALOPS_INDEXER_PREPARE_KW_KSPLIT", str(split))
    workspace = indexer_prepare_workspace(513, "cuda")
    for tokens in [16, 129, 300, 513]:
        case = make_case(tokens)
        case.kwargs["workspace"] = workspace
        check(case)


def test_activation_bounds_setting_is_not_cached(monkeypatch):
    case = prepare_case(make_case(129))
    for setting in ("1", "0", "1"):
        monkeypatch.setenv("HALOPS_INDEXER_PREPARE_ACT_BOUND", setting)
        check(case)


def test_capacity_envelope():
    previous_size, previous_prefix = 0, 0
    for tokens in range(1, 4097):
        size, prefix = indexer_prepare_workspace_bytes(tokens)
        assert previous_size <= size and previous_prefix <= prefix <= size
        previous_size, previous_prefix = size, prefix
