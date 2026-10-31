"""Compare SGLang dual stream, HALOps dual stream and fused indexer_prepare.

Default: every batch size in range(1, 513), FP8 and MXFP4, cold L2, CUPTI off,
--timing timed_graph. TimedGraph captures --graph-iters calls (default 16) in
one CUDA Graph, each between its own CUDA events and preceded by an L2 flush
outside them; a sample is the mean per-call interval of one replay, excluding
the graph launch. --timing bench_gpu_time instead surrounds the complete replay
with flashinfer.testing.bench_gpu_time's CUDA events and flushes L2 before each
replay. --l2-cache hot disables eviction for both.
HALOps dual stream uses group_wise_quant + dense_gemm for Q projection and
SGLang's EB5 dual-stream schedule/epilogues. All three paths share BF16 inputs
and original weights, but use their own quantization/rounding references.

Source the project's env.sh before running. No historical benchmark/test
helpers are imported. Workspace is allocated once for the largest batch.
"""

import argparse
import csv
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from types import MethodType, SimpleNamespace

PROJECT_ROOT = Path(
    os.environ.get(
        "DEV_OPS_ROOT", "/root/paddlejob/workspace/env_run/output/chenkailun/dev_ops"
    )
)
for directory in (PROJECT_ROOT / "sglang/python", PROJECT_ROOT / "HALOps"):
    sys.path.insert(0, str(directory))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "sglang.srt.internal.models.eb5")
os.environ.setdefault("SGLANG_PLUGINS", "eb5_internal_runtime_patch")
os.environ.setdefault("SGLANG_INTERNAL_EB5_DSA_INDEXER_FUSION", "1")
os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "1")
# Skip SGLang's DeepGEMM warmup over M=1..16384 on the first GEMM call.
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")

import torch
import torch.nn.functional as functional
import triton
import triton.language as tl
import halops
from halops import indexer_prepare, indexer_prepare_workspace
from flashinfer.testing import bench_gpu_time

HIDDEN, RANK, HEADS, HEAD_DIM, ROPE_DIM, PAGE_SIZE = 4096, 1024, 64, 128, 64, 64
WEIGHT_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
FP8 = torch.float8_e4m3fn
IMPLEMENTATIONS = ("sglang_dual_stream", "halops_dual_stream", "halops_indexer_prepare")


def group_fp8(values, group_size, floor=1e-10):
    groups = values.float().reshape(*values.shape[:-1], -1, group_size)
    exponents = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(floor) / 448))
    scales = torch.exp2(exponents)
    codes = (groups / scales[..., None]).clamp(-448, 448).to(FP8)
    return (
        codes.reshape(values.shape),
        (exponents + 127).to(torch.uint8),
        (codes.float() * scales[..., None]).reshape(values.shape),
    )


def make_weights(max_tokens, seed):
    torch.manual_seed(seed)
    projection = torch.randn(192, HIDDEN, dtype=torch.bfloat16, device="cuda")
    query_weight = torch.randn(
        HEADS * HEAD_DIM, RANK, dtype=torch.bfloat16, device="cuda"
    )
    codes, scales, decoded = group_fp8(query_weight, 32)
    # CUTLASS SfAtom: [N/128, K/128, 32, 4, 4], exposed as uint8 [N,K/32].
    swizzled = scales.reshape(64, 4, 32, 8, 4).permute(0, 3, 2, 1, 4).contiguous()
    angles = torch.arange(max_tokens + 17, device="cuda")[:, None] * torch.pow(
        10000.0, -torch.arange(0, ROPE_DIM, 2, device="cuda") / ROPE_DIM
    )
    return SimpleNamespace(
        projection=projection,
        query_weight=query_weight,
        codes=codes,
        scales=swizzled.reshape(HEADS * HEAD_DIM, RANK // 32),
        decoded=decoded,
        gain=torch.ones(HEAD_DIM, device="cuda"),
        bias=torch.zeros(HEAD_DIM, device="cuda"),
        rope=torch.cat((angles.cos(), angles.sin()), -1).contiguous(),
    )


def make_case(tokens, output_dtype, weights, workspace):
    torch.manual_seed(1000 + tokens)
    data_bytes = HEAD_DIM // 2 if output_dtype == "mxfp4" else HEAD_DIM
    cache_shape = (
        max(2, (tokens + PAGE_SIZE - 1) // PAGE_SIZE + 1),
        PAGE_SIZE * (data_bytes + 4),
    )
    return SimpleNamespace(
        tokens=tokens,
        output_dtype=output_dtype,
        static=weights,
        x=torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda"),
        q_lora=torch.randn(tokens, RANK, dtype=torch.bfloat16, device="cuda"),
        positions=torch.arange(tokens, device="cuda"),
        locations=torch.arange(tokens, device="cuda"),
        caches={
            name: torch.full(cache_shape, 0xA5, dtype=torch.uint8, device="cuda")
            for name in IMPLEMENTATIONS
        },
        workspace=workspace,
        neox=True,
        hadamard=True,
        bias=weights.bias,
    )


def call_halops(case, **options):
    weights = case.static
    return indexer_prepare(
        x=case.x,
        q_lora=case.q_lora,
        wk_w=weights.projection,
        wq_b=weights.codes,
        wq_b_sf=weights.scales,
        k_norm_weight=weights.gain,
        k_norm_bias=case.bias,
        k_norm_eps=1e-6,
        rope_cos_sin_cache=weights.rope,
        positions=case.positions,
        paged_kv_cache_with_scale=case.caches["halops_indexer_prepare"],
        out_cache_loc=case.locations,
        workspace=case.workspace,
        weight_scale=WEIGHT_SCALE,
        is_neox_style=case.neox,
        apply_hadamard=case.hadamard,
        output_dtype_str=case.output_dtype,
        **options,
    )


class DualStream:
    """Use the actual EB5 two-stream method; stub only model/cache construction."""

    def __init__(self, weights, implementation):
        self.implementation = implementation
        from sglang.jit_kernel.internal import dsa
        from sglang.srt.internal.layers.attention.mla_hca.dsa_full import (
            EB5FullMLAIndexer,
            _eb5_indexer_kernels,
        )
        from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
        from sglang.srt.layers.quantization.fp8_utils import (
            deepgemm_w8a8_block_fp8_linear_with_fallback,
            quant_weight_ue8m0,
            transform_scale_ue8m0,
        )

        self.indexer_type = EB5FullMLAIndexer
        self.method = Indexer._fused_q_prepare_and_store
        self.kernel_context = _eb5_indexer_kernels
        self.key_kernel = dsa.fused_k_indexer_norm_rope_store
        self.linear = deepgemm_w8a8_block_fp8_linear_with_fallback
        if implementation == "sglang_dual_stream":
            self.codes, block_scales = quant_weight_ue8m0(
                weights.query_weight, [128, 128]
            )
            self.scales = transform_scale_ue8m0(block_scales, mn=HEADS * HEAD_DIM)
            self.decoded = self.codes.float() * block_scales.repeat_interleave(
                128, 0
            ).repeat_interleave(128, 1)
        else:
            self.codes, self.scales, self.decoded = (
                weights.codes,
                weights.scales,
                weights.decoded,
            )
        self.stream = torch.cuda.Stream()

    def bind(self, case):
        weights = case.static
        indexer = object.__new__(self.indexer_type)
        indexer.n_heads, indexer.head_dim, indexer.q_lora_rank = HEADS, HEAD_DIM, RANK
        indexer.softmax_scale = HEAD_DIM**-0.5
        indexer.use_bf16_indexer = indexer.logits_weights_bf16 = False
        indexer.use_fp4_indexer = case.output_dtype == "mxfp4"
        indexer.use_dsa_indexer_fusion = indexer.eb5_dsa_indexer_fusion = True
        indexer.eb5_indexer_hadamard = True
        indexer.alt_stream = self.stream

        def project_query(value):
            if self.implementation == "sglang_dual_stream":
                return self.linear(value, self.codes, [128, 128], self.scales), None
            activation, scales = halops.group_wise_quant(
                value, mode="w8a8-mxfp8", use_swizzle=True
            )
            output = torch.empty(
                (case.tokens, HEADS * HEAD_DIM),
                dtype=torch.bfloat16,
                device=value.device,
            )
            halops.dense_gemm(activation, self.codes, scales, self.scales, output)
            return output, None

        indexer.wq_b = project_query
        indexer.wk_weights_proj = lambda value: (
            functional.linear(value, weights.projection),
            None,
        )
        indexer.rotary_emb = SimpleNamespace(
            cos_sin_cache=weights.rope, is_neox_style=True
        )

        def store_key(
            model,
            key_raw,
            positions,
            forward_batch,
            layer_id,
            act_quant,
            out_cache_loc=None,
        ):
            self.key_kernel(
                key_raw,
                case.caches[self.implementation],
                case.locations if out_cache_loc is None else out_cache_loc,
                weights.gain,
                weights.bias,
                1e-6,
                weights.rope,
                positions,
                PAGE_SIZE,
                hadamard=True,
                neox_style=True,
                use_bf16_indexer=False,
                use_fp4_indexer=model.use_fp4_indexer,
            )

        indexer._fused_k_prepare_and_store = MethodType(store_key, indexer)
        batch = SimpleNamespace(out_cache_loc=case.locations)

        def call(inputs):
            assert inputs is case
            with self.kernel_context(hadamard=True, neox_style=True):
                query, output_weights = self.method(
                    indexer,
                    case.x,
                    case.q_lora,
                    case.positions,
                    batch,
                    0,
                    None,
                    enable_dual_stream=True,
                )
            # The source method joins its auxiliary stream before returning.
            return (query if indexer.use_fp4_indexer else (query, None)), output_weights

        return call


def rotate(values, case):
    table = case.static.rope[case.positions.long()]
    shape = (case.tokens,) + (1,) * (values.ndim - 2) + (32,)
    cosine, sine = table[:, :32].reshape(shape), table[:, 32:].reshape(shape)
    first, second = (
        (slice(0, 32), slice(32, 64))
        if case.neox
        else (slice(0, 64, 2), slice(1, 64, 2))
    )
    result = values.clone()
    left, right = values[..., first], values[..., second]
    result[..., first], result[..., second] = (
        left * cosine - right * sine,
        left * sine + right * cosine,
    )
    if case.hadamard:
        width = 1
        while width < HEAD_DIM:
            left, right = result.reshape(*result.shape[:-1], -1, 2, width).unbind(-2)
            result = torch.stack((left + right, left - right), -2).reshape(result.shape)
            width *= 2
        result *= HEAD_DIM**-0.5
    return result


def quantize_output(values, fp4, source=False, query=False):
    # EB5 FP8 Q does not round again after RoPE; K and MXFP4 do.
    if not (source and query and not fp4):
        values = values.bfloat16().float()
    if not fp4:
        scales = values.abs().amax(-1).clamp_min(1e-4) / 448
        return (values / scales[..., None]).clamp(-448, 448).to(FP8), scales
    groups = values.reshape(*values.shape[:-1], 4, 32)
    power = torch.ceil(
        torch.log2(groups.abs().amax(-1).clamp_min(1e-4 if source else 1e-10) / 6)
    )
    scales = (power + 127).clamp(0, 254).to(torch.uint8)
    normalized = (groups / torch.exp2(scales.float() - 127)[..., None]).reshape(
        values.shape
    )
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda")
    codes = torch.bucketize(normalized.abs(), midpoints, right=True)
    if source:  # EB5 E2M1 rounds exact ties to even; HALOps rounds ties away.
        lower = torch.bucketize(normalized.abs(), midpoints, right=False)
        codes = torch.where((codes != lower) & (codes % 2 == 1), lower, codes)
    codes = codes.to(torch.uint8) | ((normalized < 0).to(torch.uint8) << 3)
    return codes[..., ::2] | (codes[..., 1::2] << 4), scales


def decode_fp4(packed, scales):
    packed = packed.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    values = levels[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def reference(case, source=None):
    projection = case.x.float() @ case.static.projection.float().T
    if source is not None:
        projection = projection.bfloat16().float()
    key, raw_weights = projection[:, :HEAD_DIM], projection[:, HEAD_DIM:]
    if case.bias is not None:
        key = key - key.mean(-1, keepdim=True)
    key = key * (key.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    key = key * case.static.gain
    if case.bias is not None:
        key = key + case.bias
    sglang_projection = (
        source is not None and source.implementation == "sglang_dual_stream"
    )
    _, _, activation = group_fp8(
        case.q_lora,
        128 if sglang_projection else 32,
        1e-4 if sglang_projection else 1e-10,
    )
    query = activation @ (source.decoded if source else case.static.decoded).T
    if source is not None:
        query = query.bfloat16().float()
    query = rotate(query.reshape(case.tokens, HEADS, HEAD_DIM), case)
    key = rotate(key, case)
    fp4 = case.output_dtype == "mxfp4"
    query_codes, query_scales = quantize_output(
        query, fp4, source is not None, query=True
    )
    key_codes, key_scales = quantize_output(key, fp4, source is not None)
    output_weights = raw_weights * WEIGHT_SCALE
    if not fp4:
        output_weights = output_weights * query_scales
    return (
        (query_codes, query_scales),
        output_weights,
        (key_codes, key_scales),
        (query.bfloat16().float(), key.bfloat16().float()),
    )


def read_cache(case, cache):
    locations = case.locations.long()
    valid = locations >= 0
    locations = locations[valid]
    page, slot = locations // PAGE_SIZE, locations % PAGE_SIZE
    data_bytes = 64 if case.output_dtype == "mxfp4" else 128
    columns = slot[:, None] * data_bytes + torch.arange(data_bytes, device="cuda")
    codes = cache[page[:, None], columns].contiguous()
    mask = torch.zeros_like(cache, dtype=torch.bool)
    mask[page[:, None], columns] = True
    columns = (
        PAGE_SIZE * data_bytes + slot[:, None] * 4 + torch.arange(4, device="cuda")
    )
    scales = cache[page[:, None], columns].contiguous()
    mask[page[:, None], columns] = True
    if case.output_dtype == "fp8":
        codes, scales = codes.view(FP8), scales.view(torch.float32).flatten()
    assert (cache[~mask] == 0xA5).all(), "K cache wrote an unselected byte"
    return codes, scales, valid


def relative_rms(actual, expected):
    if actual.numel() == 0:
        return 0.0
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert torch.isfinite(actual.float()).all(), "non-finite output"
    return (
        (actual.float() - expected.float()).square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-12)
    ).item()


def fp4_rounding_ulps(packed, scales, unquantized):
    """Distance from the reference to the actual code's rounding interval."""
    packed = packed.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    magnitudes = (codes & 7).long()
    edges = torch.tensor(
        [0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, float("inf")],
        device=packed.device,
    )
    scale = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    lower = torch.where(magnitudes == 0, -0.25, edges[magnitudes]) * scale
    upper = edges[magnitudes + 1] * scale
    signed_reference = unquantized * torch.where(codes & 8 != 0, -1.0, 1.0)
    distance = torch.maximum(
        lower - signed_reference, signed_reference - upper
    ).clamp_min(0)
    bf16_ulp = torch.exp2(
        torch.floor(torch.log2(unquantized.abs().clamp_min(1e-10))) - 7
    )
    return (distance / bf16_ulp).max().item() if distance.numel() else 0.0


def validate_result(case, result, cache, expected, source=False):
    (query, scales), output_weights = result
    (
        (query_ref, scales_ref),
        weights_ref,
        (key_ref, key_scales_ref),
        (query_unquantized, key_unquantized),
    ) = expected
    fp4 = case.output_dtype == "mxfp4"
    assert query.shape == (case.tokens, HEADS, 64 if fp4 else HEAD_DIM)
    assert query.dtype == ((torch.int8 if source else torch.uint8) if fp4 else FP8)
    assert output_weights.shape == (
        (case.tokens, HEADS, 1) if source else (case.tokens, HEADS)
    )
    assert output_weights.dtype == torch.float32
    key, key_scales, valid = read_cache(case, cache)
    key_ref, key_scales_ref = key_ref[valid], key_scales_ref[valid]
    if fp4:
        assert scales.dtype == (torch.int32 if source else torch.uint8)
        assert scales.shape == (
            (case.tokens, HEADS) if source else (case.tokens, HEADS, 4)
        )
        scales = scales.view(torch.uint8).reshape(case.tokens, HEADS, 4)
        assert (scales != 255).all() and (key_scales != 255).all()
        # Exponent changes are allowed only at an adjacent scale boundary.
        assert (scales.to(torch.int16) - scales_ref.to(torch.int16)).abs().max() <= 1
        if key_scales.numel():
            assert (
                key_scales.to(torch.int16) - key_scales_ref.to(torch.int16)
            ).abs().max() <= 1
        packed_outputs = (query, scales, key, key_scales)
        query, query_ref = decode_fp4(query, scales), decode_fp4(query_ref, scales_ref)
        key, key_ref = decode_fp4(key, key_scales), decode_fp4(key_ref, key_scales_ref)
    else:
        assert scales is None
        query, query_ref = query.float(), query_ref.float()
        key, key_ref = (
            key.float() * key_scales[:, None],
            key_ref.float() * key_scales_ref[:, None],
        )
    output_weights = output_weights.reshape(case.tokens, HEADS)
    errors = dict(
        query=relative_rms(query, query_ref),
        key=relative_rms(key, key_ref),
        weights=relative_rms(output_weights, weights_ref),
    )
    tolerance = 0.025 if fp4 else 0.01
    for name, offset, unquantized in (
        ("query", 0, query_unquantized),
        ("key", 2, key_unquantized[valid]),
    ):
        if fp4 and errors[name] > tolerance:
            # A single E2M1 boundary crossing can dominate a small tensor's
            # RMS. Accept only intervals within two BF16 ULPs of the reference,
            # accounting for projection and epilogue rounding.
            margin = fp4_rounding_ulps(
                packed_outputs[offset], packed_outputs[offset + 1], unquantized
            )
            errors[f"{name}_rounding_ulps"] = margin
            assert margin <= 2, errors
        else:
            assert errors[name] <= tolerance, errors
    assert errors["weights"] <= 0.005, errors
    if key.numel():
        scores = torch.einsum("thd,kd->thk", query, key[-8:]).relu()
        expected_scores = torch.einsum("thd,kd->thk", query_ref, key_ref[-8:]).relu()
        actual_sum = (scores * output_weights[..., None]).sum(1)
        expected_terms = expected_scores * weights_ref[..., None]
        expected_sum = expected_terms.sum(1)
        errors["scores"] = relative_rms(actual_sum, expected_sum)
        # Signed head weights can cancel. Normalize the reduction error by
        # the sum of absolute reference terms; retain the signed relative RMS.
        errors["scores_normalized"] = (
            (actual_sum - expected_sum).square().mean().sqrt()
            / expected_terms.abs().sum(1).square().mean().sqrt().clamp_min(1e-12)
        ).item()
        assert errors["scores_normalized"] <= tolerance, errors
    return errors


def flatten_output(result):
    (query, scales), weights = result
    return (query, weights) if scales is None else (query, scales, weights)


def snapshot(result, cache):
    return [
        value.view(torch.uint8).clone() for value in (*flatten_output(result), cache)
    ]


def check_snapshot(result, cache, saved):
    for actual, expected in zip((*flatten_output(result), cache), saved, strict=True):
        assert torch.equal(
            actual.view(torch.uint8), expected
        ), "Graph/eager byte mismatch"


def validate(case, name, call, adapter):
    """Check the numerical reference and replay with changed device inputs."""
    cache = case.caches[name]
    cache.fill_(0xA5)
    result = call(case)
    try:
        errors = validate_result(
            case, result, cache, reference(case, adapter), source=adapter is not None
        )
    except AssertionError as error:
        raise AssertionError(
            f"{name} {case.output_dtype} bsz={case.tokens}: {error}"
        ) from error
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call(case)
    original_input, original_query = case.x.clone(), case.q_lora.clone()
    for multiplier in (0.875, -1.0):
        case.x.mul_(multiplier)
        case.q_lora.mul_(multiplier)
        cache.fill_(0xA5)
        expected = call(case)
        saved = snapshot(expected, cache)
        cache.fill_(0xA5)
        graph.replay()
        check_snapshot(captured, cache, saved)
    case.x.copy_(original_input)
    case.q_lora.copy_(original_query)
    torch.cuda.synchronize()
    return errors


@triton.jit
def flush_l2(buffer, count: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(buffer + offsets, offsets < count, other=0, cache_modifier=".cg")
    tl.store(buffer + offsets, values + 1, offsets < count, cache_modifier=".wb")


def evict_l2(buffer):
    flush_l2[(triton.cdiv(buffer.numel(), 1024),)](buffer, buffer.numel(), BLOCK=1024)


@functools.cache
def l2_eviction_buffer():
    """int32 buffer of max(256 MiB, 2 * L2) bytes, read and rewritten by evict_l2."""
    l2_bytes = torch.cuda.get_device_properties().L2_cache_size
    buffer = torch.zeros(
        max(256 * 1024**2, 2 * l2_bytes) // 4, dtype=torch.int32, device="cuda"
    )
    evict_l2(buffer)  # Compile the Triton kernel before any graph capture.
    return buffer


class TimedGraph:
    """Capture `count` calls in one graph, each between its own start/end events.

    External events are recorded inside the graph, so each interval excludes the
    graph launch and the L2 eviction that precedes the call.
    """

    def __init__(self, call, count, stream, eviction=None):
        self.events = [
            (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for _ in range(count)
        ]
        for start, end in self.events:
            start.record(stream)
            end.record(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            for start, end in self.events:
                if eviction is not None:
                    evict_l2(eviction)
                start.record()
                call()
                end.record()

    def sample(self):
        """Replay once; return the mean per-call interval in microseconds."""
        self.graph.replay()
        self.events[-1][1].synchronize()
        intervals = [start.elapsed_time(end) * 1000 for start, end in self.events]
        if not all(math.isfinite(value) and value > 0 for value in intervals):
            raise RuntimeError(f"Invalid CUDA event intervals: {intervals}")
        return statistics.mean(intervals)


def measure(call, inputs, args):
    """Time graph replays, including both branches of dual-stream calls.

    Returns one sample per timed replay, in microseconds per call.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    if args.timing == "timed_graph":
        eviction = l2_eviction_buffer() if args.l2_cache == "cold" else None
        timed = TimedGraph(lambda: call(inputs), args.graph_iters, stream, eviction)
        for _ in range(args.warmup_replays):
            timed.sample()
        samples_us = [timed.sample() for _ in range(args.repeat_replays)]
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.graph_iters):
                call(inputs)
        torch.cuda.synchronize()

        # The chain is already captured. Use the event backend so L2 is explicitly
        # flushed before start.record(), instead of rotating tensors in input_args.
        # Each sample is start.record() -> graph.replay() -> end.record().
        samples = bench_gpu_time(
            graph.replay,
            use_cuda_graph=False,
            cold_l2_cache=args.l2_cache == "cold",
            enable_cupti=False,
            dry_run_iters=args.warmup_replays,
            repeat_iters=args.repeat_replays,
        )
        samples_us = [float(sample) * 1000 / args.graph_iters for sample in samples]
    if len(samples_us) != args.repeat_replays or not all(
        math.isfinite(value) and value > 0 for value in samples_us
    ):
        raise RuntimeError(f"Invalid CUDA event samples: {samples_us}")
    return samples_us


def command_output(*command):
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return (
        completed.stdout.strip()
        if completed.returncode == 0
        else completed.stderr.strip()
    )


def gpu_state():
    return {
        "gpus": command_output(
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm",
            "--format=csv",
        ),
        "processes": command_output(
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv",
        ),
    }


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", help="comma-separated override; default: every integer 1..512"
    )
    parser.add_argument(
        "--output-dtype", choices=("fp8", "mxfp4", "both"), default="both"
    )
    parser.add_argument(
        "--timing",
        choices=("timed_graph", "bench_gpu_time"),
        default="timed_graph",
        help="timed_graph: CUDA events inside the graph around each call; "
        "bench_gpu_time: flashinfer.testing.bench_gpu_time events around the replay",
    )
    parser.add_argument("--warmup-replays", type=positive_int, default=5)
    parser.add_argument("--repeat-replays", type=positive_int, default=25)
    parser.add_argument(
        "--graph-iters",
        type=positive_int,
        help="calls per captured graph, reusing inputs/weights; default 16 for "
        "timed_graph (each call timed and flushed separately, a sample is their mean), "
        "1 for bench_gpu_time (one event pair and one flush per replay)",
    )
    parser.add_argument(
        "--l2-cache",
        choices=("hot", "cold"),
        default="cold",
        help="cold: flush outside timing, before each call (timed_graph) or each "
        "replay (bench_gpu_time); hot: no explicit flush",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=int, default=0, help="logical CUDA device")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()
    try:
        args.batch_sizes = (
            list(dict.fromkeys(int(value) for value in args.batch_sizes.split(",")))
            if args.batch_sizes
            else list(range(1, 513))
        )
    except ValueError:
        parser.error("--batch-sizes must contain comma-separated integers")
    if not args.batch_sizes or min(args.batch_sizes) < 1:
        parser.error("batch sizes must be positive")
    if args.graph_iters is None:
        args.graph_iters = 16 if args.timing == "timed_graph" else 1
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    formats = ("fp8", "mxfp4") if args.output_dtype == "both" else (args.output_dtype,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_files = ("latency.csv", "samples.jsonl", "validation.jsonl", "metadata.json")
    if any((args.output_dir / filename).exists() for filename in result_files):
        raise FileExistsError(f"Choose a fresh --output-dir: {args.output_dir}")

    weights = make_weights(max(args.batch_sizes), args.seed)
    workspace = indexer_prepare_workspace(max(args.batch_sizes), "cuda")
    adapters = {name: DualStream(weights, name) for name in IMPLEMENTATIONS[:2]}
    timed_graph = args.timing == "timed_graph"
    if args.l2_cache == "hot":
        l2 = "no explicit flush; warmup and replay reuse the same inputs/weights"
    elif timed_graph:
        l2 = (
            f"evict_l2 reads and rewrites {l2_eviction_buffer().numel() * 4} bytes "
            "before every call, inside the graph and outside its events"
        )
    else:
        l2 = (
            "flushed before each chain, outside events; only the first call starts cold"
        )
    metadata = {
        "arguments": {**vars(args), "output_dir": str(args.output_dir)},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": False,
        "versions": {
            name: version(name)
            for name in ("torch", "flashinfer-python", "halops", "triton")
        },
        "enable_cupti": False,
        "timing_backend": "cuda_event",
        "cuda_graph": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(),
        "workspace_bytes": workspace.numel(),
        "workspace_allocations": 1,
        "gpu_before": gpu_state(),
        "revisions": {
            name: command_output(
                "git", "-C", str(PROJECT_ROOT / name), "rev-parse", "HEAD"
            )
            for name in ("HALOps", "sglang")
        },
        "repository_status": {
            name: command_output(
                "git", "-C", str(PROJECT_ROOT / name), "status", "--short"
            )
            for name in ("HALOps", "sglang")
        },
        "halops_extension_sha256": hashlib.sha256(
            Path(halops.halops_kernel.__file__).read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing": (
            "TimedGraph: external CUDA events inside the graph around each call, "
            "including dual-stream joins; a sample is the mean per-call interval "
            "of one replay, in us/call"
            if timed_graph
            else "bench_gpu_time: CUDA events outside complete graph replay, "
            "including dual-stream joins; elapsed_ms * 1000 / graph_iters = us/call"
        ),
        "l2": l2,
        "chain": (
            "graph_iters calls with the same inputs/weights, each between its own events"
            if timed_graph
            else "graph_iters calls with the same inputs/weights; no per-call timing events"
        ),
        "halops_dual_stream": "HALOps MXFP8 Q quant/GEMM + PyTorch BF16 KW + SGLang EB5 epilogues/schedule",
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    fields = [
        "output_dtype",
        "bsz",
        "timing",
        "l2_cache",
        "graph_iters",
        *(f"{name}_us" for name in IMPLEMENTATIONS),
        "sglang_over_indexer",
        "halops_dual_over_indexer",
    ]
    rows = 0
    start = time.monotonic()
    with (
        (args.output_dir / "latency.csv").open("w", newline="") as csv_file,
        (args.output_dir / "samples.jsonl").open("w") as sample_file,
        (args.output_dir / "validation.jsonl").open("w") as validation_file,
        (args.output_dir / "gpu_observations.jsonl").open("w") as gpu_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        if not args.validate_only:
            control = make_case(
                min(128, max(args.batch_sizes)), formats[0], weights, workspace
            )
            control_call = adapters["sglang_dual_stream"].bind(control)
            metadata["timing_control"] = {
                "implementation": "sglang_dual_stream",
                "bsz": control.tokens,
                "output_dtype": formats[0],
                "before_samples_us": measure(control_call, control, args),
            }
        for output_dtype in formats:
            for batch_index, tokens in enumerate(args.batch_sizes):
                inputs = make_case(tokens, output_dtype, weights, workspace)
                calls = {
                    name: adapter.bind(inputs) for name, adapter in adapters.items()
                }
                calls["halops_indexer_prepare"] = call_halops
                medians = {}
                # Rotate order to distribute drift across the three implementations.
                offset = batch_index % len(IMPLEMENTATIONS)
                order = IMPLEMENTATIONS[offset:] + IMPLEMENTATIONS[:offset]
                for name in order:
                    errors = validate(inputs, name, calls[name], adapters.get(name))
                    validation_file.write(
                        json.dumps(
                            {
                                "output_dtype": output_dtype,
                                "bsz": tokens,
                                "implementation": name,
                                "relative_rms": errors,
                                "mutable_graph_replays": 2,
                            }
                        )
                        + "\n"
                    )
                    validation_file.flush()
                    if not args.validate_only:
                        torch.cuda.nvtx.range_push(f"{name},bsz={tokens}")
                        samples = measure(calls[name], inputs, args)
                        medians[name] = statistics.median(samples)
                        sample_file.write(
                            json.dumps(
                                {
                                    "output_dtype": output_dtype,
                                    "bsz": tokens,
                                    "implementation": name,
                                    "timing": args.timing,
                                    "l2_cache": args.l2_cache,
                                    "graph_iters": args.graph_iters,
                                    "samples_us": samples,
                                }
                            )
                            + "\n"
                        )
                        sample_file.flush()
                        torch.cuda.nvtx.range_pop()
                if not args.validate_only:
                    writer.writerow(
                        {
                            "output_dtype": output_dtype,
                            "bsz": tokens,
                            "timing": args.timing,
                            "l2_cache": args.l2_cache,
                            "graph_iters": args.graph_iters,
                            **{f"{name}_us": medians[name] for name in IMPLEMENTATIONS},
                            "sglang_over_indexer": medians[IMPLEMENTATIONS[0]]
                            / medians[IMPLEMENTATIONS[2]],
                            "halops_dual_over_indexer": medians[IMPLEMENTATIONS[1]]
                            / medians[IMPLEMENTATIONS[2]],
                        }
                    )
                    csv_file.flush()
                rows += 1
                if batch_index % 32 == 0 or batch_index == len(args.batch_sizes) - 1:
                    gpu_file.write(
                        json.dumps(
                            {
                                "output_dtype": output_dtype,
                                "bsz": tokens,
                                "elapsed_seconds": time.monotonic() - start,
                                **gpu_state(),
                            }
                        )
                        + "\n"
                    )
                    gpu_file.flush()
                    print(
                        f"{output_dtype} bsz={tokens}: {medians or 'validated'} "
                        f"elapsed={time.monotonic() - start:.1f}s",
                        flush=True,
                    )
                del inputs, calls
        if not args.validate_only:
            metadata["timing_control"]["after_samples_us"] = measure(
                control_call, control, args
            )
    metadata.update(
        completed=True,
        cases=rows * len(IMPLEMENTATIONS),
        elapsed_seconds=time.monotonic() - start,
        gpu_after=gpu_state(),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed {metadata['cases']} cases: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
