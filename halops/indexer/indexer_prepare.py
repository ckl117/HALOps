# Copyright (c) 2026 WenXin Authors. All Rights Reserved.
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

"""Fused DSA indexer preparation, adapted from InferRock 8d45d147."""

from typing import Literal, Optional, Tuple

import torch

import halops.halops_kernel as halops_kernel

__all__ = ["indexer_prepare"]

FP8_E4M3 = torch.float8_e4m3fn


def indexer_prepare(
    x: torch.Tensor,
    q_lora: torch.Tensor,
    wk_w: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_sf: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: Optional[torch.Tensor],
    k_norm_eps: float,
    rope_cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache_with_scale: torch.Tensor,
    out_cache_loc: torch.Tensor,
    workspace: torch.Tensor,
    q: Optional[torch.Tensor] = None,
    sf_q: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    weight_scale: float = 0.0,
    head_dim: int = 128,
    n_heads: int = 64,
    is_neox_style: bool = True,
    apply_hadamard: bool = True,
    skip_q: bool = False,
    output_dtype_str: Literal["fp8", "mxfp4"] = "fp8",
    use_pdl: bool = True,
) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    """Project, normalize, rotate and quantize indexer Q/K in one CUDA launch.

    Args:
        x: [T, 4096] bfloat16, T > 0, unit column stride and 16-byte aligned
            row starts.
        q_lora: [T, 1024] bfloat16, unit column stride and 8-byte aligned row
            starts. Quantized to group-32 FP8 inside the kernel.
        wk_w: [192, 4096] contiguous bfloat16; the 128 key rows followed by
            the 64 weight rows.
        wq_b: [8192, 1024] contiguous float8_e4m3fn Q projection weight.
        wq_b_sf: [8192, 32] contiguous uint8 group-32 UE8M0 scales in
            CUTLASS SfAtom layout, as returned by
            ``group_wise_quant(..., mode="w8a8-mxfp8", use_swizzle=True)``.
        k_norm_weight: [128] contiguous float32 normalization gain.
        k_norm_bias: [128] contiguous float32 selects LayerNorm; None selects
            RMSNorm.
        k_norm_eps: Positive normalization epsilon.
        rope_cos_sin_cache: [max_position, 64] contiguous float32, stored as
            [cos(32) | sin(32)]. RoPE rotates the first 64 Q/K elements.
        positions: [T] contiguous int32/int64 RoPE positions. Values must
            fit int32 and index the supplied table.
        paged_kv_cache_with_scale: Contiguous uint8 page pool. In fp8 mode,
            [num_pages, page_size * 132], with all 128-byte FP8 rows followed
            by one float32 scale per row. In fp4 mode,
            [num_pages, page_size * 68], with all 64-byte packed E2M1 rows
            followed by four uint8 UE8M0 scales per row (group size 32).
            page_size must be a power of two. Written in place.
        out_cache_loc: [T] contiguous int32/int64 cache slots. Negative slots
            skip K writes. Nonnegative slots must be in range and unique
            within a launch; all values must fit int32. Slots are independent
            of positions.
        workspace: Caller-owned uint8 buffer from ``indexer_prepare_workspace``.
            Reusing calls must be serialized, including CUDA Graph replays.
        q: Optional output buffer. In fp8 mode it is a contiguous
            [T, 64, 128] float8_e4m3fn tensor. In mxfp4 mode it is a
            contiguous [T, 64, 64] uint8 packed E2M1 tensor. Even elements
            occupy the low nibble.
        sf_q: Optional contiguous [T, 64, 4] uint8 UE8M0 scale output for
            mxfp4, in linear group order. Each scale covers 32 Q elements.
            Must be None in fp8 mode, whose Q scale is folded into weights.
        weights: Optional contiguous [T, 64] float32 output.
        weight_scale: Scalar folded into weights; 0 selects
            128**-0.5 * 64**-0.5.
        head_dim: Must be 128.
        n_heads: Must be 64.
        use_pdl: Enable Programmatic Dependent Launch. Successors may initialize
            early; their dependency wait observes complete outputs and counter reset.
        is_neox_style: True (default) pairs dimensions i and i+32 in the
            rotary prefix. False selects adjacent pairs (2*i, 2*i+1).
        apply_hadamard: Apply the normalized 128-point Hadamard to Q and K.
        skip_q: Compute K only. Preallocated Q, Q scales and weights stay
            untouched; newly allocated outputs are undefined in this mode.
        output_dtype_str: "fp8" (default) or "mxfp4". The latter selects the
            packed FP4 Q and K-cache branches. Projection weights remain
            group-32 MXFP8.

    Returns:
        ``((q, sf_q), weights)``. FP8 folds each Q row's scale into ``weights``
        and returns ``sf_q=None``. MXFP4 returns separate group scales and
        ``weights_raw * weight_scale``.

    All tensors must be on the same CUDA device. This build requires SM100
    with at least 128 available SMs.
    """
    if output_dtype_str not in ("fp8", "mxfp4"):
        raise ValueError("output_dtype_str must be 'fp8' or 'mxfp4'")
    output_fp4 = output_dtype_str == "mxfp4"
    if not output_fp4 and sf_q is not None:
        raise ValueError("sf_q is only accepted with output_dtype_str='mxfp4'")
    if x.dtype != torch.bfloat16:
        raise AssertionError(f"x must be bfloat16, got {x.dtype}")
    if wk_w.dtype != torch.bfloat16:
        raise AssertionError("wk_w is bf16 only")
    if wq_b.dtype != FP8_E4M3:
        raise AssertionError(f"wq_b must be float8_e4m3fn, got {wq_b.dtype}")
    if wq_b_sf.dtype != torch.uint8:
        raise AssertionError(f"wq_b_sf must be uint8, got {wq_b_sf.dtype}")
    if not x.is_cuda:
        raise AssertionError("x must be CUDA")
    if x.dim() != 2 or wq_b.dim() != 2 or wq_b_sf.dim() != 2:
        raise AssertionError("x, wq_b and wq_b_sf must be 2-D")
    if head_dim != 128 or n_heads != 64:
        raise AssertionError("this build requires head_dim=128 and n_heads=64")
    if wq_b_sf.device != x.device:
        raise AssertionError("wq_b_sf must be on the same CUDA device as x")
    if workspace.device != x.device:
        raise AssertionError("workspace must be on the same CUDA device as x")

    num_tokens = int(x.shape[0])
    device = x.device
    if output_fp4:
        if q is None:
            q = torch.empty(
                (num_tokens, n_heads, head_dim // 2),
                dtype=torch.uint8,
                device=device,
            )
        if sf_q is None:
            sf_q = torch.empty(
                (num_tokens, n_heads, head_dim // 32), dtype=torch.uint8, device=device
            )
    else:
        if q is None:
            q = torch.empty(
                (num_tokens, n_heads, head_dim), dtype=FP8_E4M3, device=device
            )
    if weights is None:
        weights = torch.empty((num_tokens, n_heads), dtype=torch.float32, device=device)
    if weight_scale == 0.0:
        weight_scale = head_dim**-0.5 * n_heads**-0.5

    if k_norm_bias is None:
        k_norm_bias = torch.empty(0, dtype=torch.float32, device=device)
    with torch.cuda.device(device):
        halops_kernel.indexer_prepare(
            x,
            q_lora,
            wk_w,
            wq_b,
            wq_b_sf,
            k_norm_weight,
            k_norm_bias,
            float(k_norm_eps),
            rope_cos_sin_cache,
            positions,
            paged_kv_cache_with_scale,
            out_cache_loc,
            q,
            sf_q,
            weights,
            workspace,
            float(weight_scale),
            int(head_dim),
            int(n_heads),
            bool(apply_hadamard),
            bool(skip_q),
            output_dtype_str,
            bool(is_neox_style),
            bool(use_pdl),
        )
    return (q, sf_q if output_fp4 else None), weights
