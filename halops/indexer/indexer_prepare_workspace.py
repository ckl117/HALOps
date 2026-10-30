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

"""Workspace allocation for :func:`halops.indexer_prepare`."""

from typing import Tuple

import torch

import halops.halops_kernel as halops_kernel

__all__ = ["indexer_prepare_workspace_bytes", "indexer_prepare_workspace"]


def indexer_prepare_workspace_bytes(num_tokens: int) -> Tuple[int, int]:
    """Return allocation/prefix bytes covering every T in ``1..num_tokens``.

    Query on the target CUDA device with the same tuning configuration used
    for execution. The size includes the maximum scratch requirement over
    the range for both FP8 and MXFP4, even when split-K changes inside it.
    """
    total, zero_prefix = halops_kernel.indexer_prepare_workspace_bytes(int(num_tokens))
    return int(total), int(zero_prefix)


def indexer_prepare_workspace(
    num_tokens: int, device: torch.device | str
) -> torch.Tensor:
    """Allocate and initialize an ``indexer_prepare`` workspace.

    Args:
        num_tokens: Maximum positive token count served by the workspace.
        device: CUDA device on which to allocate the workspace.

    Returns:
        A uint8 workspace on ``device`` with its counter prefix initialized
        to zero. Scratch contents are undefined until written by the kernel.

    The caller owns the uncached tensor. Serialized calls and Graph replays may
    change batch size within ``1..num_tokens`` and switch formats without clearing.
    Keep the full tensor length: it determines the counter/scratch boundary.
    Concurrent calls need separate workspaces. Reallocate after device or tuning
    changes, and recreate workspaces and Graphs when upgrading the layout.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise AssertionError(f"device must be CUDA, got {device}")

    with torch.cuda.device(device):
        total, zero_prefix = indexer_prepare_workspace_bytes(num_tokens)
        workspace = torch.empty(total, dtype=torch.uint8, device=device)
        workspace[:zero_prefix].zero_()
    return workspace
