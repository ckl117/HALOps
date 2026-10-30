// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>
#include <math_constants.h>
#include <torch/extension.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>

#include "indexer_config.cuh"
#include "indexer_intrinsics.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/gemm/kernel_traits.cuh"
#include "kernel/gemm/mainloop_fwd.cuh"
#include "umma_gemm.cuh"

namespace halops::indexer::detail {

template <class Bar>
CUTE_DEVICE void wait_transaction_barrier(Bar& bar, uint32_t phase) {
    bar.wait(phase);
}

template <class Bar>
CUTE_DEVICE void sync_named_barrier() {
    cutlass::arch::NamedBarrier::arrive_and_wait(Bar::width, Bar::id);
}

struct GemmPipelineBarriers {
    transac_bar_t* tma_a;
    transac_bar_t* tma_b;
    transac_bar_t* buffer_free;
    transac_bar_t* accumulator_ready;
    transac_bar_t* accumulator_free;
};

template <int STAGES>
CUTE_DEVICE int stage_use_count(int tiles_per_split, int stage) {
    return tiles_per_split / STAGES +
           (stage < tiles_per_split % STAGES ? 1 : 0);
}

template <int STAGES>
CUTE_DEVICE void wait_stage_free(const GemmPipelineBarriers& barriers,
                                 uint32_t previous_stage_parities,
                                 uint32_t unused_stage_mask, int item_sequence,
                                 int tiles_per_split, int tile) {
    const int stage = tile % STAGES;
    const bool stage_untouched = ((unused_stage_mask >> stage) & 1u) != 0u;
    const bool first_ever =
        stage_untouched && item_sequence == 0 && tile < STAGES;
    if (first_ever) return;
    const int need =
        (int)((previous_stage_parities >> stage) & 1u) +
        item_sequence * stage_use_count<STAGES>(tiles_per_split, stage) +
        tile / STAGES;
    __syncwarp();
    wait_transaction_barrier(barriers.buffer_free[stage],
                             (uint32_t)((need - 1) & 1));
}

template <int STAGES>
CUTE_DEVICE void wait_stage_tma(const GemmPipelineBarriers& barriers,
                                uint32_t previous_stage_parities,
                                int item_sequence, int tiles_per_split,
                                int tile) {
    const int stage = tile % STAGES;
    const int need =
        (int)((previous_stage_parities >> stage) & 1u) +
        item_sequence * stage_use_count<STAGES>(tiles_per_split, stage) +
        tile / STAGES + 1;
    const uint32_t phase = (uint32_t)((need - 1) & 1);
    __syncwarp();
    wait_transaction_barrier(barriers.tma_a[stage], phase);
    wait_transaction_barrier(barriers.tma_b[stage], phase);
}

template <int ACC_STAGES>
CUTE_DEVICE int reserve_accumulator_slot(const GemmPipelineBarriers& barriers,
                                         int previous_accumulators,
                                         int accumulator_sequence) {
    const int total = previous_accumulators + accumulator_sequence;
    const int slot = total % ACC_STAGES;
    if (total >= ACC_STAGES) {
        const int lap = total / ACC_STAGES;
        __syncwarp();
        wait_transaction_barrier(barriers.accumulator_free[slot],
                                 (uint32_t)((lap - 1) & 1));
    }
    return slot;
}

template <int ACC_STAGES>
CUTE_DEVICE int wait_accumulator_slot(const GemmPipelineBarriers& barriers,
                                      int previous_accumulators,
                                      int accumulator_sequence) {
    const int total = previous_accumulators + accumulator_sequence;
    const int slot = total % ACC_STAGES;
    const int lap = total / ACC_STAGES;
    wait_transaction_barrier(barriers.accumulator_ready[slot],
                             (uint32_t)(lap & 1));
    return slot;
}

template <int STAGES, class F>
CUTE_DEVICE void dispatch_stage(int stage, F&& visit_stage, int tile) {
    static_assert(STAGES >= 1 && STAGES <= 4,
                  "Stage dispatch supports one to four stages");
    if (stage == 0) {
        visit_stage(cute::Int<0>{}, tile);
    } else if constexpr (STAGES > 2) {
        if (stage == 1)
            visit_stage(cute::Int<1>{}, tile);
        else if constexpr (STAGES > 3) {
            if (stage == 2)
                visit_stage(cute::Int<2>{}, tile);
            else
                visit_stage(cute::Int<STAGES - 1>{}, tile);
        } else {
            visit_stage(cute::Int<STAGES - 1>{}, tile);
        }
    } else {
        visit_stage(cute::Int<STAGES - 1>{}, tile);
    }
}

struct PartialBarrier {
    int* slot;
    int producer_count;

    CUTE_DEVICE PartialBarrier(int* base, int index, int producer_count)
        : slot(base + (int64_t)index * (IndexerWarpTraits::RendezvousSlotBytes /
                                        sizeof(int))),
          producer_count(producer_count) {}

    CUTE_DEVICE void arrive_warp() const {
        __syncwarp();
        if (cute::elect_one_sync())
            halops::indexer::intrinsics::gmem_red_add1_release(slot);
    }

    CUTE_DEVICE void wait(uint32_t epoch) const {
        const int target = (int)(epoch + 1) * producer_count;
        int seen;
        do {
            seen = halops::indexer::intrinsics::gmem_ld_relaxed(slot);
        } while (seen < target);
        seen = halops::indexer::intrinsics::gmem_ld_acquire(slot);
    }

    CUTE_DEVICE void wait_warp(uint32_t epoch) const {
        if (cute::elect_one_sync()) wait(epoch);
        __syncwarp();
    }

    CUTE_DEVICE bool try_wait(uint32_t epoch) const {
        const int target = (int)(epoch + 1) * producer_count;
        // Poll relaxed; acquire published data only after the producers finish.
        if (halops::indexer::intrinsics::gmem_ld_relaxed(slot) < target)
            return false;
        return halops::indexer::intrinsics::gmem_ld_acquire(slot) >= target;
    }

    CUTE_DEVICE bool try_wait_warp(uint32_t epoch) const {
        int ready_vote = 0;
        if (cute::elect_one_sync()) ready_vote = try_wait(epoch) ? 1 : 0;
        const bool ready = __any_sync(0xffffffffu, ready_vote) != 0;
        if (ready) __syncwarp();
        return ready;
    }

    CUTE_DEVICE void reset_relaxed() const {
        halops::indexer::intrinsics::gmem_st_zero_relaxed(slot);
    }
};

}  // namespace halops::indexer::detail
