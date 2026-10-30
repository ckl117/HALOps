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

#include "indexer_params.cuh"

namespace halops::indexer {

struct KeyWeightWorkItem {
    int weight_tile;  // 0: key rows; 1: head-weight rows.
    int token_group;
    int split_index;
    int first_k_tile;
    int valid_tokens;
};

template <class GemmConfig>
CUTE_DEVICE KeyWeightWorkItem key_weight_work_item_at(int work_index,
                                                      int token_groups,
                                                      int key_weight_splits,
                                                      int tiles_per_split,
                                                      int num_tokens) {
    const int token_group = work_index % token_groups;
    const int rest = work_index / token_groups;
    const int split_index = rest % key_weight_splits;
    const int weight_tile = rest / key_weight_splits;
    return KeyWeightWorkItem{
        weight_tile, token_group, split_index, split_index * tiles_per_split,
        min((int)GemmConfig::BN, num_tokens - token_group * GemmConfig::BN)};
}

template <class WeightTma, class ActivationTma>
struct KeyWeightResources {
    const WeightTma& tma_w;
    const ActivationTma& tma_act;
    char* __restrict__ gemm_shared_memory;
    halops::indexer::detail::GemmPipelineBarriers barriers;
    uint32_t tmem_base;
    int activation_rows;
    int m_total;
    int k_dim;
    int k_tiles;
    float* __restrict__ partials;
    int split_count;
};

template <class GemmConfig, class Resources>
struct KeyWeightWeightLoader {
    using GemmTypes = halops::indexer::detail::Bf16GemmTypes<GemmConfig>;
    static constexpr int STAGES = GemmConfig::TmaStages;

    const Resources& resources;
    bool elected;
    uint32_t previous_stage_parities;
    uint32_t unused_stage_mask;
    int item_sequence;
    int weight_tile;
    int first_k_tile;

    CUTE_DEVICE void begin_item(const KeyWeightWorkItem& work_item) {
        weight_tile = work_item.weight_tile;
        first_k_tile = work_item.first_k_tile;
    }
    CUTE_DEVICE void end_item() { ++item_sequence; }

    CUTE_DEVICE void issue_tile(int t) {
        halops::indexer::detail::wait_stage_free<STAGES>(
            resources.barriers, previous_stage_parities, unused_stage_mask,
            item_sequence, resources.k_tiles, t);
        if (!elected) return;
        halops::indexer::detail::dispatch_stage<STAGES>(t % STAGES, issuer(),
                                                        t);
    }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            constexpr int s = decltype(St)::value;
            resources.barriers.tma_a[s].arrive_and_expect_tx(
                halops::indexer::detail::Bf16GemmBytes<GemmConfig>::W);
            uint64_t& mbar =
                reinterpret_cast<uint64_t&>(resources.barriers.tma_a[s]);
            auto mA = resources.tma_w.get_tma_tensor(
                cute::make_shape(resources.m_total, resources.k_dim));
            auto thr_w = resources.tma_w.get_slice(cute::_0{});
            auto tA = cute::local_tile(
                mA, typename GemmTypes::TileA{},
                cute::make_coord(weight_tile, k + first_k_tile));
            cute::copy(resources.tma_w.with(mbar, 0), thr_w.partition_S(tA),
                       thr_w.partition_D(
                           halops::indexer::detail::bf16_smem_a<GemmConfig, s>(
                               resources.gemm_shared_memory)));
        };
    }
};

template <class GemmConfig, class Resources>
struct KeyWeightActivationLoader {
    using GemmTypes = halops::indexer::detail::Bf16GemmTypes<GemmConfig>;
    static constexpr int STAGES = GemmConfig::TmaStages;

    const Resources& resources;
    bool elected;
    uint32_t previous_stage_parities;
    uint32_t unused_stage_mask;
    int item_sequence;
    int n_tile;
    int first_k_tile;

    CUTE_DEVICE void begin_item(const KeyWeightWorkItem& work_item) {
        n_tile = work_item.token_group;
        first_k_tile = work_item.first_k_tile;
    }
    CUTE_DEVICE void end_item() { ++item_sequence; }

    CUTE_DEVICE void issue_tile(int t) {
        halops::indexer::detail::wait_stage_free<STAGES>(
            resources.barriers, previous_stage_parities, unused_stage_mask,
            item_sequence, resources.k_tiles, t);
        if (!elected) return;
        halops::indexer::detail::dispatch_stage<STAGES>(t % STAGES, issuer(),
                                                        t);
    }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            constexpr int s = decltype(St)::value;
            resources.barriers.tma_b[s].arrive_and_expect_tx(
                halops::indexer::detail::Bf16GemmBytes<GemmConfig>::ACT);
            uint64_t& mbar =
                reinterpret_cast<uint64_t&>(resources.barriers.tma_b[s]);
            auto mB = resources.tma_act.get_tma_tensor(
                cute::make_shape(resources.activation_rows, resources.k_dim));
            auto thr_act = resources.tma_act.get_slice(cute::_0{});
            auto tB =
                cute::local_tile(mB, typename GemmTypes::TileB{},
                                 cute::make_coord(n_tile, k + first_k_tile));
            cute::copy(resources.tma_act.with(mbar, 0), thr_act.partition_S(tB),
                       thr_act.partition_D(
                           halops::indexer::detail::bf16_smem_b<GemmConfig, s>(
                               resources.gemm_shared_memory)));
        };
    }
};

template <class GemmConfig, class Resources, class TmemCopy,
          class ThreadTmemAccumulators, class RegisterAccumulators, class Coord>
struct KeyWeightAccumulatorConsumer {
    static constexpr int ACC_STAGES =
        halops::indexer::detail::Bf16GemmAcc::Stages;

    const Resources& resources;
    TmemCopy tmem_copy;
    ThreadTmemAccumulators thread_tmem_accumulators;
    RegisterAccumulators register_accumulators;
    Coord coord;
    int previous_accumulators;
    int accumulator_sequence;
    int accumulator_slot;

    CUTE_DEVICE void wait_mma_and_load() {
        accumulator_slot =
            halops::indexer::detail::wait_accumulator_slot<ACC_STAGES>(
                resources.barriers, previous_accumulators,
                accumulator_sequence);
        halops::indexer::intrinsics::tcgen05_after_thread_sync();
        ThreadTmemAccumulators src = thread_tmem_accumulators;
        src.data() =
            src.data().get() + (uint32_t)(accumulator_slot * GemmConfig::BN);
        cute::copy(tmem_copy, src, register_accumulators);
        cutlass::arch::fence_view_async_tmem_load();
    }

    CUTE_DEVICE void store(const KeyWeightWorkItem& work_item) const {
        const int64_t stride =
            (int64_t)resources.split_count * IndexerDimensions::KEY_WEIGHT_ROWS;
        float* base =
            resources.partials +
            (int64_t)(work_item.token_group * GemmConfig::BN) * stride +
            (int64_t)work_item.split_index * IndexerDimensions::KEY_WEIGHT_ROWS;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(register_accumulators); ++i) {
            const int row =
                cute::get<0>(coord(i)) + work_item.weight_tile * GemmConfig::BM;
            const int local_token = cute::get<1>(coord(i));
            if (row < IndexerDimensions::KEY_WEIGHT_ROWS &&
                local_token < work_item.valid_tokens)
                base[(int64_t)local_token * stride + row] =
                    register_accumulators(i);
        }
    }

    CUTE_DEVICE void end_item() {
        cutlass::arch::fence_view_async_tmem_load();
        resources.barriers.accumulator_free[accumulator_slot].arrive();
        ++accumulator_sequence;
    }
};

template <class GemmConfig, class Resources>
CUTE_DEVICE auto make_key_weight_consumer(const Resources& resources,
                                          int previous_accumulators) {
    using GemmTypes = halops::indexer::detail::Bf16GemmTypes<GemmConfig>;
    using Roles = IndexerTraits::Roles;
    static_assert(Roles::DrainWarps == 4 && Roles::DrainWarpBegin % 4 == 0,
                  "the T2R copy needs one warp per TMEM datapath quadrant, "
                  "based at a multiple of 4");

    typename GemmTypes::TiledMma tiled_mma;
    cute::Tensor frag0 = cute::partition_fragment_C(
        tiled_mma,
        cute::Shape<cute::Int<GemmConfig::BM>, cute::Int<GemmConfig::BN>>{});
    frag0.data().get() = resources.tmem_base;
    cute::Tensor acc0 =
        frag0(cute::make_coord(cute::_, cute::_), cute::_0{}, cute::_0{});
    cute::TiledCopy tmem_copy =
        cute::make_tmem_copy(typename GemmTypes::CopyOpT2R{}, acc0);
    static_assert(
        (int)cute::size(decltype(tmem_copy){}) / 32 == Roles::DrainWarps,
        "the T2R copy must need exactly DrainWarps warps");

    const int warp = IndexerTraits::warp_idx();
    const int lane = IndexerTraits::lane_idx();
    cute::ThrCopy thread_tmem_copy =
        tmem_copy.get_slice((warp % Roles::DrainWarps) * 32 + lane);
    cute::Tensor thr_demo = thread_tmem_copy.partition_D(acc0);
    cute::Tensor register_accumulators =
        cute::make_tensor<float>(cute::shape(thr_demo));
    cute::Tensor coord = thread_tmem_copy.partition_D(
        cute::make_identity_tensor(cute::shape(acc0)));

    using Op = KeyWeightAccumulatorConsumer<
        GemmConfig, Resources, decltype(tmem_copy),
        decltype(thread_tmem_copy.partition_S(acc0)),
        decltype(register_accumulators), decltype(coord)>;
    return Op{resources,
              tmem_copy,
              thread_tmem_copy.partition_S(acc0),
              register_accumulators,
              coord,
              previous_accumulators,
              0,
              0};
}

template <class GemmConfig, class Resources>
CUTE_DEVICE auto make_key_weight_loader(const Resources& resources,
                                        uint32_t previous_stage_parities,
                                        uint32_t unused_stage_mask) {
    return KeyWeightWeightLoader<GemmConfig, Resources>{
        resources,
        cute::elect_one_sync() != 0,
        previous_stage_parities,
        unused_stage_mask,
        0,
        0,
        0};
}

template <class GemmConfig, class Resources>
CUTE_DEVICE auto make_key_weight_activation_loader(
    const Resources& resources, uint32_t previous_stage_parities,
    uint32_t unused_stage_mask) {
    return KeyWeightActivationLoader<GemmConfig, Resources>{
        resources,
        cute::elect_one_sync() != 0,
        previous_stage_parities,
        unused_stage_mask,
        0,
        0,
        0};
}

}  // namespace halops::indexer
