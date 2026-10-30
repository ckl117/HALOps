// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <utility>

#include "indexer_mainloop.cuh"

namespace halops::indexer {

template <size_t... I>
CUTE_DEVICE void load_q_tmem_two_lanes(float (&values)[64], uint32_t address,
                                       std::index_sequence<I...>) {
    cute::SM100_TMEM_LOAD_16dp64b64x::copy(
        address, reinterpret_cast<uint32_t&>(values[I])...);
    cutlass::arch::fence_view_async_tmem_load();
}

CUTE_DEVICE uint32_t spread_nibbles(uint32_t value) {
    value &= 0xffff;
    value = (value | (value << 8)) & 0x00ff00ffu;
    return (value | (value << 4)) & 0x0f0f0f0fu;
}

template <class KernelTraits, class Context>
CUTE_DEVICE void q_two_lane_row(Context const& context, float (&values)[64],
                                int local_token,
                                const detail::GemmWorkItem& work_item,
                                int rope_slot, bool acquire_weights,
                                int query_sequence) {
    const auto& params = context.params;
    if (local_token >= work_item.valid_tokens) return;
    const int token = work_item.n_row + local_token;
    const int parity = (threadIdx.x >> 1) & 1;
    const unsigned mask = __activemask();
    const float* rope;
    if constexpr (KernelTraits::PrefetchRope)
        rope = context.plan.query.rope[rope_slot][local_token];
    else
        rope = params.rope +
               (int64_t)params.position_of(token) * IndexerDimensions::ROPE_DIM;
    if (params.is_neox_style) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 16; i += 2) {
            const int d = i * 2 + parity;
            const float2 a = make_float2(values[i], values[i + 1]);
            const float2 b = make_float2(values[i + 16], values[i + 17]);
            const float2 cosine = make_float2(rope[d], rope[d + 2]);
            const float2 sine = make_float2(rope[32 + d], rope[34 + d]);
            const float2 left = __ffma2_rn(
                a, cosine, __fmul2_rn(make_float2(-b.x, -b.y), sine));
            const float2 right = __ffma2_rn(a, sine, __fmul2_rn(b, cosine));
            values[i] = left.x;
            values[i + 1] = left.y;
            values[i + 16] = right.x;
            values[i + 17] = right.y;
        }
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 32; ++i) {
            const float peer = __shfl_xor_sync(mask, values[i], 2);
            values[i] = parity ? peer * rope[32 + i] + values[i] * rope[i]
                               : values[i] * rope[i] - peer * rope[32 + i];
        }
    }
    if (params.apply_hadamard) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 64; i += 2) {
            const float2 peer =
                make_float2(__shfl_xor_sync(mask, values[i], 2),
                            __shfl_xor_sync(mask, values[i + 1], 2));
            const float sign = parity ? -1.f : 1.f;
            const float2 pair =
                __ffma2_rn(make_float2(sign, sign),
                           make_float2(values[i], values[i + 1]), peer);
            values[i] = pair.x;
            values[i + 1] = pair.y;
        }
        thread_hadamard<64>(values);
        scale_values<64>(values, IndexerDimensions::HADAMARD_SCALE);
    }
    round_values_bf16<64>(values);
    const int64_t row =
        (int64_t)token * IndexerDimensions::NUM_HEADS + work_item.weight_tile;
    float scale = 1.f;
    if (params.output_fp4) {
        uint32_t scale_word = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int group = 0; group < 4; ++group) {
            float maximum = thread_amax<16>(values, group * 16);
            maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2));
            float inverse;
            const uint8_t exponent = mxfp4_scale(maximum, inverse);
            scale_word |= uint32_t(exponent) << (group * 8);
            uint32_t first = 0, second = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                first |= uint32_t(pack_scaled_e2m1x2(
                             values[group * 16 + 2 * i],
                             values[group * 16 + 2 * i + 1], inverse))
                         << (i * 8);
                second |= uint32_t(pack_scaled_e2m1x2(
                              values[group * 16 + 8 + 2 * i],
                              values[group * 16 + 8 + 2 * i + 1], inverse))
                          << (i * 8);
            }
            const uint32_t peer_first = __shfl_xor_sync(mask, first, 2);
            const uint32_t peer_second = __shfl_xor_sync(mask, second, 2);
            const uint32_t even = parity ? peer_second : first;
            const uint32_t odd = parity ? second : peer_first;
            *reinterpret_cast<uint2*>(params.q_output + row * 64 + group * 16 +
                                      parity * 8) =
                make_uint2(spread_nibbles(even) | (spread_nibbles(odd) << 4),
                           spread_nibbles(even >> 16) |
                               (spread_nibbles(odd >> 16) << 4));
        }
        if (parity == 0)
            store_output_scales(params.q_scale + row * 4, scale_word,
                                params.packed_q_scales);
    } else {
        float maximum = thread_amax<64>(values);
        maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2));
        scale = fp8_row_scale(maximum);
        const float inverse = 1.f / scale;
        CUTLASS_PRAGMA_UNROLL
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint32_t low[4], high[4];
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 4; ++j) {
                const int begin = chunk * 16 + j * 4;
                const uint32_t word =
                    pack_e4m3x4(values[begin], values[begin + 1],
                                values[begin + 2], values[begin + 3], inverse);
                const uint32_t peer = __shfl_xor_sync(mask, word, 2);
                const uint32_t even = parity ? peer : word;
                const uint32_t odd = parity ? word : peer;
                low[j] = __byte_perm(even, odd, 0x5140);
                high[j] = __byte_perm(even, odd, 0x7362);
            }
            *reinterpret_cast<uint4*>(params.q_output + row * 128 + chunk * 32 +
                                      parity * 16) =
                parity ? make_uint4(low[2], high[2], low[3], high[3])
                       : make_uint4(low[0], high[0], low[1], high[1]);
        }
    }
    if (!params.independent_weights) {
        if (params.raw_weight_prefetch_mode)
            context.plan.query_weights_ready[query_sequence & 1].wait(
                (query_sequence / 2) & 1);
        else if (acquire_weights)
            wait_group_ready_active(context.barriers.weights_ready(
                work_item.n_row / KernelTraits::TokenTile));
        if (parity == 0) {
            const float raw =
                params.raw_weight_prefetch_mode
                    ? context.plan
                          .query_weights[query_sequence & 1][local_token]
                    : params.raw_weights[row];
            params.weights[row] = raw * params.weight_scale * scale;
        }
    }
}

// Four drain warps cover the two 16-row halves of each TMEM quadrant in
// successive passes. Each lane holds only 64 values throughout the epilogue.
template <class KernelTraits, class Context>
CUTE_DEVICE void drain_q_four_split(Context const& context) {
    const auto& params = context.params;
    const auto ledger =
        CollectiveIndexer<KernelTraits>::barriers_before_query(context);
    const int warp = KernelTraits::warp_idx() - 4;
    const int lane = KernelTraits::lane_idx();
    int sequence = 0, previous_group = -1, rope_group = -1, rope_sequence = -1;
    for (int item = params.query_work_start(); item < params.query_work_items;
         item += params.query_work_stride(), ++sequence) {
        const auto work_item =
            CollectiveIndexer<KernelTraits>::query_work_item(context, item);
        const int group = work_item.n_row / KernelTraits::TokenTile;
        const int slot =
            detail::wait_accumulator_slot<KernelTraits::Query::AccStages>(
                context.query_resources.barriers,
                ledger.previous_accumulators(), sequence);
        intrinsics::tcgen05_after_thread_sync();
        if constexpr (KernelTraits::PrefetchRope) {
            if (work_item.n_row != rope_group) {
                ++rope_sequence;
                context.plan
                    .rope_ready[rope_sequence % KernelTraits::RopeStages]
                    .wait((rope_sequence / KernelTraits::RopeStages) & 1);
                rope_group = work_item.n_row;
            }
        }
        CUTLASS_PRAGMA_NO_UNROLL
        for (int half = 0; half < 2; ++half) {
            if (half * 16 >= work_item.valid_tokens) break;
            float values[64];
            const uint32_t address = (uint32_t(half * 16) << 16) + slot * 128;
            load_q_tmem_two_lanes(values, address,
                                  std::make_index_sequence<64>{});
            const int token = warp * 32 + half * 16 + lane / 4 + (lane & 1) * 8;
            q_two_lane_row<KernelTraits>(
                context, values, token, work_item,
                rope_sequence % KernelTraits::RopeStages,
                group != previous_group, sequence);
        }
        previous_group = group;
        if (params.raw_weight_prefetch_mode)
            context.plan.query_weights_free[sequence & 1].arrive();
        if constexpr (KernelTraits::PrefetchRope) {
            const int next = item + params.query_work_stride();
            if (next >= params.query_work_items ||
                CollectiveIndexer<KernelTraits>::query_work_item(context, next)
                        .n_row != work_item.n_row)
                context.plan.rope_free[rope_sequence % KernelTraits::RopeStages]
                    .arrive();
        }
        cutlass::arch::fence_view_async_tmem_load();
        context.query_resources.barriers.accumulator_free[slot].arrive();
    }
}

// Eight drain warps cover 128 tokens. Each pair of lanes owns alternating
// head dimensions, directly from the 16dp64b TMEM layout.
template <class KernelTraits, class Context>
CUTE_DEVICE void drain_q_eight(Context const& context) {
    const auto& params = context.params;
    if (!params.run_query_projection()) return;
    const int warp = KernelTraits::warp_idx() - 4;
    const int lane = KernelTraits::lane_idx();
    int sequence = 0, previous_group = -1, rope_group = -1, rope_sequence = -1;
    for (int item = params.query_work_start(); item < params.query_work_items;
         item += params.query_work_stride(), ++sequence) {
        const auto work_item =
            CollectiveIndexer<KernelTraits>::query_work_item(context, item);
        const int group = work_item.n_row / KernelTraits::TokenTile;
        const int slot =
            detail::wait_accumulator_slot<KernelTraits::Query::AccStages>(
                context.query_resources.barriers, 0, sequence);
        intrinsics::tcgen05_after_thread_sync();
        if constexpr (KernelTraits::PrefetchRope) {
            if (work_item.n_row != rope_group) {
                ++rope_sequence;
                context.plan
                    .rope_ready[rope_sequence % KernelTraits::RopeStages]
                    .wait((rope_sequence / KernelTraits::RopeStages) & 1);
                rope_group = work_item.n_row;
            }
        }
        float values[64];
        const uint32_t address = (uint32_t((warp / 4) * 16) << 16) + slot * 128;
        load_q_tmem_two_lanes(values, address, std::make_index_sequence<64>{});
        const int token =
            (warp % 4) * 32 + (warp / 4) * 16 + lane / 4 + (lane & 1) * 8;
        q_two_lane_row<KernelTraits>(context, values, token, work_item,
                                     rope_sequence % KernelTraits::RopeStages,
                                     group != previous_group, sequence);
        previous_group = group;
        if (params.raw_weight_prefetch_mode)
            context.plan.query_weights_free[sequence & 1].arrive();
        if constexpr (KernelTraits::PrefetchRope) {
            const int next = item + params.query_work_stride();
            if (next >= params.query_work_items ||
                CollectiveIndexer<KernelTraits>::query_work_item(context, next)
                        .n_row != work_item.n_row)
                context.plan.rope_free[rope_sequence % KernelTraits::RopeStages]
                    .arrive();
        }
        cutlass::arch::fence_view_async_tmem_load();
        context.query_resources.barriers.accumulator_free[slot].arrive();
    }
}

}  // namespace halops::indexer
