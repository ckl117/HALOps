// Copyright (c) 2026 WenXin Authors. All Rights Reserved.
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
//
// Adapted from InferRock 8d45d147 for HALOps indexer_prepare.

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "indexer_intrinsics.cuh"
#include "kernel/gemm/kernel_traits.cuh"

// Shapes, warp roles, pipeline limits and dispatch policy are defined here.
namespace halops::indexer::detail {

using e4m3 = cutlass::float_e4m3_t;
using ue8m0 = cutlass::float_ue8m0_t;
using bf16 = cutlass::bfloat16_t;

struct IndexerWarpTraits {
    static constexpr int NumThreads = 384;
    static constexpr int WarpSize = 32;
    static constexpr int NumWarps = NumThreads / WarpSize;
    static constexpr int ThreadsPerWarpGroup = 128;
    static constexpr int NumWarpGroups = NumThreads / ThreadsPerWarpGroup;
    static constexpr int WarpsPerWarpGroup = ThreadsPerWarpGroup / WarpSize;

    static constexpr int MinTokenTile = 16;

    static constexpr int RendezvousSlotBytes = 128;
    static constexpr int SingleTokenThreads = 256;
    static constexpr int TensorMemoryColumns = 512;
    static constexpr int SharedMemoryBudget = 226 * 1024;
    static constexpr int DefaultSharedMemoryLimit = 48 * 1024;

    struct Registers {
        static constexpr int Producer = 96;
        static constexpr int LocalQueryProducer = 160;
        static constexpr int AccumulatorConsumer = 232;
        static constexpr int EightWarpConsumer = 192;
        static constexpr int VectorConsumer = 176;
    };

    static CUTE_DEVICE int warp_idx() { return (int)(threadIdx.x / WarpSize); }
    static CUTE_DEVICE int lane_idx() {
        return (int)(threadIdx.x & (WarpSize - 1));
    }

    struct Barriers {
        template <uint32_t Id, int Width>
        struct Bar {
            static constexpr uint32_t id = Id;
            static constexpr int width = Width;
            static_assert(Width % WarpSize == 0,
                          "Named barriers require whole warps");
            static_assert(Width <= NumThreads,
                          "Barrier participants exceed the block size");
        };

        using QuantizerPair01Sync = Bar<2, 2 * WarpSize>;
        using QuantizerPair23Sync = Bar<3, 2 * WarpSize>;
        using EveryoneSync = Bar<4, NumThreads>;
        using AccumulatorGroupSync = Bar<5, ThreadsPerWarpGroup>;
        using VectorGroupSync = Bar<6, ThreadsPerWarpGroup>;
    };

    struct Roles {
        static constexpr int MmaWarp = 0;
        static constexpr int WeightLoaderWarp = 1;
        static constexpr int ActivationLoaderWarp = 2;
        static constexpr int PrefetchWarp = 3;
        static constexpr int DrainWarpBegin = 4;
        static constexpr int DrainWarps = WarpsPerWarpGroup;
        static constexpr int VectorWarpBegin = 8;
        static constexpr int VectorWarps = WarpsPerWarpGroup;

        __device__ static constexpr bool is_mma(int warp) {
            return warp == MmaWarp;
        }
        __device__ static constexpr bool is_weight_loader(int warp) {
            return warp == WeightLoaderWarp;
        }
        __device__ static constexpr bool is_activation_loader(int warp) {
            return warp == ActivationLoaderWarp;
        }
        __device__ static constexpr bool is_prefetch(int warp) {
            return warp == PrefetchWarp;
        }
        __device__ static constexpr bool is_drain(int warp) {
            return warp >= DrainWarpBegin && warp < DrainWarpBegin + DrainWarps;
        }
        __device__ static constexpr bool is_vector(int warp) {
            return warp >= VectorWarpBegin &&
                   warp < VectorWarpBegin + VectorWarps;
        }
        __device__ static constexpr int drain_local_warp(int warp) {
            return warp - DrainWarpBegin;
        }

        static constexpr int roles_matching(int warp) {
            return (int)(warp == MmaWarp) + (int)(warp == WeightLoaderWarp) +
                   (int)(warp == ActivationLoaderWarp) +
                   (int)(warp == PrefetchWarp) +
                   (int)(warp >= DrainWarpBegin &&
                         warp < DrainWarpBegin + DrainWarps) +
                   (int)(warp >= VectorWarpBegin &&
                         warp < VectorWarpBegin + VectorWarps);
        }

        static constexpr bool partitions_all_warps() {
            for (int warp = 0; warp < NumWarps; ++warp)
                if (roles_matching(warp) != 1) return false;
            return true;
        }

        static_assert(
            DrainWarpBegin % WarpsPerWarpGroup == 0 &&
                VectorWarpBegin % WarpsPerWarpGroup == 0,
            "Accumulator and vector roles require aligned warp groups");
        static_assert(
            MmaWarp < WarpsPerWarpGroup &&
                WeightLoaderWarp < WarpsPerWarpGroup &&
                ActivationLoaderWarp < WarpsPerWarpGroup &&
                PrefetchWarp < WarpsPerWarpGroup,
            "Matrix and loader roles must share the first warp group");
    };

    static_assert(NumWarps == 12 && NumWarpGroups == 3);
};

template <class GemmConfig>
struct ValidateGemmConfig {
    static_assert(
        GemmConfig::BM == 128,
        "M must match the 128-row scale tile and TMEM quadrant layout");
    static_assert(GemmConfig::BN >= IndexerWarpTraits::MinTokenTile &&
                      GemmConfig::BN <= GemmConfig::ScaleRowsPerTile &&
                      (GemmConfig::BN & (GemmConfig::BN - 1)) == 0,
                  "N must be a power of two in [16, 128]");
    static_assert(GemmConfig::BK > 0 && GemmConfig::KSplit > 0 &&
                      GemmConfig::TmaStages >= 1 && GemmConfig::AccStages >= 1,
                  "Tile, split and pipeline stage counts must be positive");
    static_assert(GemmConfig::KSplit <= GemmConfig::KSplitMax,
                  "Split count exceeds partial-result capacity");
    static_assert(GemmConfig::SwapAB || (GemmConfig::BN == GemmConfig::BM),
                  "Unswapped operands require a square tile");
    static constexpr bool ok = true;
};

template <int TokenTile_>
struct PipelineTraits : IndexerWarpTraits {
    struct GemmBase {
        static constexpr int TokenTile = TokenTile_;
        static constexpr int WeightTile = 128;
        static constexpr int BK = 256;
        static constexpr int KItersPerBlock = 8;

        static constexpr int ScaleGroupSize = 32;
        static constexpr int ScaleRowsPerColumn = 32;
        static constexpr int ScaleRowsPerTile = 4 * ScaleRowsPerColumn;
        static constexpr int MinPackedScaleTile = 64;
        static constexpr bool ScaleTilePerGroup =
            TokenTile < MinPackedScaleTile;

        static constexpr int NoSwapMinTile = 128;
        static constexpr bool SwapAB = TokenTile < NoSwapMinTile;
        static constexpr int BM = SwapAB ? WeightTile : TokenTile;
        static constexpr int BN = SwapAB ? TokenTile : WeightTile;

        static constexpr int TmaStages = 2;
        static constexpr int AccStages = 1;
        static constexpr int KSplit = 1;
        static constexpr int KSplitMax = 1;
    };

    static constexpr int MaxTmaStages = 2;
    static constexpr int MaxAccStages = 2;
};

using GemmShapeDefaults = PipelineTraits<128>::GemmBase;

template <int KBlockN_, int KPrologueStages_ = 1, int KEpilogueStages_ = 1,
          int KBlockM_ = 128, int KBlockK_ = 256>
struct IndexerGemmConfig : halops::GemmKernelConfig {
    static constexpr int KBlockM = KBlockM_;
    static constexpr int KBlockN = KBlockN_;
    static constexpr int KBlockK = KBlockK_;
    static constexpr bool IsPairUmma = false;
    static constexpr int KClusterM = 1;
    static constexpr int KClusterN = 1;

    static constexpr int KPrologueStages = KPrologueStages_;

    static constexpr int KEpilogueStages = KEpilogueStages_;
    static constexpr int KBiasStages = 1;
    static constexpr int KTmemOverlapCols = 0;
    static constexpr bool UseTileSwizzle = false;
    static constexpr halops::KernelMode KernelType = halops::KernelMode::Dense;
    static constexpr bool AddBias = false;
    static constexpr bool SwapAB = true;
    static constexpr bool KIsFP4 = false;
    static constexpr bool HasAlpha = false;
};

}  // namespace halops::indexer::detail

namespace halops::indexer {

using e4m3 = cutlass::float_e4m3_t;
using ue8m0 = cutlass::float_ue8m0_t;
using bf16 = cutlass::bfloat16_t;

struct IndexerDimensions {
    static constexpr int HIDDEN = 4096;
    static constexpr int HEAD_DIM = 128;
    static constexpr int NUM_HEADS = 64;
    static constexpr int QUERY_LORA_RANK = 1024;
    static constexpr int ROPE_DIM = 64;

    static constexpr int KEY_WEIGHT_ROWS = HEAD_DIM + NUM_HEADS;
    static constexpr int PADDED_KEY_WEIGHT_ROWS = 256;
    static constexpr int QUERY_ROWS = NUM_HEADS * HEAD_DIM;

    static constexpr int CACHE_SCALE_BYTES =
        4;  // One FP32 scale or four UE8M0 scales.
    static constexpr int FP8_CACHE_TOKEN_BYTES = HEAD_DIM + CACHE_SCALE_BYTES;
    static constexpr int FP4_GROUP_SIZE = 32;
    static constexpr int FP4_SCALE_BYTES = HEAD_DIM / FP4_GROUP_SIZE;
    static constexpr int FP4_CACHE_TOKEN_BYTES = HEAD_DIM / 2 + FP4_SCALE_BYTES;

    static constexpr float HADAMARD_SCALE =
        0.08838834764831845f;  // 1/sqrt(128)
    static constexpr float FP8_MAX = 448.0f;
    static constexpr float FP8_AMAX_FLOOR = 1e-4f;

    static_assert(HEAD_DIM == 2 * ROPE_DIM,
                  "RoPE covers the first half of head_dim");
    static_assert((HEAD_DIM & (HEAD_DIM - 1)) == 0,
                  "the 128-point Hadamard needs head_dim to be a power of two");
    static_assert(
        PADDED_KEY_WEIGHT_ROWS % 128 == 0 &&
            PADDED_KEY_WEIGHT_ROWS >= KEY_WEIGHT_ROWS,
        "Padded projection rows must cover K/weights in 128-row tiles");
    static_assert(HEAD_DIM % 32 == 0,
                  "the K epilogue is one warp per token with head_dim/32 "
                  "elements per lane");
};

template <int TokenTileSize, bool UseSharedRope = false,
          bool UseFp4Output = false, bool EightWarpQuery = false,
          bool NeoX = true, int RopeBuffers = 1, bool AlignedGlobalRope = false,
          bool TwoPassQuery = false>
struct IndexerTraitsFor : halops::indexer::detail::IndexerWarpTraits {
    using Dimensions = IndexerDimensions;
    static constexpr int TokenTile = TokenTileSize;
    static constexpr bool PrefetchRope = UseSharedRope;
    static constexpr bool OutputFp4 = UseFp4Output;
    static constexpr bool EightWarpEpilogue = EightWarpQuery;
    static constexpr bool IsNeoX = NeoX;
    static constexpr int RopeStages = RopeBuffers;
    static constexpr bool RopeAligned16 = UseSharedRope || AlignedGlobalRope;
    static constexpr bool SplitQueryDrain = TwoPassQuery;
    static_assert(RopeStages == 1 || RopeStages == 2);

    using GemmBase =
        typename halops::indexer::detail::PipelineTraits<TokenTile>::GemmBase;

    struct KeyWeight : GemmBase {
        static constexpr int BK = 128;  // bf16: 128 x 2 B == fp8's 256 x 1 B
        static constexpr int TmaStages = 2;
        static constexpr int AccStages = 2;
        static constexpr int KSplit = 16;
        static constexpr int KSplitMax = 16;
    };

    struct Query : GemmBase {
        static constexpr int TmaStages = 2;
        static constexpr int AccStages = 2;
        static constexpr int KSplit = 1;
        static constexpr int KSplitMax = 1;
    };

    static constexpr int MaxTmaStages = KeyWeight::TmaStages > Query::TmaStages
                                            ? KeyWeight::TmaStages
                                            : Query::TmaStages;
    static constexpr int MaxAccStages = KeyWeight::AccStages > Query::AccStages
                                            ? KeyWeight::AccStages
                                            : Query::AccStages;

    static_assert(KeyWeight::BK * 2 == Query::BK * 1,
                  "the two ranges' per-stage A/B byte counts must match");
    static_assert(MaxTmaStages <= halops::indexer::detail::PipelineTraits<
                                      TokenTile>::MaxTmaStages &&
                      MaxAccStages <= halops::indexer::detail::PipelineTraits<
                                          TokenTile>::MaxAccStages,
                  "Pipeline depth exceeds the barrier-array capacity");

    struct Workspace {
        static constexpr int GridBarrierBytes = 128;
        static constexpr int RendezvousSlotBytes = 128;
        static constexpr int BlockAlignBytes = 128;
        static constexpr int TokenPadRows = GemmBase::ScaleRowsPerTile;
        static_assert(TokenTile == 128 || TokenTile == 64 || TokenTile == 32 ||
                          TokenTile == 16,
                      "the narrow path is only dispatched for a single tile");
    };

    static constexpr bool validated =
        halops::indexer::detail::ValidateGemmConfig<KeyWeight>::ok &&
        halops::indexer::detail::ValidateGemmConfig<Query>::ok;
};

using IndexerTraits = IndexerTraitsFor<128>;
static_assert(IndexerTraits::validated);
static_assert(IndexerTraitsFor<16>::validated);
static_assert(IndexerTraitsFor<32>::validated);
static_assert(IndexerTraits::Roles::partitions_all_warps(),
              "Roles must partition all 12 warps exactly once");

inline int select_token_tile(int num_tokens) {
    return num_tokens <= 16 ? 16 : (num_tokens <= 32 ? 32 : 128);
}

inline int64_t align128(int64_t bytes) {
    constexpr int64_t alignment = IndexerTraits::Workspace::BlockAlignBytes;
    return (bytes + alignment - 1) / alignment * alignment;
}

// Narrow batches occupy one group; workspace rows remain padded to 128.
inline int token_group_count(int num_tokens) {
    const int tile = select_token_tile(num_tokens);
    return (num_tokens + tile - 1) / tile;
}

struct IndexerExecutionPhases {
    static constexpr int KeyWeightProjection = 1;
    static constexpr int KeyEpilogue = 2;
    static constexpr int QueryQuantization = 4;
    static constexpr int QueryProjection = 8;
    static constexpr int All = 15;
};

struct RawWeightPrefetch {
    static constexpr int Disabled = 0;
    static constexpr int RopeWarp = 1;
    static constexpr int WeightLoaderWarp = 2;
};

// Read and validate diagnostic overrides once per host call.
struct IndexerTuning {
    int sm_count;
    int grid_override, key_weight_split_override, enabled_phases,
        query_rotation_override, raw_weight_prefetch_override,
        key_weight_blocks;
    int two_pass_query_override;
    bool bound_activation_rows, use_256bit_stores, packed_scales,
        output_no_allocate;
    bool small_batch_rope, use_three_way_small_batch;

    static int read(const char* suffix, int fallback, int low, int high) {
        const std::string name =
            std::string("HALOPS_INDEXER_PREPARE_") + suffix;
        const char* text = std::getenv(name.c_str());
        if (!text) return fallback;
        int value = 0;
        const char* end = text + std::strlen(text);
        const auto parsed = std::from_chars(text, end, value);
        TORCH_CHECK(parsed.ec == std::errc{} && parsed.ptr == end &&
                        value >= low && value <= high,
                    name, " must be an integer in [", low, ", ", high,
                    "], got '", text, "'");
        return value;
    }

    static IndexerTuning from_environment() {
        const auto* device = at::cuda::getCurrentDeviceProperties();
        TORCH_CHECK(device->major == 10,
                    "indexer_prepare requires an SM100-family GPU");
        TORCH_CHECK(device->multiProcessorCount >= 128,
                    "indexer_prepare requires at least 128 available SMs, got ",
                    device->multiProcessorCount);
        IndexerTuning tuning{};
        tuning.sm_count = device->multiProcessorCount;
        tuning.grid_override =
            read("GRID", 0, 0, std::numeric_limits<int>::max());
        tuning.key_weight_split_override =
            read("KW_KSPLIT", 0, 0, IndexerTraits::KeyWeight::KSplitMax);
        TORCH_CHECK(tuning.key_weight_split_override == 0 ||
                        (tuning.key_weight_split_override &
                         (tuning.key_weight_split_override - 1)) == 0,
                    "HALOPS_INDEXER_PREPARE_KW_KSPLIT must be zero (auto) or "
                    "a power of two");
        tuning.enabled_phases = read("PHASES", IndexerExecutionPhases::All, 0,
                                     IndexerExecutionPhases::All);
        TORCH_CHECK(tuning.enabled_phases == 0 || tuning.enabled_phases == 1 ||
                        tuning.enabled_phases == 3 ||
                        tuning.enabled_phases == 7 ||
                        tuning.enabled_phases == IndexerExecutionPhases::All,
                    "HALOPS_INDEXER_PREPARE_PHASES must be one of 0/1/3/7/15");
        tuning.query_rotation_override = read("WQB_ROT", -1, -1, 1);
        tuning.bound_activation_rows = read("ACT_BOUND", 1, 0, 1);
        tuning.use_256bit_stores = read("STORE256", 1, 0, 1);
        tuning.packed_scales = read("PACKED_SCALES", 1, 0, 1);
        tuning.output_no_allocate = read("OUTPUT_NO_ALLOCATE", 1, 0, 1);
        tuning.small_batch_rope = read("SMALL_ROPE", 1, 0, 1);
        tuning.use_three_way_small_batch = read("SMALL_OPT", 1, 0, 1);
        tuning.raw_weight_prefetch_override =
            read("RAW_WEIGHT_SMEM", -1, -1, 2);
        tuning.key_weight_blocks =
            read("KW_CTAS", 0, 0, std::numeric_limits<int>::max());
        tuning.two_pass_query_override = read("TWO_PASS_QUERY", -1, -1, 1);
        return tuning;
    }
};

inline int select_grid_blocks(int num_tokens, const IndexerTuning& tuning,
                              bool fp8_schedule = false) {
    const int available_sms = tuning.sm_count;
    if (tuning.grid_override)
        return std::min(available_sms, tuning.grid_override);
    if (!fp8_schedule && num_tokens > 96 && num_tokens <= 128)
        return std::min(available_sms, 112);
    if (fp8_schedule && num_tokens >= 240 && num_tokens <= 256)
        return std::min(available_sms, 144);
    if (fp8_schedule && num_tokens > 384 && num_tokens <= 400)
        return available_sms;
    if (num_tokens == 1) return std::min(available_sms, 64);
    if (num_tokens <= 4) return std::min(available_sms, 80);
    if (num_tokens <= IndexerTraits::TokenTile)
        return std::min(available_sms, 96);
    if (num_tokens <= 2 * IndexerTraits::TokenTile)
        return std::min(available_sms, 128);
    if (num_tokens <= 1024) return std::min(available_sms, 144);
    return available_sms;
}

inline int select_key_weight_splits(int tokens, int grid_blocks,
                                    const IndexerTuning& tuning,
                                    bool fp8_schedule = false) {
    if (tuning.key_weight_split_override)
        return tuning.key_weight_split_override;
    if (fp8_schedule && tokens >= 240 && tokens <= 256) return 4;
    if (tokens > 128 && tokens <= 256) return 8;
    if (tokens > 256 && tokens < 512 && tokens % 128 != 0) return 8;
    if (tokens > 256 && tokens <= 1024) return 4;
    constexpr int weight_tiles =
        IndexerDimensions::PADDED_KEY_WEIGHT_ROWS / IndexerTraits::GemmBase::BM;
    const int groups = token_group_count(tokens);
    int splits = 1;
    while (splits * 2 <= IndexerTraits::KeyWeight::KSplitMax &&
           (int64_t)weight_tiles * groups * (splits * 2) <= grid_blocks)
        splits *= 2;
    return splits;
}

template <class Visit>
void visit_workspace_schedules(int max_tokens, const IndexerTuning& tuning,
                               const Visit& visit) {
    auto include_shape = [&](int tokens) {
        for (bool fp8_schedule : {false, true}) {
            const int grid_blocks =
                select_grid_blocks(tokens, tuning, fp8_schedule);
            visit(tokens, select_key_weight_splits(tokens, grid_blocks, tuning,
                                                   fp8_schedule));
        }
    };
    // Split-K is non-monotonic: reserve the maximum across both formats.
    for (int tokens = 1; tokens <= std::min(max_tokens, 1024); ++tokens)
        include_shape(tokens);
    if (max_tokens > 1024) {
        include_shape(max_tokens);
        // Above 1024, scratch maxima occur at split-K interval ends.
        const int grid_blocks = select_grid_blocks(max_tokens, tuning);
        for (int split = 1; split <= IndexerTraits::KeyWeight::KSplitMax;
             split *= 2) {
            const int64_t endpoint = (int64_t)(grid_blocks / (2 * split)) *
                                     IndexerTraits::Workspace::TokenPadRows;
            if (endpoint > 1024 && endpoint < max_tokens)
                include_shape((int)endpoint);
        }
    }
}

inline int select_query_block_offset(int key_weight_work_items,
                                     int query_work_items, int grid_blocks,
                                     const IndexerTuning& tuning) {
    if (tuning.query_rotation_override == 0 || key_weight_work_items == 0 ||
        query_work_items == 0)
        return 0;
    // Prioritize CTAs with no KW work for the longest Q work lists.
    return std::min(grid_blocks - 1, key_weight_work_items);
}

template <class KernelTraits, class Parameters>
void configure_indexer_schedule(Parameters& params, const IndexerTuning& tuning,
                                bool query_aligned_32) {
    using Dimensions = typename KernelTraits::Dimensions;
    constexpr int WEIGHT_TILES =
        Dimensions::PADDED_KEY_WEIGHT_ROWS / KernelTraits::KeyWeight::BM;
    params.simt_projection_enabled =
        params.num_tokens == 1 && params.grid_blocks == Dimensions::NUM_HEADS;
    params.key_weight_items_per_group = WEIGHT_TILES * params.key_weight_splits;
    params.key_weight_work_items =
        params.key_weight_items_per_group * params.token_groups;
    if (params.uses_simt_projection()) params.key_weight_work_items = 0;
    const int enabled_phases = tuning.enabled_phases;
    constexpr int max_local_query_tokens = 32;
    params.local_query_enabled =
        params.num_tokens >= 2 && params.num_tokens <= max_local_query_tokens &&
        params.grid_blocks ==
            (params.num_tokens <= 4 ? 80 : 64 + 2 * params.key_weight_splits) &&
        (params.key_weight_splits == 16 ||
         (params.num_tokens > 4 && params.key_weight_splits >= 4)) &&
        enabled_phases == IndexerExecutionPhases::All && !params.skip_q;
    if (params.uses_local_query() && params.num_tokens <= 4) {
        params.key_weight_items_per_group = params.key_weight_splits;
        params.key_weight_work_items = params.key_weight_splits;
    }
    params.query_items_per_group =
        params.skip_q ? 0 : Dimensions::QUERY_ROWS / KernelTraits::Query::BM;
    params.query_work_items =
        params.query_items_per_group * params.token_groups;
    params.use_256bit_stores = query_aligned_32 && tuning.use_256bit_stores;
    params.packed_q_scales = tuning.packed_scales;
    params.output_no_allocate = tuning.output_no_allocate;
    params.prefetch_small_rope = params.uses_local_query() &&
                                 params.num_tokens > 4 &&
                                 tuning.small_batch_rope;
    params.raw_weight_prefetch_mode =
        KernelTraits::PrefetchRope && !KernelTraits::OutputFp4 &&
                !params.skip_q && enabled_phases == IndexerExecutionPhases::All
            ? (tuning.raw_weight_prefetch_override >= 0
                   ? tuning.raw_weight_prefetch_override
                   : (params.num_tokens <= 256 || params.num_tokens % 128 == 0
                          ? RawWeightPrefetch::RopeWarp
                          : RawWeightPrefetch::WeightLoaderWarp))
            : RawWeightPrefetch::Disabled;
    params.query_block_offset = halops::indexer::select_query_block_offset(
        params.key_weight_work_items, params.query_work_items,
        params.grid_blocks, tuning);
    // Eight-warp Q blocks use a separate range from K/weight blocks.
    params.key_weight_blocks =
        KernelTraits::EightWarpEpilogue
            ? std::clamp(
                  tuning.key_weight_blocks
                      ? tuning.key_weight_blocks
                      : (KernelTraits::OutputFp4 && params.num_tokens > 96
                             ? 48
                             : 32),
                  1, params.grid_blocks - 1)
            : 0;
    // More than two Q items need an independent raw-weight loader to avoid
    // deadlock.
    if (params.raw_weight_prefetch_mode ==
            RawWeightPrefetch::WeightLoaderWarp &&
        params.query_work_items >
            2 * (params.grid_blocks - params.key_weight_blocks))
        params.raw_weight_prefetch_mode = RawWeightPrefetch::RopeWarp;
    params.vector_blocks = params.key_weight_blocks ? params.key_weight_blocks
                                                    : params.grid_blocks;
    if (params.uses_local_query())
        params.vector_blocks = params.key_weight_work_items;
    params.independent_weights =
        params.output_fp4 && !params.uses_simt_projection() &&
        (!params.uses_local_query() || params.num_tokens > 4) &&
        !params.skip_q && enabled_phases == IndexerExecutionPhases::All;
    params.enabled_phases = enabled_phases;
    if (params.uses_local_query())
        params.query_block_offset = params.key_weight_work_items;
}

enum class IndexerKernelKind {
    General,
    SingleToken,
    LocalQuery,
    ThreeWaySmallBatch
};

template <class KernelTraits, class Parameters>
IndexerKernelKind select_indexer_kernel(const Parameters& params,
                                        const IndexerTuning& tuning) {
    if constexpr (KernelTraits::TokenTile == 16) {
        if (params.uses_local_query() && params.num_tokens >= 5 &&
            params.enabled_phases == IndexerExecutionPhases::All &&
            tuning.use_three_way_small_batch)
            return IndexerKernelKind::ThreeWaySmallBatch;
        if (params.uses_simt_projection() &&
            params.enabled_phases == IndexerExecutionPhases::All)
            return IndexerKernelKind::SingleToken;
    }
    if constexpr (KernelTraits::TokenTile <= 64)
        if (params.uses_local_query()) return IndexerKernelKind::LocalQuery;
    return IndexerKernelKind::General;
}

CUTE_HOST_DEVICE constexpr int query_epilogue_lanes(int tokens,
                                                    bool local_query) {
    if (tokens <= 4) return 32;
    if (!local_query) return 8;
    if (tokens <= 8) return 16;
    return tokens > 16 ? 4 : 8;
}

template <bool OutputFp4, bool NeoX, bool TwoPassQuery = false, class Launch>
void dispatch_indexer_format(int tokens, bool skip_q, bool rope_aligned_16,
                             const IndexerTuning& tuning,
                             const Launch& launch) {
    if constexpr (!TwoPassQuery) {
        const bool selected =
            (!OutputFp4 && tokens > 128 && tokens <= 144) ||
            (tokens > 256 && tokens < (OutputFp4 ? 512 : 384) &&
             tokens % 128 > 0 && tokens % 128 <= (OutputFp4 ? 48 : 16));
        if (tokens > 128 && (tuning.two_pass_query_override < 0
                                 ? selected
                                 : tuning.two_pass_query_override != 0)) {
            dispatch_indexer_format<OutputFp4, NeoX, true>(
                tokens, skip_q, rope_aligned_16, tuning, launch);
            return;
        }
    }
    const int tile = select_token_tile(tokens);
    if (tile == 16)
        launch(IndexerTraitsFor<16, false, OutputFp4, false, NeoX>{});
    else if (tile == 32)
        launch(IndexerTraitsFor<32, false, OutputFp4, false, NeoX>{});
    else if (tokens <= 128 && !skip_q &&
             select_grid_blocks(tokens, tuning) > 32)
        launch(IndexerTraitsFor<128, true, OutputFp4, true, NeoX>{});
    else {
        if constexpr (OutputFp4) {
            if (tokens > 128 && tokens <= 256) {
                if (rope_aligned_16)
                    launch(IndexerTraitsFor<128, false, true, false, NeoX, 1,
                                            true, TwoPassQuery>{});
                else
                    launch(IndexerTraitsFor<128, false, true, false, NeoX, 1,
                                            false, TwoPassQuery>{});
                return;
            }
        }
        if (tokens > 256 && tokens % 128 != 0)
            launch(IndexerTraitsFor<128, true, OutputFp4, false, NeoX, 2, false,
                                    TwoPassQuery>{});
        else
            launch(IndexerTraitsFor<128, true, OutputFp4, false, NeoX, 1, false,
                                    TwoPassQuery>{});
    }
}

}  // namespace halops::indexer
