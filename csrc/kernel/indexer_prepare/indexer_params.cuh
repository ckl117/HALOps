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

#include <algorithm>
#include <cmath>
#include <limits>

#include "indexer_config.cuh"
#include "indexer_gemm_detail.cuh"
#include "indexer_utils.cuh"

namespace halops::indexer {

template <class KernelTraits>
struct IndexerSharedStorage {
    using Arena = halops::indexer::detail::GemmArena;
    using Dimensions = typename KernelTraits::Dimensions;
    using Roles = typename KernelTraits::Roles;

    union {
        char gemm[halops::indexer::detail::pipeline_smem_bytes<
            KernelTraits::TokenTile>()];
    } gemm_storage;

    static constexpr int QStageStride = Dimensions::HEAD_DIM + 4;
    union {
        float transpose[KernelTraits::Query::SwapAB ? KernelTraits::TokenTile
                                                    : 1][QStageStride];
        alignas(16) float rope[KernelTraits::RopeStages]
                              [KernelTraits::PrefetchRope
                                   ? KernelTraits::TokenTile
                                   : 1][Dimensions::ROPE_DIM + 4];
    } query;

    cute::array_aligned<uint32_t, 1> tmem_start_addr;

    transac_bar_t tma_a_ready[Arena::MaxTmaStages];
    transac_bar_t tma_b_ready[Arena::MaxTmaStages];
    transac_bar_t gemm_buffer_free[Arena::MaxTmaStages];
    transac_bar_t accumulator_ready[Arena::MaxAccStages];
    transac_bar_t accumulator_free[Arena::MaxAccStages];
    transac_bar_t local_weight_ready;
    transac_bar_t norm_ready;
    transac_bar_t rope_ready[KernelTraits::RopeStages];
    transac_bar_t rope_free[KernelTraits::RopeStages];
    transac_bar_t query_weights_ready[2];
    transac_bar_t query_weights_free[2];
    float local_weight;
    float local_weights[64];
    alignas(16) float norm_weight[Dimensions::HEAD_DIM];
    alignas(16) float norm_bias[Dimensions::HEAD_DIM];
    alignas(16) float small_batch_rope[KernelTraits::TokenTile <= 32
                                           ? KernelTraits::TokenTile
                                           : 1][Dimensions::ROPE_DIM + 4];
    alignas(16) float query_weights[2][KernelTraits::PrefetchRope &&
                                               !KernelTraits::OutputFp4
                                           ? KernelTraits::TokenTile
                                           : 1];

    static constexpr int TMEM_COLS = KernelTraits::TensorMemoryColumns;

    static_assert(sizeof(gemm_storage) + sizeof(query) <
                      KernelTraits::SharedMemoryBudget,
                  "GEMM and Q staging exceed the shared-memory budget");

    static_assert(Arena::MaxTmaStages >= KernelTraits::MaxTmaStages,
                  "bar_gemm_tma_a must cover this kernel's deepest GEMM");
    static_assert(
        Arena::MaxAccStages >= KernelTraits::MaxAccStages,
        "bar_gemm_acc_* must cover this kernel's deepest accumulator");

    CUTE_DEVICE halops::indexer::detail::GemmPipelineBarriers gemm_barriers() {
        return {tma_a_ready, tma_b_ready, gemm_buffer_free, accumulator_ready,
                accumulator_free};
    }

    CUTE_DEVICE char* gemm_shared_memory() { return gemm_storage.gemm; }

    CUTE_DEVICE void initialize_gemm_barriers(int drain_threads) {
        CUTE_UNROLL
        for (int i = 0; i < Arena::MaxTmaStages; ++i) {
            tma_a_ready[i].init(1);
            tma_b_ready[i].init(1);
            gemm_buffer_free[i].init(1);
        }
        CUTE_UNROLL
        for (int i = 0; i < Arena::MaxAccStages; ++i) {
            accumulator_ready[i].init(1);
            accumulator_free[i].init(drain_threads);
        }
    }

    CUTE_DEVICE void initialize_barriers(int drain_threads = 128) {
        initialize_gemm_barriers(drain_threads);
        local_weight_ready.init(1);
        norm_ready.init(32);
        CUTE_UNROLL
        for (int stage = 0; stage < KernelTraits::RopeStages; ++stage) {
            rope_ready[stage].init(32);
            rope_free[stage].init(drain_threads);
        }
        if constexpr (KernelTraits::PrefetchRope && !KernelTraits::OutputFp4) {
            CUTE_UNROLL
            for (int stage = 0; stage < 2; ++stage) {
                query_weights_ready[stage].init(32);
                query_weights_free[stage].init(drain_threads);
            }
        }
        cutlass::arch::fence_barrier_init();
    }

    CUTE_DEVICE void acquire_tmem() {
        cute::TMEM::Allocator1Sm().allocate(TMEM_COLS, tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    CUTE_DEVICE void release_tmem() {
        cute::TMEM::Allocator1Sm().free(0, TMEM_COLS);
    }
};

struct IndexerPrepareArguments {
    torch::Tensor x;
    torch::Tensor q_lora;
    torch::Tensor wk_w;
    torch::Tensor wq_b;
    torch::Tensor wq_b_sf;
    torch::Tensor k_norm_weight;
    torch::Tensor k_norm_bias;
    double k_norm_eps;
    torch::Tensor rope_cos_sin_cache;
    torch::Tensor positions;
    torch::Tensor paged_k_cache;
    torch::Tensor out_cache_loc;
    torch::Tensor q;
    torch::Tensor weights;
    torch::Tensor workspace;
    double weight_scale;
    int64_t head_dim;
    int64_t n_heads;
    bool apply_hadamard;
    bool skip_q;
    torch::Tensor sf_q;
    bool output_fp4;
    bool is_neox_style;
    bool use_pdl;
};

template <class KernelTraits>
struct IndexerPrepareWorkspace {
    using Dimensions = typename KernelTraits::Dimensions;
    using WorkspaceConfig = typename KernelTraits::Workspace;

    enum : int {
        QUERY_LORA_READY,
        KEY_WEIGHT_READY,
        WEIGHTS_READY,
        QUERY_SCALE_READY
    };
    static constexpr int COUNTER_BANKS = QUERY_SCALE_READY + 1;

    struct Layout {
        int counter_bank_stride;
        int64_t counter_offset;
        int64_t counter_bytes;
        int64_t scratch_offset;
    };

    static Layout layout(int token_groups) {
        Layout geometry{};
        geometry.counter_bank_stride = std::max(1, token_groups);
        geometry.counter_offset = WorkspaceConfig::GridBarrierBytes;
        geometry.counter_bytes = (int64_t)COUNTER_BANKS *
                                 geometry.counter_bank_stride *
                                 WorkspaceConfig::RendezvousSlotBytes;
        geometry.scratch_offset = halops::indexer::align128(
            geometry.counter_offset + geometry.counter_bytes);
        return geometry;
    }

    struct ScratchRequirements {
        int key_weight_splits;
        int64_t bytes_per_token;

        int64_t bytes(const Layout& geometry, int padded_tokens) const {
            return geometry.scratch_offset +
                   (int64_t)padded_tokens * bytes_per_token;
        }
    };

    static ScratchRequirements dimensions(int key_weight_splits) {
        ScratchRequirements requirements{};
        requirements.key_weight_splits = key_weight_splits;
        requirements.bytes_per_token =
            (int64_t)key_weight_splits * Dimensions::KEY_WEIGHT_ROWS *
                4                                 // projection partials
            + (int64_t)Dimensions::NUM_HEADS * 4  // raw head weights
            + Dimensions::QUERY_LORA_RANK         // quantized query input
            + Dimensions::QUERY_LORA_RANK /
                  KernelTraits::GemmBase::ScaleGroupSize;  // input scales
        return requirements;
    }

    static int padded_token_count(int num_tokens) {
        constexpr int alignment = WorkspaceConfig::TokenPadRows;
        return (num_tokens + alignment - 1) / alignment * alignment;
    }

    static int64_t min_group_bytes() {
        return (int64_t)WorkspaceConfig::TokenPadRows *
               dimensions(1).bytes_per_token;
    }

    static Layout layout_for_capacity(int64_t capacity_bytes) {
        // Capacity alone fixes the counter/scratch boundary across batch
        // changes.
        constexpr int64_t counter_bytes =
            COUNTER_BANKS * WorkspaceConfig::RendezvousSlotBytes;
        const int64_t groups = std::max<int64_t>(
            1, (capacity_bytes - WorkspaceConfig::GridBarrierBytes) /
                   (min_group_bytes() + counter_bytes));
        return layout((int)std::min<int64_t>(
            groups,
            std::numeric_limits<int>::max() / WorkspaceConfig::TokenPadRows));
    }

    static int64_t allocation_bytes(int64_t scratch_bytes) {
        const int groups = (int)std::min<int64_t>(
            std::max<int64_t>(1, scratch_bytes / min_group_bytes()),
            std::numeric_limits<int>::max() / WorkspaceConfig::TokenPadRows);
        // layout_for_capacity recovers this group count from the allocation
        // size.
        return layout(groups).scratch_offset + scratch_bytes;
    }
};

template <class KernelTraits>
struct IndexerPrepareProblem {
    using Dimensions = typename KernelTraits::Dimensions;
    using Workspace = IndexerPrepareWorkspace<KernelTraits>;
    static constexpr int TokenTile = KernelTraits::TokenTile;

    int num_tokens, padded_tokens, grid_blocks, token_groups;
    int key_weight_splits, key_weight_tiles_per_split;
    int page_size, page_shift;
    int x_row_stride, query_lora_row_stride;
    bool positions_are_int64, cache_slots_are_int64;
    float weight_scale, k_norm_eps;
    bool use_layernorm;

    typename Workspace::ScratchRequirements workspace_dimensions;
    typename Workspace::Layout workspace_layout;

    char* base;
    float* key_weight_partials;
    float* raw_weights;
    halops::indexer::e4m3* quantized_query_lora;
    halops::indexer::ue8m0* query_lora_scales;

    static int page_size_shift(int64_t page) {
        int shift = 0;
        while ((1LL << shift) < page) ++shift;
        return shift;
    }

    static void check_indices(const char* name, const torch::Tensor& tensor,
                              int num_tokens, bool& is_int64) {
        TORCH_CHECK(tensor.scalar_type() == torch::kInt ||
                        tensor.scalar_type() == torch::kLong,
                    name, " must be int32 or int64, got ",
                    tensor.scalar_type());
        TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == num_tokens, name,
                    " must be [T]=", num_tokens, ", got ", tensor.sizes());
        TORCH_CHECK(tensor.stride(0) == 1, name, " must be contiguous");
        is_int64 = tensor.scalar_type() == torch::kLong;
    }

    static void check_storage(const char* name, const torch::Tensor& tensor,
                              const torch::Device& device, int alignment,
                              bool contiguous = true) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == device, name,
                    " must be on the same CUDA device as x");
        TORCH_CHECK(!contiguous || tensor.is_contiguous(), name,
                    " must be contiguous");
        TORCH_CHECK(
            reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0,
            name, " must be ", alignment, " B aligned");
    }

    static IndexerPrepareProblem from(const IndexerPrepareArguments& args,
                                      const IndexerTuning& tuning) {
        IndexerPrepareProblem params{};
        TORCH_CHECK(args.x.is_cuda(), "x must be CUDA");
        const auto device = args.x.device();
        check_storage("x", args.x, device, 16, false);
        check_storage("q_lora", args.q_lora, device, 8, false);
        check_storage("wk_w", args.wk_w, device, 16);
        check_storage("wq_b", args.wq_b, device, 16);
        check_storage("wq_b_sf", args.wq_b_sf, device, 16);
        check_storage("k_norm_weight", args.k_norm_weight, device, 16);
        check_storage("k_norm_bias", args.k_norm_bias, device, 16);
        check_storage("rope_cos_sin_cache", args.rope_cos_sin_cache, device, 8);
        check_storage("positions", args.positions, device,
                      args.positions.element_size());
        check_storage("out_cache_loc", args.out_cache_loc, device,
                      args.out_cache_loc.element_size());
        check_storage("paged_kv_cache_with_scale", args.paged_k_cache, device,
                      4);
        check_storage("q", args.q, device, 16);
        if (args.output_fp4) {
            TORCH_CHECK(args.sf_q.defined(),
                        "sf_q is required for MXFP4 output");
            check_storage("sf_q", args.sf_q, device, 1);
        } else {
            TORCH_CHECK(!args.sf_q.defined(),
                        "sf_q must be None for FP8 output");
        }
        check_storage("weights", args.weights, device, 16);
        check_storage("workspace", args.workspace, device, 128);
        TORCH_CHECK(std::isfinite(args.k_norm_eps) && args.k_norm_eps > 0 &&
                        std::isfinite(static_cast<float>(args.k_norm_eps)) &&
                        static_cast<float>(args.k_norm_eps) > 0,
                    "k_norm_eps must be finite and positive in float32");
        TORCH_CHECK(std::isfinite(static_cast<float>(args.weight_scale)),
                    "weight_scale must be finite in float32");
        TORCH_CHECK(
            args.x.dim() == 2 && args.x.scalar_type() == torch::kBFloat16,
            "x must be a 2-D bfloat16 [num_tokens, hidden], got ",
            args.x.sizes(), " ", args.x.scalar_type());
        TORCH_CHECK(args.x.stride(1) == 1,
                    "x's last dim must have stride 1 (the bf16 TMA reads it "
                    "K-major)");
        TORCH_CHECK(args.head_dim == Dimensions::HEAD_DIM &&
                        args.n_heads == Dimensions::NUM_HEADS,
                    "this build compiles head_dim=", (int)Dimensions::HEAD_DIM,
                    ", n_heads=", (int)Dimensions::NUM_HEADS, "; got ",
                    args.head_dim, ", ", args.n_heads);
        TORCH_CHECK(args.x.size(1) == Dimensions::HIDDEN,
                    "this build compiles hidden=", (int)Dimensions::HIDDEN,
                    ", got ", args.x.size(1));

        TORCH_CHECK(
            args.x.size(0) > 0 &&
                args.x.size(0) <= std::numeric_limits<int>::max() - TokenTile,
            "num_tokens must be positive and fit in int32 with padding");
        params.num_tokens = args.x.size(0);
        const int num_tokens = params.num_tokens;
        TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
        TORCH_CHECK(args.x.stride(0) >= Dimensions::HIDDEN &&
                        args.x.stride(0) <= std::numeric_limits<int>::max() &&
                        args.x.stride(0) % 8 == 0,
                    "x row stride must cover hidden and be 16 B aligned");
        params.x_row_stride = (int)args.x.stride(0);

        TORCH_CHECK(
            args.wk_w.dim() == 2 && args.wk_w.scalar_type() == torch::kBFloat16,
            "wk_w must be 2-D bfloat16");
        TORCH_CHECK(args.wk_w.size(0) == Dimensions::KEY_WEIGHT_ROWS,
                    "wk_w must be [head_dim + n_heads, hidden] = [",
                    (int)Dimensions::KEY_WEIGHT_ROWS, ", ",
                    (int)Dimensions::HIDDEN, "], got ", args.wk_w.sizes());
        TORCH_CHECK(args.wk_w.size(1) == Dimensions::HIDDEN &&
                        args.wk_w.stride(1) == 1 &&
                        args.wk_w.stride(0) == Dimensions::HIDDEN,
                    "wk_w must be contiguous K-major [",
                    (int)Dimensions::KEY_WEIGHT_ROWS, ", ",
                    (int)Dimensions::HIDDEN, "], got ", args.wk_w.sizes(),
                    " strides ", args.wk_w.strides());

        TORCH_CHECK(
            args.q_lora.dim() == 2 && args.q_lora.size(0) == num_tokens &&
                args.q_lora.size(1) == Dimensions::QUERY_LORA_RANK,
            "q_lora must be [T, q_lora_rank] = [", num_tokens, ", ",
            (int)Dimensions::QUERY_LORA_RANK, "], got ", args.q_lora.sizes());
        TORCH_CHECK(args.q_lora.stride(1) == 1,
                    "q_lora's last dim must have stride 1");
        TORCH_CHECK(args.q_lora.scalar_type() == torch::kBFloat16,
                    "q_lora must be bfloat16");
        TORCH_CHECK(
            args.q_lora.stride(0) >= Dimensions::QUERY_LORA_RANK &&
                args.q_lora.stride(0) <= std::numeric_limits<int>::max() &&
                args.q_lora.stride(0) % 4 == 0,
            "q_lora row stride must cover q_lora_rank and be 8 B aligned");
        params.query_lora_row_stride = (int)args.q_lora.stride(0);

        TORCH_CHECK(args.wq_b.dim() == 2 &&
                        args.wq_b.scalar_type() == torch::kFloat8_e4m3fn,
                    "wq_b must be 2-D float8_e4m3fn, got ", args.wq_b.sizes(),
                    " ", args.wq_b.scalar_type());
        TORCH_CHECK(args.wq_b.size(0) == Dimensions::QUERY_ROWS &&
                        args.wq_b.size(1) == Dimensions::QUERY_LORA_RANK &&
                        args.wq_b.stride(1) == 1 &&
                        args.wq_b.stride(0) == Dimensions::QUERY_LORA_RANK,
                    "wq_b must be contiguous K-major [",
                    (int)Dimensions::QUERY_ROWS, ", ",
                    (int)Dimensions::QUERY_LORA_RANK, "], got ",
                    args.wq_b.sizes());
        TORCH_CHECK(args.wq_b_sf.scalar_type() == torch::kByte &&
                        args.wq_b_sf.dim() == 2 &&
                        args.wq_b_sf.size(0) == Dimensions::QUERY_ROWS &&
                        args.wq_b_sf.size(1) ==
                            Dimensions::QUERY_LORA_RANK /
                                KernelTraits::GemmBase::ScaleGroupSize,
                    "wq_b_sf must be the SfAtom-swizzled uint8 [N, K/32] = [",
                    (int)Dimensions::QUERY_ROWS, ", ",
                    (int)(Dimensions::QUERY_LORA_RANK /
                          KernelTraits::GemmBase::ScaleGroupSize),
                    "] the block-scaled UMMA reads directly, got ",
                    args.wq_b_sf.sizes(), " ", args.wq_b_sf.scalar_type());

        TORCH_CHECK(args.k_norm_weight.scalar_type() == torch::kFloat &&
                        args.k_norm_weight.dim() == 1 &&
                        args.k_norm_weight.numel() == Dimensions::HEAD_DIM,
                    "k_norm_weight must be float32 [head_dim]=",
                    (int)Dimensions::HEAD_DIM);
        params.use_layernorm = args.k_norm_bias.numel() > 0;
        TORCH_CHECK(!params.use_layernorm ||
                        (args.k_norm_bias.scalar_type() == torch::kFloat &&
                         args.k_norm_bias.dim() == 1 &&
                         args.k_norm_bias.numel() == Dimensions::HEAD_DIM),
                    "k_norm_bias must be float32 [head_dim] or empty (empty "
                    "selects RMSNorm, which has no bias by definition)");
        params.k_norm_eps = (float)args.k_norm_eps;

        TORCH_CHECK(args.rope_cos_sin_cache.scalar_type() == torch::kFloat &&
                        args.rope_cos_sin_cache.dim() == 2 &&
                        args.rope_cos_sin_cache.size(0) > 0 &&
                        args.rope_cos_sin_cache.size(1) == Dimensions::ROPE_DIM,
                    "rope_cos_sin_cache must be float32 [max_position, ",
                    (int)Dimensions::ROPE_DIM,
                    "] in halves layout [cos | sin]; this build compiles "
                    "rope_dim=",
                    (int)Dimensions::ROPE_DIM, ", got ",
                    args.rope_cos_sin_cache.sizes());
        TORCH_CHECK(
            args.rope_cos_sin_cache.stride(1) == 1 &&
                args.rope_cos_sin_cache.stride(0) == Dimensions::ROPE_DIM,
            "rope_cos_sin_cache must be contiguous");
        check_indices("positions", args.positions, num_tokens,
                      params.positions_are_int64);
        check_indices("out_cache_loc", args.out_cache_loc, num_tokens,
                      params.cache_slots_are_int64);

        const int token_bytes = args.output_fp4
                                    ? Dimensions::FP4_CACHE_TOKEN_BYTES
                                    : Dimensions::FP8_CACHE_TOKEN_BYTES;
        const int q_row_bytes =
            args.output_fp4 ? Dimensions::HEAD_DIM / 2 : Dimensions::HEAD_DIM;
        TORCH_CHECK(args.paged_k_cache.scalar_type() == torch::kByte &&
                        args.paged_k_cache.dim() == 2,
                    "paged_kv_cache_with_scale must be 2-D uint8 [num_pages, "
                    "page_size * token_bytes] (132 for fp8, 68 for fp4), got ",
                    args.paged_k_cache.sizes(), " ",
                    args.paged_k_cache.scalar_type());
        const int64_t row = args.paged_k_cache.size(1);
        TORCH_CHECK(row <= std::numeric_limits<int>::max() &&
                        args.paged_k_cache.size(0) > 0 &&
                        args.paged_k_cache.size(0) <=
                            std::numeric_limits<int>::max() /
                                std::max<int64_t>(1, row / token_bytes),
                    "K cache capacity must be positive and fit in int32 slots");
        TORCH_CHECK(row % token_bytes == 0,
                    "the K cache's row must be page_size * ", token_bytes,
                    " bytes, got ", row);
        params.page_size = (int)(row / token_bytes);
        TORCH_CHECK(params.page_size > 0 &&
                        (params.page_size & (params.page_size - 1)) == 0,
                    "the page_size implied by the cache shape must be a power "
                    "of two, got ",
                    params.page_size);
        TORCH_CHECK(args.paged_k_cache.stride(1) == 1,
                    "paged_kv_cache_with_scale must be row-contiguous");
        params.page_shift = page_size_shift(params.page_size);

        TORCH_CHECK(
            args.q.scalar_type() ==
                    (args.output_fp4 ? torch::kByte : torch::kFloat8_e4m3fn) &&
                args.q.dim() == 3 && args.q.size(0) == num_tokens &&
                args.q.size(1) == Dimensions::NUM_HEADS &&
                (int)args.q.size(2) == q_row_bytes && args.q.is_contiguous(),
            "q must be contiguous ",
            args.output_fp4 ? "uint8" : "float8_e4m3fn",
            " [T, n_heads, row_bytes] = [", num_tokens, ", ",
            (int)Dimensions::NUM_HEADS, ", ", q_row_bytes, "], got ",
            args.q.sizes());
        if (args.output_fp4) {
            TORCH_CHECK(
                args.sf_q.scalar_type() == torch::kByte &&
                    args.sf_q.dim() == 3 && args.sf_q.size(0) == num_tokens &&
                    args.sf_q.size(1) == Dimensions::NUM_HEADS &&
                    args.sf_q.size(2) == Dimensions::FP4_SCALE_BYTES &&
                    args.sf_q.is_contiguous(),
                "sf_q must be contiguous uint8 [T, n_heads, head_dim/32]");
        }
        TORCH_CHECK(
            args.weights.scalar_type() == torch::kFloat &&
                args.weights.dim() == 2 && args.weights.size(0) == num_tokens &&
                args.weights.size(1) == Dimensions::NUM_HEADS &&
                args.weights.is_contiguous(),
            "weights must be a contiguous float32 [T, n_heads] = [", num_tokens,
            ", ", (int)Dimensions::NUM_HEADS, "], got ", args.weights.sizes());
        TORCH_CHECK(reinterpret_cast<uintptr_t>(args.q.data_ptr()) % 16 == 0,
                    "q must be 16 B aligned (the Q epilogue stores it as "
                    "uint4)");
        params.weight_scale = (float)args.weight_scale;

        params.grid_blocks = halops::indexer::select_grid_blocks(
            num_tokens, tuning, !KernelTraits::OutputFp4);
        params.token_groups = halops::indexer::token_group_count(num_tokens);
        TORCH_CHECK(params.grid_blocks >= 1,
                    "indexer_prepare: the grid must be positive");
        TORCH_CHECK(
            halops::indexer::select_token_tile(num_tokens) == TokenTile,
            "indexer_prepare: the tile ladder and the instantiated tile "
            "disagree (",
            halops::indexer::select_token_tile(num_tokens), " vs ", TokenTile,
            ")");
        // Workspace capacity covers both formats; execution selects this
        // format's split.
        params.key_weight_splits = halops::indexer::select_key_weight_splits(
            num_tokens, params.grid_blocks, tuning, !KernelTraits::OutputFp4);
        params.padded_tokens = Workspace::padded_token_count(num_tokens);
        params.workspace_dimensions =
            Workspace::dimensions(params.key_weight_splits);
        params.workspace_layout =
            Workspace::layout_for_capacity(args.workspace.numel());

        TORCH_CHECK(args.workspace.scalar_type() == torch::kUInt8,
                    "workspace must be uint8");
        TORCH_CHECK(args.workspace.numel() >=
                        params.workspace_dimensions.bytes(
                            params.workspace_layout, params.padded_tokens),
                    "workspace too small: need ",
                    params.workspace_dimensions.bytes(params.workspace_layout,
                                                      params.padded_tokens),
                    " bytes for T=", num_tokens,
                    ", ks_kw=", params.key_weight_splits, ", got ",
                    args.workspace.numel());
        TORCH_CHECK(
            params.workspace_layout.counter_bank_stride >= params.token_groups,
            "indexer_prepare: the buffer holds ",
            params.workspace_layout.counter_bank_stride,
            " token groups but this launch needs ", params.token_groups);

        params.base = reinterpret_cast<char*>(args.workspace.data_ptr());
        char* cursor = params.base + params.workspace_layout.scratch_offset;
        const int padded_tokens = params.padded_tokens;
        auto take = [&cursor, padded_tokens](int64_t per_token) {
            char* allocation = cursor;
            cursor += (int64_t)padded_tokens * per_token;
            return allocation;
        };
        params.key_weight_partials =
            reinterpret_cast<float*>(take((int64_t)params.key_weight_splits *
                                          Dimensions::KEY_WEIGHT_ROWS * 4));
        params.raw_weights =
            reinterpret_cast<float*>(take(Dimensions::NUM_HEADS * 4));
        params.quantized_query_lora = reinterpret_cast<halops::indexer::e4m3*>(
            take(Dimensions::QUERY_LORA_RANK));
        params.query_lora_scales = reinterpret_cast<halops::indexer::ue8m0*>(
            take(Dimensions::QUERY_LORA_RANK /
                 KernelTraits::GemmBase::ScaleGroupSize));
        return params;
    }
};

template <class KernelTraits, class Parameters>
struct IndexerBarriers {
    using Workspace = IndexerPrepareWorkspace<KernelTraits>;
    using Bar = halops::indexer::detail::PartialBarrier;
    using Roles = typename KernelTraits::Roles;
    static constexpr int TokenTile = KernelTraits::TokenTile;

    const Parameters& params;

    CUTE_DEVICE int valid_tokens(int group) const {
        return min(TokenTile, params.num_tokens - group * TokenTile);
    }
    CUTE_DEVICE Bar of(int bank, int group, int count) const {
        return Bar(params.group_counters,
                   bank * params.counter_bank_stride + group, count);
    }

    CUTE_DEVICE Bar query_lora_ready(int group) const {
        return of(Workspace::QUERY_LORA_READY, group, valid_tokens(group));
    }
    CUTE_DEVICE Bar key_weight_ready(int group) const {
        return of(Workspace::KEY_WEIGHT_READY, group,
                  params.uses_simt_projection()
                      ? IndexerDimensions::NUM_HEADS
                      : params.key_weight_items_per_group);
    }
    CUTE_DEVICE Bar query_scale_ready(int group) const {
        return of(Workspace::QUERY_SCALE_READY, group,
                  params.query_items_per_group);
    }
    CUTE_DEVICE Bar weights_ready(int group) const {
        return of(Workspace::WEIGHTS_READY, group, valid_tokens(group));
    }

    CUTE_DEVICE void reset_active_warp() const {
        // Inactive slots stay zero. Interleaving banks keeps division constant.
        constexpr int banks = Workspace::COUNTER_BANKS;
        for (int i = KernelTraits::lane_idx(); i < banks * params.token_groups;
             i += KernelTraits::WarpSize)
            Bar(params.group_counters,
                (i % banks) * params.counter_bank_stride + i / banks, 1)
                .reset_relaxed();
    }
};

template <class KernelTraits, class KeyWeightTma, class InputTma>
struct IndexerParams {
    using Dimensions = typename KernelTraits::Dimensions;

    int num_tokens, padded_tokens, token_groups, grid_blocks;
    int key_weight_work_items, key_weight_items_per_group, query_work_items,
        query_items_per_group;
    int key_weight_tiles_per_split, query_tiles_per_split;
    int key_weight_splits;
    int padded_key_weight_rows, input_hidden_size, query_rows;
    int page_size, page_shift, cache_row_bytes;
    int x_row_stride, query_lora_row_stride;
    int positions_are_int64, cache_slots_are_int64;
    int use_layernorm, apply_hadamard, skip_q;
    static constexpr bool is_neox_style = KernelTraits::IsNeoX;
    static constexpr bool output_fp4 = KernelTraits::OutputFp4;
    static constexpr bool eight_warp_query = KernelTraits::EightWarpEpilogue;
    bool use_pdl, independent_weights, use_256bit_stores;
    bool packed_q_scales, output_no_allocate;
    bool prefetch_small_rope;
    int raw_weight_prefetch_mode;
    static constexpr bool prefetch_norm = true;
    static constexpr bool all_warps_process_keys =
        KernelTraits::EightWarpEpilogue;
    int simt_projection_enabled;
    int local_query_enabled;
    int query_block_offset;
    int key_weight_blocks;
    int vector_blocks;
    int enabled_phases;
    float k_norm_eps, weight_scale;

    CUTE_HOST_DEVICE bool uses_simt_projection() const {
        if constexpr (KernelTraits::TokenTile == 16)
            return simt_projection_enabled;
        return false;
    }
    CUTE_HOST_DEVICE bool uses_local_query() const {
        if constexpr (KernelTraits::TokenTile <= 32) return local_query_enabled;
        return false;
    }

    CUTE_DEVICE bool run_key_weight_projection() const {
        return (enabled_phases & IndexerExecutionPhases::KeyWeightProjection) !=
               0;
    }
    CUTE_DEVICE bool run_key_epilogue() const {
        return (enabled_phases & IndexerExecutionPhases::KeyEpilogue) != 0;
    }
    CUTE_DEVICE bool run_query_quantization() const {
        return (enabled_phases & IndexerExecutionPhases::QueryQuantization) !=
               0;
    }
    CUTE_DEVICE bool run_query_projection() const {
        return (enabled_phases & IndexerExecutionPhases::QueryProjection) !=
                   0 &&
               !skip_q;
    }

    CUTE_DEVICE int query_work_start() const {
        if (key_weight_blocks)
            return blockIdx.x < key_weight_blocks
                       ? query_work_items
                       : (int)blockIdx.x - key_weight_blocks;
        const int b = (int)blockIdx.x + grid_blocks - query_block_offset;
        return b >= grid_blocks ? b - grid_blocks : b;
    }
    CUTE_DEVICE int key_weight_work_start() const {
        return key_weight_blocks && blockIdx.x >= key_weight_blocks
                   ? key_weight_work_items
                   : (int)blockIdx.x;
    }
    CUTE_DEVICE int key_weight_work_stride() const {
        return key_weight_blocks ? key_weight_blocks : grid_blocks;
    }
    CUTE_DEVICE int query_work_stride() const {
        return grid_blocks - key_weight_blocks;
    }

    int* __restrict__ grid_completion_counter;
    int* __restrict__ group_counters;
    int counter_bank_stride;

    float* __restrict__ key_weight_partials;
    float* __restrict__ raw_weights;
    e4m3* __restrict__ quantized_query_lora;
    ue8m0* __restrict__ query_lora_scales;

    const __nv_bfloat16* __restrict__ x;
    const __nv_bfloat16* __restrict__ key_weight_projection;
    const __nv_bfloat16* __restrict__ q_lora;
    const float* __restrict__ k_norm_weight;
    const float* __restrict__ k_norm_bias;
    const float* __restrict__ rope;
    const void* __restrict__ positions;
    const void* __restrict__ out_cache_loc;
    uint8_t* __restrict__ key_cache;
    uint8_t* __restrict__ q_output;
    uint8_t* __restrict__ q_scale;
    float* __restrict__ weights;

    KeyWeightTma key_weight_tma;
    InputTma input_tma;
    halops::indexer::detail::WeightDescriptors<KernelTraits::TokenTile>
        query_weight_descriptors;
    halops::indexer::detail::ActivationDescriptors<KernelTraits::TokenTile>
        query_lora_descriptors;

    CUTE_DEVICE void prefetch_descriptors() const {
        if (!uses_simt_projection() &&
            (!uses_local_query() || blockIdx.x < key_weight_work_items)) {
            cute::prefetch_tma_descriptor(key_weight_tma.get_tma_descriptor());
            cute::prefetch_tma_descriptor(input_tma.get_tma_descriptor());
        }
        if (skip_q ||
            (uses_local_query() && blockIdx.x < key_weight_work_items))
            return;
        if constexpr (KernelTraits::Query::SwapAB) {
            cute::prefetch_tma_descriptor(
                query_weight_descriptors.tma_a.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                query_weight_descriptors.tma_sfa.get_tma_descriptor());
            if (!uses_simt_projection() &&
                (!uses_local_query() || num_tokens > 4)) {
                cute::prefetch_tma_descriptor(
                    query_lora_descriptors.tma_b.get_tma_descriptor());
                cute::prefetch_tma_descriptor(
                    query_lora_descriptors.tma_sfb.get_tma_descriptor());
            }
        } else {
            cute::prefetch_tma_descriptor(
                query_weight_descriptors.tma_b.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                query_weight_descriptors.tma_sfb.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                query_lora_descriptors.tma_a.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                query_lora_descriptors.tma_sfa.get_tma_descriptor());
        }
    }

    CUTE_DEVICE int position_of(int token) const {
        return positions_are_int64
                   ? (int)__ldg(reinterpret_cast<const int64_t*>(positions) +
                                token)
                   : (int)__ldg(reinterpret_cast<const int*>(positions) +
                                token);
    }
    CUTE_DEVICE int cache_loc_of(int token) const {
        return cache_slots_are_int64
                   ? (int)__ldg(
                         reinterpret_cast<const int64_t*>(out_cache_loc) +
                         token)
                   : (int)__ldg(reinterpret_cast<const int*>(out_cache_loc) +
                                token);
    }
};

template <class KernelTraits, class Parameters, class KeyWeightResources,
          class QueryResources, class Barriers>
struct IndexerContext {
    const Parameters& params;
    halops::indexer::IndexerSharedStorage<KernelTraits>& plan;
    const KeyWeightResources& key_weight_resources;
    const QueryResources& query_resources;
    const Barriers& barriers;
};

}  // namespace halops::indexer
