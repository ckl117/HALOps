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

#include <optional>
#include <string>

#include "common/cuda_utils.h"
#include "indexer_prepare_kernel.cuh"
#include "indexer_small.cuh"

static_assert(HALOPS_INDEXER_ARCH_GATE_CHECKED,
              "indexer_prepare requires the SM100 architecture gate");

namespace halops::indexer {

namespace {

using KernelTraits = halops::indexer::IndexerTraits;
using Workspace = IndexerPrepareWorkspace<KernelTraits>;

template <class KernelTraits>
auto make_indexer_params(const IndexerPrepareArguments& args,
                         const IndexerPrepareProblem<KernelTraits>& problem,
                         const IndexerTuning& tuning) {
    using Dimensions = typename KernelTraits::Dimensions;
    using KeyWeightConfig = typename KernelTraits::KeyWeight;
    using QueryConfig = typename KernelTraits::Query;
    constexpr int TokenTile = KernelTraits::TokenTile;

    auto key_weight_tma =
        halops::indexer::detail::make_bf16_weight_tma<KeyWeightConfig>(
            args.wk_w.data_ptr(), Dimensions::KEY_WEIGHT_ROWS,
            Dimensions::HIDDEN);
    auto input_tma =
        halops::indexer::detail::make_bf16_activation_tma<KeyWeightConfig>(
            args.x.data_ptr(), problem.num_tokens, problem.x_row_stride,
            Dimensions::HIDDEN);

    const int key_weight_k_tiles = Dimensions::HIDDEN / KeyWeightConfig::BK;
    TORCH_CHECK(key_weight_k_tiles % problem.key_weight_splits == 0,
                "the kw K-split must divide the k-tile count; got ",
                key_weight_k_tiles,
                " tiles for KSplit=", problem.key_weight_splits);
    const int key_weight_tiles_per_split =
        key_weight_k_tiles / problem.key_weight_splits;
    TORCH_CHECK(key_weight_tiles_per_split >= KeyWeightConfig::TmaStages,
                "Each split needs at least one K tile per stage; got tiles=",
                key_weight_tiles_per_split,
                " stages=", (int)KeyWeightConfig::TmaStages);

    const halops::indexer::detail::GemmShape query_shape =
        halops::indexer::detail::make_gemm_shape<
            QueryConfig::BM, QueryConfig::BK, QueryConfig::KSplit,
            QueryConfig::TmaStages>(Dimensions::QUERY_ROWS,
                                    Dimensions::QUERY_LORA_RANK);

    halops::indexer::IndexerParams<KernelTraits, decltype(key_weight_tma),
                                   decltype(input_tma)>
        params{};
    params.num_tokens = problem.num_tokens;
    params.padded_tokens = problem.padded_tokens;
    params.token_groups = problem.token_groups;
    params.grid_blocks = problem.grid_blocks;
    params.key_weight_tiles_per_split = key_weight_tiles_per_split;
    params.query_tiles_per_split = query_shape.tiles_per_split;
    params.key_weight_splits = problem.key_weight_splits;
    params.padded_key_weight_rows = Dimensions::PADDED_KEY_WEIGHT_ROWS;
    params.input_hidden_size = Dimensions::HIDDEN;
    params.query_rows = query_shape.m_total;
    params.page_size = problem.page_size;
    params.page_shift = problem.page_shift;
    params.cache_row_bytes = (int)args.paged_k_cache.size(1);
    params.x_row_stride = problem.x_row_stride;
    params.query_lora_row_stride = problem.query_lora_row_stride;
    params.positions_are_int64 = problem.positions_are_int64 ? 1 : 0;
    params.cache_slots_are_int64 = problem.cache_slots_are_int64 ? 1 : 0;
    params.use_layernorm = problem.use_layernorm ? 1 : 0;
    params.apply_hadamard = args.apply_hadamard ? 1 : 0;
    params.skip_q = args.skip_q ? 1 : 0;
    params.use_pdl = args.use_pdl;
    configure_indexer_schedule<KernelTraits>(
        params, tuning,
        (reinterpret_cast<uintptr_t>(args.q.data_ptr()) & 31) == 0);
    params.k_norm_eps = problem.k_norm_eps;
    params.weight_scale = problem.weight_scale;

    params.grid_completion_counter = reinterpret_cast<int*>(problem.base);
    params.group_counters = reinterpret_cast<int*>(
        problem.base + problem.workspace_layout.counter_offset);
    params.counter_bank_stride = problem.workspace_layout.counter_bank_stride;

    params.key_weight_partials = problem.key_weight_partials;
    params.raw_weights = problem.raw_weights;
    params.quantized_query_lora = problem.quantized_query_lora;
    params.query_lora_scales = problem.query_lora_scales;

    params.x = reinterpret_cast<const __nv_bfloat16*>(args.x.data_ptr());
    params.key_weight_projection =
        reinterpret_cast<const __nv_bfloat16*>(args.wk_w.data_ptr());
    params.q_lora =
        reinterpret_cast<const __nv_bfloat16*>(args.q_lora.data_ptr());
    params.k_norm_weight = args.k_norm_weight.data_ptr<float>();
    params.k_norm_bias = problem.use_layernorm
                             ? args.k_norm_bias.data_ptr<float>()
                             : args.k_norm_weight.data_ptr<float>();
    params.rope = args.rope_cos_sin_cache.data_ptr<float>();
    params.positions = args.positions.data_ptr();
    params.out_cache_loc = args.out_cache_loc.data_ptr();
    params.key_cache =
        reinterpret_cast<uint8_t*>(args.paged_k_cache.data_ptr());
    params.q_output = reinterpret_cast<uint8_t*>(args.q.data_ptr());
    params.q_scale = args.output_fp4 ? args.sf_q.data_ptr<uint8_t>() : nullptr;
    params.weights = args.weights.data_ptr();
    params.weights_dtype = problem.weights_dtype;

    params.key_weight_tma = key_weight_tma;
    params.input_tma = input_tma;
    if (!args.skip_q) {
        params.query_weight_descriptors =
            halops::indexer::detail::make_weight_descriptors<TokenTile>(
                args.wq_b.data_ptr(), args.wq_b_sf.data_ptr(),
                query_shape.m_total, query_shape.k_dim);
        params.query_lora_descriptors =
            halops::indexer::detail::make_activation_descriptors<TokenTile>(
                problem.quantized_query_lora, problem.query_lora_scales,
                query_shape.m_total, problem.padded_tokens,
                Dimensions::QUERY_LORA_RANK, problem.padded_tokens,
                tuning.bound_activation_rows ? problem.num_tokens
                                             : problem.padded_tokens);
    }
    return params;
}

template <class KernelTraits>
void indexer_prepare_launch(const IndexerPrepareArguments& args,
                            const IndexerTuning& tuning) {
    const IndexerPrepareProblem<KernelTraits> problem =
        IndexerPrepareProblem<KernelTraits>::from(args, tuning);
    const auto params =
        make_indexer_params<KernelTraits>(args, problem, tuning);

    const auto kernel_kind =
        select_indexer_kernel<KernelTraits>(params, tuning);
    const size_t shared_memory_bytes =
        sizeof(IndexerSharedStorage<KernelTraits>);
    if constexpr (KernelTraits::TokenTile == 16) {
        if (kernel_kind == IndexerKernelKind::ThreeWaySmallBatch) {
            const SmallBatchParams<std::remove_cv_t<decltype(params)>> small{
                params};
            auto* kernel =
                indexer_small_batch_kernel<KernelTraits, decltype(small), 8>;
            if (query_epilogue_lanes(params.num_tokens, true) == 16)
                kernel = indexer_small_batch_kernel<KernelTraits,
                                                    decltype(small), 16>;
            HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_memory_bytes));
            halops::LaunchKernelWithPdl(
                kernel, dim3(problem.grid_blocks),
                dim3(KernelTraits::NumThreads), shared_memory_bytes,
                GetCurrentCUDAStream(args.x), args.use_pdl, small);
            HALOPS_CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    auto* kernel = indexer_prepare_kernel<KernelTraits, decltype(params)>;
    int threads = KernelTraits::NumThreads;
    if constexpr (KernelTraits::TokenTile <= 64) {
        if constexpr (KernelTraits::TokenTile == 16) {
            if (kernel_kind == IndexerKernelKind::SingleToken) {
                kernel = indexer_prepare_kernel<KernelTraits, decltype(params),
                                                true>;
                threads = KernelTraits::SingleTokenThreads;
            }
        }
        if (kernel_kind == IndexerKernelKind::LocalQuery) {
            kernel = indexer_prepare_kernel<KernelTraits, decltype(params),
                                            false, true>;
        }
    }
    if (shared_memory_bytes > KernelTraits::DefaultSharedMemoryLimit)
        HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_memory_bytes));
    halops::LaunchKernelWithPdl(kernel, dim3(problem.grid_blocks, 1, 1),
                                dim3(threads), shared_memory_bytes,
                                GetCurrentCUDAStream(args.x), args.use_pdl,
                                params);
    HALOPS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

std::vector<int64_t> IndexerPrepareWorkspaceBytes(int64_t num_tokens) {
    TORCH_CHECK(
        num_tokens > 0 && num_tokens <= std::numeric_limits<int>::max() -
                                            KernelTraits::TokenTile,
        "indexer_prepare_workspace_bytes: num_tokens must be positive "
        "and fit in int32 with padding");
    const auto tuning = IndexerTuning::from_environment();
    const int max_tokens = (int)num_tokens;
    int64_t scratch_bytes = 0;
    visit_workspace_schedules(
        max_tokens, tuning, [&](int tokens, int key_weight_splits) {
            const auto requirements = Workspace::dimensions(key_weight_splits);
            scratch_bytes = std::max(
                scratch_bytes, (int64_t)Workspace::padded_token_count(tokens) *
                                   requirements.bytes_per_token);
        });
    const int64_t total = Workspace::allocation_bytes(scratch_bytes);
    return {total, Workspace::layout_for_capacity(total).scratch_offset};
}

void IndexerPrepare(torch::Tensor x, torch::Tensor q_lora, torch::Tensor wk_w,
                    torch::Tensor wq_b, torch::Tensor wq_b_sf,
                    torch::Tensor k_norm_weight, torch::Tensor k_norm_bias,
                    double k_norm_eps, torch::Tensor rope_cos_sin_cache,
                    torch::Tensor positions, torch::Tensor paged_k_cache,
                    torch::Tensor out_cache_loc, torch::Tensor q,
                    std::optional<torch::Tensor> sf_q, torch::Tensor weights,
                    torch::Tensor workspace, double weight_scale,
                    int64_t head_dim, int64_t n_heads, bool apply_hadamard,
                    bool skip_q, const std::string& output_dtype_str,
                    bool is_neox_style, bool use_pdl) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    const CudaDeviceGuard device_guard(x.device());
    const auto tuning = IndexerTuning::from_environment();
    TORCH_CHECK(output_dtype_str == "fp8" || output_dtype_str == "mxfp4",
                "output_dtype_str must be 'fp8' or 'mxfp4', got ",
                output_dtype_str);
    const bool output_fp4 = output_dtype_str == "mxfp4";
    const IndexerPrepareArguments args{x,
                                       q_lora,
                                       wk_w,
                                       wq_b,
                                       wq_b_sf,
                                       k_norm_weight,
                                       k_norm_bias,
                                       k_norm_eps,
                                       rope_cos_sin_cache,
                                       positions,
                                       paged_k_cache,
                                       out_cache_loc,
                                       q,
                                       weights,
                                       workspace,
                                       weight_scale,
                                       head_dim,
                                       n_heads,
                                       apply_hadamard,
                                       skip_q,
                                       sf_q.value_or(torch::Tensor{}),
                                       output_fp4,
                                       is_neox_style,
                                       use_pdl};
    const bool rope_aligned_16 =
        (reinterpret_cast<uintptr_t>(args.rope_cos_sin_cache.data_ptr()) &
         15) == 0;
    const auto launch = [&](auto kernel_traits) {
        using SelectedTraits = decltype(kernel_traits);
        indexer_prepare_launch<SelectedTraits>(args, tuning);
    };
    if (output_fp4) {
        if (is_neox_style)
            dispatch_indexer_format<true, true>(
                args.x.size(0), args.skip_q, rope_aligned_16, tuning, launch);
        else
            dispatch_indexer_format<true, false>(
                args.x.size(0), args.skip_q, rope_aligned_16, tuning, launch);
    } else {
        if (is_neox_style)
            dispatch_indexer_format<false, true>(
                args.x.size(0), args.skip_q, rope_aligned_16, tuning, launch);
        else
            dispatch_indexer_format<false, false>(
                args.x.size(0), args.skip_q, rope_aligned_16, tuning, launch);
    }
}

}  // namespace halops::indexer
