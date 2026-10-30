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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "indexer_config.cuh"
#include "indexer_intrinsics.cuh"
#include "kernel/gemm/kernel_traits.cuh"
#include "kernel/gemm/mainloop_fwd.cuh"

namespace halops::indexer::detail {

// BF16 maxima permit exact exponent selection for divisors 6 and 448.
template <int SHIFT, uint32_t MANTISSA>
CUTE_DEVICE uint8_t quant_scale_exponent(float amax) {
    const uint32_t bits = __float_as_uint(fmaxf(amax, 1e-10f));
    if (bits >= 0x7f800000u) return 254;
    return static_cast<uint8_t>(max(
        0, min(254,
               int(bits >> 23) - SHIFT + int((bits & 0x7fffffu) > MANTISSA))));
}

CUTE_DEVICE float quant_scale_inverse(uint8_t exponent) {
    return __uint_as_float(exponent == 254 ? 0x00400000u
                                           : uint32_t(254 - exponent) << 23);
}

template <int N_PAD>
CUTE_HOST_DEVICE constexpr int sf_row_of(int t) {
    static_assert(N_PAD >= IndexerWarpTraits::MinTokenTile &&
                      N_PAD <= GemmShapeDefaults::ScaleRowsPerTile &&
                      (N_PAD & (N_PAD - 1)) == 0,
                  "token tile must be a power of two within the SfAtom tile");
    if constexpr (N_PAD < GemmShapeDefaults::MinPackedScaleTile)
        return (t / N_PAD) * GemmShapeDefaults::ScaleRowsPerTile + (t % N_PAD);
    else
        return t;
}

template <int TokenTile_, int STAGES = GemmShapeDefaults::TmaStages,
          int ACC_STAGES = 1, int KBlockM = 128, int KBlockK = 256>
struct GemmInstance {
    using Config =
        IndexerGemmConfig<TokenTile_, STAGES, ACC_STAGES, KBlockM, KBlockK>;
    using Traits =
        halops::GemmKernelTraits<e4m3, e4m3, bf16, bf16, ue8m0, float, Config>;
    using Mainloop = halops::CollectiveMainloop<Traits>;

    static constexpr int TokenTile = TokenTile_;
    static_assert(Traits::KBlockM == GemmShapeDefaults::BM, "M tile mismatch");
    static_assert(Config::SwapAB,
                  "The reused GEMM traits require one epilogue warp group");
    static_assert(Traits::KBlockK == GemmShapeDefaults::BK, "K block mismatch");
    static_assert(Traits::kItersK == GemmShapeDefaults::KItersPerBlock,
                  "K iter count mismatch");

    static constexpr bool SEPARATE_SCALE_TILE_PER_GROUP =
        TokenTile_ < GemmShapeDefaults::MinPackedScaleTile;

    static constexpr int TOKEN_TILES_PER_SCALE_TILE =
        Traits::CTA_N_SF / TokenTile_;

    static constexpr int SCALE_B_COLUMNS_PER_TOKEN_TILE =
        TokenTile_ / GemmShapeDefaults::ScaleRowsPerColumn;

    static_assert(Traits::CTA_N_SF == GemmShapeDefaults::ScaleRowsPerTile,
                  "Each token group requires one 128-row scale tile");
    static_assert(SEPARATE_SCALE_TILE_PER_GROUP ||
                      SCALE_B_COLUMNS_PER_TOKEN_TILE % 2 == 0,
                  "Packed scale slots must start at even TMEM columns");

    using SmemLayoutAStaged = typename Traits::SmemLayoutA;
    using SmemLayoutBStaged = typename Traits::SmemLayoutB;
    using SmemLayoutSFAStaged = typename Traits::SmemLayoutSFA;
    using SmemLayoutSFBStaged = typename Traits::SmemLayoutSFB;

    static constexpr int OPERAND_A_STAGE_BYTES =
        cute::cosize_v<typename Traits::SmemLayoutA> * sizeof(e4m3) / STAGES;
    static constexpr int OPERAND_B_STAGE_BYTES =
        cute::cosize_v<typename Traits::SmemLayoutB> * sizeof(e4m3) / STAGES;
    static constexpr int SCALE_A_STAGE_BYTES =
        cute::cosize_v<typename Traits::SmemLayoutSFA> * sizeof(ue8m0) / STAGES;
    static constexpr int SCALE_B_STAGE_BYTES =
        cute::cosize_v<typename Traits::SmemLayoutSFB> * sizeof(ue8m0) / STAGES;

    static constexpr int BUFFER_STAGE_COUNT =
        PipelineTraits<TokenTile_>::MaxTmaStages;
    static_assert(
        BUFFER_STAGE_COUNT >= STAGES,
        "Shared A/B/scale offsets must use the maximum pipeline depth");

    static constexpr int SHARED_MEMORY_BYTES =
        BUFFER_STAGE_COUNT * (OPERAND_A_STAGE_BYTES + OPERAND_B_STAGE_BYTES +
                              SCALE_A_STAGE_BYTES + SCALE_B_STAGE_BYTES);

    static constexpr int OPERAND_A_OFFSET = 0;
    static constexpr int OPERAND_B_OFFSET =
        OPERAND_A_OFFSET + BUFFER_STAGE_COUNT * OPERAND_A_STAGE_BYTES;
    static constexpr int SCALE_A_OFFSET =
        OPERAND_B_OFFSET + BUFFER_STAGE_COUNT * OPERAND_B_STAGE_BYTES;
    static constexpr int SCALE_B_OFFSET =
        SCALE_A_OFFSET + BUFFER_STAGE_COUNT * SCALE_A_STAGE_BYTES;

    static constexpr uint32_t TMA_BYTES_A =
        Mainloop::TmaTransactionBytesA + Mainloop::TmaTransactionBytesSFA;
    static constexpr uint32_t TMA_BYTES_B =
        Mainloop::TmaTransactionBytesB + Mainloop::TmaTransactionBytesSFB;

    static_assert(cute::cosize_v<SmemLayoutAStaged> * (int)sizeof(e4m3) %
                          STAGES ==
                      0,
                  "A staged layout is not dense");
    static_assert(cute::cosize_v<SmemLayoutBStaged> * (int)sizeof(e4m3) %
                          STAGES ==
                      0,
                  "B staged layout is not dense");
    static_assert(cute::cosize_v<SmemLayoutSFAStaged> * (int)sizeof(ue8m0) %
                          STAGES ==
                      0,
                  "SFA staged layout is not dense");
    static_assert(cute::cosize_v<SmemLayoutSFBStaged> * (int)sizeof(ue8m0) %
                          STAGES ==
                      0,
                  "SFB staged layout is not dense");
};

template <class GemmConfig>
using GemmInstanceFor =
    GemmInstance<GemmConfig::BN, GemmConfig::TmaStages, GemmConfig::AccStages,
                 GemmConfig::BM, GemmConfig::BK>;

template <int TokenTile>
CUTE_HOST_DEVICE constexpr int pipeline_smem_bytes() {
    using Profile = PipelineTraits<TokenTile>;
    return GemmInstance<TokenTile, Profile::MaxTmaStages,
                        Profile::MaxAccStages>::SHARED_MEMORY_BYTES;
}

struct GemmArena {
    // Barrier capacities cover every token tile.
    static constexpr int Total = pipeline_smem_bytes<128>();
    static constexpr int MaxTmaStages = PipelineTraits<128>::MaxTmaStages;
    static constexpr int MaxAccStages = PipelineTraits<128>::MaxAccStages;
};

struct GemmShape {
    int m_total;
    int k_dim;
    int m_tiles;
    int k_tiles;
    int tiles_per_split;
};

template <int BM, int BK, int KSplit, int Stages>
CUTE_HOST GemmShape make_gemm_shape(int m_total, int k_dim) {
    TORCH_CHECK(m_total > 0 && m_total % BM == 0,
                "a GEMM's M must be a positive multiple of ", BM, ", got ",
                m_total);
    TORCH_CHECK(k_dim > 0 && k_dim % BK == 0,
                "a GEMM's K must be a positive multiple of ", BK, ", got ",
                k_dim);
    const int k_tiles = k_dim / BK;
    TORCH_CHECK(k_tiles % KSplit == 0,
                "the K-split must divide the k-tile count; got ", k_tiles,
                " tiles for KSplit=", KSplit);
    const int tiles_per_split = k_tiles / KSplit;
    TORCH_CHECK(tiles_per_split >= Stages,
                "Each split needs at least one K tile per stage; got tiles=",
                tiles_per_split, " stages=", Stages);
    GemmShape s{};
    s.m_total = m_total;
    s.k_dim = k_dim;
    s.m_tiles = m_total / BM;
    s.k_tiles = k_tiles;
    s.tiles_per_split = tiles_per_split;
    return s;
}

static_assert(GemmArena::MaxAccStages * GemmShapeDefaults::BM <= 512 - 128,
              "Accumulator stages must leave 128 TMEM columns for scales");

template <int TokenTile_>
struct GemmOperandADescriptors {
    using Instance = GemmInstance<TokenTile_>;
    typename Instance::Mainloop::TMALoad_A tma_a;
    typename Instance::Mainloop::TMALoad_SFA tma_sfa;
    typename Instance::Traits::LayoutSFA layout_sfa;
    int m_total;
    int k_dim;
};

template <int TokenTile_>
CUTE_HOST GemmOperandADescriptors<TokenTile_> make_atoms_a(
    const void* weight, const void* weight_sf, int M, int K,
    int N_PAD = GemmInstance<TokenTile_>::TokenTile, int M_ROWS = 0) {
    using Traits = typename GemmInstance<TokenTile_>::Traits;
    using Mainloop = typename GemmInstance<TokenTile_>::Mainloop;

    const int MMA_M = M;
    const int n_pad = N_PAD;
    const int m_rows = M_ROWS > 0 ? M_ROWS : M;

    auto layout_a = cute::make_layout(cute::make_shape(m_rows, K, 1),
                                      cute::make_stride(K, cute::Int<1>{}, 0));
    auto layout_sfa = Traits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(MMA_M, n_pad, K, 1));

    cute::Tensor tensor_a = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<const e4m3*>(weight)), layout_a);
    cute::Tensor tensor_sfa =
        cute::make_tensor(static_cast<const ue8m0*>(weight_sf), layout_sfa);

    auto cluster_layout_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma::AtomThrID{}));

    GemmOperandADescriptors<TokenTile_> atoms{
        cute::make_tma_atom_A_sm100(
            typename Mainloop::GmemTiledLoad{}, tensor_a,
            typename Traits::SmemLayoutA{}(cute::_, cute::_, cute::_,
                                           cute::Int<0>{}),
            typename Traits::TileShape{}, typename Traits::TiledMma{},
            cluster_layout_vmnk),
        cute::make_tma_atom_A_sm100<uint16_t>(
            typename Mainloop::GmemTiledLoad{}, tensor_sfa,
            typename Traits::SmemLayoutSFA{}(cute::_, cute::_, cute::_,
                                             cute::Int<0>{}),
            typename Traits::TileShape{}, typename Traits::TiledMma{},
            cluster_layout_vmnk),
        layout_sfa,
        M,
        K,
    };
    return atoms;
}

template <int TokenTile_>
struct GemmOperandBDescriptors {
    using Instance = GemmInstance<TokenTile_>;
    typename Instance::Mainloop::TMALoad_B tma_b;
    typename Instance::Mainloop::TMALoad_SFB tma_sfb;
    typename Instance::Traits::LayoutSFB layout_sfb;
    int n_pad;
    int k_dim;
};

template <int TokenTile_>
CUTE_HOST GemmOperandBDescriptors<TokenTile_> make_atoms_b(const void* act_fp8,
                                                           const void* act_sf,
                                                           int M, int N_PAD,
                                                           int K,
                                                           int N_PAD_SF) {
    using Traits = typename GemmInstance<TokenTile_>::Traits;
    using Mainloop = typename GemmInstance<TokenTile_>::Mainloop;

    auto layout_b = cute::make_layout(cute::make_shape(N_PAD, K, 1),
                                      cute::make_stride(K, cute::Int<1>{}, 0));
    auto layout_sfb = Traits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N_PAD_SF, K, 1));

    cute::Tensor tensor_b = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<const e4m3*>(act_fp8)), layout_b);
    cute::Tensor tensor_sfb =
        cute::make_tensor(static_cast<const ue8m0*>(act_sf), layout_sfb);

    auto cluster_layout_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma::AtomThrID{}));
    auto cluster_layout_sfb_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma_SF::AtomThrID{}));

    GemmOperandBDescriptors<TokenTile_> atoms{
        cute::make_tma_atom_B_sm100(
            typename Mainloop::GmemTiledLoad{}, tensor_b,
            typename Traits::SmemLayoutB{}(cute::_, cute::_, cute::_,
                                           cute::Int<0>{}),
            typename Traits::TileShape{}, typename Traits::TiledMma{},
            cluster_layout_vmnk),
        cute::make_tma_atom_B_sm100<uint16_t>(
            typename Mainloop::GmemTiledLoad{}, tensor_sfb,
            typename Traits::SmemLayoutSFB{}(cute::_, cute::_, cute::_,
                                             cute::Int<0>{}),
            typename Traits::TileShape_SF{}, typename Traits::TiledMma_SF{},
            cluster_layout_sfb_vmnk),
        layout_sfb,
        N_PAD,
        K,
    };
    return atoms;
}

template <int BN>
CUTE_HOST_DEVICE constexpr bool bn_swap_ab() {
    return PipelineTraits<BN>::GemmBase::SwapAB;
}

template <int BN>
using WeightDescriptors =
    cute::conditional_t<bn_swap_ab<BN>(), GemmOperandADescriptors<BN>,
                        GemmOperandBDescriptors<BN>>;
template <int BN>
using ActivationDescriptors =
    cute::conditional_t<bn_swap_ab<BN>(), GemmOperandBDescriptors<BN>,
                        GemmOperandADescriptors<BN>>;

template <int BN>
CUTE_HOST WeightDescriptors<BN> make_weight_descriptors(const void* w,
                                                        const void* w_sf,
                                                        int m_total, int k) {
    if constexpr (bn_swap_ab<BN>())
        return make_atoms_a<BN>(w, w_sf, m_total, k);
    else
        return make_atoms_b<BN>(w, w_sf, m_total, m_total, k, m_total);
}

template <int BN>
CUTE_HOST ActivationDescriptors<BN> make_activation_descriptors(
    const void* a, const void* a_sf, int m_total, int padded_tokens, int k,
    int t_pad_sf, int rows_valid = 0) {
    if constexpr (bn_swap_ab<BN>())
        return make_atoms_b<BN>(a, a_sf, m_total, padded_tokens, k, t_pad_sf);
    else
        return make_atoms_a<BN>(a, a_sf, padded_tokens, k,
                                GemmInstance<BN>::TokenTile, rows_valid);
}

template <int BN>
CUTE_HOST_DEVICE auto const& sf_layout(const GemmOperandADescriptors<BN>& at) {
    return at.layout_sfa;
}
template <int BN>
CUTE_HOST_DEVICE auto const& sf_layout(const GemmOperandBDescriptors<BN>& at) {
    return at.layout_sfb;
}

template <int N_PAD, class LayoutSFB>
CUTE_DEVICE void quantize_group32_from_regs(const __nv_bfloat16 (&v)[4],
                                            e4m3* __restrict__ ws_fp8,
                                            ue8m0* __restrict__ ws_sf,
                                            LayoutSFB const& layout_sfb,
                                            int n_row, int k_dim, int group,
                                            int lane) {
    const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(v);
    const __nv_bfloat162 m2 = __hmax2(__habs2(v2[0]), __habs2(v2[1]));
    float amax = fmaxf(fmaxf(__bfloat162float(__low2bfloat16(m2)),
                             __bfloat162float(__high2bfloat16(m2))),
                       1e-10f);
    const unsigned mask = 0xffu << (lane & ~7);
    CUTLASS_PRAGMA_UNROLL
    for (int o = 1; o < 8; o <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(mask, amax, o));

    const uint8_t biased_exp = quant_scale_exponent<8, 0x600000u>(amax);
    const __nv_bfloat162 inv2 =
        __float2bfloat162_rn(quant_scale_inverse(biased_exp));

    const int base = group * GemmShapeDefaults::ScaleGroupSize;
    const int j = lane & 7;
    const __nv_bfloat162 p0 = __hmul2(v2[0], inv2);
    const __nv_bfloat162 p1 = __hmul2(v2[1], inv2);
    const uint32_t packed =
        (uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
            *reinterpret_cast<const __nv_bfloat162_raw*>(&p0), __NV_SATFINITE,
            __NV_E4M3) |
        ((uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
             *reinterpret_cast<const __nv_bfloat162_raw*>(&p1), __NV_SATFINITE,
             __NV_E4M3)
         << 16);
    *reinterpret_cast<uint32_t*>(ws_fp8 + (int64_t)n_row * k_dim + base +
                                 j * 4) = packed;

    if (j == 0) {
        cute::Tensor mSFB = cute::make_tensor(
            cute::make_gmem_ptr(reinterpret_cast<uint8_t*>(ws_sf)), layout_sfb);
        mSFB(sf_row_of<N_PAD>(n_row), base, cute::_0{}) =
            static_cast<uint8_t>(biased_exp);
    }
}

template <int TokenTile_, int STAGES = GemmShapeDefaults::TmaStages,
          int ACC_STAGES = 1>
CUTE_DEVICE auto make_gemm_tmem(uint32_t tmem_base) {
    using Mainloop =
        typename GemmInstance<TokenTile_, STAGES, ACC_STAGES>::Mainloop;
    auto tmem_storage = Mainloop::init_tmem_tensors();
    Mainloop::set_tmem_offsets(tmem_storage, tmem_base);
    return tmem_storage;
}

}  // namespace halops::indexer::detail
