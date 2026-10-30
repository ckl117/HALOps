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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <torch/extension.h>

#include <cute/config.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cuda_host_adapter.hpp>
#include <string>
#include <vector>

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
#    define KERUTILS_ENABLE_SM100A
#endif

#if (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
#    define KERUTILS_ENABLE_SM100A
#endif

#define HALOPS_INDEXER_ARCH_GATE_CHECKED 1

#ifndef KU_TRAP_ONLY_DEVICE_ASSERT
#    define KU_TRAP_ONLY_DEVICE_ASSERT(cond) \
        do {                                 \
            if (not(cond)) asm("trap;");     \
        } while (0)
#endif

using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

namespace halops::indexer::intrinsics {

CUTE_DEVICE
void gmem_red_add1_release(int* p) {
    asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(p) : "memory");
}

CUTE_DEVICE
int gmem_ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                 : "=r"(v)
                 : "l"(p)
                 : "memory");
    return v;
}

CUTE_DEVICE
int gmem_ld_relaxed(const int* p) {
    int v;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];"
                 : "=r"(v)
                 : "l"(p)
                 : "memory");
    return v;
}

CUTE_DEVICE
int gmem_atom_add1_acq_rel(int* p) {
    int prev;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
                 : "=r"(prev)
                 : "l"(p)
                 : "memory");
    return prev;
}

CUTE_DEVICE
void gmem_st_zero_relaxed(int* p) {
    asm volatile("st.global.s32 [%0], 0;" ::"l"(p) : "memory");
}

CUTE_DEVICE
void umma_arrive_noelect(transac_bar_t& bar) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
        "[%0];\n"
        :
        : "r"(bar_intptr));
}

CUTE_DEVICE void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

template <typename TiledMMA, typename TensorA, typename TensorB,
          typename TensorFragC>
CUTE_DEVICE void utcmma_ss(TiledMMA& tiled_mma, TensorA sA, TensorB sB,
                           TensorFragC tC_frag, bool clear_accum) {
    using namespace cute;
    tiled_mma.accumulate_ =
        clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;

    ThrMMA thr_mma = tiled_mma.get_slice(_0{});
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));
    static_assert(size<1>(sA_frag) == size<1>(tC_frag));
    static_assert(size<1>(sB_frag) == size<2>(tC_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template <int MN, int K, int SWIZZLE, typename T>
static constexpr auto make_umma_canonical_k_major_layout() {
    using namespace cute;
    using base_atom_type = std::conditional_t<
        SWIZZLE == 0 || SWIZZLE == 16, UMMA::Layout_K_INTER_Atom<T>,
        std::conditional_t<
            SWIZZLE == 32, UMMA::Layout_K_SW32_Atom<T>,
            std::conditional_t<
                SWIZZLE == 64, UMMA::Layout_K_SW64_Atom<T>,
                std::conditional_t<SWIZZLE == 128, UMMA::Layout_K_SW128_Atom<T>,
                                   void>>>>;
    static_assert(!std::is_same_v<base_atom_type, void>,
                  "Invalid SWIZZLE value");
    return coalesce(tile_to_shape(base_atom_type{}, Shape<Int<MN>, Int<K>>{},
                                  Step<_1, _2>{}),
                    Shape<_1, _1>{});
}

}  // namespace halops::indexer::intrinsics

namespace cute {

template <class a_type, class b_type, class c_type, int M, int N>
struct SM100_MMA_F16BF16_NOELECT_TraitsBase {
    using ValTypeD = c_type;
    using ValTypeA = a_type;
    using ValTypeB = b_type;
    using ValTypeC = c_type;

    static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

    using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
    using ThrID = Layout<_1>;
    using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>,
                           Stride<_0, Stride<_1, Int<M>>>>;
    using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>,
                           Stride<_0, Stride<_1, Int<N>>>>;
    using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>,
                           Stride<_0, Stride<_1, Int<M>>>>;
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS_NOELECT {
    static_assert(
        M == 64 || M == 128,
        "SM100_MMA_F16BF16_SS_NOELECT M-mode size should be 64 or 128 for "
        "1 CTA cluster MMA.");
    static_assert(
        (M == 64 && (N % 8 == 0) && (8 <= N) && (N <= 256)) ||
            (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
        "SM100_MMA_F16BF16_SS_NOELECT N-mode size should be a multiple of 8 "
        "between 8 and 256 for M=64, or a multiple of 16 for M=128.");

    using DRegisters = void;
    using ARegisters = uint64_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                     uint64_t const& desc_b,
                                     uint32_t const& tmem_c,
                                     uint32_t const& scaleC,
                                     uint64_t const& idescE) {
        uint32_t mask[4] = {0, 0, 0, 0};
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, "
            "%8}, p; \n\t"
            "}\n"
            :
            : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
              "r"(uint32_t(idescE >> 32)), "r"(scaleC), "r"(mask[0]),
              "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_SS_NOELECT<a_type, b_type, c_type, M, N,
                                               a_major, b_major, a_neg, b_neg>>
    : SM100_MMA_F16BF16_NOELECT_TraitsBase<a_type, b_type, c_type, M, N> {
    static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                      cute::sizeof_bits_v<b_type> == 16,
                  "SM100_MMA_F16BF16_SS_NOELECT supports 16bit types");

    using FrgTypeA = UMMA::smem_desc<a_major>;
    using FrgTypeB = UMMA::smem_desc<b_major>;
    using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

    UMMA::InstrDescriptor idesc_ =
        UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                              a_neg, b_neg>();

    UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

    template <class TD, class DLayout, class TA, class ALayout_, class TB,
              class BLayout_, class TC, class CLayout_>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(
        MMA_Traits const& traits, Tensor<TD, DLayout>& Dimensions,
        Tensor<TA, ALayout_> const& A, Tensor<TB, BLayout_> const& B,
        Tensor<TC, CLayout_> const& C) {
        static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
        static_assert(is_rmem<TA>::value,
                      "Expected desc registers in MMA_Atom::call");
        static_assert(is_rmem<TB>::value,
                      "Expected desc registers in MMA_Atom::call");
        static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

        uint64_t desc_a = A[0];
        uint64_t desc_b = B[0];
        uint32_t tmem_c = raw_pointer_cast(Dimensions.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
        SM100_MMA_F16BF16_SS_NOELECT<a_type, b_type, c_type, M, N, a_major,
                                     b_major, a_neg,
                                     b_neg>::fma(desc_a, desc_b, tmem_c,
                                                 uint32_t(traits.accumulate_),
                                                 idesc);
    }
};

}  // namespace cute
