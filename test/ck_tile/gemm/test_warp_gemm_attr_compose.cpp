// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include "ck_tile/ops/gemm/warp/detail/warp_gemm_attribute_mfma_compose.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"
// For real kernel run and verification
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

using namespace ck_tile;

// ---------------------- Typed tests for Dispatcher coverage ----------------------
template <typename A,
          typename B,
          typename Acc,
          index_t M,
          index_t N,
          index_t K,
          bool TransposeC,
          bool SwizzleA              = false,
          bool UseStructuredSparsity = false,
          WGAttrNumAccessEnum NA     = WGAttrNumAccessEnum::Single>
struct WGDispCase
{
    using AType                              = A;
    using BType                              = B;
    using AccType                            = Acc;
    static constexpr index_t MPerWave        = M;
    static constexpr index_t NPerWave        = N;
    static constexpr index_t KPerWave        = K;
    static constexpr bool kTransposeC        = TransposeC;
    static constexpr bool kSwizzleA          = SwizzleA;
    static constexpr bool kUSS               = UseStructuredSparsity;
    static constexpr WGAttrNumAccessEnum kNA = NA;
};

template <typename Case>
class WGCompileTimeTest : public ::testing::Test
{
};

using WGDispatcherTypesList = ::testing::Types<
    // clang-format off

    // fp16
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 8, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 8, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false, false, false, WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, true, false, false, WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float,16, 16, 32, false, false, false, WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, true, false, false, WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 4, 64, 16, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 64, 4, 16, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 16, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 16, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 8, false, true>,
    // WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 8, true, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, true, true>,

    // fp16 2:4 structural sparsity
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false, false, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, false, false, true>,

    // bf16
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 8, false>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 8, true>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, false>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, true>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, false, false, false, WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, true, false, false, WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 32, false>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 32, true>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 32, false, false, false, WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 32, true, false, false, WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 4, 64, 16, false>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 64, 4, 16, false>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 16, false>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 16, true>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 8, false, true>,
    // WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, false, true>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 8, true, true>,
    WGDispCase<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, true, true>,

    // fp8
    WGDispCase<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32, 16, false>,
    WGDispCase<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32, 32, false>,
    WGDispCase<ck_tile::fp8_t, ck_tile::fp8_t, float, 16, 16, 32, false>,
    WGDispCase<ck_tile::fp8_t, ck_tile::fp8_t, float, 16, 16, 64, false>,
    WGDispCase<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32, 16, true>,
    WGDispCase<ck_tile::fp8_t, ck_tile::fp8_t, float, 16, 16, 32, true>,
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32, 16, false>,
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32, 16, true>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32, 16, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32, 16, true>,
    WGDispCase<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32, 16, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32, 32, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::bf8_t, float, 16, 16, 32, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::bf8_t, float, 16, 16, 32, true>,
    WGDispCase<ck_tile::bf8_t, ck_tile::bf8_t, float, 16, 16, 64, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32, 16, true>,
    WGDispCase<ck_tile::fp8_t, ck_tile::fp8_t, float, 16, 16, 128, false>,
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 16, 16, 128, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 16, 16, 128, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::bf8_t, float, 16, 16, 128, false>,

    // fp8/bf8
    WGDispCase<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32, 64, false>,
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32, 64, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32, 64, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32, 64, false>,
    WGDispCase<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32, 64, false, false, false, WGAttrNumAccessEnum::Quad>,
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32, 64, false, false, false, WGAttrNumAccessEnum::Quad>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32, 64, false, false, false, WGAttrNumAccessEnum::Quad>,
    WGDispCase<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32, 64, false, false, false, WGAttrNumAccessEnum::Quad>,
    WGDispCase<ck_tile::fp8_t, ck_tile::fp8_t, float, 16, 16, 128, false, false, false, WGAttrNumAccessEnum::Quad>,
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 16, 16, 128, false, false, false, WGAttrNumAccessEnum::Quad>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 16, 16, 128, false, false, false, WGAttrNumAccessEnum::Quad>,
    WGDispCase<ck_tile::bf8_t, ck_tile::bf8_t, float, 16, 16, 128, false, false, false, WGAttrNumAccessEnum::Quad>,

    // int8
    WGDispCase<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 32, 32, 16, false>,
    WGDispCase<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 32, 32, 16, true>,
    WGDispCase<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 16, 16, 32, false>,
    WGDispCase<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 16, 16, 32, true>>;
// clang-format on

TYPED_TEST_SUITE(WGCompileTimeTest, WGDispatcherTypesList);

TYPED_TEST(WGCompileTimeTest, Instantiate)
{
    using Case = TypeParam;
    // Dispatcher-selected WarpGemm
    using Disp = typename ck_tile::WarpGemmDispatcher<typename Case::AType,
                                                      typename Case::BType,
                                                      typename Case::AccType,
                                                      Case::MPerWave,
                                                      Case::NPerWave,
                                                      Case::KPerWave,
                                                      Case::kTransposeC,
                                                      Case::kSwizzleA,
                                                      Case::kUSS,
                                                      Case::kNA>;

    // Factory-selected WarpGemm (MakeWarpGemm)
    using Make = typename ck_tile::MakeWarpGemm<Case::kTransposeC,
                                                Case::kSwizzleA,
                                                typename Case::AType,
                                                typename Case::BType,
                                                typename Case::AccType,
                                                Case::MPerWave,
                                                Case::NPerWave,
                                                Case::KPerWave,
                                                Case::kUSS,
                                                Case::kNA>::Type;

    // 1) Scalar compile-time constants must match
    static_assert(Disp::kM == Make::kM, "kM differs between Dispatcher and MakeWarpGemm");
    static_assert(Disp::kN == Make::kN, "kN differs between Dispatcher and MakeWarpGemm");
    static_assert(Disp::kK == Make::kK, "kK differs between Dispatcher and MakeWarpGemm");
    static_assert(Disp::kKPerThread == Make::kKPerThread,
                  "kKPerThread differs between Dispatcher and MakeWarpGemm");
    static_assert(Disp::get_num_of_access() == Make::get_num_of_access(),
                  "get_num_of_access() differs between Dispatcher and MakeWarpGemm");

    // 2) Data types must match
    static_assert(std::is_same_v<typename Disp::ADataType, typename Make::ADataType>,
                  "ADataType differs between Dispatcher and MakeWarpGemm");
    static_assert(std::is_same_v<typename Disp::BDataType, typename Make::BDataType>,
                  "BDataType differs between Dispatcher and MakeWarpGemm");
    static_assert(std::is_same_v<typename Disp::CDataType, typename Make::CDataType>,
                  "CDataType differs between Dispatcher and MakeWarpGemm");

    // 3) Distribution encodings must match (ensures identical warp tiling/layout)
    static_assert(
        std::is_same_v<typename Disp::AWarpDstrEncoding, typename Make::AWarpDstrEncoding>,
        "AWarpDstrEncoding differs between Dispatcher and MakeWarpGemm");
    static_assert(
        std::is_same_v<typename Disp::BWarpDstrEncoding, typename Make::BWarpDstrEncoding>,
        "BWarpDstrEncoding differs between Dispatcher and MakeWarpGemm");
    static_assert(
        std::is_same_v<typename Disp::CWarpDstrEncoding, typename Make::CWarpDstrEncoding>,
        "CWarpDstrEncoding differs between Dispatcher and MakeWarpGemm");

    // 4) Final tensor types must match (encodes DataType + Distribution)
    static_assert(std::is_same_v<typename Disp::AWarpTensor, typename Make::AWarpTensor>,
                  "AWarpTensor differs between Dispatcher and MakeWarpGemm");
    static_assert(std::is_same_v<typename Disp::BWarpTensor, typename Make::BWarpTensor>,
                  "BWarpTensor differs between Dispatcher and MakeWarpGemm");
    static_assert(std::is_same_v<typename Disp::CWarpTensor, typename Make::CWarpTensor>,
                  "CWarpTensor differs between Dispatcher and MakeWarpGemm");

    SUCCEED();
}

// ---------------------- Runtime tests: Compare Dispatcher (MFMA and SMFMA only) vs MakeWarpGemm vs
// CPU ----------------------
// ---------------------- Runtime operator() behavior tests on GPU ----------------------
template <bool UseMakeWarpGemm,
          typename AType,
          typename BType,
          typename CType,
          index_t M,
          index_t N,
          index_t K,
          bool TransposeC,
          bool SwizzleA,
          bool UseStructuredSparsity,
          WGAttrNumAccessEnum NumAccess>
struct WarpGemmKernel
{
    static constexpr int kBlockSize = 64;
    __device__ void operator()(const AType* A, const BType* B, CType* C) const
    {
        using WarpGemm = std::conditional_t<UseMakeWarpGemm,
                                            typename ck_tile::MakeWarpGemm<TransposeC,
                                                                           SwizzleA,
                                                                           AType,
                                                                           BType,
                                                                           CType,
                                                                           M,
                                                                           N,
                                                                           K,
                                                                           UseStructuredSparsity,
                                                                           NumAccess>::Type,
                                            ck_tile::WarpGemmDispatcher<AType,
                                                                        BType,
                                                                        CType,
                                                                        M,
                                                                        N,
                                                                        K,
                                                                        TransposeC,
                                                                        SwizzleA,
                                                                        UseStructuredSparsity,
                                                                        NumAccess>>;

        // A: [M,K] row-major (packed)
        const auto a_view =
            ck_tile::make_naive_tensor_view_packed<ck_tile::address_space_enum::global>(
                const_cast<AType*>(A), ck_tile::make_tuple(M, K));
        // B: expose as logical [N,K] with strides (1, N) over the original row-major [K,N] buffer
        const auto b_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            const_cast<BType*>(B), ck_tile::make_tuple(N, K), ck_tile::make_tuple(1, N));
        // C: [M,N] row-major (packed)
        const auto c_view =
            ck_tile::make_naive_tensor_view_packed<ck_tile::address_space_enum::global>(
                const_cast<CType*>(C), ck_tile::make_tuple(M, N));

        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr auto a_len = AWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto b_len = BWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto c_len = CWarpTensor::get_tile_distribution().get_lengths();

        auto a_win = ck_tile::make_tile_window(
            a_view, a_len, ck_tile::make_multi_index(0, 0), AWarpTensor::get_tile_distribution());
        auto b_win = ck_tile::make_tile_window(
            b_view, b_len, ck_tile::make_multi_index(0, 0), BWarpTensor::get_tile_distribution());
        auto c_win = ck_tile::make_tile_window(
            c_view, c_len, ck_tile::make_multi_index(0, 0), CWarpTensor::get_tile_distribution());

        AWarpTensor a_tile;
        BWarpTensor b_tile;
        ck_tile::load_tile(a_tile, a_win);
        ck_tile::load_tile(b_tile, b_win);

        CWarpTensor c_tile;
        c_tile = WarpGemm{}(a_tile, b_tile);
        ck_tile::store_tile(c_win, c_tile);
    }
};

// ---------------------- New runtime helper: run a WG on device with given A/B into C
// ----------------------
template <typename Case, bool UseMakeWarpGemm>
static void RunWarpGemmCase(const ck_tile::HostTensor<typename Case::AType>& A,
                            const ck_tile::HostTensor<typename Case::BType>& B,
                            ck_tile::HostTensor<typename Case::AccType>& C)
{
    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType; // CDataType equals Acc for these tests

    ck_tile::DeviceMem Ad(A.get_element_space_size_in_bytes());
    ck_tile::DeviceMem Bd(B.get_element_space_size_in_bytes());
    ck_tile::DeviceMem Cd(C.get_element_space_size_in_bytes());

    Ad.ToDevice(A.data());
    Bd.ToDevice(B.data());
    Cd.SetZero();

    dim3 grid(1);
    dim3 block{64};

    using Kernel = WarpGemmKernel<UseMakeWarpGemm,
                                  AType,
                                  BType,
                                  CType,
                                  Case::MPerWave,
                                  Case::NPerWave,
                                  Case::KPerWave,
                                  Case::kTransposeC,
                                  Case::kSwizzleA,
                                  Case::kUSS,
                                  Case::kNA>;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, true},
        ck_tile::make_kernel(Kernel{},
                             grid,
                             block,
                             0,
                             static_cast<const AType*>(Ad.GetDeviceBuffer()),
                             static_cast<const BType*>(Bd.GetDeviceBuffer()),
                             static_cast<CType*>(Cd.GetDeviceBuffer())));

    Cd.FromDevice(C.mData.data());
}

template <typename Case>
class WGRuntimeTest : public ::testing::Test
{
};

// enforce 2:4 sparsity on A for SMFMA runtime cases (only meaningful for half_t here)
template <typename AType>
static inline void make_2to4_sparse_A(ck_tile::HostTensor<AType>& A)
{
    // zero half the values in each consecutive group of 4 along K for each row m
    const index_t M = A.mDesc.get_lengths()[0];
    const index_t K = A.mDesc.get_lengths()[1];
    for(index_t m = 0; m < M; ++m)
    {
        for(index_t k = 0; k + 3 < K; k += 4)
        {
            // keep entries at k and k+2, zero k+1 and k+3 (simple 2:4 pattern)
            A(m, k + 1) = ck_tile::type_convert<AType>(0);
            A(m, k + 3) = ck_tile::type_convert<AType>(0);
        }
    }
}

TYPED_TEST_SUITE(WGRuntimeTest, WGDispatcherTypesList);

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_MakeWG)
{
    using Case = TypeParam;

    // Equivalent MakeWarpGemm
    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType;

    constexpr index_t M = Case::MPerWave;
    constexpr index_t N = Case::NPerWave;
    constexpr index_t K = Case::KPerWave;

    ck_tile::HostTensor<AType> A({M, K});
    ck_tile::HostTensor<BType> B({K, N});
    ck_tile::HostTensor<CType> C_disp({M, N});
    ck_tile::HostTensor<CType> C_make({M, N});

    for(index_t m = 0; m < M; ++m)
        for(index_t k = 0; k < K; ++k)
            A(m, k) = ck_tile::type_convert<AType>((m + 1) * 0.1f + (k + 1) * 0.01f);

    if constexpr(Case::kUSS)
    {
        // ensure A satisfies 2:4 sparsity for SMFMA
        make_2to4_sparse_A(A);
    }

    for(index_t k0 = 0; k0 < K; ++k0)
        for(index_t n = 0; n < N; ++n)
            B(k0, n) = ck_tile::type_convert<BType>((k0 + 1) * 0.2f - (n + 1) * 0.03f);

    C_disp.SetZero();
    C_make.SetZero();
    RunWarpGemmCase<Case, /*UseMakeWarpGemm=*/false>(A, B, C_disp);
    RunWarpGemmCase<Case, /*UseMakeWarpGemm=*/true>(A, B, C_make);

    EXPECT_TRUE(ck_tile::check_err(C_disp, C_make, "Dispatcher vs MakeWarpGemm mismatch", 0, 0));
}
