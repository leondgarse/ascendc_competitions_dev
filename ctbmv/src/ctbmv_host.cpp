/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ctbmv_host.cpp
 * \brief aclblasCtbmv (complex64 triangular banded matrix-vector multiply) host entry, arch22.
 */

#include <algorithm>
#include <mutex>
#include <cstddef>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "ctbmv_tiling_data.h"
#include "ctbmv_kernel.h"

namespace {

constexpr uint32_t FLOATS_PER_COMPLEX = 2U;

aclblasStatus_t ValidateCtbmvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const aclblasComplex *a, int lda, const aclblasComplex *x, int incx)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasCtbmv", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasCtbmv", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        diag == ACLBLAS_UNIT || diag == ACLBLAS_NON_UNIT,
        OP_LOGE("aclblasCtbmv", "invalid diag=%d", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(k >= 0, OP_LOGE("aclblasCtbmv", "invalid k=%d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        lda >= k + 1, OP_LOGE("aclblasCtbmv", "invalid lda=%d, k=%d", lda, k);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasCtbmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(a != nullptr, OP_LOGE("aclblasCtbmv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasCtbmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

CtbmvTilingData CalCtbmvTilingData(
    uint32_t useCoreNum, int n, int k, int lda, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int incx)
{
    CtbmvTilingData tilingData{};
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.k = static_cast<uint32_t>(k);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.useCoreNum = useCoreNum;
    tilingData.incx = static_cast<int64_t>(incx);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.trans = static_cast<uint32_t>(trans);
    tilingData.diag = static_cast<uint32_t>(diag);
    return tilingData;
}

// Six gather tables, uploaded once per process and reused by every call:
//   [0, maxCnt)          real lanes of a strided (32B-padded) load
//   [maxCnt, 2*maxCnt)   imag lanes of the same
//   [2*maxCnt, 4*maxCnt) reversed variants (kept for symmetry with the kernels)
//   [4*maxCnt, 6*maxCnt) interleave table that re-packs result planes into x
// Gather consumes byte offsets. The buffer is immutable, so a single device
// allocation is shared for the lifetime of the process; freeing it per call
// would force a host/device sync on every invocation.
std::once_flag g_ctbmvOffsetOnce;
uint8_t *g_ctbmvOffsetDevice = nullptr;

void BuildCtbmvGatherOffset()
{
    const uint32_t mc = CTBMV_MAX_DATA_COUNT;
    const uint32_t lanes = mc * 6U;
    const size_t byteSize = static_cast<size_t>(lanes) * sizeof(uint32_t);
    constexpr uint32_t F = static_cast<uint32_t>(sizeof(float));
    constexpr uint32_t PADDED = 8U;  // floats per 32B UB block

    uint32_t *hostOffset = new (std::nothrow) uint32_t[lanes];
    if (hostOffset == nullptr) {
        OP_LOGE("aclblasCtbmv", "offset alloc failed");
        return;
    }
    for (uint32_t i = 0U; i < mc; ++i) {
        hostOffset[i] = i * PADDED * F;
        hostOffset[mc + i] = (i * PADDED + 1U) * F;
        hostOffset[2U * mc + i] = (mc - 1U - i) * PADDED * F;
        hostOffset[3U * mc + i] = ((mc - 1U - i) * PADDED + 1U) * F;
        hostOffset[4U * mc + FLOATS_PER_COMPLEX * i] = i * F;
        hostOffset[4U * mc + FLOATS_PER_COMPLEX * i + 1U] = (mc + i) * F;
    }

    void *devPtr = nullptr;
    aclError aclRet = aclrtMalloc(&devPtr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        delete[] hostOffset;
        OP_LOGE("aclblasCtbmv", "aclrtMalloc for gather offset failed, ret=%d", aclRet);
        return;
    }
    aclRet = aclrtMemcpy(devPtr, byteSize, hostOffset, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    delete[] hostOffset;
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFree(devPtr);
        OP_LOGE("aclblasCtbmv", "aclrtMemcpy for gather offset failed, ret=%d", aclRet);
        return;
    }
    g_ctbmvOffsetDevice = reinterpret_cast<uint8_t *>(devPtr);
}

}  // namespace

aclblasStatus_t aclblasCtbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const aclblasComplex *A, int lda, aclblasComplex *x, int incx)
{
    CHECK_RET(n >= 0, OP_LOGE("aclblasCtbmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasCtbmv", "handle is nullptr");
              return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;  // legal no-op per Netlib ctbmv
    }

    aclblasStatus_t st = ValidateCtbmvParams(uplo, trans, diag, n, k, A, lda, x, incx);
    CHECK_RET(st == ACLBLAS_STATUS_SUCCESS, return st);

    // k == 0 with a unit diagonal means op(A) == I, so x is unchanged.
    if (k == 0 && diag == ACLBLAS_UNIT) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto *h = handle;
    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0U, OP_LOGE("aclblasCtbmv", "GetAivCoreCount failed");
              return ACLBLAS_STATUS_EXECUTION_FAILED);
    if (aivCoreNum > CTBMV_MAX_CORE_NUM) {
        aivCoreNum = CTBMV_MAX_CORE_NUM;
    }

    // Bands are the parallel dimension. Strided x forces a single core so the
    // scalar gather/scatter path stays deterministic.
    const uint32_t taskCount = static_cast<uint32_t>(k) + 1U;
    uint32_t useCoreNum = (incx == 1) ? std::min(taskCount, aivCoreNum) : 1U;
    if (useCoreNum == 0U) {
        useCoreNum = 1U;
    }

    CtbmvTilingData tilingData = CalCtbmvTilingData(useCoreNum, n, k, lda, uplo, trans, diag, incx);

    // Workspace holds two float planes (real, imag) of length n, followed by a
    // contiguous copy of x used when incx != 1.
    const bool strided = (incx != 1);
    size_t workspaceSize = static_cast<size_t>(n) * FLOATS_PER_COMPLEX * sizeof(float);
    if (strided) {
        workspaceSize += static_cast<size_t>(n) * FLOATS_PER_COMPLEX * sizeof(float);
    }
    if (k == 0 && !strided) {
        workspaceSize = 0U;
    }
    uint8_t *workspaceDevice = nullptr;
    if (workspaceSize > 0U) {
        CHECK_RET(
            workspaceSize <= GetEffectiveWorkspaceSize(h),
            OP_LOGE("aclblasCtbmv", "workspace %zu > handle %zu", workspaceSize, GetEffectiveWorkspaceSize(h));
            return ACLBLAS_STATUS_EXECUTION_FAILED);
        workspaceDevice = reinterpret_cast<uint8_t *>(GetEffectiveWorkspace(h));
    }

    std::call_once(g_ctbmvOffsetOnce, BuildCtbmvGatherOffset);
    CHECK_RET(g_ctbmvOffsetDevice != nullptr, OP_LOGE("aclblasCtbmv", "gather offset table unavailable");
              return ACLBLAS_STATUS_ALLOC_FAILED);

    OP_LOGD(
        "aclblasCtbmv", "tiling: n=%u k=%u lda=%u uplo=%u trans=%u diag=%u incx=%ld useCoreNum=%u",
        tilingData.n, tilingData.k, tilingData.lda, tilingData.uplo, tilingData.trans, tilingData.diag,
        static_cast<long>(tilingData.incx), tilingData.useCoreNum);

    // Asynchronous: the caller synchronises the stream before reading x back.
    ctbmv_arch22_kernel_do(
        reinterpret_cast<uint8_t *>(const_cast<aclblasComplex *>(A)), reinterpret_cast<uint8_t *>(x),
        g_ctbmvOffsetDevice, workspaceDevice, tilingData, useCoreNum, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
