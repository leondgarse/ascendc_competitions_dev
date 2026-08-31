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
 * \file ctbmv_kernel.cpp
 * \brief aclblasCtbmv (complex64 triangular banded matrix-vector multiply) AIV kernel, arch22.
 *
 * Execution model (mirrors stbmv/arch35):
 *   ctbmv_compute_kernel : zero workspace, then accumulate every diagonal band into it.
 *   ctbmv_copy_kernel    : interleave the split planes back into x (honouring incx).
 *   ctbmv_diag_kernel    : k == 0 shortcut, A is diagonal so scale x in place.
 *
 * Ordering between compute and copy comes from stream serialisation, so no cross-core
 * barrier is needed there. The SyncAll inside compute only separates zeroing from
 * accumulation.
 *
 * Complex values live interleaved in GM ([re0,im0,re1,im1,...]). They are split into
 * real/imag planes with GatherMask so all arithmetic stays vectorised, and re-interleaved
 * with Gather using a host-provided offset table.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "ctbmv_tiling_data.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2U;
constexpr uint32_t FLOATS_PER_COMPLEX = 2U;
constexpr uint32_t BYTES_PER_FLOAT = 4U;
constexpr uint32_t FLOATS_PER_BLOCK = 8U;      // 32B / 4B
constexpr uint32_t FLOATS_PER_REPEAT = 64U;    // 256B / 4B
// A strided DataCopyPad pads each 8B block to a 32B UB block => 8 floats per complex.
constexpr uint32_t FLOATS_PER_PADDED_BLOCK = 8U;
// uint32 lanes per 32B UB block; Gather sources must start on such a boundary.
constexpr uint32_t UINT32_LANES_PER_BLOCK = 8U;
constexpr uint32_t MAX_REPEAT = 255U;          // repeatTimes is uint8 in the vector ISA

// GatherMask pattern selectors: 1 keeps even lanes (real), 2 keeps odd lanes (imag).
constexpr uint8_t PATTERN_REAL = 1U;
constexpr uint8_t PATTERN_IMAG = 2U;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (b == 0U) ? 0U : ((a + b - 1U) / b);
}

__aicore__ inline uint32_t AlignUpU32(uint32_t a, uint32_t align)
{
    return (align == 0U) ? a : (((a + align - 1U) / align) * align);
}
}  // namespace

class CtbmvComputeAIV {
public:
    __aicore__ inline CtbmvComputeAIV() = default;

    __aicore__ inline void Init(GM_ADDR aBanded, GM_ADDR x, GM_ADDR gatherOffset, GM_ADDR workspace,
                                const CtbmvTilingData &tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ZeroWorkspace();
    __aicore__ inline void ProcessBand(uint32_t bandIdx);
    __aicore__ inline void CopyBackTiles();
    __aicore__ inline void ProcessBandChunk(uint32_t bandIdx, uint32_t col, uint32_t cnt, uint32_t aRowBase,
                                            uint32_t yStart);
    __aicore__ inline void LoadComplexChunk(const GlobalTensor<float> &src, uint64_t elemOffset, uint32_t cnt,
                                            const LocalTensor<float> &planeRe, const LocalTensor<float> &planeIm);
    __aicore__ inline void LoadBandChunk(uint32_t aRow, uint32_t aColBase, uint32_t cnt,
                                          const LocalTensor<float> &planeRe, const LocalTensor<float> &planeIm);
    __aicore__ inline void LoadXChunk(uint32_t logicalStart, uint32_t cnt);
    __aicore__ inline void SplitInterleaved(const LocalTensor<float> &aos, uint32_t cnt,
                                            const LocalTensor<float> &planeRe, const LocalTensor<float> &planeIm);
    __aicore__ inline void ComplexMul(uint32_t cnt, bool conjugate);
    __aicore__ inline void StoreBandResult(uint32_t yStart, uint32_t cnt);

    TPipe pipe_;

    TQue<QuePosition::VECIN, BUFFER_NUM> aosQueue_;   // staging for interleaved GM reads
    TBuf<QuePosition::VECCALC> planeBuf_;             // ar, ai, xr, xi, yr, yi, t0..t3
    TBuf<QuePosition::VECCALC> tableBuf_;             // stride-8 gather tables

    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> wsReGm_;
    GlobalTensor<float> wsImGm_;

    LocalTensor<float> ar_, ai_, xr_, xi_, yr_, yi_, t0_, t1_, t2_, t3_;
    LocalTensor<uint32_t> tblRe_, tblIm_, tblRevRe_, tblRevIm_, tblOut_;

    uint32_t blockIdx_ = 0U;
    uint32_t n_ = 0U;
    uint32_t k_ = 0U;
    uint32_t lda_ = 0U;
    uint32_t useCoreNum_ = 1U;
    uint32_t maxCnt_ = 0U;
    uint32_t tileCnt_ = 0U;
    int64_t incx_ = 1;
    uint32_t absIncx_ = 1U;
    uint32_t uplo_ = ACLBLAS_LOWER;
    uint32_t trans_ = ACLBLAS_OP_N;
    uint32_t diag_ = ACLBLAS_NON_UNIT;
};

__aicore__ inline void CtbmvComputeAIV::Init(GM_ADDR aBanded, GM_ADDR x, GM_ADDR gatherOffset, GM_ADDR workspace,
                                             const CtbmvTilingData &tiling)
{
    blockIdx_ = GetBlockIdx();
    n_ = tiling.n;
    k_ = tiling.k;
    lda_ = tiling.lda;
    useCoreNum_ = (tiling.useCoreNum == 0U || tiling.useCoreNum > CTBMV_MAX_CORE_NUM) ? 1U : tiling.useCoreNum;
    incx_ = tiling.incx;
    uplo_ = tiling.uplo;
    trans_ = tiling.trans;
    diag_ = tiling.diag;
    absIncx_ = (incx_ >= 0) ? static_cast<uint32_t>(incx_) : static_cast<uint32_t>(-incx_);
    maxCnt_ = CTBMV_MAX_DATA_COUNT;
    // Never tile wider than the problem: keeps the per-call table prefix and the
    // UB traffic proportional to n instead of to CTBMV_MAX_DATA_COUNT.
    tileCnt_ = (n_ < maxCnt_) ? AlignUpU32(n_, FLOATS_PER_BLOCK) : maxCnt_;
    if (tileCnt_ > maxCnt_) {
        tileCnt_ = maxCnt_;
    }

    // x is contiguous here: strided inputs are normalised by ctbmv_normalise_kernel.
    const uint64_t xFloats = static_cast<uint64_t>(n_) * FLOATS_PER_COMPLEX;
    const uint64_t aFloats = static_cast<uint64_t>(lda_) * n_ * FLOATS_PER_COMPLEX;

    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aBanded), aFloats);
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), xFloats);

    __gm__ float *wsBase = reinterpret_cast<__gm__ float *>(workspace);
    wsReGm_.SetGlobalBuffer(wsBase, n_);
    wsImGm_.SetGlobalBuffer(wsBase + n_, n_);

    pipe_.InitBuffer(aosQueue_, BUFFER_NUM, maxCnt_ * FLOATS_PER_PADDED_BLOCK * BYTES_PER_FLOAT);
    pipe_.InitBuffer(planeBuf_, maxCnt_ * 10U * BYTES_PER_FLOAT);
    pipe_.InitBuffer(tableBuf_, maxCnt_ * 6U * static_cast<uint32_t>(sizeof(uint32_t)));

    LocalTensor<float> planes = planeBuf_.Get<float>();
    ar_ = planes[0U * maxCnt_];
    ai_ = planes[1U * maxCnt_];
    xr_ = planes[2U * maxCnt_];
    xi_ = planes[3U * maxCnt_];
    yr_ = planes[4U * maxCnt_];
    yi_ = planes[5U * maxCnt_];
    t0_ = planes[6U * maxCnt_];
    t1_ = planes[7U * maxCnt_];
    t2_ = planes[8U * maxCnt_];
    t3_ = planes[9U * maxCnt_];

    // Gather tables live at the head of the host-provided offset buffer:
    // [0, maxCnt)          -> real lanes (byte offset i * 8 floats)
    // [maxCnt, 2 * maxCnt) -> imag lanes (byte offset i * 8 + 1 floats)
    GlobalTensor<uint32_t> tblGm;
    tblGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gatherOffset), maxCnt_ * 6U);
    LocalTensor<uint32_t> tbl = tableBuf_.Get<uint32_t>();
    tblRe_ = tbl[0U];
    tblIm_ = tbl[maxCnt_];
    tblRevRe_ = tbl[2U * maxCnt_];
    tblRevIm_ = tbl[3U * maxCnt_];
    tblOut_ = tbl[4U * maxCnt_];
    // Only the first tileCnt_ entries of each table are ever indexed; copying
    // that prefix instead of the full table removes a 16 KB per-core preamble
    // from every small call.
    DataCopyPadExtParams<uint32_t> tpad{true, 0U, 0U, 0U};
    DataCopyExtParams tp{1U, tileCnt_ * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPad(tblRe_, tblGm[0U], tp, tpad);
    DataCopyPad(tblIm_, tblGm[maxCnt_], tp, tpad);
    DataCopyExtParams tpo{1U, tileCnt_ * FLOATS_PER_COMPLEX * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPad(tblOut_, tblGm[4U * maxCnt_], tpo, tpad);
}

// Split [re0,im0,re1,im1,...] into two contiguous float planes.
__aicore__ inline void CtbmvComputeAIV::SplitInterleaved(const LocalTensor<float> &aos, uint32_t cnt,
                                                         const LocalTensor<float> &planeRe,
                                                         const LocalTensor<float> &planeIm)
{
    const uint32_t interleaved = cnt * FLOATS_PER_COMPLEX;
    uint32_t totalRepeat = CeilDivU32(interleaved, FLOATS_PER_REPEAT);
    uint32_t doneRepeat = 0U;
    uint64_t rsvdCnt = 0U;
    const uint32_t mask = 0U;  // 0 selects "use the full 256B repeat" counting mode

    // repeatTimes is a uint8 field, so walk the chunk in <=255 repeat batches.
    while (doneRepeat < totalRepeat) {
        uint32_t batch = totalRepeat - doneRepeat;
        if (batch > MAX_REPEAT) {
            batch = MAX_REPEAT;
        }
        const uint32_t srcElem = doneRepeat * FLOATS_PER_REPEAT;
        const uint32_t dstElem = doneRepeat * (FLOATS_PER_REPEAT / FLOATS_PER_COMPLEX);
        GatherMask<float>(planeRe[dstElem], aos[srcElem], PATTERN_REAL, false, mask,
                          {1U, static_cast<uint16_t>(batch), FLOATS_PER_BLOCK, FLOATS_PER_BLOCK}, rsvdCnt);
        GatherMask<float>(planeIm[dstElem], aos[srcElem], PATTERN_IMAG, false, mask,
                          {1U, static_cast<uint16_t>(batch), FLOATS_PER_BLOCK, FLOATS_PER_BLOCK}, rsvdCnt);
        doneRepeat += batch;
    }
}

__aicore__ inline void CtbmvComputeAIV::LoadComplexChunk(const GlobalTensor<float> &src, uint64_t elemOffset,
                                                          uint32_t cnt, const LocalTensor<float> &planeRe,
                                                          const LocalTensor<float> &planeIm)
{
    LocalTensor<float> aos = aosQueue_.AllocTensor<float>();
    const uint32_t floats = cnt * FLOATS_PER_COMPLEX;
    DataCopyExtParams copyParams{1U, floats * BYTES_PER_FLOAT, 0U, 0U, 0U};
    DataCopyPadExtParams<float> padParams{true, 0U, 0U, 0.0f};
    DataCopyPad(aos, src[elemOffset * FLOATS_PER_COMPLEX], copyParams, padParams);
    aosQueue_.EnQue(aos);

    LocalTensor<float> ready = aosQueue_.DeQue<float>();
    SplitInterleaved(ready, cnt, planeRe, planeIm);
    aosQueue_.FreeTensor(ready);
}

// Banded A is stored column-major: element (row, col) lives at A[row + col * lda].
// Walking one diagonal band holds `row` fixed and advances `col`, so successive
// elements are lda complex values apart. A strided DataCopyPad gathers them, but
// the hardware pads each 8-byte block up to a 32-byte UB block, so element i
// lands at float offset i * 8 (real) and i * 8 + 1 (imag). Two Gather calls with
// precomputed offset tables compact those into dense real/imag planes.
__aicore__ inline void CtbmvComputeAIV::LoadBandChunk(uint32_t aRow, uint32_t aColBase, uint32_t cnt,
                                                       const LocalTensor<float> &planeRe,
                                                       const LocalTensor<float> &planeIm)
{
    LocalTensor<float> aos = aosQueue_.AllocTensor<float>();
    const uint64_t firstElem = static_cast<uint64_t>(aRow) + static_cast<uint64_t>(aColBase) * lda_;
    DataCopyExtParams copyParams{static_cast<uint16_t>(cnt), FLOATS_PER_COMPLEX * BYTES_PER_FLOAT,
                                 (lda_ - 1U) * FLOATS_PER_COMPLEX * BYTES_PER_FLOAT, 0U, 0U};
    DataCopyPadExtParams<float> padParams{true, 0U, 0U, 0.0f};
    DataCopyPad(aos, aGm_[firstElem * FLOATS_PER_COMPLEX], copyParams, padParams);
    aosQueue_.EnQue(aos);

    LocalTensor<float> ready = aosQueue_.DeQue<float>();
    Gather(planeRe, ready, tblRe_, 0U, cnt);
    Gather(planeIm, ready, tblIm_, 0U, cnt);
    aosQueue_.FreeTensor(ready);
}

__aicore__ inline void CtbmvComputeAIV::LoadXChunk(uint32_t logicalStart, uint32_t cnt)
{
    // x has been normalised into a contiguous logical-order buffer by
    // ctbmv_gather_kernel when incx != 1, so this path only ever sees stride 1.
    LoadComplexChunk(xGm_, logicalStart, cnt, xr_, xi_);
}

// (ar + i*ai) * (xr + i*xi), or its conjugate form when trans == OP_C.
__aicore__ inline void CtbmvComputeAIV::ComplexMul(uint32_t cnt, bool conjugate)
{
    Mul(t0_, ar_, xr_, cnt);
    Mul(t1_, ai_, xi_, cnt);
    Mul(t2_, ar_, xi_, cnt);
    Mul(t3_, ai_, xr_, cnt);
    if (conjugate) {
        Add(yr_, t0_, t1_, cnt);  // ar*xr + ai*xi
        Sub(yi_, t2_, t3_, cnt);  // ar*xi - ai*xr
    } else {
        Sub(yr_, t0_, t1_, cnt);  // ar*xr - ai*xi
        Add(yi_, t2_, t3_, cnt);  // ar*xi + ai*xr
    }
}

__aicore__ inline void CtbmvComputeAIV::StoreBandResult(uint32_t yStart, uint32_t cnt)
{
    DataCopyExtParams copyParams{1U, cnt * BYTES_PER_FLOAT, 0U, 0U, 0U};
    DataCopyPad(wsReGm_[yStart], yr_, copyParams);
    DataCopyPad(wsImGm_[yStart], yi_, copyParams);
}

__aicore__ inline void CtbmvComputeAIV::ProcessBandChunk(uint32_t bandIdx, uint32_t col, uint32_t cnt,
                                                          uint32_t aRowBase, uint32_t yStart)
{
    const bool isUnitDiag = (diag_ == ACLBLAS_UNIT) && (bandIdx == 0U);
    LoadXChunk(col, cnt);

    if (isUnitDiag) {
        // A_diag == (1, 0) so op(A)*x reduces to x; skip the A load and the multiply.
        DataCopyExtParams copyParams{1U, cnt * BYTES_PER_FLOAT, 0U, 0U, 0U};
        DataCopyPad(wsReGm_[yStart], xr_, copyParams);
        DataCopyPad(wsImGm_[yStart], xi_, copyParams);
        return;
    }

    uint32_t aColBase = col;
    const bool isTransposed = (trans_ == ACLBLAS_OP_T) || (trans_ == ACLBLAS_OP_C);
    if (isTransposed && uplo_ == ACLBLAS_LOWER) {
        aColBase = col - bandIdx;
    } else if (isTransposed && uplo_ == ACLBLAS_UPPER) {
        aColBase = col + bandIdx;
    }
    LoadBandChunk(aRowBase, aColBase, cnt, ar_, ai_);

    ComplexMul(cnt, trans_ == ACLBLAS_OP_C);
    StoreBandResult(yStart, cnt);
}

__aicore__ inline void CtbmvComputeAIV::ProcessBand(uint32_t bandIdx)
{
    uint32_t firstCol = 0U;
    uint32_t bandLen = n_;
    uint32_t aRowBase = 0U;
    const bool isTransposed = (trans_ == ACLBLAS_OP_T) || (trans_ == ACLBLAS_OP_C);

    if (uplo_ == ACLBLAS_LOWER && !isTransposed) {
        bandLen = n_ - bandIdx;
        aRowBase = bandIdx;
    } else if (uplo_ == ACLBLAS_LOWER && isTransposed) {
        firstCol = bandIdx;
        aRowBase = bandIdx;
    } else if (uplo_ == ACLBLAS_UPPER && !isTransposed) {
        firstCol = bandIdx;
        aRowBase = k_ - bandIdx;
    } else {
        bandLen = n_ - bandIdx;
        aRowBase = k_ - bandIdx;
    }

    const bool yShiftsForward = (uplo_ == ACLBLAS_LOWER && !isTransposed) || (uplo_ == ACLBLAS_UPPER && isTransposed);

    for (uint32_t col = firstCol; col < bandLen; col += tileCnt_) {
        uint32_t cnt = ((col + tileCnt_) <= bandLen) ? tileCnt_ : (bandLen - col);
        if (cnt == 0U) {
            continue;
        }
        const uint32_t yStart = yShiftsForward ? (col + bandIdx) : (col - bandIdx);
        ProcessBandChunk(bandIdx, col, cnt, aRowBase, yStart);
    }
}

__aicore__ inline void CtbmvComputeAIV::ZeroWorkspace()
{
    const uint32_t perCore = CeilDivU32(n_, useCoreNum_);
    const uint32_t start = blockIdx_ * perCore;
    if (start >= n_) {
        return;
    }
    uint32_t cnt = ((start + perCore) <= n_) ? perCore : (n_ - start);

    LocalTensor<float> zeros = yr_;
    uint32_t done = 0U;
    while (done < cnt) {
        uint32_t batch = ((cnt - done) > maxCnt_) ? maxCnt_ : (cnt - done);
        Duplicate(zeros, 0.0f, batch);
        DataCopyExtParams copyParams{1U, batch * BYTES_PER_FLOAT, 0U, 0U, 0U};
        DataCopyPad(wsReGm_[start + done], zeros, copyParams);
        DataCopyPad(wsImGm_[start + done], zeros, copyParams);
        done += batch;
    }
}

__aicore__ inline void CtbmvComputeAIV::Process()
{
    if (blockIdx_ >= useCoreNum_) {
        return;
    }

    ZeroWorkspace();
    SyncAll();

    SetAtomicAdd<float>();
    for (uint32_t bandIdx = blockIdx_; bandIdx <= k_; bandIdx += useCoreNum_) {
        ProcessBand(bandIdx);
    }
    SetAtomicNone();

    // Publish the accumulator into x from this same dispatch. A separate copy
    // kernel costs a full launch (measured ~14 us on 910B3, versus 0.33 us for
    // an idle stream sync), which dominates at these problem sizes. One
    // cross-core barrier is far cheaper than a second dispatch.
    SyncAll();
    CopyBackTiles();
}

// Re-interleave the accumulator planes and write them over x. Split across the
// same cores that ran the bands; each core owns a disjoint slice of y.
__aicore__ inline void CtbmvComputeAIV::CopyBackTiles()
{
    const uint32_t perCore = CeilDivU32(n_, useCoreNum_);
    const uint32_t start = blockIdx_ * perCore;
    if (start >= n_) {
        return;
    }
    const uint32_t total = ((start + perCore) <= n_) ? perCore : (n_ - start);

    uint32_t done = 0U;
    while (done < total) {
        const uint32_t cnt = ((total - done) > tileCnt_) ? tileCnt_ : (total - done);
        const uint32_t base = start + done;

        DataCopyExtParams in{1U, cnt * BYTES_PER_FLOAT, 0U, 0U, 0U};
        DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0f};
        DataCopyPad(yr_, wsReGm_[base], in, pad);
        DataCopyPad(yi_, wsImGm_[base], in, pad);

        const int32_t evt = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evt);
        WaitFlag<HardEvent::MTE2_V>(evt);

        // yr_ and yi_ are adjacent planes maxCnt_ apart, which is exactly what
        // the interleave table encodes.
        LocalTensor<float> aos = aosQueue_.AllocTensor<float>();
        Gather(aos, yr_, tblOut_, 0U, cnt * FLOATS_PER_COMPLEX);
        aosQueue_.EnQue(aos);
        LocalTensor<float> ready = aosQueue_.DeQue<float>();
        DataCopyExtParams out{1U, cnt * FLOATS_PER_COMPLEX * BYTES_PER_FLOAT, 0U, 0U, 0U};
        DataCopyPad(xGm_[static_cast<uint64_t>(base) * FLOATS_PER_COMPLEX], ready, out);
        aosQueue_.FreeTensor(ready);
        done += cnt;
    }
}

// ---------------------------------------------------------------------------
// Copy-back: interleave the workspace planes into x, honouring incx.
// ---------------------------------------------------------------------------
class CtbmvCopyAIV {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gatherOffset, GM_ADDR workspace, const CtbmvTilingData &tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t XPhysicalPos(uint32_t logical) const
    {
        return (incx_ >= 0) ? (logical * absIncx_) : ((n_ - 1U - logical) * absIncx_);
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> buf_;
    TBuf<QuePosition::VECCALC> offsetBuf_;

    GlobalTensor<float> xGm_;
    GlobalTensor<float> wsReGm_;
    GlobalTensor<float> wsImGm_;
    GlobalTensor<uint32_t> offsetGm_;

    LocalTensor<float> re_, im_, aos_;
    LocalTensor<uint32_t> offsets_;

    uint32_t blockIdx_ = 0U;
    uint32_t n_ = 0U;
    uint32_t useCoreNum_ = 1U;
    uint32_t maxCnt_ = 0U;
    uint32_t tileCnt_ = 0U;
    int64_t incx_ = 1;
    uint32_t absIncx_ = 1U;
};

__aicore__ inline void CtbmvCopyAIV::Init(GM_ADDR x, GM_ADDR gatherOffset, GM_ADDR workspace,
                                          const CtbmvTilingData &tiling)
{
    blockIdx_ = GetBlockIdx();
    n_ = tiling.n;
    useCoreNum_ = (tiling.useCoreNum == 0U || tiling.useCoreNum > CTBMV_MAX_CORE_NUM) ? 1U : tiling.useCoreNum;
    incx_ = tiling.incx;
    absIncx_ = (incx_ >= 0) ? static_cast<uint32_t>(incx_) : static_cast<uint32_t>(-incx_);
    maxCnt_ = CTBMV_MAX_DATA_COUNT;

    const uint64_t xFloats = static_cast<uint64_t>(n_) * FLOATS_PER_COMPLEX;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), xFloats);
    __gm__ float *wsBase = reinterpret_cast<__gm__ float *>(workspace);
    wsReGm_.SetGlobalBuffer(wsBase, n_);
    wsImGm_.SetGlobalBuffer(wsBase + n_, n_);
    // Interleave table starts after the two band-gather tables.
    offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gatherOffset) + 4U * maxCnt_,
                              maxCnt_ * FLOATS_PER_COMPLEX);

    pipe_.InitBuffer(buf_, maxCnt_ * 4U * BYTES_PER_FLOAT);
    pipe_.InitBuffer(offsetBuf_, maxCnt_ * FLOATS_PER_COMPLEX * sizeof(uint32_t));

    LocalTensor<float> all = buf_.Get<float>();
    re_ = all[0U];
    im_ = all[maxCnt_];
    aos_ = all[2U * maxCnt_];
    offsets_ = offsetBuf_.Get<uint32_t>();
}

__aicore__ inline void CtbmvCopyAIV::Process()
{
    if (blockIdx_ >= useCoreNum_) {
        return;
    }
    const uint32_t perCore = CeilDivU32(n_, useCoreNum_);
    const uint32_t start = blockIdx_ * perCore;
    if (start >= n_) {
        return;
    }
    const uint32_t total = ((start + perCore) <= n_) ? perCore : (n_ - start);

    // Offset table maps plane lanes back to interleaved positions; load once.
    DataCopyExtParams offParams{1U, maxCnt_ * FLOATS_PER_COMPLEX * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<uint32_t> offPad{true, 0U, 0U, 0U};
    DataCopyPad(offsets_, offsetGm_, offParams, offPad);

    uint32_t done = 0U;
    while (done < total) {
        const uint32_t cnt = ((total - done) > maxCnt_) ? maxCnt_ : (total - done);
        const uint32_t base = start + done;

        DataCopyExtParams inParams{1U, cnt * BYTES_PER_FLOAT, 0U, 0U, 0U};
        DataCopyPadExtParams<float> inPad{true, 0U, 0U, 0.0f};
        DataCopyPad(re_, wsReGm_[base], inParams, inPad);
        DataCopyPad(im_, wsImGm_[base], inParams, inPad);

        // Re-interleave: aos[2i] = re[i], aos[2i+1] = im[i].
        // re_ and im_ are adjacent in UB, so a single Gather with the precomputed
        // table produces the interleaved layout.
        Gather(aos_, re_, offsets_, 0U, cnt * FLOATS_PER_COMPLEX);

        DataCopyExtParams outParams{1U, cnt * FLOATS_PER_COMPLEX * BYTES_PER_FLOAT, 0U, 0U, 0U};
        DataCopyPad(xGm_[static_cast<uint64_t>(base) * FLOATS_PER_COMPLEX], aos_, outParams);
        done += cnt;
    }
}

// ---------------------------------------------------------------------------
// k == 0: A is diagonal, scale x in place (no workspace needed).
// ---------------------------------------------------------------------------
class CtbmvDiagAIV {
public:
    __aicore__ inline void Init(GM_ADDR aBanded, GM_ADDR x, GM_ADDR gatherOffset, const CtbmvTilingData &tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t XPhysicalPos(uint32_t logical) const
    {
        return (incx_ >= 0) ? (logical * absIncx_) : ((n_ - 1U - logical) * absIncx_);
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> buf_;
    TBuf<QuePosition::VECCALC> offsetBuf_;

    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<uint32_t> offsetGm_;

    LocalTensor<float> aos_, ar_, ai_, xr_, xi_, yr_, yi_, t0_, t1_, t2_, t3_;
    LocalTensor<uint32_t> offsets_, tblRe_, tblIm_, tblRevRe_, tblRevIm_;
    TBuf<QuePosition::VECCALC> bandTblBuf_;

    uint32_t blockIdx_ = 0U;
    uint32_t n_ = 0U;
    uint32_t lda_ = 0U;
    uint32_t useCoreNum_ = 1U;
    uint32_t maxCnt_ = 0U;
    uint32_t tileCnt_ = 0U;
    int64_t incx_ = 1;
    uint32_t absIncx_ = 1U;
    uint32_t trans_ = ACLBLAS_OP_N;
    uint32_t diag_ = ACLBLAS_NON_UNIT;
};

__aicore__ inline void CtbmvDiagAIV::Init(GM_ADDR aBanded, GM_ADDR x, GM_ADDR gatherOffset,
                                          const CtbmvTilingData &tiling)
{
    blockIdx_ = GetBlockIdx();
    n_ = tiling.n;
    lda_ = tiling.lda;
    useCoreNum_ = (tiling.useCoreNum == 0U || tiling.useCoreNum > CTBMV_MAX_CORE_NUM) ? 1U : tiling.useCoreNum;
    incx_ = tiling.incx;
    absIncx_ = (incx_ >= 0) ? static_cast<uint32_t>(incx_) : static_cast<uint32_t>(-incx_);
    trans_ = tiling.trans;
    diag_ = tiling.diag;
    maxCnt_ = CTBMV_MAX_DATA_COUNT;

    const uint64_t xFloats = static_cast<uint64_t>(n_) * FLOATS_PER_COMPLEX;
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aBanded),
                         static_cast<uint64_t>(lda_) * n_ * FLOATS_PER_COMPLEX);
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), xFloats);
    offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gatherOffset) + 4U * maxCnt_,
                              maxCnt_ * FLOATS_PER_COMPLEX);

    // aos_ stages a strided A load, where every 8-byte complex occupies a full
    // 32-byte UB block, so it needs FLOATS_PER_PADDED_BLOCK floats per element.
    pipe_.InitBuffer(buf_, maxCnt_ * (FLOATS_PER_PADDED_BLOCK + 10U) * BYTES_PER_FLOAT);
    pipe_.InitBuffer(offsetBuf_, maxCnt_ * FLOATS_PER_COMPLEX * sizeof(uint32_t));

    LocalTensor<float> all = buf_.Get<float>();
    aos_ = all[0U];                                  // FLOATS_PER_PADDED_BLOCK * maxCnt
    ar_ = all[(FLOATS_PER_PADDED_BLOCK + 0U) * maxCnt_];
    ai_ = all[(FLOATS_PER_PADDED_BLOCK + 1U) * maxCnt_];
    xr_ = all[(FLOATS_PER_PADDED_BLOCK + 2U) * maxCnt_];
    xi_ = all[(FLOATS_PER_PADDED_BLOCK + 3U) * maxCnt_];
    yr_ = all[(FLOATS_PER_PADDED_BLOCK + 4U) * maxCnt_];
    yi_ = all[(FLOATS_PER_PADDED_BLOCK + 5U) * maxCnt_];
    t0_ = all[(FLOATS_PER_PADDED_BLOCK + 6U) * maxCnt_];
    t1_ = all[(FLOATS_PER_PADDED_BLOCK + 7U) * maxCnt_];
    t2_ = all[(FLOATS_PER_PADDED_BLOCK + 8U) * maxCnt_];
    t3_ = all[(FLOATS_PER_PADDED_BLOCK + 9U) * maxCnt_];
    offsets_ = offsetBuf_.Get<uint32_t>();

    pipe_.InitBuffer(bandTblBuf_, maxCnt_ * 4U * static_cast<uint32_t>(sizeof(uint32_t)));
    GlobalTensor<uint32_t> bandTblGm;
    bandTblGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gatherOffset), maxCnt_ * 4U);
    LocalTensor<uint32_t> bt = bandTblBuf_.Get<uint32_t>();
    tblRe_ = bt[0U];
    tblIm_ = bt[maxCnt_];
    tblRevRe_ = bt[2U * maxCnt_];
    tblRevIm_ = bt[3U * maxCnt_];
    DataCopyExtParams btp{1U, maxCnt_ * 4U * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<uint32_t> btpad{true, 0U, 0U, 0U};
    DataCopyPad(bt, bandTblGm, btp, btpad);
}

__aicore__ inline void CtbmvDiagAIV::Process()
{
    if (blockIdx_ >= useCoreNum_) {
        return;
    }
    if (diag_ == ACLBLAS_UNIT) {
        return;  // op(A) == I, x unchanged
    }

    const uint32_t perCore = CeilDivU32(n_, useCoreNum_);
    const uint32_t start = blockIdx_ * perCore;
    if (start >= n_) {
        return;
    }
    const uint32_t total = ((start + perCore) <= n_) ? perCore : (n_ - start);

    DataCopyExtParams offParams{1U, maxCnt_ * FLOATS_PER_COMPLEX * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<uint32_t> offPad{true, 0U, 0U, 0U};
    DataCopyPad(offsets_, offsetGm_, offParams, offPad);

    uint32_t done = 0U;
    while (done < total) {
        const uint32_t cnt = ((total - done) > maxCnt_) ? maxCnt_ : (total - done);
        const uint32_t base = start + done;

        // Diagonal of a k==0 band matrix is row 0, column j, stored column-major
        // at A[0 + j * lda]; stride between consecutive j is lda complex values.
        DataCopyExtParams aParams{static_cast<uint16_t>(cnt), FLOATS_PER_COMPLEX * BYTES_PER_FLOAT,
                                  (lda_ - 1U) * FLOATS_PER_COMPLEX * BYTES_PER_FLOAT, 0U, 0U};
        DataCopyPadExtParams<float> aPad{true, 0U, 0U, 0.0f};
        DataCopyPad(aos_, aGm_[static_cast<uint64_t>(base) * lda_ * FLOATS_PER_COMPLEX], aParams, aPad);

        // A was fetched with a strided (32B-padded) copy, so use the stride-8 tables.
        Gather(ar_, aos_, tblRe_, 0U, cnt);
        Gather(ai_, aos_, tblIm_, 0U, cnt);
        uint64_t rsvd = 0U;
        const uint32_t repeats = CeilDivU32(cnt * FLOATS_PER_COMPLEX, FLOATS_PER_REPEAT);

        // x is contiguous, so this is a dense copy: the interleaved layout is
        // [re0,im0,re1,im1,...] with no 32B padding, which GatherMask splits
        // directly (the stride-8 tables above only apply to strided loads).
        DataCopyExtParams xParams{1U, cnt * FLOATS_PER_COMPLEX * BYTES_PER_FLOAT, 0U, 0U, 0U};
        DataCopyPad(aos_, xGm_[static_cast<uint64_t>(base) * FLOATS_PER_COMPLEX], xParams, aPad);
        GatherMask<float>(xr_, aos_, PATTERN_REAL, false, 0U,
                          {1U, static_cast<uint16_t>(repeats), FLOATS_PER_BLOCK, FLOATS_PER_BLOCK}, rsvd);
        GatherMask<float>(xi_, aos_, PATTERN_IMAG, false, 0U,
                          {1U, static_cast<uint16_t>(repeats), FLOATS_PER_BLOCK, FLOATS_PER_BLOCK}, rsvd);

        Mul(t0_, ar_, xr_, cnt);
        Mul(t1_, ai_, xi_, cnt);
        Mul(t2_, ar_, xi_, cnt);
        Mul(t3_, ai_, xr_, cnt);
        if (trans_ == ACLBLAS_OP_C) {
            Add(yr_, t0_, t1_, cnt);
            Sub(yi_, t2_, t3_, cnt);
        } else {
            Sub(yr_, t0_, t1_, cnt);
            Add(yi_, t2_, t3_, cnt);
        }

        Gather(aos_, yr_, offsets_, 0U, cnt * FLOATS_PER_COMPLEX);

        DataCopyExtParams outParams{1U, cnt * FLOATS_PER_COMPLEX * BYTES_PER_FLOAT, 0U, 0U, 0U};
        DataCopyPad(xGm_[static_cast<uint64_t>(base) * FLOATS_PER_COMPLEX], aos_, outParams);
        done += cnt;
    }
}

// ---------------------------------------------------------------------------
// incx != 1 support: copy x into (and out of) a contiguous logical-order buffer.
// Doing this once keeps every other kernel on the simple stride-1 path and
// avoids per-chunk reverse tables, whose alignment depends on the chunk size.
// Each element is one 8-byte complex; blockCount elements are moved per call
// with srcStride/dstStride expressed in bytes.
// ---------------------------------------------------------------------------
__aicore__ inline void CtbmvCopyStrided(GM_ADDR dst, GM_ADDR src, uint32_t n, int64_t incx, bool toContiguous)
{
    if (GetBlockIdx() != 0U) {
        return;
    }
    const uint32_t absIncx = (incx >= 0) ? static_cast<uint32_t>(incx) : static_cast<uint32_t>(-incx);
    const uint64_t stridedFloats = (static_cast<uint64_t>(n - 1U) * absIncx + 1U) * FLOATS_PER_COMPLEX;

    GlobalTensor<float> dstGm;
    GlobalTensor<float> srcGm;
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst),
                          toContiguous ? static_cast<uint64_t>(n) * FLOATS_PER_COMPLEX : stridedFloats);
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src),
                          toContiguous ? stridedFloats : static_cast<uint64_t>(n) * FLOATS_PER_COMPLEX);

    TPipe pipe;
    TBuf<QuePosition::VECCALC> buf;
    pipe.InitBuffer(buf, CTBMV_MAX_DATA_COUNT * FLOATS_PER_PADDED_BLOCK * BYTES_PER_FLOAT);
    LocalTensor<float> ub = buf.Get<float>();

    const uint32_t elemBytes = FLOATS_PER_COMPLEX * BYTES_PER_FLOAT;
    const uint32_t skipBytes = (absIncx - 1U) * elemBytes;
    DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0f};

    // Walk one element at a time in the DMA's block dimension: blockCount moves
    // `cnt` complexes, and for negative incx the physical addresses descend as
    // the logical index rises. DataCopyPad can only stride upwards, so process
    // the chunk anchored at its lowest address and undo the resulting reversal
    // by mirroring the logical range that the chunk covers.
    for (uint32_t done = 0U; done < n; done += CTBMV_MAX_DATA_COUNT) {
        const uint32_t cnt = ((n - done) > CTBMV_MAX_DATA_COUNT) ? CTBMV_MAX_DATA_COUNT : (n - done);

        if (incx >= 0) {
            const uint64_t firstPos = static_cast<uint64_t>(done) * absIncx;
            if (toContiguous) {
                DataCopyExtParams in{static_cast<uint16_t>(cnt), elemBytes, skipBytes, 0U, 0U};
                DataCopyPad(ub, srcGm[firstPos * FLOATS_PER_COMPLEX], in, pad);
                DataCopyExtParams flat{static_cast<uint16_t>(cnt), elemBytes, 0U, 0U, 0U};
                DataCopyPad(dstGm[static_cast<uint64_t>(done) * FLOATS_PER_COMPLEX], ub, flat);
            } else {
                DataCopyExtParams flat{static_cast<uint16_t>(cnt), elemBytes, 0U, 0U, 0U};
                DataCopyPad(ub, srcGm[static_cast<uint64_t>(done) * FLOATS_PER_COMPLEX], flat, pad);
                DataCopyExtParams out{static_cast<uint16_t>(cnt), elemBytes, 0U, skipBytes, 0U};
                DataCopyPad(dstGm[firstPos * FLOATS_PER_COMPLEX], ub, out);
            }
            continue;
        }

        // incx < 0: logical [done, done+cnt) occupies descending physical slots.
        // The lowest address belongs to logical done+cnt-1, and an ascending DMA
        // therefore yields logical order reversed. Copy element by element so the
        // logical ordering is preserved without needing a reversal table.
        for (uint32_t i = 0U; i < cnt; ++i) {
            const uint32_t logical = done + i;
            const uint64_t phys = static_cast<uint64_t>(n - 1U - logical) * absIncx;
            DataCopyExtParams one{1U, elemBytes, 0U, 0U, 0U};
            if (toContiguous) {
                DataCopyPad(ub, srcGm[phys * FLOATS_PER_COMPLEX], one, pad);
                DataCopyPad(dstGm[static_cast<uint64_t>(logical) * FLOATS_PER_COMPLEX], ub, one);
            } else {
                DataCopyPad(ub, srcGm[static_cast<uint64_t>(logical) * FLOATS_PER_COMPLEX], one, pad);
                DataCopyPad(dstGm[phys * FLOATS_PER_COMPLEX], ub, one);
            }
        }
    }
}

extern "C" __global__ __aicore__ void ctbmv_normalise_kernel(GM_ADDR xContig, GM_ADDR x, uint32_t n, int64_t incx)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtbmvCopyStrided(xContig, x, n, incx, true);
}

extern "C" __global__ __aicore__ void ctbmv_restore_kernel(GM_ADDR x, GM_ADDR xContig, uint32_t n, int64_t incx)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtbmvCopyStrided(x, xContig, n, incx, false);
}

extern "C" __global__ __aicore__ void ctbmv_compute_kernel(GM_ADDR aBanded, GM_ADDR x, GM_ADDR gatherOffset,
                                                            GM_ADDR workspace, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtbmvComputeAIV op;
    op.Init(aBanded, x, gatherOffset, workspace, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void ctbmv_copy_kernel(GM_ADDR x, GM_ADDR gatherOffset, GM_ADDR workspace,
                                                         CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtbmvCopyAIV op;
    op.Init(x, gatherOffset, workspace, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void ctbmv_diag_kernel(GM_ADDR aBanded, GM_ADDR x, GM_ADDR gatherOffset,
                                                         CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtbmvDiagAIV op;
    op.Init(aBanded, x, gatherOffset, tiling);
    op.Process();
}

void ctbmv_arch22_kernel_do(GM_ADDR aBanded, GM_ADDR x, GM_ADDR gatherOffset, GM_ADDR workspace,
                            const CtbmvTilingData &tiling, uint32_t numBlocks, void *stream)
{
    // Workspace layout: [ yr(n) | yi(n) | xContig(2n floats) ]
    GM_ADDR xContig = workspace + static_cast<uint64_t>(tiling.n) * 2U * sizeof(float);
    const bool strided = (tiling.incx != 1);
    GM_ADDR xWork = strided ? xContig : x;

    if (strided) {
        ctbmv_normalise_kernel<<<1, nullptr, stream>>>(xContig, x, tiling.n, tiling.incx);
    }

    if (tiling.k == 0U) {
        ctbmv_diag_kernel<<<numBlocks, nullptr, stream>>>(aBanded, xWork, gatherOffset, tiling);
    } else {
        ctbmv_compute_kernel<<<numBlocks, nullptr, stream>>>(aBanded, xWork, gatherOffset, workspace, tiling);
    }

    if (strided) {
        ctbmv_restore_kernel<<<1, nullptr, stream>>>(x, xContig, tiling.n, tiling.incx);
    }
}
