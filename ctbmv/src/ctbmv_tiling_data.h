/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CTBMV_TILING_DATA_H
#define CTBMV_TILING_DATA_H

#include <cstdint>

// Upper bound on cores used by ctbmv; mirrors the stbmv/arch22 convention.
constexpr uint32_t CTBMV_MAX_CORE_NUM = 50;

// Number of complex elements handled per UB tile.
// A strided DataCopyPad (used to gather one diagonal band out of column-major
// banded storage) pads every 8-byte block up to a full 32-byte UB block, so the
// staging buffer costs 8 floats per complex rather than 2. Budget per complex:
//   aosQueue (8 floats, double buffered) = 64 B
//   split planes ar/ai/xr/xi/yr/yi + t0..t3 (10 floats) = 40 B
//   gather tables (real, imag, interleave; 3 x uint32) = 12 B
//                                                total = 116 B
// 1024 * 116 = 118784 B ~= 116 KB, comfortably inside the 192 KB UB.
constexpr uint32_t CTBMV_MAX_DATA_COUNT = 1024;

struct CtbmvTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t useCoreNum;
    int64_t incx;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
};

#endif  // CTBMV_TILING_DATA_H
