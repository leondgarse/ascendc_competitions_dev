/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CTBMV_KERNEL_H
#define CTBMV_KERNEL_H

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "ctbmv_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

void ctbmv_arch22_kernel_do(
    GM_ADDR aBanded, GM_ADDR x, GM_ADDR gatherOffset, GM_ADDR workspace, const CtbmvTilingData &tiling,
    uint32_t numBlocks, void *stream);

#endif  // CTBMV_KERNEL_H
