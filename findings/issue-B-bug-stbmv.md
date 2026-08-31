# [Bug-Report|缺陷反馈]: aclblasStbmv (arch22) 带状矩阵按行主序索引，与 BLAS 列主序约定不符

## Describe the current behavior / 问题描述

`blas/tbmv/arch22/stbmv_kernel.cpp:113-114` 按**行主序** `A[band * lda + col]`
读取带状矩阵，一条对角带被当作连续内存（stride=1）：

```cpp
uint64_t r = static_cast<uint64_t>(rowOffset) * lda + colOffset;
DataCopy(LocalA, aGM[r], dataCount);          // 连续读，stride=1
```

但 BLAS 的带状存储约定是**列主序** `A[row + col * lda]`：
一条对角带的 stride 应为 `lda`，一**列**才是连续的。

该约定见于 Netlib `stbmv`、cuBLAS `cublasStbmv`，
以及本仓自己的 golden —— `test/tbmv/stbmv/stbmv_golden.h:38`
调用的正是 `cblas_stbmv(CblasColMajor, ...)`。

因此 `aclblasStbmv` 的结果与 `cblas_stbmv(CblasColMajor,...)` 不一致。

## Environment / 环境信息

- 硬件：Atlas 800I A2 / Ascend910B3
- CANN：9.1.0
- 分支：master
- 涉及文件：`blas/tbmv/arch22/stbmv_kernel.cpp`

## Steps to reproduce the issue / 重现步骤

用 4×4、UPPER、k=1、NON_UNIT、x 全 1 的输入，
分别调用 `aclblasStbmv` 与 `cblas_stbmv(CblasColMajor, ...)` 比对
（完整复现代码见附件 `stbmv_bug.cpp`）：

1. 按列主序带状约定填充 A（即 `A[i-j+k + j*lda]`）
2. 调用 `aclblasStbmv`
3. 用相同数据调用 `cblas_stbmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, ...)`
4. 比对结果

## Describe the expected behavior / 预期结果

两者结果一致，即 `aclblasStbmv` 与 `cblas_stbmv(CblasColMajor,...)` 相同。

## Related log / screenshot / 日志 / 截图

```
lda=2 (k+1=2)  cblas=[12 14 16 4]  stbmv=[12 13 14 3]  maxdiff=2.0    不一致
lda=4 (k+1=2)  cblas=[12 14 16 4]  stbmv=[12  2  0 0]  maxdiff=16.0   不一致
```

`lda > k+1`（带状矩阵有 padding）时偏差更大，因为行主序索引把 padding 也当成了数据。

## Special notes for this issue / 备注

**现网 UT 40/40 通过，但没有覆盖到这个问题**：

`test/tbmv/stbmv/arch22/stbmv_test.cpp:125,132` 生成测试数据时用的是
`a[bandIdx * lda + col]`，与 kernel 的行主序约定一致；
且用手写参考值比对，**没有走同目录下的 `stbmv_golden.h`（cblas golden）**。
两处用了同一个错误约定，因此互相验证通过。

建议在修复 kernel 索引的同时，把该 UT 切换到 `stbmv_golden.h` 的 cblas 参考实现。

补充：我们在实现 `aclblasCtbmv`（complex64，arch22）时踩到同一个坑，
改用列主序（一条带 stride=`lda`）后，与 `cblas_ctbmv(CblasColMajor)` 比对
**1872/1872 用例通过**（12 种 uplo×trans×diag × 13 组 shape ×
incx ∈ {1,2,-1,-3} × 3 组 lda padding），可以佐证列主序才是正确约定。
