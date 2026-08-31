# aclblasStbmv (arch22) misreads column-major banded storage

**Component**: `ops-blas` @ `4307872`, `blas/tbmv/arch22/stbmv_kernel.cpp`
**Hardware**: Atlas 800I A2 (Ascend910B3), CANN 9.1.0
**Severity**: correctness — results disagree with Netlib/cuBLAS for all tested shapes

## Summary

`aclblasStbmv_legacy` on arch22 indexes the banded array as if it were
**row-major**, but BLAS banded storage (Netlib `stbmv`, cuBLAS `cublasStbmv`,
and the repo's own golden `cblas_stbmv(CblasColMajor, ...)`) is **column-major**.

The in-tree test suite does not catch this because its data generator uses the
same row-major convention as the kernel and validates against a hand-written
reference rather than the cblas golden that already exists in the same directory.

## Evidence

Reproducer compares `aclblasStbmv_legacy` against `cblas_stbmv(CblasColMajor,...)`
on a 4x4 UPPER, k=1, NON_UNIT case with x = all ones:

```
lda=2 (k+1=2)  cblas=[12 14 16 4]  stbmv=[12 13 14 3]  maxdiff=2.0   MISMATCH
lda=4 (k+1=2)  cblas=[12 14 16 4]  stbmv=[12  2  0 0]  maxdiff=16.0  MISMATCH
```

Source: `findings/stbmv_bug.cpp` (in this directory).

## Root cause

`stbmv_kernel.cpp:113-114`

```cpp
uint64_t r = static_cast<uint64_t>(rowOffset) * lda + colOffset;
DataCopy(LocalA, aGM[r], dataCount);          // contiguous, stride 1
```

This computes `A[band * lda + col]` and reads consecutive elements, i.e. it
assumes a band is contiguous. In column-major banded storage element
`(row, col)` lives at `A[row + col * lda]`, so a band is `fixed row, varying
col` and has **stride `lda`**, not 1.

The same assumption appears in `ProcessGeneralBandCol` (`aOffset = aRowBase *
lda + aColBase`) and in `CopyInPad`.

## Why the tests pass

`test/tbmv/stbmv/arch22/stbmv_test.cpp:125,132` generates A as

```cpp
a[static_cast<size_t>(bandIdx) * lda + col] = dist(rng);
```

which is the kernel's row-major convention, and line 198 reads it back the same
way for the expected value. So generator, reference and kernel share one
non-standard layout and agree with each other.

`test/tbmv/stbmv/stbmv_golden.h:38` does call
`cblas_stbmv(CblasColMajor, ...)`, but the arch22 test path does not use it.

## Suggested fix

Read bands with `stride = lda` (e.g. `DataCopyPad` with
`blockCount = cnt, blockLen = sizeof(T), srcStride = (lda - 1) * sizeof(T)`),
or restructure to load whole banded columns (which are contiguous) and select
the wanted row in UB.

Note the naive strided form pads every element to a 32-byte UB block, so the
column-wise form is preferable for bandwidth.

## Cross-reference

`aclblasCtbmv` (arch22, complex64) hit the same trap during development; it now
uses the strided read and is verified against `cblas_ctbmv(CblasColMajor,...)`
over 1872 cases (12 uplo/trans/diag combos x 13 shapes x incx in {1,2,-1,-3} x
3 lda paddings).
