# aclblasCtbmv — attempt log, failures and findings

**Hardware:** built `SOC_VERSION=ascend910b3` (A2); executed on a devenv reporting
`socVersion=Ascend910_9362` (A3). Both map to `NPU_ARCH=dav-2201` / `arch22`, so
one implementation covers A2 and A3 — which is what "A2/A3" in the task title means.
Perf targets are specified on A2 (910B3).

Chronological record of what was tried on `blas/tbmv/arch22/ctbmv_*`, so future
work does not repeat it. Current state: **correct (1872/1872), 3.5-5x too slow**.

## Current committed state

Branch `feat/aclblasCtbmv-arch22`, HEAD `39fcf1d`.
- 84/84 basic + **1872/1872** extended (12 uplo/trans/diag x 13 shapes x
  incx {1,2,-1,-3} x 3 lda paddings), verified against `cblas_ctbmv(CblasColMajor)`
- 63.6 / 71.8 / 101.8 us vs targets 10 / 16.51 / 19.77
- Structure: band-parallel. Each core owns a set of diagonal bands, accumulates
  into a GM workspace with atomics, then a second kernel re-interleaves into x.

`3d9a7bf` on the same branch holds the fused column-tile attempt (failure 5).
It compiles but is **functionally wrong** — kept for reference only.

## Correctness bugs found and fixed (all on hardware)

### 1. Column-major banded storage
Indexed `A[band*lda + col]` (row-major). BLAS banded storage is column-major:
`A[row + col*lda]`, so a band has stride `lda`. Symptom: results drifted by
1, 2, 3... Fixed with a strided `DataCopyPad`.

**The in-repo `stbmv/arch22` has this exact bug and ships with it** — see
`stbmv-arch22-banded-layout-bug.md`. Its tests pass only because the generator
uses the same wrong convention. Do not copy its band-read code.

### 2. DataCopyPad pads every burst to 32 B in UB
A probe kernel dumped actual UB contents:
```
dstStride=0 : 0 1 0 0 0 0 0 0 4 5 4 4 4 4 4 4 ...
```
An 8-byte block lands at float offset `i*8`, not `i*2`, and **no `dstStride`
value packs densely**. Fixed with stride-8 `Gather` tables (the `cgerc/arch22`
idiom).

### 3. Diag kernel staging buffer undersized
Sized for dense complex (2 floats/elem) but the strided load needs the padded
layout (8 floats/elem) — silent overflow into the adjacent plane for n > 256.

### 4. Strided incx faulted
Per-element Alloc/EnQue/DeQue/Free inside a loop. Replaced by normalising x into
a contiguous workspace buffer once, running everything at stride 1, then
scattering back. Also removed every per-element `SetValue`/`GetValue` (rule R1).

## Performance attempts — all reverted

| # | Approach | Result |
|---|---|---|
| 1 | Row-tiled (output-stationary) | wrong (36 fails) **and** slower: k+1 tiny DMAs per tile |
| 2 | Merge compute+copy into one kernel | 63 -> 59 us, then 55 us; cost moved, not removed |
| 3 | Column-burst load per band | correct but **worse** (91/162/364 us): A traffic becomes (k+1)^2 * n |
| 4 | Fused column-tile, first cut | wrong and 3x slower — block DMA left **inside** the band loop |
| 5 | Fused column-tile, hoisted | **compiles**, DMA correctly hoisted, but 45/84 fail — see below |

### Wins that were kept
- cache the gather tables process-wide and drop the per-call
  `aclrtMalloc`/`Memcpy`/**`SynchronizeStream`**/`Free` — **~40% faster** and
  restores the documented async contract
- tile to the problem size and copy only the needed table prefix (removes a
  fixed 16 KB per-core preamble)

### Failure 5 in detail (the most valuable finding)

```cpp
Add(yr_[lo - rowStart], yr_[lo - rowStart], t0_, cnt);   // dstOff = d floats
```

**AscendC vector ops require 32-byte aligned operands.** Band `d` writes at
element offset `d` = `4*d` bytes, aligned only when `d % 8 == 0`. So `d=0` is
correct and every other band corrupts. Traced on a 4x4 LOWER/N k=1 case where
`y[0]`, `y[2]`, `y[3]` were right and only `y[1]` was wrong.

**`Gather` does NOT have this restriction** — it takes arbitrary byte offsets.
That asymmetry is why the bug is easy to hit: the same shift is legal in one
primitive and silently wrong in the other. The same alignment class also broke
an earlier reverse-lookup gather (tables indexed at `maxCnt - cnt`).

**Offline validation caught none of this**, because the Python models
(`data/verify_*.py`, 324/324 and 240/240 configs) validate the *algorithm*.
The algorithm was right every time; the failures were all hardware constraints.
**Model the memory layout and the primitive constraints, not just the math.**

## Alignment rule — confirmed against CANN built-ins

Searching the 266 built-in Ascend C kernels for offset vector writes found
`ops_math/ascendc/exp_segsum_grad/exp_segsum_grad.h:311`:

```cpp
int64_t calNumAlign = CeilA2B(calCount, blockSize) * blockSize;   // aligned unit
int64_t offset      = (row - i - 1) * calNumAlign;                // multiple of it
Add(currentRow[offset], currentRow[offset], lastRow, calNumAlign);
```

So offset vector operands **are** legal — but only when the offset is a multiple
of a 32-byte-aligned stride. Built-ins universally construct offsets as
`i * alignedStride`. Our failing code used `yr_[lo - rowStart]` where the offset
is the band index `d` (arbitrary 0..k), which is the violation.

`grouped_bias_add_grad/arch32/...:225` shows the same discipline:
`Add(src[0], src[0], src[halfRows * cols], ...)` — offset is a multiple of `cols`.

## Validated fix (offline, not yet on hardware)

Shift the **source**, never the destination. For each band process the full tile
width with x loaded from a clamped GM offset and invalid lanes zeroed, so every
vector op writes at offset 0. `DataCopyPad` accepts arbitrary *element* offsets
in GM, so the shift costs nothing there.

`data/verify_align.py`: **528/528 configs** (11 shapes x 12 enum combos x 4 tile
widths) match the dense cblas-equivalent reference.

Extra cost: each band processes `rows` lanes instead of `cnt <= rows`. For
rows=1024, k=32 that is ~3% more vector work — negligible against the ~14 us
saved by removing a dispatch.

## Recommended next step

Fix failure 5 by keeping every vector op at offset 0: process the full tile
width for each band, load x from a clamped source offset, and zero the invalid
prefix with `Duplicate` (safe at offset 0). Then the fused kernel from `3d9a7bf`
should be correct, and it removes a dispatch (~14 us), the atomics, the
`SyncAll` and the workspace round trip.

Predicted by the cost model: intercept drops from ~58.5 us toward ~30 us, and
`serialDMA` falls by roughly `k+1`x. That plausibly reaches the k>=16 targets.

**Case 1 (n=512, k=8, target 10 us) still looks unreachable**: the measured
floor for *any* call on this hardware is 25 us, and the A100 itself needs 13.9 us.
If that holds after the rewrite, report it with evidence rather than submit a
known miss — the test README permits amending anomalous perf cases with
justification.

## Process notes

- Five restructures failed; three were the same alignment class. When a fix
  touches operand offsets, check 32 B alignment **first**.
- Always keep a known-good commit and re-verify 1872 cases after every revert.
- The traced 4x4 case (`data/dbg.cpp`) localised two separate bugs in minutes
  where the 84-case suite only said "45 failed". Build the small traceable case.
