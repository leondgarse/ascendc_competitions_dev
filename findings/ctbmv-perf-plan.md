# aclblasCtbmv arch22 — performance gap analysis and plan

## Status

Correct: 1872/1872 (12 uplo/trans/diag x 13 shapes x incx {1,2,-1,-3} x 3 lda paddings).
Speed: 64 / 72 / 103 us against targets 10 / 16.51 / 19.77 us — **3.5–5x short**.

Measured A100 baseline (real hardware, 200/200 cases, `a100_ctbmv_measured.csv`):
13.87 / 16.18 / 19.66 us, i.e. the published targets are genuine end-to-end
`cublasCtbmv` times. Median measured/CSV ratio 1.254 across all 200 cases.

## Root cause (measured, not inferred)

msprof, n=512 k=8: `ctbmv_compute_kernel` 41.9 us, `ctbmv_copy_kernel` 14.7 us.
The copy kernel does a trivial memcpy of 512 complex, so ~14 us is fixed
per-dispatch cost; the rest is the band loop.

The band load is the problem:

```cpp
// blockCount = cnt, blockLen = 8 bytes (ONE complex per burst)
DataCopyExtParams cp{cnt, 8, (lda-1)*8, 0, 0};
```

`agent-skills/.../performance-optim/references/data-copy-prof.md`:
- §2.1 each `DataCopy` should move **>= 16 KB**; we move 8 KB useful per DMA
- §2.2 GM 512B alignment is worth up to 30%; our band start is arbitrary
- §2.3 use stride params not loops — already satisfied

On top of that `DataCopyPad` rounds every burst up to a 32-byte UB block, so an
8-byte payload costs 32 bytes of UB write: **4x waste** before bandwidth even
enters the picture.

## What was tried and rejected (all reverted, each verified)

| Attempt | Outcome |
|---|---|
| Row-tiled (output-stationary) | wrong (36 fails) and slower — k+1 tiny DMAs per tile |
| Merge compute+copy kernels | 63 -> 59 us, then 55 us single kernel; cost moved, not removed |
| Column-burst per band | correct but **worse** (91/162/364 us): A traffic becomes (k+1)^2 * n |
| Fused column-tile (first cut) | wrong and 3x slower — block DMA left inside the band loop |

The fused idea is right; the implementation put the DMA in the wrong place.

## Reference implementations found

**`ops-blas/blas/trmv/arch22/ctrmv_kernel.cpp`** — closest in-repo analogue.
- loads whole matrix columns per burst (`loda_matrix_gm2ub`: nBurst = n_real,
  lenBurst = m_real * 8 bytes), not one element per burst
- splits complex with the `vreducev2` intrinsic on densely packed data
- uses `gm_to_ub_align` with an explicit `dstGap`

**CANN built-in `as_strided`** (`cann-A3-ops-math`, 266 Ascend C kernels total)
— the canonical strided-gather operator. It picks between **five** strategies by
tiling key: `MOVE_ALIGN_B8/16/32/64` (101/102/104/108), `DUAL_CUT` (200),
`SIMT` (400), `GATHER` (500), `ZERO_STRIDE`.

Its `GATHER` path is exactly our shape:
1. `AsCopyGM2Ub` — **one** `blockCount = 1` contiguous `DataCopyPad` of the whole
   working region (no per-element bursts at all)
2. `AsComputeIdx` / `GenDimNIndex` — build index vectors on device
3. `AsGather` — UB-local gather via `MicroAPI` vector ops
4. `AsCopyOut` — one strided store

The lesson: **do not express a strided read as many small DMA bursts. Bring in a
contiguous superset once, then gather inside UB.**

## Attempt 5: fused column-tile (compiled, tested, reverted)

Implemented the plan below using the ops-blas idiom. It **compiled cleanly** and
the block DMA was correctly hoisted out of the band loop, but 45/84 cases failed.

Root cause found by tracing a 4x4 LOWER/N k=1 case (`y[0]` and `y[2..3]` right,
`y[1]` wrong):

```cpp
Add(yr_[lo - rowStart], yr_[lo - rowStart], t0_, cnt);   // dstOff = 1 float
```

**AscendC vector ops require 32-byte aligned operands.** Band d writes its
contribution at element offset `lo - rowStart == d`, which is 4*d bytes — only
aligned when d is a multiple of 8. This is the same alignment class that broke
the earlier reverse-gather attempt.

Note this constraint does **not** apply to `Gather`, which takes arbitrary byte
offsets — that asymmetry is what makes the fix non-obvious.

Two candidate fixes, neither yet implemented:
1. process the full tile width for every band, loading x from a clamped source
   offset and zeroing the invalid prefix (`Duplicate` at offset 0 is safe), so
   every vector op stays at offset 0
2. keep per-band ranges but stage the contribution through a `Gather` that
   applies the shift, since Gather tolerates unaligned byte offsets

Option 1 is simpler and is the recommended next step.

## Plan

Per output-row tile of R rows:
1. one contiguous `DataCopyPad` of banded columns covering `[r0-k, r0+R+k)`
   (blockCount = R, blockLen = (k+1)*8 bytes — 18–57 KB for k = 8/16/32,
   comfortably past the 16 KB threshold)
2. keep the x window resident in UB for the tile
3. for each band d, a **UB-local** `Gather` at the right row offset — no DMA
4. accumulate in UB, write the tile back once

Expected: A read once per tile instead of once per band; DMA count per tile
drops from 2*(k+1) to 2; no atomics, no `SyncAll`, no accumulator workspace.

The scheme is already validated offline (`/tmp/verify_fused.py`, 324/324 configs
including the `trans='C'` conjugate case). What failed was hoisting: the block
DMA must live **outside** the band loop, covering the union of column ranges.

## Environment note

`cann_env/extract.sh` selectively extracts the local CANN run packages
(`./extract.sh list`, `./extract.sh --perf`, or by substring). Already extracted:
`asc-devkit` (tikcpp headers), `asc-tools`, `opbase`, `A3-ops-math`
(266 production Ascend C kernels — the most useful reference found so far).
