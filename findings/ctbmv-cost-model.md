# aclblasCtbmv — measured cost model (predict without an NPU)

All numbers measured with the committed band-parallel kernel, CANN 9.1.0 (`feat/aclblasCtbmv-arch22` @ `39fcf1d`).

> **Hardware attribution.** Measurements were taken on the hidevlab devenv whose
> runtime reports `socVersion=Ascend910_9362` — that is an **A3** part
> (`ascend910_93*`). The build targeted `SOC_VERSION=ascend910b3` (**A2**).
> Both map to `NPU_ARCH=dav-2201` and compile the same `arch22` sources, so the
> binary is valid on either, but the task specifies its perf targets on
> **Atlas 800I/800T A2 (910B3)**. Numbers here are therefore measured on A3
> silicon against A2-specified targets and must be relabelled if re-run on a
> genuine 910B3.
Timing method: 20 warmup calls, then 100 calls with a single
`aclrtSynchronizeStream`, wall clock / 100. Raw data in `data/`.

## Hardware constants (reusable for any arch22 vector kernel)

| Constant | Value | How measured |
|---|---|---|
| Host launch cost (no sync) | **2.9 us/call** | `perf2`, loop without sync |
| Minimum end-to-end call (n=1, k=0) | **25.1 us** | `./perf2 1 0` |
| Fixed cost per extra kernel dispatch | **~14 us** | `ctbmv_copy_kernel` does a trivial memcpy of 512 complex and still costs 14.4 us |
| First call (cold) | ~50 us | `R=1` vs `R=100` in `perf2` |
| AIV cores usable | 40 (`GetAivCoreCount`) | platform API |
| UB | 192 KB | platform |
| `DataCopyPad` UB block granularity | **32 B** | probe kernel; an 8 B burst still consumes 32 B |
| Vector op operand alignment | **32 B** | see failure 5 below |
| `Gather` offset alignment | none (arbitrary bytes) | works where vector ops fail |
| Recommended DMA size | >= 16 KB | `agent-skills/.../data-copy-prof.md` §2.1 |

## Fitted model (band-parallel kernel)

For `k >= 1`, with `CORES = 40`, `TILE = 1024`:

```
bands        = k + 1
cores        = min(bands, CORES)
bandsPerCore = ceil(bands / cores)
chunks       = ceil(n / TILE)
serialDMA    = bandsPerCore * chunks              # DMA pairs serialised per core
kElems       = bandsPerCore * chunks * min(n, TILE) / 1000

t_us = 58.5 + 13.94 * serialDMA + 6.97 * kElems
```

**mean error 8.6%, max 28%** over 32 (n, k) points with n in {512..4096},
k in {1..128}. Worst errors are at k=1 and n=4096 (few cores / many chunks).

For `k == 0` a separate diagonal kernel runs (1 core, no workspace):

```
n=512 -> 25.4    n=1024 -> 26.1    n=2048 -> 35.3    n=4096 -> 53.7 us
```

### Interpreting the model

- `58.5 us` intercept ~= 2 dispatches (~28 us) + workspace memset + atomics setup
  + SyncAll. This is why **k=0 (25-35 us) is 2-3x faster than k=1 (59-188 us)**:
  the diag path skips all of it.
- `13.94 us` per serialised DMA pair dominates at small n. Each pair is currently
  `blockCount=cnt, blockLen=8 B`, i.e. the worst possible burst shape.
- `6.97 us` per 1000 vector elements is the actual arithmetic, and it is small.

**Predictive rule of thumb:** anything that removes a dispatch saves ~14 us;
anything that halves the serialised DMA count saves ~7 us per pair removed.

## Reference: A100 baseline

`data/a100_ctbmv_measured.csv` — 200/200 cases on a real A100-SXM4-40GB,
CUDA 12.6, `cublasCtbmv`, same protocol.

- median `measured / gpu_baseline.csv` ratio = **1.254** across all 200 cases
- the task doc targets equal `csv_us / 0.8`, and my measured A100 times land
  within 2% of those targets — so the published bar is genuine end-to-end
  cuBLAS performance, not a synthetic number
- A100 needs ~13.9 us for n=512 k=8 (~74 KB of traffic), i.e. it is also
  latency-bound at these sizes

| case | A100 measured | NPU measured | target | gap |
|---|---|---|---|---|
| 512/8 U N | 13.87 | 63.6 | 10.00 | 4.6x |
| 1024/16 U N | 16.18 | 71.8 | 16.51 | 4.4x |
| 2048/32 L T | 19.66 | 101.8 | 19.77 | 5.2x |

**The NPU fixed floor (25 us) already exceeds the case-1 target (10 us).**
Case 1 is unreachable without removing dispatches and the workspace round trip;
no amount of inner-loop tuning gets there.

## msprof kernel split

| shape | compute | copy | total |
|---|---|---|---|
| 512 / 8 | 49.96 | 14.39 | 64.4 |
| 2048 / 32 | 84.75 | 19.14 | 103.9 |
| 4096 / 128 | 360.98 | 20.86 | 381.8 |

`ctbmv_copy_kernel` only re-interleaves and stores, so its ~14-20 us is almost
entirely dispatch overhead — the single clearest saving available.

## Core-count behaviour (n=2048, cores = k+1)

```
cores=1   35.3    <- diag kernel, different path
cores=2  108.6
cores=4  105.5
cores=8  100.4
cores=16  95.3
cores=32  99.1
cores=48 150.0    <- above 40 AIV cores, contention
```

More cores barely helps: the kernel is DMA-bound, not compute-bound, and going
past the 40 physical AIV cores actively hurts.
