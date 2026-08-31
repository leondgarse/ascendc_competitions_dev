# Where the time actually goes — and what the target implies

Measured on the real acceptance device (`npu-smi` reports **910B3**), CANN 9.1.0.

## 1. The physics floor is ~1000x below what we measure

Ascend 910B HBM peak bandwidth is **800 GB/s**
([Parallel Scan on Ascend, arXiv 2505.15112](https://arxiv.org/html/2505.15112v1)).
Traffic for the three acceptance cases, at complex64 = 8 bytes:

| case | A bytes | x read+write | total | pure-BW time |
|---|---|---|---|---|
| 512/8 | 36.9 KB | 8.2 KB | 44 KB | **0.056 us** |
| 1024/16 | 139.3 KB | 16.4 KB | 152 KB | **0.195 us** |
| 2048/32 | 540.7 KB | 32.8 KB | 560 KB | **0.717 us** |

We measure 61 / 69 / 100 us. **Data movement is not the constraint** — the
problem is three orders of magnitude away from being bandwidth-bound.

## 2. The floor is per-dispatch, not per-byte

`ctbmv` at k=0 (single kernel, no workspace, no atomics), sweeping n:

```
n=1      23.87 us       n=512    23.45 us
n=8      23.22 us       n=4096   49.95 us
n=64     22.82 us       n=32768 291.64 us
```

**Flat at ~23 us until n=4096**, then slope 8.6 us per 1024 elements. So:
- fixed cost per kernel dispatch: **~23 us**
- marginal cost: ~8.6 us / 1024 complex

For comparison, on the same device and stream:
- `aclrtSynchronizeStream` on an idle stream: **0.33 us**
- 4 KB `aclrtMemcpyAsync` D2D, back-to-back: **2.87 us**

So a raw async DMA costs 2.9 us while our simplest kernel costs 23 us. The ~20 us
delta is **kernel launch + TPipe/UB setup + tiling-data load**, not data movement.

## 3. The decisive comparison: other merged ops-blas complex kernels

All measured identically (20 warmup, 100 timed, one sync), n=512 and n=4096.
Every one is **flat in n** at these sizes, i.e. entirely overhead-dominated:

| op | n=512 | n=4096 |
|---|---|---|
| **our ctbmv, k=0 path** | **23.5** | 50.0 |
| `ccopy` | 28.5 | 28.7 |
| `cswap` | 52.0 | 52.2 |
| `cdotu` | 58.1 | 58.2 |
| **our ctbmv, 512/8 (2 kernels)** | **61.0** | — |
| `caxpy` | 122.6 | 129.6 |
| `cscal` | 157.0 | 157.1 |

`cscal` computes `x = alpha*x` — the simplest possible complex BLAS-1 op — and
takes **157 us for a single element**. `ccopy`, a pure memcpy, takes 28.5 us.

### What this means

1. **~25-30 us is the normal floor for a merged ops-blas complex kernel on this
   chip.** It is not something our implementation does badly.
2. **Our single-kernel path (23.5 us) is the fastest complex op measured**,
   including ones that do far less work.
3. **The acceptance targets (10 / 16.51 / 19.77 us) are below what any shipped
   complex op in this repo achieves**, including `ccopy`.

## 4. Consequence for the perf gate

The A100 numbers are real (I measured them: 13.87 / 16.18 / 19.66 us, matching
the published targets within 2%). But at these problem sizes both platforms are
latency-bound, and the two platforms' fixed costs differ by roughly an order of
magnitude for this class of kernel.

Removing our second dispatch and the workspace round trip is still worth doing —
it is ~14 us of real, measured saving, and it would put us at roughly `ccopy`
level. But `ccopy` itself is 28.5 us against a 10 us target.

**No amount of kernel tuning reaches 10 us if a device-to-device memcpy of the
same data costs 28.5 us through the same interface.**

## 5. CORRECTION — the per-call alloc, and what the "14 us dispatch" really was

Two follow-up experiments revised the picture:

### 5a. Why `cscal`/`ccopy` are slow: a blocking H2D memcpy per call

Measured on 910B3:

| operation | cost |
|---|---|
| `aclrtMalloc` + `aclrtFree`, 64 B | 3.49 us |
| **blocking `aclrtMemcpy` H2D, 64 B** | **13.78 us** |
| malloc + memcpy + free, 64 B | 20.95 us |

`ccopy_host.cpp:119-132` and `cscal_host.cpp` do exactly malloc + blocking
memcpy + free **per call** to ship their tiling struct. That accounts for
`ccopy` = 28.5 us (~21 us alloc path + ~7 us kernel) almost exactly.

Our `ctbmv` already avoids this: the gather tables are uploaded once under
`std::call_once`. **That is why our 23.5 us k=0 path beats `ccopy`'s 28.5 us.**

### 5b. Merging the two kernels saved ~1 us, not ~14 us

Implemented and verified (1872/1872 on 910B3, commit `55bdd87`). msprof:

```
before:  compute 47.39 + copy 13.44 = 60.83 us   (2 dispatches)
after:   compute 60.00              = 60.00 us   (1 dispatch)
```

**The copy kernel's 13.4 us was its own work — the `SyncAll` barrier,
re-interleave and store — not launch overhead.** Merging relocated it.

This invalidates the earlier "~14 us per dispatch" estimate that appeared in
`ctbmv-perf-plan.md` and `ctbmv-attempts-log.md`. A bare dispatch is closer to
**<10 us** (from `ccopy` minus its alloc path), and the ~23 us floor is mostly
kernel-side setup (TPipe, InitBuffer, tiling load, GM binding), which merging
does not touch.

**Consequence:** the projected "~30-35 us after removing dispatch + workspace"
was too optimistic. The measured result of removing one dispatch is 60.1 us for
case 1, against a 10 us target.

## 5c. What would still be worth trying

- **single dispatch** (fuse compute+copy): ~14 us, measured
- **drop workspace memset + atomics + SyncAll**: ~14 us, from the cost-model intercept
- both together predict ~30-35 us for case 1 — competitive with `ccopy`, still
  3x the target

The remaining gap is the per-dispatch floor, which is a property of the runtime
and this kernel-invocation path, not of the algorithm.

## Sources

- [Parallel Scan on Ascend AI Accelerators (arXiv 2505.15112)](https://arxiv.org/html/2505.15112v1) — 910B: 20 cube / 40 vector cores, 800 GB/s HBM, split cube/vector architecture
- [Ascend 910B architecture overview (ResearchGate)](https://www.researchgate.net/figure/Architecture-of-Ascend-910B-Training-series-accelerators-Each-AI-core-contains-one-cube_fig1_391953195) — each AI Core = 1 AIC + 2 AIV
