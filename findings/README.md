# aclblasCtbmv findings

Working notes for the 8月社区任务 `aclblasCtbmv` (complex64 triangular banded
matrix-vector multiply, Atlas A2/A3, arch22).

## Status

**Correct but too slow — not submittable.** Measurement shows the shortfall is a
per-dispatch floor shared by every merged complex op in this repo, not a defect
in this kernel: `cscal` (x = alpha*x) costs 157 us and `ccopy` 28.5 us on the
same device, against targets of 10-20 us. See `ctbmv-overhead-analysis.md`.
 The task doc makes performance a
hard gate (§3.3.2 "应不高于", §7.1 "确认符合验收标准后再提交").

| | value |
|---|---|
| Correctness | **1872/1872 on real 910B3** (the acceptance device), CANN 9.1.0 |
| Performance | **61.0 / 69.2 / 100.0 us** on 910B3 vs targets 10 / 16.51 / 19.77; full 200-case sweep **0/200 pass**, median 6.0x over |
| Code | `work/ops-blas`, branch `feat/aclblasCtbmv-arch22` @ `39fcf1d` |

## Documents

| File | Contents |
|---|---|
| `ctbmv-overhead-analysis.md` | **Why the gap exists.** Physics floor, per-dispatch cost, and a comparison against merged ops-blas complex kernels. Read this before any further tuning. |
| `ctbmv-910b3-measurements.md` | **Measurements on the real acceptance device (910B3)**, incl. the full 200-case sweep. Start here. |
| `ctbmv-cost-model.md` | **Measured hardware constants + fitted cost model.** Predict runtime without an NPU (8.6% mean error). |
| `ctbmv-attempts-log.md` | Every attempt, failure and root cause. Read before touching the kernel. |
| `ctbmv-perf-plan.md` | Root-cause analysis and the planned fix. |
| `stbmv-arch22-banded-layout-bug.md` | Upstream bug in the repo's own `stbmv`, with reproducer. |

## Data and harnesses (`data/`)

| File | Purpose |
|---|---|
| `npu_ctbmv_scaling.csv` | 32 (n,k) NPU timings — the cost-model fit set |
| `a100_ctbmv_measured.csv` | 200/200 real A100 `cublasCtbmv` timings |
| `smoke.cpp` / `smoke2.cpp` | 84- and 1872-case correctness suites vs cblas golden |
| `perf.cpp` / `perf2.cpp` | benchmark harnesses (task protocol / fixed-vs-variable split) |
| `dbg.cpp` / `one.cpp` | small traceable cases — fastest way to localise a bug |
| `gperf.cu` / `gsweep.cu` | A100 cuBLAS baselines |
| `verify_*.py` | offline algorithm models (band mapping, fused scheme, band slicing) |

## Three things worth knowing before resuming

1. **32-byte alignment.** AscendC vector ops require it on every operand;
   `Gather` does not. Three of five failed rewrites were this one constraint.
2. **`DataCopyPad` pads each burst to 32 B in UB.** An 8-byte burst costs 32 B.
   Never express a strided read as many tiny bursts — bring in a contiguous
   superset and gather in UB (the `cgemv`/`as_strided` idiom).
3. **Offline models validate the algorithm, not the hardware.** Every Python
   model passed while the kernel failed. Model layout and primitive constraints.
