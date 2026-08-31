# ascendc_competitions_dev

Working notes, measurements and reusable tooling from CANN 社区算子任务
(Ascend C operator development tasks) at
<https://www.hiascend.com/activities/task-center>.

Everything here was measured on real hardware — an **Atlas 800I A2
(Ascend910B3)** devenv with CANN 9.1.0, and an **NVIDIA A100-SXM4-40GB** for the
cuBLAS baselines.

## Layout

| path | contents |
|---|---|
| `ctbmv/` | our `aclblasCtbmv` implementation (complex64, arch22): sources, design doc, and a patch against `cann/ops-blas@4307872` |
| `findings/` | measurements, cost models, failure analyses, and the questions raised upstream |
| `cann-comp-skills/` | a reusable Claude Code skill for running one of these tasks end to end |
| `tasks/` | the task statement and supplied test cases, kept for context |

## Status of `aclblasCtbmv`

- **Correct**: 1872/1872 cases pass on real 910B3, verified against
  `cblas_ctbmv(CblasColMajor, ...)` — 12 uplo×trans×diag combinations × 13
  shapes × `incx` ∈ {1, 2, −1, −3} × 3 `lda` paddings.
- **Too slow to submit**: 60.1 / 69.2 / 100.0 us against targets of
  10 / 16.51 / 19.77 us. A full 200-case sweep passes 0/200 (median 6.0× over).
- The shortfall is not specific to this kernel. On the same device a kernel
  launch through the best available path costs **~14.4 us**, which already
  exceeds the first target. See `findings/ctbmv-cost-decomposition.md`.

## Headline measurements

n=512, complex64, 910B3, CANN 9.1.0 (20–30 warmup, 100–200 timed, one sync,
first timing window discarded):

| call path | us |
|---|---|
| `aclrtMemcpyAsync` D2D — no kernel | 2.8 |
| `asdBlasCcopy` (SIP, plan-based) | 17.2 |
| **our `aclblasCtbmv`, k=0** | **23.0** |
| `aclblasCcopy` (ops-blas, stock) | 28.5 |
| `aclblasCscal` (ops-blas, stock) | 157.0 |

Our k=0 path is the fastest complex operator measured in `ops-blas` — stock ops
pay ~13.8 us per call for a blocking H2D of their tiling struct, which the
plan-based model hoists out of the hot path.

## Also here

- **A reproducible correctness bug in merged `aclblasStbmv` (arch22)**: it
  indexes banded storage row-major, while BLAS (and the repo's own cblas golden)
  is column-major. In-tree tests pass only because the data generator shares the
  same wrong convention. See `findings/stbmv-arch22-banded-layout-bug.md` and
  `findings/data/stbmv_bug.cpp`.
- **A cost model** that predicts kernel runtime without an NPU to within ~9%
  mean error (`findings/ctbmv-cost-model.md`).
- **Five failed optimisation attempts**, each with its root cause
  (`findings/ctbmv-attempts-log.md`) — three of them the same 32-byte alignment
  constraint on vector operands.

## Building `ctbmv`

The sources drop into a `cann/ops-blas` checkout:

```bash
git clone --depth 1 https://gitcode.com/cann/ops-blas.git
cd ops-blas && git apply /path/to/ctbmv/ctbmv-arch22.patch
bash install_deps.sh                     # needs libblas-dev for the cblas golden
bash build.sh --ops=tbmv --soc=ascend910b3
```

Correctness and benchmark harnesses are in `findings/data/` (`smoke.cpp`,
`smoke2.cpp`, `perf.cpp`, `floor.cpp`, …).

## Licence / attribution

`ctbmv/` sources follow the CANN Open Software License Agreement 2.0 headers of
the upstream repository they are intended for. `tasks/` contains material
supplied with the community task and is included unmodified for context.
