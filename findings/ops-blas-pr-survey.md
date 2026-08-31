# ops-blas PRs — how other complex-dtype work reports performance

Surveyed 8 pages of `ops-blas` PRs (all states) for complex-dtype submissions.

## The decisive pattern: fast complex BLAS-2 numbers are all arch35, not arch22

| PR | operator | target | reported perf |
|---|---|---|---|
| **#365** | `aclblasCtrmv` | **arch35 / Ascend 950PR** | n=512 **9.47 us**, n=1024 18.76, n=2048 48.21, n=4096 92.34 |
| **#369** | `aclblasCtrmv` (社区任务) | **arch35 / Ascend 950PR** | "全部满足 实测 ≤ 标杆"; 3.5-4.2x over a per-row baseline of 208/377/778 us |
| #315 | `aclblasCaxpy` | arch22 | benchmarks added, no absolute us in the description |
| #334 | `aclblasCaxpy` (follow-up) | arch22 / 910B3 | "性能证据" referenced, no absolute us in the description |
| #347 | `aclblasCherk` | arch22 + arch35 | "4 个 A100 性能基准点实测 **1.27x~2.38x** 达标" |

**`Ctrmv` — the closest analogue to our `Ctbmv` — reaches 9.47 us at n=512, but
only on arch35 (Ascend 950PR).** Nobody has published a comparable arch22
number for a complex BLAS-2 op.

This matters because our task is **arch22 (A2/A3, 910B3)** and our measured
floor on that chip is ~23.5 us for *any* call. PR #365's 9.47 us on arch35 is
below our arch22 floor, which is consistent with arch35 being a different
generation (dav-3510) with a different dispatch cost — not with our kernel being
4x worse than theirs.

### How the arch35 Ctrmv PRs get their speed

- #365: **dual-path design** — an AIV fast path for the common case
  (64-row band split, dual-core band parallelism) plus a SIMT general path
- #369: row-blocking across **all** AIV cores, 4 threads per row doing column
  blocking, **warp shuffle (`asc_shfl_down`)** for in-group reduction

Both rely on arch35-only features. `asc_shfl_down` and the SIMT programming
model are not available on arch22 (dav-2201).

## How complex PRs report performance

Three distinct conventions, all accepted by reviewers:

1. **Absolute us against 标杆** (#369): a table of 标杆(us) vs 实测(us) with the
   claim "全部满足 实测 ≤ 标杆". This is the format our task expects.
2. **Ratio against an A100 baseline** (#347, cherk): "4 个 A100 性能基准点实测
   1.27x~2.38x 达标", with `ratio = A100耗时/NPU耗时` and a note that the
   harness uses "预热+计时窗口（含 300ms cooldown 规避 Cube 降频）".
3. **Relative speedup vs a naive baseline** (#369): "相比每行单线程基线
   (208/377/778 us) 加速 3.5~4.2x".

Useful detail from #347: they insert a **300 ms cooldown between timing windows
to avoid Cube frequency throttling**. Our harness does not do this — worth
adding for the self-test report, though it would not change conclusions at our
current 4-6x gap.

## Corroborating finding: the migrated SIP ops kept AIV_ONLY

PR (approved+lgtm) "feat(cdgmm): 迁移 ColwiseMul 为行主序 Cdgmm；新增
extensions/ 目录迁移 ComplexMatDot" ported two ops **out of SIP into ops-blas**.

Checking the merged result:
- `extensions/complexmatdot/arch22/complexmatdot_kernel.cpp:238` →
  `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`
- `blas/dgmm/arch22/cdgmm_host.cpp` → three `aclrtMalloc` + two `aclrtMemcpy`
  per call

So when SIP's `ColwiseMul`/`ComplexMatDot` (both compiled `_mix_aic`+`_mix_aiv`
in `libasdsip_core.so`) were migrated into ops-blas, they became **AIV_ONLY with
per-call host allocation**. The MIX design did not survive the port.

That weakens the "just use MIX mode" idea: someone already moved SIP kernels
into this repo and did not carry MIX across. Whether that was a deliberate
constraint of the ops-blas kernel-direct-call model or simply the porter's
choice is unknown, and is a good question for the Ascend C developers.

## Net conclusions

1. **No arch22 complex BLAS-2 op has published a sub-20 us number.** The fast
   numbers (9.47 us) are arch35, using arch35-only features.
2. Our 60 us on arch22 should be compared against arch22 peers, and by that
   measure (`ccopy` 28.5, `cswap` 52.0, `cdotu` 58.1, `caxpy` 122.6,
   `cscal` 157.0) it is mid-pack, with our k=0 path (23.5 us) the fastest.
3. Reviewers accept ratio-based and relative-speedup reporting, not only
   absolute-vs-标杆. That is relevant if we end up reporting a miss with context.
