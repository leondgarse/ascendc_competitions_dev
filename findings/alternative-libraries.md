# Alternative implementations checked — asdBlas (SIP) and aclnnRoll

Two leads investigated for a faster path than the ~23 us per-call floor.

## 1. asdBlas (Ascend SIP / NNAL) — plan-based complex BLAS

`cann_env/ASCEND/sip/include/blas_api.h`, installed on the devenv at
`/opt/home/developer/Ascend/nnal/asdsip/9.0.0/`. **80 entry points**, a
completely separate library from `ops-blas`, with a genuinely different
invocation model:

```cpp
asdBlasCreate(handle);
asdBlasSetStream(handle, stream);
asdBlasMakeCtrmvPlan(handle, uplo, n);      // setup hoisted OUT of the call
asdBlasGetWorkspaceSize(handle, &wsSize);
asdBlasSetWorkspace(handle, ws);
asdBlasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);   // hot path
```

**Why this looked promising:** the `MakeXxxPlan` step is exactly the fix for our
problem — tiling/setup is done once at plan time instead of per call. That is
the structural difference we measured as ~23 us of per-dispatch cost.

Complex ops present: `Cgemm`, `Cgemv`, `Ctrmv`, `Ccopy`, `Cscal`, `Csscal`,
`Cdotu`, `Cdotc`, `Cgerc`, `Caxpy`, `Cswap`, `Csrot`, `CgemvBatched`,
`ComplexMatDot`, `ColwiseMul`.

### SOLVED — SIP works; the call order was wrong

The official docs
([SIP_API_0006](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/SiPAPI/SIP_API_0006.html))
give the required sequence:

> 创建handle → 调用plan接口 → 获取workspace大小 → 设置workspace → 执行计算 → 销毁plan

**`asdBlasSetStream` must come AFTER the plan and workspace are set**, not right
after `Create`. My original code called `SetStream` immediately after `Create`,
which is what produced `500000` for every op. With the documented order
everything returns rc=0:

```
Create rc=0 ; MakeCopyPlan rc=0 ; GetWs rc=0 size=1024 ; SetWs rc=0
SetStream rc=0 ; Scopy rc=0 ; asdBlasSynchronize rc=0 ; Ccopy rc=0
```

There is also an `asdBlasSynchronize(handle)` for waiting on the op.

### MEASURED: SIP is 1.7x faster than ops-blas for the same op

910B3, CANN 9.1.0, identical protocol (20 warmup, 100 timed, one sync):

| op | n=512 | n=4096 |
|---|---|---|
| **`asdBlasCcopy` (SIP, plan-based)** | **16.78 us** | **16.83 us** |
| `aclblasCcopy` (ops-blas, per-call setup) | 28.51 us | 28.78 us |

Same device, same data, same measurement. **The plan-based model is ~11.7 us
cheaper per call** — consistent with the ~13.8 us we measured for a single
blocking 64 B H2D memcpy, which is exactly what ops-blas does per call to ship
its tiling struct and SIP does once at plan time.

This is direct evidence that **the ~23 us per-call floor in ops-blas is an
artifact of its invocation model, not a hardware limit.** SIP reaches 16.8 us
for a complex copy on the same silicon.

### (superseded) earlier failure notes

The devenv actually has **asdsip 9.1.0** at `~/Ascend/nnal/asdsip/9.1.0`
(I had been looking at a stale 9.0.0 copy under `/opt/home/...`). Retested with
the matched version, correct `ASDSIP_HOME_PATH`, and its own `set_env.sh`:

```
MakeCopyPlan   rc=0
GetWorkspaceSize rc=0  size=1024
SetWorkspace   rc=0
Scopy  rc=500000     <- REAL float, also fails
Ccopy  rc=500000
```

So it is **not** version skew, and **not** complex-specific — plain `Scopy`
fails identically. Plan/workspace calls all succeed; only the op invocation
fails, with a generic `ACL_ERROR_INTERNAL_ERROR`.

`strings libasdsip_core.so` confirms `Ascend910B3` is among the supported SoCs,
and the kernel binaries are embedded in the `.so` (no `.o` files on disk).
No samples or tests ship with the package, and MKI/ASDSIP debug logging produced
no additional output.

**Conclusion: asdBlas needs an invocation detail that is not discoverable from
the headers.** This is a good, concrete question for the Ascend C developers:
what else does `asdBlasScopy` require beyond Create / SetStream / MakeCopyPlan /
GetWorkspaceSize / SetWorkspace, with `aclTensor`s built by `aclCreateTensor`?

### (superseded) version-skew hypothesis

The devenv had asdsip **9.0.0** while the toolkit was CANN 9.1.0. Our local
`cann_env/ASCEND/sip` is **9.1.0** (`br_release_cann_9.1.0_20261223`), i.e. the
matched version. So the 500000 failure is most plausibly that skew, and my
earlier statement that SIP "is not runnable" was too strong — it was untested on
a matched install. Retesting with 9.1.0 on the box is worth doing.

### What SIP actually is

**SIP = Signal Processing.** Headers: `fft_api.h`, `filter_api.h`,
`interp_api.h`, `domain/rs_api.h` (resampling by sinc), plus `blas_api.h`.
BLAS is a *supporting* component of a DSP library, not a general BLAS
replacement. It ships inside the **NNAL** package, which we also have locally.

### Result: could not get it to execute

Handle creation, `MakeCopyPlan`, `GetWorkspaceSize` (1024 B) and `SetWorkspace`
all return **rc=0 (success)**, but `asdBlasCcopy` returns **500000**
(`ACL_ERROR_INTERNAL_ERROR`, a generic code) for both n=512 and n=4096.

Runtime confirms the device is `socVersion=Ascend910B3`, so it is not an
unsupported-SoC rejection at the ACL level.

Most likely cause: **version skew.** The installed asdsip is **9.0.0**
(`br_release_cann_9.0.0_20260928`) while the toolkit is **CANN 9.1.0**. Linking
also required pulling `libasdsip_host`, `libasdsip_core` and `libmki` explicitly,
which suggests the packaging expects its own matched environment.

### No banded support anyway

**`grep -i "tbmv|band"` over all SIP headers returns nothing.** There is
`Ctrmv` (triangular, dense) but no triangular *banded* variant. So even working,
asdBlas would not provide `Ctbmv` directly — it would only be evidence that the
plan model beats the per-call model, and a possible implementation target if the
task ever allowed delegating to NNAL (it does not: the task requires an Ascend C
kernel in `ops-blas/blas/tbmv/arch22/`).

### Still worth knowing

The existence of a plan-based API for the same operator class is strong evidence
that **Huawei themselves treat per-call setup as the thing to eliminate** for
small BLAS calls. If the perf targets were derived from a plan-based
implementation, that would explain the gap entirely. Worth asking about.

## 2. aclnnRoll — no complex64 specialisation to learn from

Source: `cann-A3-ops-math/.../ops_math/ascendc/roll/` (roll_apt.cpp + 6 arch35
headers: `roll_gather_simd.h`, `roll_hsplit.h`, `roll_simd.h`, `roll_simt.h`,
`roll_unaligned_simd.h`, `roll_struct.h`).

`grep -i "complex64|DT_COMPLEX"` over all of them: **no matches.** Roll is
written generically over `DTYPE_X` and dispatches on *shape/alignment*, not
dtype — it is pure data movement, so a complex64 element is just an 8-byte
value. There is no complex arithmetic to learn from.

What *is* transferable is its dispatch structure: Roll selects among
`SIMD_SMALL_TAIL_SHIFTW`, `SIMD_BEFOR_H`, `SIMD_AFTER_H_ALIGN`,
`SIMD_AFTER_H_UNALIGN`, `SIMD_SPLIT_W`, `SIMD_ONE_DIM`, `SIMT`, ... by tiling
key — the same multi-strategy pattern as `as_strided`. That reinforces the
earlier finding that CANN built-ins branch heavily on alignment and access
shape rather than using one general path.

## 3. What SIP does differently — two ideas worth borrowing

Symbol inspection of `libasdsip_core.so`:

```
CtrmvC64Kernel, CtrmvOperation, GetKernelCtrmvC64Kernel
ctrmv_0_mix_aic$local   ctrmv_0_mix_aiv$local
```

### Idea A: every SIP kernel is built in MIX (AIC+AIV) mode

Ops compiled `_mix_aic` **and** `_mix_aiv` include: `ctrmv`, `cgemv_do_trans`,
`cgemv_no_trans`, `cgemm`, `cgerc`, `caxpy`, `cscal`, `cswap`, `csrot`, `cdot`,
`colwise_mul`, `complex_mat_dot`, `dft_c2c`, `convolve`.

Note `cscal` and `cswap` are in that list — pure elementwise ops that need no
matrix unit. So SIP compiles **everything** as a fused AIC+AIV kernel.

By contrast `ops-blas/arch22` is overwhelmingly single-unit:

```
43 x KERNEL_TYPE_AIV_ONLY
 3 x KERNEL_TYPE_AIC_ONLY
```
and our `ctbmv` is `AIV_ONLY` throughout.

The `agent-skills` pipeline reference (§5.3) documents MIX mode as
"异步 Iterate（MIX 模式，AIC+AIV）" — the AIC and AIV halves run concurrently and
hand off through L2/GM. For a memory-bound op this can overlap the vector work
with the next tile's loads in a way a single-unit kernel cannot.

**MIX modes are available in CANN 9.1.0.** From
`tikcpp/ascendc_kernel_cmake/legacy_modules/util/host_stub_util.py`:

```
KERNEL_TYPE_AIV_ONLY   = 5
KERNEL_TYPE_AIC_ONLY   = 6
KERNEL_TYPE_MIX_AIV_1_0 = 7
KERNEL_TYPE_MIX_AIC_1_0 = 8
KERNEL_TYPE_MIX_AIC_1_1 = 9
KERNEL_TYPE_MIX_AIC_1_2 = 10
```

The `1_0 / 1_1 / 1_2` suffixes match SIP's `ctrmv_0_mix_aic` / `_mix_aiv` symbol
pairs, i.e. one AIC block paired with N AIV blocks.

**However, no arch22 op in ops-blas uses MIX** — not even `chemm`, which has an
AIV preprocess feeding an AIC GEMM. It splits those into *separate* kernels
(`chemm_preprocess_kernel` AIV_ONLY, `chemm_gemm_cube_kernel` AIC_ONLY) rather
than one MIX kernel. Combined with the SIP→ops-blas migration dropping MIX
(see §4), that is two independent signals that MIX is not the idiom for
kernel-direct-call ops in this repo. Worth asking why before investing in it.

### Idea B: plan-based setup (already discussed)

`MakeXxxPlan` + `GetWorkspaceSize` + `SetWorkspace` hoists all tiling and
workspace out of the hot path. Measured relevance: our per-call floor is ~23 us,
and a blocking 64 B H2D memcpy alone is 13.78 us.

## 4. Complex ops in ops-blas/arch22 — the full list

13 complex operators exist to learn from:

| op | kernel lines | measured (n=512) |
|---|---|---|
| `caxpy` | 226 | 122.6 us |
| `cdgmm` | 374 | not measured |
| `cdot` | 323 | 58.1 us (`cdotu`) |
| `cgemv` (2 kernels) | 310 + 333 | not measured |
| `cgemv_batched` | 641 | not measured |
| `cgerc` | 241 | not measured |
| `chemm` | **1106** | not measured |
| `csrot` | 199 | not measured |
| `cscal` | 389 | **157.0 us** |
| `cswap` | 164 | 52.0 us |
| **`ctbmv` (ours)** | 818 | 60.1 us |
| `ctrmv` | 663 | not measured |
| `ccopy` | — | **28.5 us** |

### Why `cscal` is 157 us — and why we are already better

`cscal_host.cpp:83-105` builds a **mask table on the host and uploads it on
every call**:

```cpp
uint32_t* maskHost = new uint32_t[MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64];
CreateMaskData(maskHost);
aclrtMalloc(&tilingDevice, ...);      // alloc 1
aclrtMalloc(&maskDevice, ...);        // alloc 2
aclrtMemcpy(tilingDevice, ...);       // blocking H2D 1
aclrtMemcpy(maskDevice, ...);         // blocking H2D 2  (the big one)
```

Two allocs + two blocking H2D per call. At 13.78 us per blocking 64 B H2D, that
is the whole 157 us.

This is exactly the antipattern we removed from `ctbmv` with `std::call_once`.
**Our 23.5 us single-kernel path is the fastest complex op measured in this
repo** precisely because we upload the gather tables once per process.

`ccopy` (28.5 us) is fastest of the stock ops because it uploads only a small
tiling struct — one alloc + one memcpy, no mask table.

## Conclusion

- **asdBlas**: cannot be *called* from our submission (the task requires an
  Ascend C kernel in `ops-blas/blas/tbmv/arch22/`), and it has no banded op
  anyway. But its **design** is borrowable, and two ideas are worth testing:
  **MIX AIC+AIV kernels** and **plan-based setup**.
- **aclnnRoll**: no complex64 path; nothing to borrow beyond multi-strategy
  dispatch on alignment/shape.
- **ops-blas complex ops**: 13 exist. The spread (28.5 us to 157 us) is almost
  entirely explained by how much host-to-device setup each does per call. We
  already sit at the good end of that spectrum.

### Untested idea with the best odds

Compile the ctbmv kernel in **MIX (AIC+AIV)** mode rather than `AIV_ONLY`.
Every SIP complex op does this, including ones with no matrix work. It is a
build/annotation change plus restructuring into the AIC/AIV halves, not an
algorithm change, and it is the one structural difference between SIP's kernels
and ops-blas's that we have not tried.
