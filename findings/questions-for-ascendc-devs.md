# Ascend C 开发问题

以下所有数据均在 **Atlas 800I A2 / Ascend910B3、CANN 9.1.0** 实测，
测试方法：预热 20\~30 次，正式采样 100\~200 次，每轮结束后 `aclrtSynchronizeStream`
取平均；丢弃进程首次计时窗口（冷启动会偏高约 2 倍），重复运行结果稳定在 ±0.5us 内。

---

## 问题一（最关键）：arch22 上 10us 的性能标杆如何达成？

任务书《8月社区任务-aclblasCtbmv算子开发（A2A3）》§3.3 给出的标杆为：

| case | n | k | uplo/trans/diag | 标杆耗时 |
|---|---|---|---|---|
| 1 | 512 | 8 | UPPER/N/NON_UNIT | **10 us** |
| 2 | 1024 | 16 | UPPER/N/NON_UNIT | 16.51 us |
| 3 | 2048 | 32 | LOWER/T/NON_UNIT | 19.77 us |

我们实测了同一台设备上各种调用路径的**固定开销**（n=512、complex64、4KB 数据）：

| 调用路径 | 耗时 | 说明 |
|---|---|---|
| `aclrtMemcpyAsync` D2D（**不启动 kernel**） | **2.8 us** | runtime 下限 |
| `asdBlasCcopy`（SIP，plan 模式） | **17.2 us** | 平台上最优的调用模型 |
| `aclblasCtbmv` k=0（我们的实现） | **23.0 us** | n=1/2/4/8/512 均为 ~23us，与规模无关 |
| `aclblasCcopy`（ops-blas 现网） | **28.5 us** | |
| `aclblasCscal`（ops-blas 现网） | **157.0 us** | |

由此推算：**一次 kernel 下发的固定成本约 14.4us**（17.2 − 2.8），
这已经是平台上最优调用模型（SIP plan 模式）的开销。

**问题：case 1 的标杆是 10us，低于我们测到的 14.4us kernel 下发开销。
在 arch22 上，一个 complex64 的 BLAS-2 算子应通过什么路径达到 10us？**

补充：我们注意到 ops-blas 已合入的 `aclblasCtrmv`（PR #365 / #369）报告
n=512 为 **9.47us**，但那是 **arch35（Ascend 950PR）** 实现，依赖 SIMT 模型与
`asc_shfl_down` warp shuffle，这些在 arch22（dav-2201）上不可用。

**请问 Ctbmv 的标杆数据是在 arch22 实测的，还是参考了 arch35 的结果？**

---

## 问题二：ops-blas 现网算子每次调用都做一次阻塞 H2D，是否可以避免？

我们实测的单次固定开销：

| 操作 | 耗时 |
|---|---|
| `aclrtMalloc` + `aclrtFree`（64B） | 3.49 us |
| **阻塞 `aclrtMemcpy` H2D（64B）** | **13.78 us** |
| malloc + memcpy + free（64B） | 20.95 us |

`blas/copy/arch22/ccopy_host.cpp:119-132`、`blas/scal/arch22/cscal_host.cpp:83-105`
等算子在**每次调用**里做 malloc + 阻塞 memcpy + free 来下发 tiling
（`cscal` 还额外每次构造并上传一张 mask 表，这解释了它的 157us）。

SIP 的 `asdBlasMakeXxxPlan` / `asdBlasSetWorkspace` 把这部分提前到 plan 阶段，
因此同样是 complex64 copy，SIP 17.2us、ops-blas 28.5us，**相差约 11.7us**，
与单次阻塞 H2D 的 13.78us 基本吻合。

**问题：**
1. ops-blas 的 kernel 直调模型下，是否有官方推荐的方式把 tiling / 常量表的
   上传提前到"一次性"阶段？（我们目前用 `std::call_once` 缓存常量表，并把
   tiling 通过 `<<<>>>` 按值传参，这是我们 23.0us 优于现网 `ccopy` 28.5us 的原因）
2. 现网这些算子的 per-call malloc+memcpy 是有意为之的约束，还是可以优化的点？

---

## 问题三：arch22 上是否应该使用 MIX（AIC+AIV）模式？

我们在 CANN 9.1.0 的
`tikcpp/ascendc_kernel_cmake/legacy_modules/util/host_stub_util.py` 中看到：

```
KERNEL_TYPE_AIV_ONLY    = 5
KERNEL_TYPE_AIC_ONLY    = 6
KERNEL_TYPE_MIX_AIV_1_0 = 7
KERNEL_TYPE_MIX_AIC_1_0 = 8
KERNEL_TYPE_MIX_AIC_1_1 = 9
KERNEL_TYPE_MIX_AIC_1_2 = 10
```

而 `libasdsip_core.so` 中 SIP 的**所有**复数算子都编译为 mix 形态
（`ctrmv_0_mix_aic` / `ctrmv_0_mix_aiv`，包括 `cscal`、`cswap` 这类纯 element-wise 算子）。

但 ops-blas/arch22 中：43 个算子是 `AIV_ONLY`、3 个是 `AIC_ONLY`，**没有一个用 MIX**。
即使是 `chemm`（AIV 预处理 + AIC GEMM）也拆成了两个独立 kernel。
另外，已合入的 `cdgmm` PR 把 SIP 的 `ColwiseMul` / `ComplexMatDot` 迁移进 ops-blas 后，
也变成了 `AIV_ONLY` + per-call malloc。

**问题：arch22 的 kernel 直调算子是否可以/应该使用 MIX 模式？
如果不建议，原因是什么？（是直调模型的限制，还是收益不明显？）**

---

## 问题四（缺陷报告）：已合入的 `aclblasStbmv`（arch22）带状存储索引有误

**这是一个与我们任务无关的独立问题，但可复现。**

`blas/tbmv/arch22/stbmv_kernel.cpp:113-114`：

```cpp
uint64_t r = static_cast<uint64_t>(rowOffset) * lda + colOffset;
DataCopy(LocalA, aGM[r], dataCount);          // 连续读，stride=1
```

这按 `A[band * lda + col]` 索引，即**行主序**。但 BLAS 带状存储
（Netlib `stbmv`、cuBLAS `cublasStbmv`，以及仓内自己的 golden
`test/tbmv/stbmv/stbmv_golden.h:38` 调用的 `cblas_stbmv(CblasColMajor, ...)`）
是**列主序** `A[row + col * lda]`，一条对角带的 stride 应为 `lda`，不是 1。

与 `cblas_stbmv(CblasColMajor,...)` 对比（4×4，UPPER，k=1，NON_UNIT，x 全 1）：

```
lda=2 (k+1=2)  cblas=[12 14 16 4]  stbmv=[12 13 14 3]  maxdiff=2.0   不一致
lda=4 (k+1=2)  cblas=[12 14 16 4]  stbmv=[12  2  0 0]  maxdiff=16.0  不一致
```

现网 UT 40/40 通过，是因为
`test/tbmv/stbmv/arch22/stbmv_test.cpp:125,132` 的数据生成用
`a[bandIdx * lda + col]`，与 kernel 的行主序约定一致，且用手写参考值比对，
**没有走同目录下的 cblas golden**。

复现代码见附件 `stbmv_bug.cpp`。

---

## 附：我们当前的状态

`aclblasCtbmv`（complex64，arch22）功能已完成：

- **1872/1872 用例通过**（12 种 uplo×trans×diag 组合 × 13 组 shape ×
  incx ∈ {1,2,-1,-3} × 3 组 lda padding），golden 由 `cblas_ctbmv(CblasColMajor)` 生成
- 性能：60.1 / 69.2 / 100.0 us，对应标杆 10 / 16.51 / 19.77 us
- 200 条性能用例全量实测：**0/200 达标**，最好的一条为标杆的 1.90 倍，中位数 6.00 倍
- 我们的 k=0 单 kernel 路径（23.0us）是实测中**最快的 ops-blas 复数算子**
  （`ccopy` 28.5、`cswap` 52.0、`cdotu` 58.1、`caxpy` 122.6、`cscal` 157.0）

也就是说，我们的实现并不比现网算子差，但离标杆仍有 3.5~6 倍差距，
且固定开销（23us）本身已超过 case 1 和 case 2 的标杆。
在提交验收前，希望先确认标杆的可达性。
