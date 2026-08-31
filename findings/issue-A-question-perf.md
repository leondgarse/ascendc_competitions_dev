# [Question|问题咨询]: arch22 上 complex64 BLAS-2 算子如何达到 10us 级别的性能标杆？

## 问题描述

我们在做社区任务《8月社区任务-aclblasCtbmv算子开发（A2A3）》，
实现已功能完成（**1872/1872 用例通过**，golden 为 `cblas_ctbmv(CblasColMajor)`），
但性能距标杆有 3.5~6 倍差距。定位后发现瓶颈**不在 kernel 计算内部**，
因此想请教一下 arch22 上的正确做法。

**环境**：Atlas 800I A2 / Ascend910B3，CANN 9.1.0。
测试方法：预热 20~30 次，正式采样 100~200 次，每轮 `aclrtSynchronizeStream` 后计时取平均；
丢弃进程首次计时窗口（冷启动偏高约 2 倍）。**所有数据在两台独立 devenv 上分别复现，差异 <2%。**

### 1. 核心现象：空 kernel 的下发开销已超过标杆

我们把 kernel 函数体清空（只保留 `return`）后测量纯下发耗时：

| blockDim | 机器 1 | 机器 2 |
|---|---|---|
| 1 | 23.57 us | 23.77 us |
| 2 | **59.45 us** | **58.93 us** |
| 3 | 59.49 us | 59.30 us |
| 4 | 59.83 us | 59.03 us |
| 8 | 59.17 us | 58.78 us |
| 16 | 61.47 us | 61.97 us |
| 32 | 73.39 us | 77.77 us |

两个现象：

- **单 block 空 kernel 就要 ~23.6us**
- **blockDim 从 1 增加到 2，多付约 36us**；2~8 之间基本持平

任务书 §3.3 的标杆为：

| case | n | k | uplo/trans/diag | 标杆 |
|---|---|---|---|---|
| 1 | 512 | 8 | UPPER/N/NON_UNIT | **10 us** |
| 2 | 1024 | 16 | UPPER/N/NON_UNIT | 16.51 us |
| 3 | 2048 | 32 | LOWER/T/NON_UNIT | 19.77 us |

**case 1 的标杆 10us，低于我们测到的单 block 空 kernel 下发耗时 23.6us。**

对 case 1 做逐段拆解（总 60.1us）：约 23.6us 单 block 下发
+ 约 36us 多 block 下发 + **不足 1us 的实际计算**。
ZeroWorkspace、两次 `SyncAll()`、原子累加、条带累加、拷回，逐段测量均在噪声范围内。

### 2. 横向对比：同设备上其他调用路径

n=512、complex64、4KB 数据：

| 调用路径 | 耗时 |
|---|---|
| `aclrtMemcpyAsync` D2D（**不启动 kernel**） | 2.8 us |
| `asdBlasCcopy`（SIP，plan 模式） | 16.78 us |
| **`aclblasCtbmv` k=0（我们的实现）** | **23.0 us** |
| `aclblasCcopy`（ops-blas 现网） | 28.5 us |
| `aclblasCswap` | 52.0 us |
| `aclblasCdotu` | 58.1 us |
| `aclblasCaxpy` | 122.6 us |
| `aclblasCscal` | 157.0 us |

我们的实现已是实测中最快的 ops-blas 复数算子，但仍与标杆差 6 倍。

### 想请教的问题

1. **arch22 上单次 kernel 下发约 23.6us 的固定开销，是否有办法降低？**
   是否存在我们没用上的下发路径（persistent kernel、图下沉/graph capture、
   算子融合下发等），能让 BLAS-2 小算子绕开这个开销？

2. **blockDim 1→2 多付约 36us，是预期行为吗？**
   如果是，那么在 n=512 这类规模上多核切分反而是负优化
   （标杆本身只有 10us，起第二个核就已超标）。
   arch22 上小规模 BLAS-2 算子的推荐 blockDim 策略是什么？

3. **Ctbmv 的标杆数据是在 arch22 实测得到的，还是参考了 arch35 的结果？**
   我们注意到已合入的 `aclblasCtrmv`（PR #365 / #369）报告 n=512 为 9.47us，
   与 case 1 的 10us 接近；但那是 **arch35（Ascend 950PR）** 实现，
   依赖 SIMT 模型与 `asc_shfl_down` warp shuffle，这些在 arch22（dav-2201）上不可用。

4. **标杆的口径**：任务书给出的 CSV 参考值，达标条件是 `实测 ≤ 参考值/0.8`
   还是 `实测 ≤ 参考值×0.8`？两种理解相差 1.56 倍，
   我们看到其他参与者的设计文档采用了与我们不同的解读，希望确认官方口径。

### 补充：两个可能相关的观察

**(a) 现网算子每次调用都做一次阻塞 H2D。**
实测单次开销：`aclrtMalloc`+`aclrtFree`(64B) = 3.49us；
**阻塞 `aclrtMemcpy` H2D(64B) = 13.78us**；三者合计 20.95us。

`blas/copy/arch22/ccopy_host.cpp:119-132`、`blas/scal/arch22/cscal_host.cpp:83-105`
在**每次调用**里做 malloc + 阻塞 memcpy + free 来下发 tiling
（`cscal` 还每次构造并上传一张 mask 表，这应该是它 157us 的主因）。

我们改用 `std::call_once` 缓存常量表 + tiling 经 `<<<>>>` 按值传参，
这是我们 23.0us 优于现网 `ccopy` 28.5us 的原因。
**请问 kernel 直调模型下，是否有官方推荐的方式把 tiling / 常量表上传提前到一次性阶段？
现网的 per-call malloc+memcpy 是有意为之的约束，还是可优化点？**

**(b) arch22 没有算子使用 MIX 模式。**
CANN 9.1.0 的 `tikcpp/ascendc_kernel_cmake/legacy_modules/util/host_stub_util.py` 中定义了
`KERNEL_TYPE_MIX_AIC_1_0/1_1/1_2`，且 `libasdsip_core.so` 里 SIP 的**所有**复数算子
都编译为 mix 形态（`ctrmv_0_mix_aic` / `ctrmv_0_mix_aiv`，连 `cscal`、`cswap`
这类纯 element-wise 算子也是）。

但 ops-blas/arch22 中 43 个算子为 `AIV_ONLY`、3 个为 `AIC_ONLY`，**无一使用 MIX**；
即使 `chemm`（AIV 预处理 + AIC GEMM）也拆成两个独立 kernel。
**请问 arch22 的直调算子是否可以/应该使用 MIX 模式？如果不建议，是直调模型的限制还是收益不明显？**

---

如需要，我们可以提供完整的复现代码（空 kernel 扫描、各算子对比测量、200 条性能用例全量结果）。
