# 重要更正：PR #361（aclblasCtpmv, arch22）推翻了我们的核心结论

来源：<https://gitcode.com/cann/ops-blas/pull/361>
**状态：已开启（open），尚未合入。** 页面横幅明确写着
"This PR does not yet meet the following requirements: lgtm (requires ≥ 2 persons
per module)、approve (requires ≥ 1 person per module)"。
我最初把审批表格的**列标题**（"lgtm status" / "approve status"）误读成了已达成的状态，
在此更正 —— 这是一个**未经评审的 PR，其性能数据为贡献者自报**。

**另注：`aclblasCtpmv` 与我们的 `aclblasCtbmv` 不是同一个算子。**
- Ctpmv = 三角**打包**（packed），存储 `n(n+1)/2` 连续元素，无 k
- Ctbmv = 三角**带状**（banded），存储 `lda × n`，半带宽 k

同族、同 dtype、同架构，但存储布局与每元素工作量不同：
他们 n=512 的 packed 用例约 1 MB 数据，我们 n=512/k=8 的带状用例只有 44 KB。
**两者的绝对耗时不可直接比较。**
需求：<https://gitcode.com/cann/ops-blas/issues/377>

## 实测验证（910B3，两次运行稳定）

用 PR #361 描述的方法拆分我们自己的算子：

| case | dispatch | device | total |
|---|---|---|---|
| n=1 k=0 | **4.9 us** | **18.2 us** | 23.2 us |
| n=8 k=0 | 5.0 us | 18.0 us | 23.0 us |
| n=512 k=0 | 4.7 us | 18.5 us | 23.1 us |
| n=512 k=8 | 4.9 us | 55.5 us | 60.4 us |
| n=2048 k=32 | 5.1 us | 93.5 us | 98.5 us |

对比 PR #361 自报的 Ctpmv：n=1 dispatch 2.71us + device 2.25us = 5.08us。

**结论：**
- **dispatch 不是问题**：我们 4.9us vs 他们 2.71us，同一量级（1.8 倍），
  且与 n 无关。
- **device 才是问题**：我们 n=1 的 device 时间 **18.2us**（算一个复数元素！），
  他们 2.25us —— 差 8 倍。

### 进一步排查：不是 gather 表搬入的开销

假设是每次调用把 gather 表搬进 UB（最多 16KB）导致，
但实测 k=0 时耗时对 n 完全平坦：

```
n=1 23.5 | n=8 23.0 | n=32 23.0 | n=64 22.7 | n=128 22.6
n=256 22.7 | n=512 23.1 | n=1024 23.7 | n=2048 32.6
```

`tileCnt_` 随 n 增长（表搬入量也随之增长），但耗时不变 →
**表搬入不是主要成本**。18us 的 device 固定开销来自别处：
TPipe 构造、5 次 InitBuffer、多个 GlobalTensor 绑定，或 kernel 启动本身在
device 侧的固定代价。需要 msprof 进一步定位。

## 我们错在哪

我们此前的结论是：**arch22 上 kernel 下发的固定开销约 14.4~23us，
标杆（10us）物理不可达**。

PR #361 是一个**已合入的、arch22 的、complex64 BLAS-2 算子**，
它把每次调用拆成 host 下发与 device 执行两段实测：

| 测量项 | PR #361 实测 | 我们的实测 |
|---|---|---|
| n=1，host 下发 | **2.71 µs** | — |
| n=1，device 执行 | **2.25 µs** | — |
| n=1，合计 | **5.08 µs** | **23.0 µs**（我们的 ctbmv k=0） |
| `aclblasStpmv_legacy` n=1 合计 | 3.28 µs（下发 2.33） | — |

**他们的 n=1 下发是 2.71us，我们的整体固定开销是 23us —— 差 8.5 倍。**

所以那 23us **不是平台地板，是我们 kernel 自己的开销**
（TPipe 构造、多次 InitBuffer、GM 绑定、gather 表搬入 UB 等）。

我们据此推出的"14.4us 下发地板"也是错的：那是
`asdBlasCcopy`(17.2) − `aclrtMemcpyAsync`(2.8) 的差值，
但 `asdBlasCcopy` 自身也包含它的 kernel 执行时间，不是纯下发。
**我们把"别人的 kernel 总耗时"当成了"平台下发成本"。**

## 他们的性能结果（arch22, A100×0.8 口径）

| n | uplo/trans/diag | 本实现 | 标杆 | 余量 |
|---|---|---|---|---|
| 512 | UPPER/N/NON_UNIT | **16.33 µs** | 18.53 µs | 12% |
| 1024 | LOWER/N/NON_UNIT | **22.84 µs** | 39.34 µs | 42% |
| 2048 | UPPER/T/NON_UNIT | **52.14 µs** | 86.02 µs | 39% |
| 4096 | UPPER/N/NON_UNIT | **131.02 µs** | 181.10 µs | 38% |

**200 条基线用例 189 条达标；n ≥ 257 的 186 条全部达标。**

对比我们的 ctbmv：n=512 **60.1 µs**（标杆 10 µs）。
同为 arch22 复数 BLAS-2，他们 16.33 µs 达标，我们 60.1 µs。

## 他们怎么做到的（可直接借鉴）

1. **复数处理**：搬入后**一次**拆成实/虚两个平面 → 分别向量运算 →
   算完**一次**重交织。共轭只改两个交叉项的符号，折进 `trans` 分支，不额外花指令。
   （与我们的做法一致，这不是差距来源。）

2. **按数据流分两种并行方式**：
   - `trans=N`（**AXPY 形式**）：一列贡献到连续的输出行，核间写重叠 →
     每核发布部分结果到 workspace，之后各核归约自己负责的行区间
   - `trans=T/C`（**DOT 形式**）：一列归约成一个输出元素，核独占其输出 →
     只需一次屏障

3. **按"权重"而非列数分核**（关键）：
   一列的权重 = 列长 + 一次搬运的固定成本（**实测约等于 512 个元素**）。
   按列数均分会让拿到短列的核承担约十倍搬运次数，拖住所有人。
   核数 = 总权重 / 3800（常数由实测扫描确定），核数上限用
   `GetAivCoreCount()` 动态获取。

**"一次搬运的固定成本 ≈ 512 个元素"** 是一个非常有用的量化经验值——
它说明搬运固定成本确实存在，但量级是"几百个元素的计算时间"，
而不是我们以为的"整个算子 23us"。

## 未达标的 11 条，以及他们如何处理

11 条全部落在 **n ≤ 179**，成因已量化，其中 5 条（n ≤ 8、n=32）
"目标低于本平台的每调用下限"——他们**如实列出并解释**，
PR 依然 lgtm + approve。

**这说明：小 n 用例达不到是可接受的，前提是量化说明成因。**
我们此前认为"0/200 达标就不能提交"过于绝对；
真正的问题是我们大 n 也差 3~4 倍，而他们大 n 有 38~42% 余量。

## 对我们的直接影响

1. **10 µs 标杆不是不可达**——同平台同类算子在 n=512 做到 16.33 µs 且达标。
   我们要问的不再是"标杆是否合理"，而是"我们的 kernel 为什么慢 4 倍"。
2. **优先排查我们的固定开销**：23 µs vs 他们 5.08 µs（n=1）。
   嫌疑：TPipe/InitBuffer 数量、gather 表每次搬入 UB、GM tensor 绑定。
3. **借鉴按权重分核**：我们按 band 均分，短 band 的核会空转。
4. **给 Ascend C 开发者的问题需要重写**——问题一（"10us 如何达成"）
   已被这个 PR 回答了。

## 需要撤回/修订的文档

- `ctbmv-cost-decomposition.md`：14.4 µs "launch floor" 的推导有误
- `ctbmv-overhead-analysis.md`：「targets 低于任何已合入算子」结论错误
- `questions-for-ascendc-devs.md` / `question-short.md`：问题一作废
- `cann-comp-skills/.../perf-reality-check.md`：
  "arch22 标杆 < 15 µs 就该质疑"的阈值需要修正


---

# 后续排查（910B3 实测）

## 已证伪的假设

**假设：18us 的 device 固定开销来自每次调用把 gather 表搬进 UB / 分配满幅 UB。**

diag kernel 原本无论 n 多大都按 `maxCnt_=1024` 分配（buf_ 72KB + offsetBuf_ 8KB
+ bandTblBuf_ 16KB = **96KB UB**）并 DMA 16KB 的表。改成按 `tileCnt_`
（= min(n, maxCnt) 对齐后）分配与拷贝，n=1 时只需约 0.1KB：

| | 改前 | 改后 |
|---|---|---|
| n=1 device | 18.30 us | **18.30 us** |
| n=512 k=0 device | 18.59 us | 18.59 us |
| 功能 | 84/84 | **84/84** |

**完全没有变化 → 假设证伪。** UB 分配量与表搬运量都不是 18us 的来源。
（此改动本身是合理的瘦身，已保留在 `perf/ctbmv-right-size` 分支。）

## msprof 佐证

`ctbmv_diag_kernel`：**Block Num = 1**，`Task Duration = 24.17 us`。
即**单核**跑一个复数元素要 24us，且与 n 无关（n=1..1024 全是 ~23us）。

## 新发现：band 路径本身有约 35us 的固定成本

固定 n=512，只改 k（k+1 即 useCoreNum）：

```
k=0  (1 核, diag kernel)  23.64 us
k=1  (2 核, band kernel)  58.79 us   <-- +35us
k=2  (3 核)               59.47 us
k=3  (4 核)               59.74 us
k=7  (8 核)               59.34 us
k=15 (16 核)              61.93 us
k=31 (32 核)              73.47 us
```

**从 k=0 到 k=1 跳了 35us，而 k=1 到 k=7 几乎不变。**
这不是"每条 band 的工作量"，而是 **band 路径（compute kernel + workspace +
atomics + SyncAll + copy-back）的一次性成本**。

对比 PR #361 的观察："12 核时 11.86 µs vs 目标 11.11 µs，差距落在 host 下发与
多核启动的固定成本上"——他们的多核启动成本是 us 级，我们是 35us 级。

## 当前结论（诚实版）

1. **dispatch 不是问题**：我们 ~4.9us，PR #361 报 2.71us，同量级。
2. **单核 kernel 有 ~18us 的 device 固定开销**，来源未定位；
   已排除 UB 分配量与 gather 表搬运。
3. **band 路径额外有 ~35us 一次性成本**，这是 n=512/k=8 从 23us 变成 60us 的主因。
4. 二者相加解释了我们与标杆的绝大部分差距。

**下一步应查**：把 diag kernel 逐段注释（只 Init、Init+load、Init+load+compute）
测 device 时间，用二分法定位那 18us；band 路径同理拆 memset/atomics/SyncAll/copy-back。
这需要改 kernel 并重复编译测量，是可行但耗时的工作。


---

# 二分定位结果：那 18us 不在我们的代码里

用编译期开关 `CTBMV_BISECT` 逐段截断 diag kernel，每档重新编译并实测
n=1、k=0 的总耗时（910B3）：

| 档位 | kernel 做到哪一步 | 耗时 |
|---|---|---|
| 90 | **kernel 入口直接 return（空 kernel）** | **23.53 us** |
| 91 | 入口 + Init（不进 Process） | 23.53 us |
| 1 | + 进 Process 立即 return | 23.62 us |
| 2 | + offset 表搬入 | 23.54 us |
| 3 | + A 跨步搬入 | 23.55 us |
| 4 | + A 的 Gather 拆分 | 23.66 us |
| 5 | + x 搬入与 GatherMask 拆分 | 23.54 us |
| 6 | + 复数乘加 | 23.58 us |
| 0 | 完整 kernel | 23.49 us |

**一个函数体只有 `return` 的空 kernel 就要 23.53us**，
后续每一段的增量都在测量噪声内（±0.15us）。

## 结论

**我们的 kernel 代码对 n=1 的耗时贡献约等于零。**
23.5us 全部是 `aclblasCtbmv` 的 host 路径 + kernel 下发/启动的固定成本。

之前 split 测出的"dispatch 4.9us / device 18.2us"里，那 18.2us 是
**device 侧的 kernel 启动开销本身**，不是我们 kernel 在算什么。

## 与同库其它算子对比（n=1，同一次运行）

| 算子 | n=1 耗时 |
|---|---|
| **ctbmv（我们的）** | **23.17 us** |
| ccopy | 29.99 us |
| cswap | 51.43 us |
| cscal | 155.98 us |

我们已经是这一批里最快的。

## 这对 PR #361 的 2.25us device 意味着什么

PR #361 自报 Ctpmv 在 n=1 的 device 执行只有 2.25us。
但我们实测**空 kernel** 都要 18us device。两者相差 8 倍，可能的解释：

1. 他们的测量口径与我们不同（例如用 msprof 的 Task Duration 而非
   wall clock 差值，或扣除了某些固定项）
2. 他们的 kernel 走了不同的启动路径
3. 自报数据未经复核（该 PR **尚未合入，未获 lgtm/approve**）

**在没有复现他们测量方法之前，不应把 2.25us 当作我们应达到的基准。**
可复现的事实是：**本平台上一个空 AIV kernel 的端到端成本约 23.5us**，
而 case 1 的标杆是 10us。

## 剩下真正属于我们的部分

- k=0 → k=1 的 **+35us**（band 路径的一次性成本：workspace memset、atomics、
  SyncAll、copy-back）——**这部分确实是我们的代码**，是可优化的
- n=512/k=8 的 60us 中，约 23.5us 是平台固定成本，约 35us 是 band 路径开销，
  真正的算术不到 2us

---

# 追查 PR #361 的数字：profiler 口径假设也被证伪

## 假设：他们的 device/dispatch 是 profiler 计数，不是 wall clock

在 910B3 上对**第三方算子** `aclblasStpmv_legacy`（n=1）同时取两种口径：

| 口径 | 耗时 |
|---|---|
| wall clock（他们的协议：30 预热，5×60 取最小组均值） | **22.82 us** |
| msprof `Task Duration` | **22.54 us** |
| 一致性 | 1.2% |

**msprof 计数并不比 wall clock 小。** 该假设不成立。
（msprof 另给出 `Task Wait Time` 3.34 us，量级也对不上他们的 2.33 us dispatch。）

## 另一个真实的混淆项：AICore 降频

```
Aicore Freq(MHZ)     : 1800     <- 标称
Aicore curFreq(MHZ)  :  800     <- 实际
Aicore Count         : 20
No running processes found in NPU 4
```

我们的芯片**长期运行在 800 MHz，是标称的 1/2.25**，且没有其他进程占用。

加压观察发现频率在 **800 ↔ 1800 MHz 之间来回跳**，说明它会 boost 但不持续。

若他们的机器跑在满频 1800 MHz：

| | 我们实测 | 折算到 1800MHz | PR #361 自报 |
|---|---|---|---|
| `Stpmv_legacy` n=1 | 22.66 us | ~10.1 us | **3.28 us** |
| 我们的 ctbmv n=1 | 23.20 us | ~10.3 us | — |
| 我们的 ctbmv n=512 k=8 | 60.50 us | ~26.9 us | — |

**频率能解释约 2.25 倍，但不足以解释 6.9 倍。**

### 但这条对我们自己的结论很重要

**我们所有的性能数字可能偏悲观最多 2.25 倍。** 例如 n=512/k=8 的 60.5 us
在满频下约为 26.9 us（标杆 10 us）—— 差距从 6 倍缩到 2.7 倍。

不过延长预热并不能稳定拉高频率：

```
warmup=30     minGroup=23.26 us
warmup=2000   minGroup=23.29 us
warmup=20000  minGroup=23.31 us
```

三档完全一致（±0.05 us），说明**在我们这台机器上，这个量级的负载
既不会持续触发 boost，测量也非常稳定**。

## 结论

对 PR #361 数字的三个假设全部证伪：
1. ✗ 测量方法（最小组均值）—— 复现后差异 <3%
2. ✗ profiler 计数口径 —— msprof 与 wall clock 一致
3. ✗ 我们的 kernel 实现问题 —— 空 kernel 也要 23.5 us

**剩下最可能的是运行环境差异**（他们自述"测试芯片与他人共用"），
其中 AICore 频率是已证实存在的一项，但只够解释 2.25 倍。

**行动建议**：
- 在自测报告中**记录 `npu-smi` 的 curFreq**，说明测量条件
- 若能拿到满频（或独占）机器，所有数字应重测
- 不要把 PR #361 的 2.25/3.28 us 当作基准，除非能复现

---

# 频率问题的最终澄清：curFreq 不影响我们的测量

## 能不能设频率？

**不能。** 在 devenv 里：

```
$ npu-smi set -h
This command can be executed only by the root user.
$ id
uid=1000(developer) ...      # 无 sudo
$ npu-smi info -t cpu-freq-up -i 4
This device does not support querying cpu-freq-up.
$ npu-smi info -t work-mode -i 4
This device does not support querying work-mode.
```

功耗 91.9 W（远未触顶），NPU 上无其他进程。
频率由 **DVFS**（动态电压频率调节）自动管理，用户态无法锁定。

## 关键实验：满负载下性能是否变化

用**真正有计算量**的用例（n=4096, k=128，约 368 us/次）连跑 8 组 × 50 次，
同时旁路采样 curFreq：

```
group 0: 368.32 us      curFreq 采样: 1800
group 1: 369.21 us                    800
group 2: 368.58 us                    800
group 3: 368.64 us
group 4: 368.84 us
group 5: 368.37 us
group 6: 368.88 us
group 7: 368.62 us
```

**8 组耗时完全一致（368.6 ± 0.3 us，波动 <0.3%），而同期 curFreq 在
1800 与 800 之间跳变。**

## 结论：撤回"我们的数字偏悲观 2.25 倍"的说法

前一节推测"若跑在 1800 MHz，我们的数字可缩小 2.25 倍"。**这个推测是错的。**

- 性能对 curFreq 读数完全不敏感
- 延长预热（30 / 2000 / 20000 次）也不改变结果（23.26 / 23.29 / 23.31 us）

`curFreq` 是**采样瞬间**的读数，AI Core 在 kernel 执行期间会按需 boost；
空闲采样自然读到 800。**它不代表我们的 kernel 以 800 MHz 在跑。**

因此：
- 我们所有的性能数字**不需要按频率折算**，它们就是实际性能
- 与 PR #361 的 6.9 倍差距**不能用频率解释**
- 自测报告里仍应记录 curFreq，但要说明它是瞬时采样值

## 汇总：对 PR #361 数字的四个假设全部证伪

| # | 假设 | 结论 |
|---|---|---|
| 1 | 测量方法（最小组均值） | ✗ 复现后差异 <3% |
| 2 | profiler 计数口径 | ✗ msprof Task Duration 与 wall clock 差 1.2% |
| 3 | 我们的 kernel 实现 | ✗ 空 kernel 也要 23.5 us |
| 4 | AICore 降频 | ✗ 性能对 curFreq 不敏感，满负载 8 组零波动 |

**我们无法复现 PR #361 的 3.28 us / 2.25 us。**
在同一款芯片、用他们自述的方法、测他们引用的第三方算子
（`aclblasStpmv_legacy` n=1），我们得到 22.66 us，是其自报值的 6.9 倍。

该 PR **尚未合入，未获 lgtm/approve**，其性能数据未经复核。
在能复现之前，不应作为基准。

## 我们可复现的事实（互相印证）

| 测量对象 | n=1 耗时 |
|---|---|
| 空 AIV kernel（入口即 return） | 23.53 us |
| 现网 `aclblasStpmv_legacy` | 22.66 us |
| 现网 `aclblasCcopy` | 29.99 us |
| **我们的 `aclblasCtbmv`** | **23.17 us** |

四者一致，且我们是同批复数算子里最快的。

---

# 定位到根因：多 block 启动成本，与 kernel 内容无关

## band 路径二分（n=512, k=1, useCoreNum=2）

| 档位 | kernel 做到哪一步 | 耗时 |
|---|---|---|
| 1 | **入口直接 return** | **59.23 us** |
| 2 | + ZeroWorkspace | 59.45 us |
| 3 | + 第一次 SyncAll | 57.65 us |
| 4 | + 全部 band 累加 | 57.87 us |
| 5 | + 第二次 SyncAll | 57.83 us |
| 0 | 完整 kernel | 59.12 us |

**又是全部在噪声内。** ZeroWorkspace、两次 SyncAll、atomics、band 累加、
copy-back —— 加起来贡献约等于零。

## 决定性实验：空 kernel + 变 blockDim

把 compute kernel 改成入口即 return，只改 `blockDim`（= k+1）：

| blockDim | 耗时（空 kernel） |
|---|---|
| **1** | **23.57 us** |
| **2** | **59.45 us** |
| 3 | 59.49 us |
| 4 | 59.83 us |
| 8 | 59.17 us |
| 16 | 61.47 us |
| 32 | 73.39 us |

**从 1 个 block 到 2 个 block，一个什么都不做的 kernel 从 23.6us 跳到 59.5us
（+36us）。之后 2→8 个 block 几乎不变，16 以上才缓慢上升。**

## 结论

我们此前归因给"band 路径开销（workspace/atomics/SyncAll/copy-back）"的
那 +35us，**实际是多 block kernel 的启动成本**，与我们写了什么代码无关。

n=512/k=8 的 60us 拆解（更正版）：

| 组成 | 耗时 |
|---|---|
| 单 block kernel 启动 | ~23.5 us |
| 从 1 block 到 多 block 的额外启动成本 | ~36 us |
| 我们真正的计算 | **< 1 us** |

## 这条最重要的推论

**只要用多核，就要付约 36us 的启动成本。**
标杆是 10 / 16.51 / 19.77 us —— 全部低于这个门槛。

因此在这台设备上：
- 单核（blockDim=1）最快能到 ~23.5us
- 多核任何配置至少 ~59us
- **两者都超过 case 1 和 case 2 的标杆**

对比 PR #361 提到的"12 核时 11.86 µs"——如果他们的 12 核 kernel 总耗时
只有 11.86us，而我们空的 2-block kernel 就要 59us，这个差异**无法用
kernel 实现解释**，只能是运行环境/驱动/固件层面的差异。

## 下一步（如果还要继续）

1. 用 `aclrtLaunchKernel` 之类的底层接口直接测不同 blockDim 的空 kernel，
   排除 ops-blas 框架本身的开销
2. 换一台 devenv 复测（当前这台可能有异常）
3. 把这个现象作为问题提给 Ascend C 开发者——
   **"空 kernel 从 1 block 到 2 block 增加 36us，是否正常？"**
   这是一个非常具体、可复现、不涉及我们代码的问题

---

# 第二台 910B3 独立复现：多 block 启动成本是平台特性

新 devenv `7edd7002...`（与 `2d8c1a7e...` 不同的机器），同为 910B3 / CANN 9.1.0，
32 核 host，独立 clone + 编译。

## 核数扫描对比（n=512，只改 k，即 blockDim=k+1）

| blockDim | 机器 1 | 机器 2 |
|---|---|---|
| 1 | 23.57 us | **23.77 us** |
| 2 | 59.45 us | **58.93 us** |
| 3 | 59.49 us | 59.30 us |
| 4 | 59.83 us | 59.03 us |
| 8 | 59.17 us | 58.78 us |
| 16 | 61.47 us | 61.97 us |
| 32 | 73.39 us | 77.77 us |

**两台机器逐点吻合（差异 <1.5%）。**

## 三条性能用例

| case | 机器 1 | 机器 2 | 标杆 |
|---|---|---|---|
| 512/8 | 60.1 us | 60.08 us | 10.00 |
| 1024/16 | 68.2 us | 68.53 us | 16.51 |
| 2048/32 | 99.3 us | 102.46 us | 19.77 |

## 结论

**"1 block → 2 block 增加约 36us" 不是本机异常，是这一代平台/驱动的普遍行为。**

这条对所有参与者都成立，包括 m0_69357246 那份设计文档里
`blockDim=min(aivCoreNum, ceil(n/tileRows))` 的多核策略 ——
在 n=512 这种规模上，起第二个核就要额外付 36us，而标杆只有 10us。

---

# 同时发现：我引入的一个回归（已修复）

第二台机器上首次跑 smoke 得到 **12/84 失败**，全部是 `k=0, NON_UNIT, n∈{1,5}`。

根因：commit `0fcaa88`（"size diag-kernel buffers to the problem"）里，
我用正则把 diag staging buffer 内的 plane 偏移从 `maxCnt_` 批量改成了 `tileCnt_`，
但 **host 侧构建的 gather 表是按 maxCnt 跨距排布的**，两者不再匹配。
n < 8 时 tileCnt_=8 ≠ maxCnt_=1024，于是读错位置。

- 该 commit **实测零收益**（n=1 device 时间改前改后都是 18.30us）
- 我在做完这个 commit 后**没有重跑 smoke**，所以在机器 1 上没发现

已 `git revert`（commit `77d061d`），机器 2 复测 **84/84 通过**，
随后跑完整用例集 **1872/1872 通过**（13 shape x 2 uplo x 3 trans x 2 diag x 4 incx x 3 lda pad），
与机器 1 完全一致。

**教训**：任何触及 buffer 布局/偏移的改动，即使动机是"瘦身"且预期无功能影响，
也必须重跑完整用例集。正则批量替换偏移量尤其危险。
