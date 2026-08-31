# aclblasCtbmv 算子设计文档

| 版本 | 日期 | 修改人 | 修改内容 |
|------|------|--------|----------|
| v1.0 | 2026-08-30 | leondgarse | 初稿 |

---

> **提交说明**：设计文档 PR 提交至 `cann-ops-competitions`
> （`cann-competitions` 为同一仓库的别名，两者 HEAD 相同，任一 URL 均可）。
> 路径：`04_tasks/01_community-task-2026/tasklist/<任务名>/<用户名>/docs/design.md`

# 一、需求背景

## 1.1 需求来源

8 月社区任务——`aclblasCtbmv` 算子开发（Atlas A2/A3）。参考 cuBLAS `cublasCtbmv`
（语义以 Netlib `ctbmv.f` 为准），在昇腾 NPU 上用 Ascend C 实现单精度复数
（complex64）三角带状矩阵-向量乘，交付 Ascend C Kernel + `aclblasCtbmv` 句柄式
BLAS 接口，验收后合入 `ops-blas` 的 `blas/tbmv/arch22/`。

## 1.2 背景介绍

TBMV（Triangular Banded Matrix-Vector multiply）是 BLAS Level-2 的基础算子，
用于三角带状线性系统求解、带状预条件子等场景。计算定义：

```
x := op(A) * x          （原地覆写 x）
```

- A 为 n×n 三角带状矩阵，半带宽 k，按**带状格式列主序**存于 lda×n 数组
- `op(A) = A`（OP_N）/ `Aᵀ`（OP_T）/ `Aᴴ`（OP_C，共轭转置）
- 无 alpha/beta 标量；x 为长度 n、步长 incx 的向量

## 1.3 现状分析

`ops-blas` 仓现状（基于 master 核查）：

| 项 | 状态 |
|---|---|
| `blas/tbmv/arch22/` | **已存在**，为实数版 `stbmv`（host 95 行 + kernel 393 行 + tiling 26 行） |
| `blas/tbmv/arch35/` | 已存在，含 `stbmv` 通用路径 + SIMD fastpath |
| `include/cann_ops_blas.h` 中 `aclblasCtbmv` | **无声明**，complex 路径为全新接口 |
| `test/tbmv/stbmv/` | 已存在，含 `stbmv_param.h`/`stbmv_golden.h`/`arch22/`、CSV 驱动 GTest |
| 复数类型 `aclblasComplex` | **已定义**于 `include/cann_ops_blas_common.h`（实部/虚部各 float32） |
| 复数辅助运算 | `blas/common/helper/complex.h` 已提供 `+ - * /`、共轭、取模等 |
| **`blas/gerc/arch22/`（cgerc）** | **已存在，complex64 BLAS-2 on arch22，467 行**——本设计最直接的参照实现 |
| `blas/gemv_batched/arch22/`（cgemv_batched） | 已存在，complex64 矩阵-向量类，可参考其 GatherMask 用法 |
| `blas/hemm/arch22/`（chemm） | 已存在，complex64 BLAS-3，含 GatherMask 拆分与精度补偿 |

**结论**：本任务是在已有 `stbmv` 同族目录下**新增 complex64 路径**。数据通路、
带状索引、多核切分、CSV 测试框架均可复用；新增工作集中在
①复数运算 ②`OP_C` 共轭语义 ③接口声明与测试适配。

关键复用点：现有 `stbmv_kernel.cpp` 的核心类已经是
`template <typename T> class TbmvAIV`，实例化为 `TbmvAIV<float>`。本设计沿用该
模板结构，**不新造轮子**。

---

# 二、需求分析

## 2.1 需求描述

实现 `aclblasCtbmv`，与 `cublasCtbmv` 参数序列、语义完全对齐；支持
uplo(2) × trans(3) × diag(2) = 12 种枚举组合全覆盖，支持负步长 incx，
精度满足生态算子开源精度标准，性能不低于任务书标杆。

## 2.2 需求拆解

1. **接口层**：`include/cann_ops_blas.h` 新增 `aclblasCtbmv` 声明，与
   `aclblasStbmv` 同型（`float*` → `aclblasComplex*`），禁止定义 A2/A3 私有平行接口。
2. **Host 层**：参数校验 + tiling 计算 + kernel 下发，落在 `blas/tbmv/arch22/`。
3. **Kernel 层**：复数带状矩阵-向量乘，AIV（vector）实现。
4. **测试层**：`test/tbmv/ctbmv/arch22/`，复用任务书提供的 1200 条 CSV 用例。

## 2.3 输入输出规格

| 名称 | 角色 | 类型 | 说明 |
|------|------|------|------|
| handle | 输入 | `aclblasHandle_t` | 携带 stream，Host 内存 |
| uplo | 输入 | `aclblasFillMode_t` | UPPER / LOWER |
| trans | 输入 | `aclblasOperation_t` | OP_N / OP_T / OP_C |
| diag | 输入 | `aclblasDiagType_t` | NON_UNIT / UNIT |
| n, k | 输入 | `int` | n ≥ 0；k ≥ 0，有效 k ∈ [0, n-1] |
| A | 输入 | `const aclblasComplex*` | lda×n，带状列主序，Device，只读 |
| lda | 输入 | `int` | lda ≥ k+1 |
| x | 输入/输出 | `aclblasComplex*` | 长度 n，步长 incx，**原地覆写** |
| incx | 输入 | `int` | ≠ 0，支持负值 |

## 2.4 接口原型

```cpp
aclblasStatus_t aclblasCtbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int n, int k,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx);
```

## 2.5 带状存储索引规则（1-based，源自 Netlib）

| uplo | 主对角所在行 | 元素 A(i,j) 位置 | 不引用区域 |
|---|---|---|---|
| LOWER | 第 1 行 | `A(1+i-j, j)` | 数组右下角 k×k |
| UPPER | 第 k+1 行 | `A(1+k+i-j, j)` | 数组左上角 k×k |

`diag = UNIT` 时对角元素不读取，视为 `(1.0f, 0.0f)`。

---

# 三、方案设计

## 3.1 实现路径选型

| 候选 | 结论 |
|---|---|
| **AIV（vector）逐带处理** | ✅ **选用**。BLAS-2，计算量 O(n·k)，访存受限，无需 Cube |
| Cube（Mmad） | ❌ 带状矩阵稀疏度高，转稠密块浪费算力，且 BLAS-2 无复用 |
| ACLNN 封装 | ❌ 无现成内置复数 TBMV 算子 |

沿用 `stbmv/arch22` 的 **按对角带（band）切分 + 多核并行 + 原子累加** 方案。

## 3.2 复数运算方案

Ascend vector 单元**无原生复数乘**。采用**实虚分离**：设
`a = ar + i·ai`，`x = xr + i·xi`，则

```
op = N/T :  (a·x).re = ar*xr - ai*xi      (a·x).im = ar*xi + ai*xr
op = C   :  conj(a)·x：
            (·).re = ar*xr + ai*xi        (·).im = ar*xi - ai*xr
```

即 `OP_C` 相对 `OP_T` **只需对 ai 取反**，无需额外分支路径——这是本设计的关键简化：
`trans` 的三种取值中，`OP_T`/`OP_C` 共用同一套带状索引，仅在复数乘时对虚部符号做区分。

**存储布局选择**：`aclblasComplex` 为 `{float real; float imag;}`，GM 上是
**交织（interleaved）** 排布。两种处理方式：

| 方案 | 说明 | 结论 |
|---|---|---|
| A. 交织直算 | 直接把 `aclblasComplex` 当 `T` 搬入 UB，用标量循环算复数乘 | ❌ 违反 R1（逐元素） |
| **B. 拆分为实虚两路** | 搬入后用 `GatherMask` 拆成 ar/ai/xr/xi 四个 float 向量，4 次 `Mul` + 2 次 `Add/Sub` 算完，再用 `Gather` 重新交织写回 | ✅ **选用**，全程矢量化 |

**拆分/交织 API（arch22 已验证可用）**：

```cpp
// 拆分：交织 [re0,im0,re1,im1,...] → 实部向量 + 虚部向量
// pattern=1 取偶数位（实部），pattern=2 取奇数位（虚部）
uint32_t mask = 0; uint64_t rsvdCnt = 0;
AscendC::GatherMask<float>(ubReal, ubIn, 1, false, mask,
                           {1, (uint16_t)repeatTime, 8, 8}, rsvdCnt);
AscendC::GatherMask<float>(ubImag, ubIn, 2, false, mask,
                           {1, (uint16_t)repeatTime, 8, 8}, rsvdCnt);

// 交织回写：用预先构造的 offset 表
AscendC::Gather(ubOut, ubCalc, gatherOffset, 0U, repeatTimes * 64);
```

> 依据：`blas/gerc/arch22/cgerc_kernel_impl.h:103-106,158` 与
> `blas/hemm/arch22/chemm_kernel.cpp:218-219,745-746` 均为 arch22 上的既有用法。
> 注意 `DeInterleave`/`Interleave` 是 **arch35 专有**（见 `herk/arch35`），arch22 不可用。

方案 B 的向量运算序列（每个 band、每批 colBatchSize 个元素）：

```cpp
Mul(t0, ar, xr, cnt);          // ar*xr
Mul(t1, ai, xi, cnt);          // ai*xi
Sub(yr, t0, t1, cnt);          // re = ar*xr - ai*xi     (OP_N/OP_T)
// OP_C: Add(yr, t0, t1, cnt);  re = ar*xr + ai*xi

Mul(t2, ar, xi, cnt);          // ar*xi
Mul(t3, ai, xr, cnt);          // ai*xr
Add(yi, t2, t3, cnt);          // im = ar*xi + ai*xr     (OP_N/OP_T)
// OP_C: Sub(yi, t2, t3, cnt);  im = ar*xi - ai*xr
```

## 3.3 Kernel 划分与执行模型

### 3.3.1 为什么需要 workspace 与两段 kernel

`aclblasCtbmv` 语义为**原地覆写** `x := op(A)·x`。但带状乘累加过程中，同一个
`x[i]` 既作为输入被多个 band 读取、又是输出位置——若直接就地写，先完成的 band
会污染尚未读取的输入。

仓内 `stbmv/arch35` 已给出标准解法（`stbmv_fallback_kernel.cpp:259-267`）：
**device workspace 存中间结果 + 两段 kernel 串行下发**。

```cpp
void ctbmv_arch22_kernel_do(const aclblasComplex* a, aclblasComplex* x,
                            uint8_t* workspace, const CtbmvTilingData& tiling,
                            uint32_t numBlocks, void* stream)
{
    if (tiling.k == 0U) {                    // 退化为对角缩放，无需 workspace
        ctbmv_diag_kernel<<<numBlocks, nullptr, stream>>>(a, x, tiling);
        return;
    }
    ctbmv_compute_kernel<<<numBlocks, nullptr, stream>>>(a, x, workspace, tiling);
    ctbmv_copy_kernel<<<numBlocks, nullptr, stream>>>(x, workspace, tiling);
}
```

**同步语义**：两个 kernel 下发到**同一 stream**，由 stream 的顺序执行保证
compute 全部完成后 copy 才启动——**不需要 `SyncAll` 或跨核屏障**。这是本设计
采用两段 kernel 而非单 kernel + 核间同步的原因。

| kernel | 职责 | 前置条件 |
|---|---|---|
| `ctbmv_compute_kernel` | 各 band 并行计算，原子累加到 workspace | workspace 已清零 |
| `ctbmv_copy_kernel` | workspace → x 回写（按 incx 展开） | compute 已完成（stream 保序） |
| `ctbmv_diag_kernel` | `k == 0` 专用：x[i] *= A[对角]，直接就地 | 无 |

### 3.3.2 workspace 布局与清零

**布局**：采用**实虚分离的两个 float 平面**，而非交织存放：

```
workspace: [ yr(n 个 float) | yi(n 个 float) ]
偏移       0                  n*4 字节
大小       n * 2 * sizeof(float) = n * 8 字节
```

选择分离布局的原因：compute 阶段的向量运算产出就是分离的 `yr`/`yi`
（见 §3.2），分离存放可直接 `DataCopyPad` 落盘，无需先交织；交织留到
copy 阶段用 `Gather` 一次完成。

**清零**：原子累加要求初值为 0。workspace 由 handle 提供（可能残留上次数据），
因此在 `ctbmv_compute_kernel` 入口按核分片用 `Duplicate` 清零，再
`SyncAll()` 一次，然后进入累加。

> 备选：Host 侧 `aclrtMemsetAsync` 清零。本设计选 kernel 内清零以省一次下发。
> 注意此处的 `SyncAll()` 是**清零与累加之间**的核间同步，与 §3.3.1 的
> compute↔copy 顺序无关（后者由 stream 保序）。

### 3.3.3 原子累加的类型

不同 band 会写到 y 的重叠位置，需原子加。workspace 按 `float` 平面组织，
因此使用 `SetAtomicAdd<float>()` / `SetAtomicNone()`——**作用对象是 float，
不是 complex**。这正是选择分离布局的另一个理由：交织布局下对
`aclblasComplex` 做原子加无对应原语。

> 参照：`stbmv/arch22` 的 `Process()` 用 `SetAtomicAdd<T>()`（T=float）包裹全过程。

### 3.3.4 与 stbmv/arch22 的接口差异（重要）

现有 `stbmv/arch22` 走的是 `aclblasStbmv_legacy`，签名里 **x 与 y 是两个独立
buffer**（`const float* x, float* y`），由调用方预先把 y 清零
（见 `test/tbmv/stbmv/arch22/stbmv_npu_wrapper.h:56-64` 的 `yZero`）。

而本任务的 `aclblasCtbmv` 是**单 buffer 原地**语义，与
`aclblasStbmv`（arch35 实现的现代签名）一致。因此：

- **算法与带状索引**复用 `stbmv/arch22`（kernel 内的 band 映射）
- **接口形态、workspace 与两段 kernel 结构**复用 `stbmv/arch35`（现代签名）

这是本设计需要"跨 arch 取材"的关键点，不能只照抄 arch22。

## 3.4 带状索引与并行切分

> **关于 `ProcessFast`**：`stbmv/arch22` 对 `LOWER+OP_N` 有一条复用 `xLocal`
> 的连续访存快路径。复数版拆分为 6 路 float 平面后，UB 占用上升（§4.3），
> 该复用模式是否仍然成立**需编码阶段实测确认**；本设计不将其作为达标前提，
> 先实现通用路径，快路径作为性能优化项（§6.2）在基线跑通后再引入。

**并行维度**：按对角带 `bandIdx ∈ [0, k]` 切分到多核，
`for (bandIdx = vecIdx; bandIdx <= k; bandIdx += useCoreNum)`。
任务数 `taskCount = k + 1`。

**各组合的 A 行基址 / 列范围**（复用 `stbmv` 已验证的映射）：

| uplo | trans | aRowBase | firstCol | bandLen |
|---|---|---|---|---|
| LOWER | OP_N | bandIdx | 0 | n - bandIdx |
| LOWER | OP_T/C | bandIdx | bandIdx | n |
| UPPER | OP_N | k - bandIdx | bandIdx | n |
| UPPER | OP_T/C | k - bandIdx | 0 | n - bandIdx |

**y 写回起点**：`LOWER+OP_N` 与 `UPPER+OP_T/C` 为 `col + bandIdx`，其余为 `col - bandIdx`。

**多核累加**：见 §3.3.3——workspace 为 float 平面，用 `SetAtomicAdd<float>()` /
`SetAtomicNone()` 包裹累加过程。

## 3.5 负步长与 incx 处理

物理位置映射（复用 `stbmv` 的 `XPhysicalPos`）：

```cpp
pos(logical) = (incx >= 0) ? logical * |incx|
                           : (n - 1 - logical) * |incx|
```

等价于 Netlib 的 `kx = 1 - (n-1)*incx` 反向遍历。

`incx != 1` 时访存不连续 → tiling 中 `useCoreNum = 1`（与 `stbmv` 一致），
保证正确性优先；连续场景（incx == 1）才开多核与向量化快路径。

---

# 四、Tiling 设计

## 4.1 TilingData 结构

```cpp
constexpr uint32_t TBMV_MAX_CORE_NUM = 50;

struct CtbmvTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t useCoreNum;
    int64_t  incx;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
};
```

> ⚠️ Host 与 Kernel 两侧字段顺序/类型必须**逐一对齐**——kernel 侧直接按结构体
> 解释 GM 内存，不一致只在运行时表现为错误结果，编译期不报错。

## 4.2 核间切分（Block 级）

```cpp
taskCount = k + 1;                                   // 带数
availableCoreNum = GetAivCoreCount();                // R2：动态获取，禁止硬编码
availableCoreNum = min(availableCoreNum, TBMV_MAX_CORE_NUM);
useCoreNum = (incx == 1) ? min(taskCount, availableCoreNum) : 1;
useCoreNum = max(useCoreNum, 1);
```

## 4.3 核内切分（UB 级）

复数需要 6 路 float 缓冲（ar, ai, xr, xi, yr, yi）+ 2 个临时量，
且 double buffer（BUFFER_NUM = 2）。

| Buffer | 用途 | 数量 | 单份字节 |
|---|---|---|---|
| aQueue | A 带元素（交织 complex） | 2 | `maxCnt * 8` |
| xQueue | x 分片（交织 complex） | 2 | `maxCnt * 8` |
| yQueue | y 输出（交织 complex） | 2 | `maxCnt * 8` |
| tmpBuf | ar/ai/xr/xi/yr/yi + t0~t3 | 1 | `maxCnt * 4 * 10` |
| gatherOffset | 交织回写用 offset 表（uint32） | 1 | `maxCnt * 2 * 4` |

总占用 ≈ `maxCnt * (8*2*3 + 40 + 8)` = `maxCnt * 96` 字节。

取 UB 可用 ≈ 192KB，留余量按 160KB 计：
`maxCnt = 160*1024 / 96 ≈ 1706` → 向下对齐到 8 个 complex → **maxCnt = 1704**（约 159.75KB）。

`gatherOffset` 表在 `Init()` 中构造一次并复用，不随分块重建。

> 实际编码时用平台 API 查询 UB 大小，不硬编码 192KB。

## 4.4 Host 侧 workspace 申请与校验

```cpp
size_t workspaceSize = (k > 0) ? static_cast<size_t>(n) * 2U * sizeof(float) : 0;
if (k > 0) {
    CHECK_RET(workspaceSize <= GetEffectiveWorkspaceSize(h),
        OP_LOGE("aclblasCtbmv", "workspace %zu > handle %zu",
                workspaceSize, GetEffectiveWorkspaceSize(h));
        return ACLBLAS_STATUS_EXECUTION_FAILED);
    workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
}
```

`k == 0` 走 `ctbmv_diag_kernel`，不需要 workspace（对齐 `stbmv/arch35:153`）。

**提前返回**：`k == 0 && diag == ACLBLAS_UNIT` 时 `op(A) = I`，x 不变，
直接返回 `ACLBLAS_STATUS_SUCCESS`（对齐 `stbmv/arch35:133-135`）。

## 4.5 对齐与边界

- UB 32B 对齐：complex64 单元素 8B → 每 4 个 complex 对齐一次
- 尾块用 `DataCopyPad` + `DataCopyPadExtParams` 补齐
- **`repeatTime` ≤ 255**：`Mul`/`Add`/`Sub` 的高维切分重载需分批，
  单批元素数不超过 `255 * elementsPerRepeat`
- `DataCopyExtParams.blockLen` 为 uint16 → 单次搬运字节数 ≤ 65535

---

# 五、精度方案

## 5.1 Golden

由 cblas（Netlib `ctbmv`）生成，随测试工程提供，实部/虚部分别比对。

## 5.2 判定标准

| dtype | rtol | atol | matched_ratio | max_abs_error_limit |
|---|---|---|---|---|
| COMPLEX64（实/虚部按 FLOAT32） | 2⁻¹⁰ (9.77e-4) | 2⁻¹⁶ (1.53e-5) | ≥ 0.99 | 1e-2 或 32×ULP |

逐元素：`|actual - golden| ≤ atol + rtol × |golden|`。

## 5.3 精度风险与对策

带状乘累加最多 `k+1` 项，k ≤ 4096 量级，累加深度有限，**预期朴素 fp32 累加即可达标**。

若实测不达标，备选：对每个输出元素的 `k+1` 项累加改用 **Kahan / double-float 补偿求和**
（仓内 `chemm/arch22` 已有 `TwoSum`/`TwoProduct` 实现可参考），代价是吞吐下降。
**先按朴素实现验证，不达标再启用。**

## 5.4 特殊值

INF/NaN 按精度标准文档规则验收；`diag=UNIT` 用例对角不参与比对。

---

# 六、性能方案

## 6.1 标杆

| case | n | k | uplo | trans | diag | incx | 标杆(us) |
|---|---|---|---|---|---|---|---|
| 1 | 512 | 8 | UPPER | N | NON_UNIT | 1 | 10 |
| 2 | 1024 | 16 | UPPER | N | NON_UNIT | 1 | 16.51 |
| 3 | 2048 | 32 | LOWER | T | NON_UNIT | 1 | 19.77 |

采集口径：warmup 后有效采样 > 50 次取平均（任务书 §3.3/§7.4）。

## 6.2 优化点

1. **Double buffer**（BUFFER_NUM=2）隐藏 MTE2/MTE3 搬运延迟
2. **按 band 多核并行**，`taskCount = k+1`；标杆 case 的 k ∈ {8,16,32} → 可用 9/17/33 核
3. **连续访存快路径**（可选优化）：`LOWER+OP_N && incx==1` 复用 xLocal，避免逐列偏移计算；需实测 UB 是否容纳，见 §3.4
4. **实虚分离后全向量化**，杜绝逐元素 `SetValue`/`GetValue`（R1）
5. `diag=UNIT` 时对角带**跳过 A 的搬入与复数乘**，直接把 x 累加到 y
   （因为 `A_diag = (1,0)` ⇒ `A_diag · x = x`），比"置 1 再乘"更省一次搬运和两次 Mul。
   若仍需显式构造，则为**两次** `Duplicate`：`ar=1.0f`、`ai=0.0f`。

> 现有 `stbmv` 在 `diag==UNIT` 分支用了逐元素 `LocalA.SetValue(i, 1.0f)` 循环
> （`stbmv_kernel.cpp:132-135, 303-306`），**违反 R1**。本设计不沿用该写法。

## 6.3 性能风险

k 较小时（如 k=0 退化为对角矩阵）并行度受限（taskCount=1）。
对策：k 小且 n 大时改按 **列方向**切分而非按 band 切分。

---

# 七、测试方案

## 7.1 用例来源

复用任务书 `test_cases/` 提供的 **1200 条** CSV 用例（`ctbmv_test.csv`），
固定随机种子，含：

- 精度 1000 条：uplo×trans×diag 全 12 组合 × 尺寸扫描 × 带宽扫描 × lda/incx/填充/边界
- 性能 200 条（`TC_PF` 前缀）：含 4 条与任务书 §3.3 精确匹配的 case

尺寸范围 [1, 2048]（精度）/ [1, 4096]（性能）；带宽覆盖 0 / 1 / 小值 / 半带 / 满带 n-1；
含奇数(3/5/7/65)、边界值(1)、非对齐值、大尺寸。

**实测用例构成**（解析 `ctbmv_test.csv` 得到，共 1200 条）：

| 前缀 | 条数 | 含义 |
|---|---|---|
| `TC_EX` | 808 | 扩展组合扫描 |
| `TC_PF` | 200 | 性能/内存 |
| `TC_SQ` | 92 | 尺寸扫描 |
| `TC_CV` | 24 | 中等尺寸覆盖 |
| `TC_BD` | 22 | 带宽边界 |
| `TC_FL` | 14 | 填充模式 |
| `TC_L0` | 12 | 基础全枚举组合 |
| `TC_INC` | 12 | 步长（含负值） |
| `TC_ED` | 10 | 边界/负向 |
| `TC_LD` | 6 | 前导维 padding |

精度 1000 条 + 性能 200 条，与任务书 §3.5 要求的覆盖面一致。

## 7.2 工程结构

```
blas/tbmv/arch22/
  ├── ctbmv_host.cpp          # 参数校验 + tiling + launch
  ├── ctbmv_kernel.cpp        # AIV kernel
  ├── ctbmv_kernel.h
  └── ctbmv_tiling_data.h
test/tbmv/ctbmv/
  ├── CMakeLists.txt
  ├── ctbmv_param.h           # CSV 解析（对齐 stbmv_param.h）
  ├── ctbmv_golden.h          # cblas golden
  └── arch22/
      ├── ctbmv_test.cpp      # GTest 主体
      ├── ctbmv_npu_wrapper.h # Host↔Device 搬运封装
      └── ctbmv_test.csv      # 1200 条用例
include/cann_ops_blas.h       # 新增 aclblasCtbmv 声明
```

## 7.3 覆盖场景

| 类别 | 场景 |
|---|---|
| 基础 | 12 种枚举组合 × 小 shape |
| 尺寸扫描 | n ∈ [1,2048]，含奇数/边界/非对齐 |
| 带宽 | k = 0 / 1 / 中间值 / n-1 满带 |
| 前导维 | lda = k+1 紧凑 + 多组 padding |
| 步长 | incx = ±1/±2/±3 |
| 边界负向 | n=0 no-op、A/x 空指针、incx=0、n<0、k<0、lda<k+1、非法枚举 |
| 特殊值 | INF/NaN |
| 性能 | 3 条标杆 case + TC_PF 集 |

## 7.4 编译与运行

```bash
bash build.sh --ops=tbmv --soc=ascend910b3 --run
python verify_accuracy.py --repo <ops-blas> --soc ascend910b3
python verify_performance.py --repo <ops-blas> --soc ascend910b3
```

---

# 八、产品支持与交付

## 8.1 产品支持表

| 产品 | 支持 |
|---|---|
| Atlas A2 训练系列 / A2 推理系列（dav-2201 / arch22） | √ |
| Atlas A3 训练系列 / A3 推理系列（arch22） | √ |
| Ascend 950PR / 950DT（arch35） | 本任务不涉及 |

## 8.2 交付清单

- [ ] `include/cann_ops_blas.h` 新增 `aclblasCtbmv` 声明
- [ ] `blas/tbmv/arch22/ctbmv_*`（host / kernel / tiling）
- [ ] `blas/tbmv/README.md` 补充 Ctbmv 章节（接口、约束、产品支持、调用示例）
- [ ] `test/tbmv/ctbmv/arch22/`（GTest + CSV + wrapper）
- [ ] 自测报告（用例参数、实部/虚部精度截图、性能数据、内存占用）

## 8.3 编码规范自查

提交前按 `agent/skills/repo-coding-rules/references/checklist.md` 8 步自查，
重点：R1（禁逐元素）、R2（动态取核数）、R3（TPipe 非成员）、R4（TilingData 禁数组）、
R5-R10（圈复杂度/嵌套/行数/除零/许可证头/extern），并执行
`sh scripts/oat_check.sh <变更文件>`。
