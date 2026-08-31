# 硬约束速查（违反 = 评审打回）

分三类来源：① 上游仓 `agent/skills/repo-coding-rules`（评审依据，权威）
② Ascend C 硬件/API 限制 ③ 已合入代码里的实测经验。

**权威性顺序**：上游仓 rules > 本地 `agent-skills/` references > 论文/推测。
冲突时以上游仓为准。

---

## 一、上游 R1-R10（`repo-coding-rules`，必须零违规）

| 编号 | 规则 | 说明 |
|---|---|---|
| **R1** | 禁止逐元素操作 | `SetValue`/`GetValue`/逐元素 `DataCopy` 会让 kernel 退化成串行 CPU 代码，丧失矢量并行与 DMA 批量搬移 |
| **R2** | 动态获取 CoreNum | 用平台 API 取核数，**禁止硬编码** 40/48 之类 |
| **R3** | TPipe 禁止作成员变量 | |
| **R4** | TilingData 禁止用数组做核间分配 | |
| R5 | 圈复杂度 ≤ 20 | 1 + if/else if/for/while/case/&&/\|\| 数量 |
| R6 | 嵌套深度 ≤ 5 | |
| R7 | NBNC ≤ 50（函数非空非注释行） | |
| R8 | 除法/取模前校验除数非零 | |
| R9 | 所有源文件含许可证头 | |
| R10 | 禁止 extern 函数/变量声明 | |

> ⚠️ **R1 与现存代码的矛盾**：`sparse/spmm/arch22/spmm_kernel.cpp` 里大量用
> `colIndicesGm.GetValue(idx)`。那是历史实现，**不要当范例**。新代码走 Cube/批量搬移。

## 二、MR 安全编码规则

- `references/mr-rules-essential.md`：严重/致命级（G.PRE.05、G.INC.*、G.FUU.09/10/12/13/15、
  G.MEM.04、G.STD.*、OAT 等约 18 条）—— **提交前必查，须零违规**
- `references/mr-rules-general.md`：一般/建议级（G.EXP.*、G.CTL.03、G.AST.03、CQ.* 等约 17 条）

常见致命项：`realloc`/`alloca` 禁用；安全函数返回值必须检查；`destMax` 要正确；
禁 `exit`/`abort`/`atexit`/`kill`；格式化字符串与参数类型匹配。

**提交前跑**：`sh scripts/oat_check.sh <变更文件列表>`

## 三、Ascend C API / 硬件限制

| 限制 | 值 | 后果 |
|---|---|---|
| `repeatTime` (Add/Sub/Mul/Cast/Duplicate/LoadData…) | **uint8，≤255** | 传 256 静默截断为 0：**不计算也不报错** |
| `DataCopyParams.blockLen` | **uint16** | 直接限制单次搬运长度（cube_spmm 的 N≤32752 就源于此） |
| GM↔UB 搬运 | 必须 `DataCopyPad` | 普通 `DataCopy` 处理不了非对齐 |
| UB 对齐 | 32B | 单值缓冲也要开 32B |
| Cache Line | 512B | Block 级切分对齐用 |
| `InitBuffer` 总数 | ≤ 64 | 超了要合并 buffer |
| `Compare` | 数据区 256B 整数倍 | 不足需 padding |
| UB / L1 / L0A/B / L0C | 192KB / 512KB / 64KB×2 / 128KB | A2/A3；实际用平台 API 取 |

**Kernel 侧禁止**：`std::` 数学函数（`std::min/max/abs/sqrt/exp/log`）、`<cmath>`、
动态内存（`new`/`malloc`/`std::vector`）。标量比较用三元表达式，向量用 AscendC API。

**Host/Kernel 头文件隔离**：
- `op_host/*.cpp` 可 include `<cmath>`/`<algorithm>`/tiling 头；**不可** include `kernel_operator.h`
- `op_kernel/*.cpp` 可 include `kernel_operator.h`；**不可** include `<cmath>`/`<algorithm>`/tiling 头

## 三点五、向量指令的操作数偏移必须 32B 对齐（实测，代价最大的一条）

**这条上游 skill 没有明确写，但它让同一个改造失败了三次。**

`Add`/`Sub`/`Mul`/`Adds` 等向量指令的**每个操作数**（含目的操作数）起始地址
必须 32 字节对齐。写成 `dst[i]` 时，只有 `i * sizeof(T)` 是 32 的整数倍才合法。

```cpp
// ✗ 错：offset = d 是任意值（band 号 0..k），只有 d % 8 == 0 时才对
Add(yr_[lo - rowStart], yr_[lo - rowStart], t0_, cnt);

// ✓ 对：built-in 的通用写法 —— 偏移永远是某个"已对齐步长"的整数倍
int64_t calNumAlign = CeilA2B(calCount, blockSize) * blockSize;
int64_t offset      = i * calNumAlign;
Add(currentRow[offset], currentRow[offset], lastRow, calNumAlign);
```

依据：CANN 内置算子
`ops_math/ascendc/exp_segsum_grad/exp_segsum_grad.h:311`、
`grouped_bias_add_grad/arch32/...:225`（`Add(src[0], src[0], src[halfRows*cols], ...)`，
偏移是 `cols` 的整数倍）。**内置算子无一例外地把偏移构造成 `i * 对齐步长`。**

### 关键的不对称：`Gather` 没有这个限制

`Gather` 接受**任意字节偏移**，向量指令不接受。同一个"错位"操作，用 `Gather`
合法、用 `Add` 静默出错。这个不对称是最容易踩的坑。

**规避手法（推荐）**：移动**源**而不是目的。让每个向量指令都写在 offset 0，
把位移交给 GM 侧的 `DataCopyPad`（GM 元素偏移任意合法）或 `Gather`。

**自查**：任何形如 `Op(dst[expr], ...)` 的向量调用，先问 `expr` 是不是某个
32B 对齐量的整数倍；不是就改写。

## 三点六、DataCopyPad 会把每个 burst 补齐到 32B（UB 侧）

实测（probe kernel dump UB 内容）：`blockLen = 8` 字节的 burst，元素 i 落在
float 偏移 `i*8` 而不是 `i*2`；**没有任何 `dstStride` 取值能让它紧凑排布**。

后果：
- 8 字节 burst 实际占 32 字节 UB 带宽 —— **4 倍浪费**
- 想在这种"补齐"布局上用 `GatherMask`（它假设紧凑交织）会读到错位数据

**正确姿势**（`cgemv/arch22` 与内置 `as_strided` 的共同做法）：
**不要把跨步读表达成大量小 burst。一次连续搬入一个超集，再在 UB 内 `Gather`。**
参考 `blas/common/helper/kernel_utils.h` 的 `matrix_gm2ubuf`：按列做**连续**
拷贝并用显式 `dstStride` 控制 UB 落位。

## 四、实测经验（来自已合入代码 & 论文）

- **Host↔Kernel TilingData 字段必须逐一对齐**。kernel 里是
  `reinterpret_cast<__gm__ XxxTilingData*>(tilingGm)` 直接解释 GM 内存——
  字段顺序/类型不一致**只在运行时炸**，编译期不报错。改 tiling 结构时两侧同步改。
  （AgenticCANN 论文 §D.2 把这列为 LLM 生成代码的高频失败模式）
- **fp16/bf16 必须 Cast 到 fp32 计算**再 Cast 回去；vector 单元不直接算 bf16。
- **Hermitian 对角线虚部显式置零**（见 proven-patterns 套路 1）。
- **索引乘积用 int64**，防 BCSR/大 shape 溢出。
