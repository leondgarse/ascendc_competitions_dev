# 已验证实现套路（来自上游已合入代码）

这些不是推测，是从 `ops-blas` / `ops-sparse` **已合入 master 的代码**里读出来的。
遇到对应形态的算子，**照着改，不要从零发明**。

---

## 套路 1：复数（complex64）矩阵乘 → 拆实虚 + 4 次实数 GEMM

**出处**：`ops-blas/blas/hemm/arch22/`（chemm，已合入，A2，1357 行）
另见 `ops-blas/blas/herk/arch35/`（cherk，arch35 版，同思路）

Ascend 的 Cube/vector 单元**没有原生复数乘**。已验证做法是拆成实数矩阵：

```
输入 A(复) → A_r, A_i      输入 B(复) → B_r, B_i
4 次实数 GEMM：RR = A_r·B_r,  II = A_i·B_i,  RI = A_r·B_i,  IR = A_i·B_r
合并：C_real = RR - II,  C_imag = RI + IR
```

**Workspace 布局**（chemm 原样）：

```
A_r(M*K) + A_i(M*K) + B_r(K*N) + B_i(K*N) + RR(M*N) + II(M*N) + RI(M*N) + IR(M*N)
总计 (2MK + 2KN + 4MN) * sizeof(float)，32B 对齐
```

⚠️ 4 个 M×N 中间矩阵是**显著的显存开销**，大 shape 时要算清楚。

**三段式 kernel**（每段绑定计算单元）：

| kernel | KERNEL_TASK_TYPE | 职责 |
|---|---|---|
| `*_preprocess_kernel` | `KERNEL_TYPE_AIV_ONLY` | 拆实虚；Hermitian/对称三角展开成满阵 |
| `*_gemm_cube_kernel` | `KERNEL_TYPE_AIC_ONLY` | 4 次实数 GEMM |
| `*_gemm_aiv_fallback_kernel` | `KERNEL_TYPE_AIV_ONLY` | 小 shape / 非常规情况的 vector 兜底 |
| postprocess | AIV | alpha/beta 缩放、按 uplo 写回三角 |

**Hermitian 访问器**（chemm 的 `ChemmGetHermVal`）：
- `row == col` → 取实部，**虚部强制 0**
- 在存储三角内 → 直接取
- 在镜像侧 → 取转置位置并**对虚部取负**（共轭）

**cherk 的坑（README 明确写了）**：Hermitian 对角线虚部必须显式置零，
因为 `t3[diag]` 和 `t4[diag]` 来自两次独立 GEMM，浮点差值是 O(0.1~1.0)，不会自动抵消。

---

## 套路 2：Cube 单元编程（arch22）

**出处**：`chemm_kernel.cpp` 的 `ChemmBasicMmad`；`ops-sparse/sparse/cube_spmm/arch22/`

Ascend C 的 vector 类 skill（`ascendc-operator-*`）**不覆盖 Cube**。完整阶梯：

```cpp
// 1) 在各级 buffer 上建 LocalTensor（注意 TPosition）
LocalTensor<float> a1(TPosition::A1, 0U, aSizeAlignL1);
LocalTensor<float> b1(TPosition::B1, aSizeAlignL1 * sizeof(float), bSizeAlignL1);
LocalTensor<float> a2(TPosition::A2, 0U, aSizeAlignL0);
LocalTensor<float> b2(TPosition::B2, 0U, bSizeAlignL0);
LocalTensor<float> c1(TPosition::CO1, 0U, cSizeAlignL0);

// 2) L1 -> L0：LoadData（fp32 NZ->ZZ 用 LoadData3DParamsV2）
LoadData3DParamsV2<float> aParams; /* ... */ LoadData(a2, a1, aParams);

// 3) 矩阵乘累加
MmadParams params = {};
Mmad(c1, a2, b2, params);            // 首次
Mmad(cAcc, a2, b2, cAcc, params);    // 累加（K 方向分块时）

// 4) CO1 -> GM
FixpipeParamsV220 fp;
Fixpipe(output[mBase * n_ + nBase], c1, fp);
```

存储层级（A2/A3）：L1 512KB；L0A/L0B 各 64KB；L0C 128KB；UB 192KB。

⚠️ `LoadData` 的 `repeatTimes` 是 **uint8 → 最大 255**，必须分批：
```cpp
constexpr int64_t kMaxLoadDataRepeats = 255;   // cube_spmm 的做法
```

---

## 套路 3：稀疏 × 稠密 上 Cube → COO/CSR 转 BCSR 分块

**出处**：`ops-sparse/sparse/cube_spmm/arch22/`（1464 行，A2）

**为什么重要**：老的 `sparse/spmm/arch22/` 是纯 vector 实现——每个非零元做
`GetValue()` 读 GM + `Muls`/`Add`。这既违反 R1，也**打不过 GPU 基线**。
Cube 路线才是达标的方向。

做法：preprocess 把 COO 转成 **BCSR**（16×16 块），kernel 用 `Mmad` 吃块。

```cpp
constexpr int32_t kTileM = 16, kTileK = 16, kTileN = 16;
constexpr int32_t kNChunkSize = 512;   // N 方向分块，控制单核 buffer
```

**它的实测约束（照抄它的 `AreDimensionsSupported`）**：
- B 为 `ACL_FLOAT16`，C 为 `ACL_FLOAT`，computeType `ACL_FLOAT`
- **N 必须是 16 的倍数**，且 N ≤ 32752 —— 因为 `DataCopyParams.blockLen` 是 uint16
- M/K/N ≤ INT32_MAX；BCSR 块索引乘积用 **int64** 存，防溢出
- 越界返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` / `INVALID_VALUE`

---

## 套路 4：fp32 精度补偿（达不到精度标准时用）

**出处**：`chemm_kernel.cpp` 顶部

任务书的 fp32 精度门槛（如 rtol=2⁻¹⁰、atol=2⁻¹⁶，且每元素误差 ≤ max(A, 32×ULP)）
**朴素 fp32 累加往往过不了**。chemm 用 Dekker/Knuth 补偿算法：

```cpp
struct ChemmDoubleFloat { float high; float low; };

ChemmTwoSum(a, b)      // 无误差加法：返回 high + low
ChemmTwoProduct(a, b)  // 无误差乘法，splitter = 4097.0f
ChemmDfAdd(x, y)       // double-float 加
ChemmDfLinearCombination(a0,b0,a1,b1)  // a0*b0 + a1*b1，补偿精度
```

另见 `ops-sparse` 的 `ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG`：
用 **Kahan 补偿求和** 抑制长行累加的舍入/抵消误差，牺牲吞吐换精度。

**建议**：先用朴素实现跑精度测试；不达标再上补偿算法（有性能代价）。

---

## 套路 5：三段式稀疏接口（GetBufferSize / Preprocess / Compute）

**出处**：`ops-sparse/sparse/spmm/arch22/`

对标 cuSPARSE 的调用流程：
1. `*GetBufferSize` → 查 workspace 字节数
2. `*Preprocess` → 行重排 + 分桶 + 建 tiling，结果存 workspace，标记为 matA 的 active buffer
3. `*` (compute) → 下发 kernel，**异步**，调用方自己 `aclrtSynchronizeStream`

参数校验集中在一个函数里，非法组合返回明确错误码
（`ACL_SPARSE_STATUS_NOT_SUPPORTED` / `MATRIX_TYPE_NOT_SUPPORTED`）。
