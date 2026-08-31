# aclblasCtbmv 算子开发任务书

## 1. 任务概述

在昇腾 NPU（Atlas A2/A3 系列产品）上使用 Ascend C 编程语言开发单精度复数（complex64）三角带状矩阵-向量乘算子 `aclblasCtbmv`，完成算子设计、开发、测试全流程工作。验收通过后合入昇腾算子开源仓（请参考链接：https://gitcode.com/cann/ops-blas ）。

## 2. 核心开发要求

### 2.1 功能实现要求

1. 与 cuBLAS `cublasCtbmv` 核心功能、参数语义完全对齐，计算 `x = op(A) * x`（原地覆写 x），其中 A 为 n×n 三角带状单精度复数矩阵：
   - `trans = ACLBLAS_OP_N` 时 `op(A) = A`；
   - `trans = ACLBLAS_OP_T` 时 `op(A) = Aᵀ`（不取共轭）；
   - `trans = ACLBLAS_OP_C` 时 `op(A) = Aᴴ`（共轭转置，复数路径专有语义）。
2. A 按带状格式（banded storage）列主序存储于 lda×n 数组，仅带内元素被引用：
   - `uplo = ACLBLAS_LOWER`：主对角存于第 1 行，第 i 条次对角存于第 i+1 行，元素 A(i,j) 位于 A(1+i-j, j)（1-based）；数组右下角 k×k 三角区域不被引用；
   - `uplo = ACLBLAS_UPPER`：主对角存于第 k+1 行，元素 A(i,j) 位于 A(1+k+i-j, j)（1-based）；数组左上角 k×k 三角区域不被引用。
3. `diag = ACLBLAS_UNIT` 时 A 的对角元素不被读取，视为 (1, 0)；`diag = ACLBLAS_NON_UNIT` 时读取实际对角元素。
4. k 为半带宽（sub/super-diagonals 条数），取值 k ∈ [0, n-1]；k = 0 时 A 退化为对角矩阵；k ≥ n-1 时带内覆盖整个三角矩阵。
5. 复数类型 `aclblasComplex` 以 ops-blas 仓 `include/cann_ops_blas_common.h` 中定义为准（实部/虚部各 float32）。
6. `n = 0` 时为合法 no-op（直接返回 `ACLBLAS_STATUS_SUCCESS`，不执行计算；语义依据 Netlib `ctbmv` 参考实现，请参考链接：https://www.netlib.org/blas/ctbmv.f ）。
7. 负步长 incx < 0 时按 Netlib 语义从向量末端反向遍历（起始偏移 kx = 1-(n-1)*incx）。
8. 本算子无 alpha/beta 标量参数，无确定性计算要求。
9. 对标基线接口：cuBLAS `cublasCtbmv`（语义参考 Netlib `ctbmv`，请参考链接：https://www.netlib.org/blas/ctbmv.f ）。
10. 接口声明放入 `include/cann_ops_blas.h`，禁止定义 Atlas A2/A3 产品私有平行接口。

### 2.2 算子工程模式

使用 Ascend C kernel 直调方式开发：基于 ops-blas 开源仓（请参考链接：https://gitcode.com/cann/ops-blas ）工程框架，实现 `aclblasCtbmv` 句柄式 BLAS 接口，通过 handle 绑定 stream 直调 NPU kernel，实现代码放在 `blas/tbmv/arch22/`（Atlas A2/A3 对应架构目录，与现有 `aclblasStbmv` 实现同族目录）。

### 2.3 接口定义

ops-blas 仓 `include/cann_ops_blas.h` 当前**无 `aclblasCtbmv` 声明**（complex 路径为新增接口，头文件需新增该声明）。新增声明须与 cuBLAS `cublasCtbmv` 参数序列一致、与仓内 `aclblasStbmv` 声明同型（float→aclblasComplex），定义如下：

```cpp
aclblasStatus_t aclblasCtbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx);
```

### 2.4 参数说明

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| handle | 输入 | ops-blas 库上下文句柄，携带 stream，Host 内存 | scalar | - | - | - | 指向已创建的有效句柄 | handle 为 nullptr 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | 输入 | A 矩阵存储模式：ACLBLAS_UPPER（上三角带状）或 ACLBLAS_LOWER（下三角带状），Host 内存 | attr | int（枚举） | - | - | {ACLBLAS_UPPER, ACLBLAS_LOWER} | 取值不在上述枚举时返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| trans | 输入 | 矩阵操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，复数路径取共轭），Host 内存 | attr | int（枚举） | - | - | {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C} | 取值不在上述枚举时返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| diag | 输入 | 对角线类型：ACLBLAS_NON_UNIT（非单位对角）或 ACLBLAS_UNIT（单位对角，对角元素不读、视为 1），Host 内存 | attr | int（枚举） | - | - | {ACLBLAS_NON_UNIT, ACLBLAS_UNIT} | 取值不在上述枚举时返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| n | 输入 | 三角带状矩阵 A 的行数和列数，Host 内存 | scalar | int | - | - | n ≥ 0 | n < 0 时返回 `ACLBLAS_STATUS_INVALID_VALUE`；n = 0 为合法 no-op |
| k | 输入 | 三角带状矩阵的半带宽，Host 内存 | scalar | int | - | - | k ≥ 0（有效范围 k ∈ [0, n-1]，k ≥ n-1 时带内覆盖整个三角矩阵） | k < 0 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| A | 输入 | 三角带状矩阵复数数组，按带状格式列主序存储，Device 内存，只读；带外及未映射三角区域不被引用 | tensor | COMPLEX64 | ND（列主序带状） | lda × n | 实部/虚部取值于 FLOAT32 全集 | n > 0 时 A 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| lda | 输入 | 矩阵 A 带状存储的前导维度，Host 内存 | scalar | int | - | - | lda ≥ k + 1 | lda < k + 1 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| x | 输出（原地输入/输出） | 复数向量，包含 n 个元素；输入为原始向量，输出为计算结果（原地覆盖），Device 内存 | tensor | COMPLEX64 | ND（步长 incx） | n | 实部/虚部取值于 FLOAT32 全集 | n > 0 时 x 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | 输入 | x 中连续元素之间的步长，Host 内存；支持负步长（反向遍历） | scalar | int | - | - | incx ≠ 0 | incx = 0 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |

**返回值**：`aclblasStatus_t`，状态码语义与 ops-blas 仓 `include/cann_ops_blas_common.h` 定义一致（`ACLBLAS_STATUS_SUCCESS`=0 / `ACLBLAS_STATUS_INVALID_VALUE`=3 / `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`=9 / `ACLBLAS_STATUS_INVALID_ENUM`=10）。

### 2.5 算子实现约束

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | n ≥ 0；k ≥ 0；lda ≥ k + 1；incx ≠ 0；A、x 在 n > 0 时不可为 nullptr；非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`，非法枚举（uplo/trans/diag）返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| 三角带状引用 | 仅引用 uplo 指定三角的带内元素；UPPER 的左上 k×k、LOWER 的右下 k×k 未映射区域不得读取；diag=UNIT 时不得读取对角元素 |
| 非连续 Tensor 支持 | 不要求（本批次不支持超出 lda/incx 语义的非连续内存访问） |
| broadcast 规则 | 不涉及，A/x 为独立操作数，无广播 |
| dynamic shape 要求 | 不要求，n/k 为运行时入参 |
| 原地与视图语义 | x 原地覆写（输入即输出），不返回视图 |
| 确定性计算要求 | 不要求 |
| 空 Tensor 与 0 维处理 | n = 0 为合法 no-op，返回成功且不执行计算 |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream |

## 3. 验收标准

### 3.1 软硬件环境要求

- **适配硬件**：Atlas A2 系列产品（性能测试设备：Atlas 800T A2 (910B3)）
- **CANN 版本**：CANN 9.1.0
- **三方软件版本**：精度比对 golden 由 cblas（Netlib BLAS 复数实现）生成，随测试工程提供，无其他三方软件依赖

### 3.2 精度要求

1. golden 由 cblas（Netlib `ctbmv`）单标杆比对生成，输出向量 x（n 元素）全量验证，实部、虚部分别比对。
2. 算子计算精度需满足生态算子开源精度标准（请参考链接：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ），本算子输入输出为单精度复数，比对时实部、虚部分别按 FLOAT32 标准判定：

    | 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
    |----------|------|------|------------------------|---------------------|
    | COMPLEX64（实部/虚部按 FLOAT32 分量） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |

    逐元素通过条件：`|actual - golden| ≤ atol + rtol × |golden|`；当用例同时满足 matched_ratio ≥ required_matched_ratio 且 max_abs_error ≤ max_abs_error_limit 时，判定该用例精度通过。

**补充说明**：

- 本算子为浮点矩阵-向量乘累加运算，非 bit-exact；输入矩阵 A 按三角带状约束生成（仅带内元素有效，diag=UNIT 用例对角不参与比对）。
- 本算子不含随机数生成，§3.5 的正态/均匀分布是测试输入数据的生成规则，不是算子行为。

### 3.3 性能要求

1. 测试设备：Atlas 800I A2（910B3）。性能数据为 COMPLEX64 输入场景下的平均单次耗时（Avg time，单位 us），须先 warmup 再有效采样 >50 次取平均。
2. 算子在各性能 case 下的平均单次耗时应不高于下表标杆耗时：

| case | n | k | uplo | trans | diag | incx | 标杆耗时（Avg time，us） |
|---|---|---|---|---|---|---|---|
| 1 | 512 | 8 | UPPER | N | NON_UNIT | 1 | 10 |
| 2 | 1024 | 16 | UPPER | N | NON_UNIT | 1 | 16.51 |
| 3 | 2048 | 32 | LOWER | T | NON_UNIT | 1 | 19.77 |

3. 更多参考性能用例及测试指导见测试用例目录。

### 3.4 内存要求
不涉及。

### 3.5 自验要求

1. 测试工具与方法：使用 ops-blas 仓 test 目录的测试框架完成自验——测试用例以 CSV 文件描述，提供的python 脚本调用C++ GTest 工程加载 CSV 调用 `aclblasCtbmv` 接口执行，精度 golden 由 cblas（Netlib BLAS 复数实现）生成。请根据随任务提供的自测用例与测试指导完成自测，并输出自测报告。
2. 本算子参数序列与对标接口 `cublasCtbmv` 的参数序列一致（handle 及参数顺序一一对应，维数参数为 int），无需额外映射说明。
3. 测试用例入参生成规则（基于 §2.4 参数说明）：

| 参数名 | Tensor 值域分布 | Attr 覆盖规则 |
|---|---|---|
| handle | - | 固定为已创建的有效句柄，不随机生成 |
| uplo | - | ACLBLAS_UPPER / ACLBLAS_LOWER 全覆盖，与 trans、diag 正交组合（2×3×2 = 12 组合全覆盖） |
| trans | - | ACLBLAS_OP_N / OP_T / OP_C 全覆盖（含复数共轭转置路径） |
| diag | - | ACLBLAS_NON_UNIT / ACLBLAS_UNIT 全覆盖 |
| n | - | 覆盖 0、1、小质数、2 的幂及 2 的幂 ±1、非对齐值，直至大规模 |
| k | - | 覆盖 0、1、小值、k = n-1（满带）、中间值 |
| A | 均匀分布 [-5, 5] 占 50%、正态分布（μ∈[-5,5]，σ∈[0.1,2]）占 50%，实部/虚部独立采样；含 Inf/NaN/极端值特殊用例 | - |
| lda | - | 覆盖等于最小约束值（k+1）的紧凑场景及多个 padding 场景 |
| x | 均匀/正态各 50%，实部/虚部独立采样；含 Inf/NaN 特殊值用例 | - |
| incx | - | 覆盖 ±1/±2/±3 步长 |

4. 自验用例须覆盖：小shape基础用例、shape扫描、填充模式、对齐偏移、边界与负向用例（如零维、空指针 x/y、非法步长、负维度等）、规格允许的INF/NAN场景，以及性能/内存用例。如果任务配套提供的用例没有覆盖要求的场景需要自行补充相应的用例，边界、负向用例及特殊值用例的行为对齐cublas。

## 4. 验收交付件

在社区任务IT系统中提交验收时，需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ；<br> 2. 在 cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以 PR 形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md |
| 2 | 自测用例及测试代码 | 1. 覆盖随任务提供的全部自测用例，需要清晰列出精度测试 case 和性能测试 case；<br> 2. 测试代码中的 readme 文件需要说明测试步骤，保证验收人可以复现测试结果 |
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图（实部/虚部分别）、性能数据及截图、内存占用数据 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号 Ascend-CANN 作为开发者；<br> 2. 需要根据 ops-blas 仓库规范提供算子 readme 文档；README 产品支持表标注 Atlas A2/A3 系列产品：支持；接口声明放入 `include/cann_ops_blas.h`，供其他产品线共用 |

## 5. PR 申请合入

测试通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入该目录：https://gitcode.com/cann/ops-blas （目录 `blas/tbmv/arch22/`）；测试代码合入该目录：https://gitcode.com/cann/ops-blas （目录 `test/tbmv/ctbmv/arch22/`），文件结构参考主仓blas算子测试代码结构（包含csv文件）。

## 6. 参考资料

1. Ascend C算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html ；
2. 算子开发接口文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html ；
3. Ascend C在线课程：https://www.hiascend.com/developer/courses/detail/1691696509765107713 ；
4. ops-blas 开源仓：https://gitcode.com/cann/ops-blas ；
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ；
6. cuBLAS 参考文档（cublasCtbmv）：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-tbmv ；
7. Netlib BLAS 参考实现（ctbmv）：https://www.netlib.org/blas/ctbmv.f 。

## 7. 特别注意事项

1. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
2. 开发前请务必阅读【社区任务】流程及注意事项：https://gitcode.com/org/cann/discussions/39 ；
3. 接口签名须可与其他产品线共用（与仓内 `aclblasStbmv` 同型），禁止定义 Atlas A2/A3 产品私有平行 API；
4. 性能测试须先 warmup 再有效采样 >50 次取平均。

## 8. 环境获取（无需修改，使用模板原始内容）

1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend 。

- **【补充说明】填写示例：本人gitcode账号是 yolo，现在参与社区任务"7月社区任务-aclnnRoll算子开发"，需要申请A2/A3算力进行任务开发。**

  ![环境截图](./pics/zaixiankaifa1.png)
  ![环境截图](./pics/apply.png)

2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](./pics/yunkaifa.png)
3. 如需额外环境资源，请联系昇腾CANN小助手。
