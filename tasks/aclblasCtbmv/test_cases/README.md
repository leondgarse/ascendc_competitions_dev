# ctbmv 测试用例及测试指导参考（Atlas 800I A2/A3 版）

> **任务书**: ./aclblasCtbmv任务书
> **适配硬件**: Atlas A2/A3 系列产品（性能测试设备 Atlas 800I A2 (910B3)，对应 arch22）
> **CANN 版本**: 9.1.0
> **数据类型**: complex64（单精度复数）
> **内存预算**: 单用例 host 侧 ≤ 512MB（设计选择，非硬件限制）
> **复用说明**: 用例集与 950 版（output/batch3/aclblasCtbmv/）一致（同种子同用例，CSV 逐字节相同），仅 SOC 映射/设备说明按 A2/A3（ascend910b3/arch22）调整

## 算子说明

**接口**: `aclblasCtbmv`（三角带状矩阵-向量乘，带状格式列主序存储）
**公式**: `x = op(A) * x`（x 原地覆写；A 为 n×n 三角带状矩阵）
**枚举**: uplo=UPPER/LOWER，trans=N/T/C（复数下 C 为共轭转置），diag=NON_UNIT/UNIT（UNIT 时对角不读、视为 1）
**标量**: 无 alpha/beta
**维度**: A: lda×n（带状存储，lda ≥ k+1），x: n（步长 incx ≠ 0，支持负步长）
**带宽**: k ∈ [0, n-1]；k=0 退化为对角矩阵；k=n-1 为满带（覆盖整个三角矩阵）

## 用例规模

`gen_csv.py` 使用固定随机种子生成 **1200 条**用例（条数可扩展）：

- 精度：1000 条，覆盖 uplo×trans×diag 全枚举组合（2×3×2=12）× 尺寸扫描 × 带宽扫描 × 前导维/步长/填充/边界
- 性能/内存：200 条（TC_PF 前缀），含 4 条与任务书 §3.3 精确匹配的典型 case，其余按 512MB 内存预算分布（含小尺寸 1~512），均为连续访存（incx=1、lda=k+1 紧凑）

尺寸范围 [1, 2048]（精度）/ [1, 4096]（性能），带宽覆盖 0/1/小值/半带/满带 n-1，尺寸含奇数（3/5/7/65）、边界值（1）、非对齐值、大尺寸（2048/4096）。

生成/重新生成用例（条数可灵活扩展）：

```bash
python gen_csv.py                            # 默认 1000 精度 + 200 性能
python gen_csv.py --accuracy 1500 --perf 300 # 扩展条数
python gen_csv.py --seed 12345               # 更换随机种子
```

生成文件：

- `ctbmv_test.csv`：1200 条 CSV 驱动用例，ops-blas C++ GTest 直接加载。
- `gpu_baseline.csv`：200 条性能/内存基线占位（`gpu_ms` 列待基线测试后回填，前 4 条与任务书 §3.3 典型 case 对应）。

## 用例分类

| 类别 | 前缀 | 条数 | 说明 |
|------|------|------|------|
| L0 基础 | TC_L0 | 12 | uplo×trans×diag 全枚举组合（12）× 小尺寸 (n=8, k=2) |
| L1 尺寸 | TC_SQ | 92 | 23 种尺寸（1→2048，含奇数/边界/非对齐）× 枚举组合轮转 |
| L2 带宽 | TC_BD | 22 | k=0/1/小值/半带/满带 n-1 × n∈{4,8,16,32,64} × 枚举轮转 |
| L4 前导维/步长 | TC_LD/TC_INC | 18 | lda=k+1+padding 场景；incx ∈ ±1/±2/±3（含负步长） |
| L5 填充 | TC_FL | 14 | 均匀/全零/交替/极端值/Inf/NaN；含 diag=UNIT 对角 NaN/Inf 不读验证 |
| L5b 覆盖 | TC_CV | 24 | 中等尺寸（24/96）× uplo×trans×diag 全组合 |
| L6 边界 | TC_ED | 10 | n=0 quick return(SUCCESS)；空指针/非法前导维/负维度/负带宽/零步长(INVALID_VALUE)；非法枚举 999(INVALID_ENUM) |
| EX 扩展 | TC_EX | 808 | 尺寸池×带宽×枚举组合×步长×padding 的确定性采样（扩展精度条数的主力类别） |
| PF 性能 | TC_PF | 200 | 4 条任务书典型 case + 小尺寸延迟区 + 规模对数扫描 + 带宽网格（含满带）+ 枚举组合性能 + 预算内混合（均为连续访存） |

> 扩展条数时，固定类别（L0~L6）保持全集覆盖，TC_EX 与 TC_PF 的补足部分随 `--accuracy` / `--perf` 自动伸缩。

## 精度验收

`verify_accuracy.py` 调用 ops-blas 仓 `build.sh` 编译算子测试，执行 C++ GTest 二进制，解析逐条 PASS/FAIL（自动排除 TC_PF_ 前缀的性能用例）：

```bash
# 前置：将 ctbmv_test.csv 安装到仓内测试目录 test/tbmv/ctbmv/<arch>/（--csv 自动复制并备份原文件）
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --csv ./ctbmv_test.csv
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --skip-build --device 1
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --filter TC_L0 --timeout 3600
```

### 精度阈值

遵循生态算子开源精度标准（complex64 实部/虚部分别按 FLOAT32 判定）：

- atol = 2⁻¹⁶ ≈ 1.5259e-5，rtol = 2⁻¹⁰ ≈ 9.7656e-4
- matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32×ULP

参考: https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md

## 性能验收

`verify_performance.py` 执行 TC_PF 用例采集 NPU 耗时，与 `gpu_baseline.csv` 关联比对（按 n/k/uplo/trans/diag 五元组匹配）：

```bash
python verify_performance.py --repo /path/to/ops-blas --soc ascend910b3 --timeout 3600
python verify_performance.py --repo /path/to/ops-blas --soc ascend910b3 --skip-build --device 1
```

- **判定口径**：NPU 平均单次耗时 ≤ 标杆耗时 / 倍率（标杆耗时为`gpu_baseline.csv` 的 `gpu_ms` 列）；基线回填前全部标记 NO_REF，仅采集 NPU 耗时。
- 性能测试须先 warmup 再有效采样 >50 次取平均（由测试工程执行）。
- 注：GTest 输出耗时含 host 准备 + kernel + golden 计算 + 比对，为保守上界；精确 kernel 耗时可配合 msprof 采集。

## 补充说明
- 代码上库时需要同步提交测试工程代码，可能需要新增、复用或者改造当前库上test目录下测试工程，详细要求以仓库最新代码规范和贡献指南要求为准

- 部分特殊场景下自动生成的case可能导致标杆出现异常行为使测试行为无意义，此时可根据实际场景过滤掉或者修改这些case并给出相应的说明
