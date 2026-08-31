---
name: cann-competition-workflow
description: |
  端到端交付一个 CANN 社区算子任务（cann-ops-competitions 社区任务）：从任务选型、
  可行性核查、设计文档 PR，到 ops-blas / ops-sparse 的代码 + 测试 + 自测报告 PR。
  触发：用户提到 CANN 社区任务、算子竞赛、任务书（task_doc）、aclblas*/aclsparse* 算子开发、
  设计文档评审、提交算子 PR、验收交付件时加载。
---

# CANN 社区算子任务交付流程

本 skill 编排一个社区算子任务的完整交付。它**不替代**上游仓自带的
`agent/skills/`（repo-coding-rules / repo-build-guide / repo-test-develop /
repo-op-templates / repo-knowledge）——那些是评审依据，**必须**在对应阶段加载。
本 skill 负责的是它们之间的**流程、选型与验收**。

## 核心事实（务必先读，避免走错仓）

| 交付件 | 去向 | 说明 |
|---|---|---|
| 设计文档 | `cann-ops-competitions` PR | **仅 markdown**；历史 500+ 合入全部是设计文档 |
| 算子代码+UT | `ops-blas` 或 `ops-sparse` 的 `master` | BLAS 类→ops-blas；稀疏类→ops-sparse |
| 自测报告 | 腾讯文档表格（任务书给链接） | 非仓库 |
| 待验收代码 | **个人 fork**，邀请 `Ascend-CANN` 为开发者 | 不是主仓分支 |

**顺序是硬性的：设计文档 PR 先过评审，再写代码。** 不要跳过。

## 阶段 0：任务选型与竞争度核查（MANDATORY，最省成本的一步）

多数失败来自"选了一个已被做完的任务"。**在任何编码或申请算力之前**执行
`references/task-recon.md` 的核查流程，产出一份结论：

- [ ] 目标算子在目标仓 `master` 上**是否已存在**（`ls <repo>/<族>/<op>/arch*`）
- [ ] 目标 arch 是否已覆盖（`arch22`=A2/A3，`arch35`=A5/950）——**同名不同 arch 不算已完成**
- [ ] 目标仓是否有**进行中的 PR**（competitions 仓的设计文档 PR ≠ 代码仓 PR，两个都要查）
- [ ] 任务在社区任务 IT 系统是否**仍可报名**（agent 无法确认 → 必须让用户去问）

> ⚠️ 已知教训：只查 competitions 仓的设计文档会**严重低估**竞争度。
> 代码仓 PR 往往远早于设计文档 PR。两个仓都要查。

**输出**：`竞争度报告`（已完成/在做/空白 + 推荐任务 + 理由）。**由用户拍板**，不要自作主张开工。

### 0.5 性能可达性预检（同样在开工前）

竞争度之外，还要判断**标杆是否物理可达**。执行
`references/perf-reality-check.md`：算固定开销占比、实测平台 dispatch 地板、
量一个同类已合入算子、确认标杆的架构来源。

> ⚠️ 真实教训：我们完整做完一个算子（功能 1872/1872 通过）后才发现，标杆
> （10us）低于平台 kernel 下发开销（~14.4us），无论怎么优化都不可能达标。
> 这个预检一小时内就能得出同样结论。

## 阶段 1：任务书解析

读任务书（`*_task_doc.md`），抽取成一张结构化表：

- 算子数学定义、参数表、dtype、layout、约束
- 目标硬件（A2/A3 → arch22；A5/950 → arch35）
- **精度标准**（rtol/atol/A 值、golden 用什么精度算）
- **性能标准**（对标 GPU 基线、倍率门槛、预热/采样次数）
- 验收交付件清单

同时读任务书自带的 `*_testCase/` 目录——用例、基线 CSV、测试脚本通常**已经给好**，
不要自己造。

## 阶段 2：可行性核查（对照已有实现）

**MANDATORY**：先在目标仓找**最接近的已有实现**当作参照物，再判断难度。
参考 `references/proven-patterns.md`（复数 GEMM、Cube SpMM、精度补偿等已验证套路）。

判断三件事：
1. 数学形态 → 落到哪个已知 pattern（vector / Cube GEMM / 复数拆分 / 稀疏分块）
2. 与现有 baseline 的 **gap 清单**（dtype、layout、arch、算法枚举、索引）
3. 性能门槛能否够到（vector-only 打不过 Cube；见 proven-patterns.md）

**输出**：可行性结论 + gap 清单 + 工作量分档（小/中/大）。

## 阶段 3：设计文档 → PR 到 competitions 仓

模板：`cann-ops-competitions/04_tasks/01_community-task-2026/resources/design_template.md`
（章节：需求背景 / 需求分析 / 方案设计 / Tiling / 精度与性能 / 测试方案 / 交付计划）

写作要点见 `references/design-doc-guide.md`。提交路径：
`04_tasks/01_community-task-2026/tasklist/<任务名>/<你的用户名>/docs/design.md`

PR 标题沿用社区惯例：`【社区任务】<算子名>算子设计文档`

**门控**：评审通过（`approved` / `lgtm`）后才进入阶段 4。

## 阶段 4：实现（加载上游仓 skill，不要自创规范）

**MANDATORY 顺序**：
1. 加载目标仓 `agent/skills/repo-op-templates` → 目录骨架与命名
2. 加载 `agent/skills/repo-knowledge` → 领域规则
3. 写 host + kernel，全程对照 `references/arch22-cube-cookbook.md`（若用 Cube）
4. 加载 `agent/skills/repo-coding-rules` → **按 `checklist.md` 8 步自查**
5. 加载 `agent/skills/repo-build-guide` → `bash build.sh --ops=<op> --soc=<soc> [--run]`

硬约束速记（违反直接被打回，详见 `references/hard-constraints.md`）：
- **R1 禁止逐元素 SetValue/GetValue/DataCopy**（最常见退回原因）
- **R2 动态获取 CoreNum**，禁止硬编码
- **R3 TPipe 禁止作成员变量**
- **R4 TilingData 禁止用数组做核间分配**
- R5-R10 圈复杂度≤20 / 嵌套≤5 / NBNC≤50 / 除零防御 / 许可证头 / 禁 extern
- `repeatTime` 是 uint8 → **≤255**，超出静默截断为 0（不报错、不计算）
- GM↔UB 用 `DataCopyPad`；`DataCopyParams.blockLen` 是 uint16

## 阶段 5：测试

加载目标仓 `agent/skills/repo-test-develop`（含 `op_golden.h` / `op_param.h` /
`op_npu_wrapper.h` 模板）。优先复用任务书 `*_testCase/` 已给的用例与基线。

覆盖任务书"自验证用例"表的每一类；精度按任务书 dtype 参数表逐项对齐
（golden 用更高精度算：fp16/bf16→fp32，fp32→fp64，complex64→complex128）。

## 阶段 6：性能采集

按任务书口径（通常：预热≥10、正式≥30、报中位数与 P90、设备同步后计时、
不含 H2D 与首次编译）。NPU 侧用 `msprof` / `torch_npu.profiler` 出 Kernel 总耗时，
并保留 **Profiler 证据**证明未回退 CPU。

## 阶段 7：提交

- 代码 PR → 目标仓 `master`（个人 fork + 邀请 `Ascend-CANN`）
- 提交前跑 `sh scripts/oat_check.sh <变更文件>`
- 自测报告填腾讯文档模板，含用例参数/精度截图/性能数据/Profiler 证据/失败项说明

## 反模式

- ❌ 跳过阶段 0 直接开工（最贵的错误）
- ❌ 只查 competitions 仓就断定"无人认领"
- ❌ 设计文档没过评审就写代码
- ❌ 自创编码规范而不加载上游 `agent/skills/repo-coding-rules`
- ❌ 用 vector 逐元素循环实现矩阵乘类算子，然后指望达标性能
- ❌ 自造测试用例而无视任务书 `*_testCase/`
- ❌ 不做性能可达性预检就开工（最贵的错误，比选错任务更贵）
- ❌ 把跨步访问写成大量小 burst 的 DMA（见 hard-constraints 三点六）
- ❌ 在向量指令里用任意偏移 `dst[i]`（见 hard-constraints 三点五）
- ❌ 声称任务"可报名"——agent 查不到，必须让用户确认
