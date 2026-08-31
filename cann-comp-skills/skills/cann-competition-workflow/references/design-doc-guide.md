# 设计文档写作与提交

## 提交位置

仓库：`https://gitcode.com/cann/cann-ops-competitions`
路径：`04_tasks/01_community-task-2026/tasklist/<任务名>/<用户名>/docs/design.md`
模板：`04_tasks/01_community-task-2026/resources/design_template.md`
PR 标题：`【社区任务】<算子名>算子设计文档`（沿用社区惯例）

> 注意：部分任务书写 `cann-competitions`（无 `-ops-`），部分写 `cann-ops-competitions`。
> **两者是同一个仓库**（gitcode 上互为别名/重定向，`git ls-remote` 返回相同 HEAD）。
> 任一 URL 均可访问，不必纠结；推荐统一用 `cann-ops-competitions`。

## 章节骨架（模板 + 社区已合入文档的共性）

1. **需求背景**：需求来源（哪个月社区任务）、背景、**现状分析**
2. **需求分析**：需求描述、需求拆解、输入输出规格表
3. **方案设计**：实现路径选型、算法描述、kernel 划分
4. **Tiling 设计**：核间切分 + 核内切分、UB/L1 分配表、tiling 结构体字段
5. **精度方案**：golden 怎么算、容差参数、是否需要补偿算法
6. **性能方案**：对标基线、预期倍率、优化点
7. **测试方案**：用例分类与覆盖
8. **交付计划**：文件清单、支持的产品型号表

## 写作要点（从已合入文档总结）

**现状分析要诚实且具体**。参考 SpGEMM 设计文档的写法：

> `ops-sparse` 仓库中已有 SpMV、SpMM、Snnz 等算子，**无 `src/spgemm/`**；
> SpGEMM 为全新算子，接口与 Kernel 均需从零交付。

即：明确说清"已有什么 / 缺什么 / 本任务补什么"。评审看的就是这个 gap 判断准不准。

**明确产品型号支持表**：

| 产品 | 支持 |
|---|---|
| Ascend 950PR (dav-3510 / arch35) | √ |
| Atlas A2 (dav-2201 / arch22) | 暂不支持 |

**给出编译命令**：`bash build.sh --ops=<op> --soc=<soc> [--run]`

**不要在设计文档里贴大段代码**（catlass 系 skill 明确禁止；competitions 的
文档也以表格+说明为主）。用**选型表格**和伪代码描述。

## 复用本地资产

- 本地 `agent-skills/skills/ascendc-operator-design/templates/design-template.md`
  提供 Tiling/UB 分配表的写法，可以借结构，但**章节名以 competitions 模板为准**
- Tiling 参考：`ascendc-operator-design/references/*-tiling.md`（elementwise/reduction/
  index/sort/pooling + general-tiling-principles）

## 门控

PR 拿到 `approved` / `lgtm` 后再进入编码阶段。CI 标签 `ci-pipeline-passed`
只代表格式检查过了，不等于评审通过。
