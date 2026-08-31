# cann-comp-skills

面向 **CANN 社区算子任务（cann-ops-competitions）** 的自建 skill 集。

## 用法

放到 Claude Code 能发现 skill 的位置（如项目 `.claude/skills/` 或
`~/.claude/skills/`），然后：

```
/cann-competition-workflow
```

或直接描述需求（"我要做这个 CANN 社区任务"、"分析这份任务书"），
skill 的 description 会触发加载。

## 内容

```
skills/cann-competition-workflow/
├── SKILL.md                        # 主流程：阶段 0-7
└── references/
    ├── task-recon.md               # 阶段 0：竞争度核查（最省成本的一步）
    ├── proven-patterns.md          # 已验证实现套路（复数GEMM/Cube/BCSR/精度补偿）
    ├── hard-constraints.md         # R1-R10 + API 限制 + 实测经验
    ├── design-doc-guide.md         # 设计文档写作与提交
    └── delivery-checklist.md       # 验收交付件逐项清单
```

## 与其他 skill 的关系

**这套 skill 不重复造轮子**，它编排三层已有资产：

| 层 | 来源 | 角色 |
|---|---|---|
| 上游仓 skill | `ops-blas/agent/skills/`、`ops-sparse/agent/skills/` | **评审依据，权威**。编码/测试/构建阶段必须加载 |
| 本地通用 skill | `agent-skills/skills/ascendc-operator-*` | Ascend C 通用开发（vector 类为主）、tiling、精度/性能评测模板 |
| 本 skill | `cann-comp-skills` | 竞赛特有的流程、选型、跨仓协调、验收 |

**冲突时权威性**：上游仓 rules > 本地 agent-skills > 论文/推测。

## 知识来源

- `ops-blas` / `ops-sparse` 已合入代码（chemm arch22、cube_spmm arch22、spmm arch22、
  cherk arch35、spgemm arch35）
- 两仓自带 `agent/skills/repo-*`（R1-R10、MR 安全规则、8 步 checklist、build/test 模板）
- `cann-ops-competitions` 已合入设计文档与 PR 惯例
- 论文 *AgenticCANN*（arXiv 2607.26661v1）：知识分层注入（L0-L5）、
  阶段自适应 agent 模式、host-kernel TilingData 失配等失败模式

## 维护

每次做完一个任务，把新踩到的坑回写进 `references/`——尤其是
`proven-patterns.md`（新套路）和 `hard-constraints.md`（新限制）。
