# 阶段 0：任务竞争度核查

目的：**在申请 NPU 算力之前**，用纯本地/网页操作判断一个任务是否还值得做。
这一步成本几乎为零，跳过它的代价是几十小时算力 + 一个已被做完的 PR。

## 教训（真实案例）

某次核查只看了 `cann-ops-competitions` 的设计文档 PR 列表，得出
"chemm / csyrk 无人认领"的结论。实际情况是：

- `ops-blas` master 上 **`blas/hemm/arch22/` 已经合入**（chemm，1357 行，complex64，A2）
- 同一作者在 `ops-blas` 有 4 个进行中 PR：cherk/csyrk/cher2k/csymm 的 arch22 实现，
  每个 +3500~3900 行、46~58 条评审意见
- 另一作者还有一个"五算子打包"PR

**结论：代码仓的 PR 比设计文档 PR 领先数天到数周。只查设计文档仓会严重低估竞争度。**

第二次踩坑（同一类错误的另一面）：查 `cann-ops-competitions` 时**只看了 PR 列表**，
漏掉了已经合入 `tasklist/2025-08/08-aclblasCtbmv-A2A3/m0_69357246/docs/design.md`
——一份针对本任务的完整竞品设计文档。它不在任何 open PR 里，因为它已经合入了。

**结论二：PR 列表 ≠ 仓库内容。已合入的东西必须靠列目录树才能发现。**

## 核查清单

### 1. 目标仓 master 上是否已存在

```bash
git clone --depth 1 https://gitcode.com/cann/ops-blas.git      # BLAS 类
git clone --depth 1 https://gitcode.com/cann/ops-sparse.git    # 稀疏类

# 算子族目录是否存在，覆盖了哪些 arch
for op in hemm herk syrk symm her2k; do
  echo -n "$op: "; ls ops-blas/blas/$op/ 2>/dev/null | tr '\n' ' '; echo
done

# 公开头文件里接口是否已声明
grep -c "aclblasChemm\|aclsparseSpMM" ops-blas/include/cann_ops_blas.h
```

**关键**：`arch22` = Atlas A2/A3（dav-2201）；`arch35` = A5 / Ascend 950PR（dav-3510）。
**同名算子只有 arch35 ≠ 你的 arch22 任务已完成**，反之亦然。这是最容易误判的一点。

### 2. 两个仓的 PR 都要查

```bash
# 代码仓（更重要，领先于设计文档仓）
for p in 1 2 3 4 5 6; do
  curl -sL "https://gitcode.com/cann/ops-blas/pulls?state=all&page=$p" -o ob_$p.html
done

# 设计文档仓
curl -sL "https://gitcode.com/cann/cann-ops-competitions/pulls?state=all&page=1" -o comp_1.html
```

页面是 SSR 渲染，用去标签的方式抽标题：

```python
import re, html, glob
for f in sorted(glob.glob('ob_*.html')):
    s = open(f, encoding='utf-8').read()
    t = re.sub(r'<script.*?</script>', '', s, flags=re.S)
    t = re.sub(r'<[^>]+>', '\n', t)
    t = html.unescape(t)
    for l in (x.strip() for x in t.split('\n')):
        if re.search(r'(cherk|csyrk|csymm|chemm|spmm)', l, re.I) and 6 < len(l) < 160:
            print(f, '|', l)
```

注意分页：单页只有约 25 条。**必须翻多页**，否则会漏。
计数可从页面里 `opened` / `merged` 字段粗略取到。

### 2b. 已合入的设计文档目录树（最容易漏的一步）

**PR 列表只显示"还开着"的东西。已经合入的设计文档不在 PR 列表里，
但它同样代表一个正在做这个任务的对手。**

`cann-ops-competitions` 的设计文档合入后落在 `tasklist/` 目录树里，
路径形如 `tasklist/<期号>/<NN>-<算子名>-<arch>/<作者 ID>/docs/design.md`。
必须**直接列目录**，不能只看 PR：

```bash
git clone --depth 1 https://gitcode.com/cann/cann-ops-competitions.git
find cann-ops-competitions/tasklist -maxdepth 3 -type d | grep -i ctbmv
# tasklist/2025-08/08-aclblasCtbmv-A2A3/m0_69357246/
```

或者不 clone，直接翻网页目录：

```bash
curl -sL "https://gitcode.com/cann/cann-ops-competitions/tree/master/tasklist" -o tl.html
```

**每个作者 ID 子目录 = 一个已提交设计文档的竞争者。** 逐个读他们的
`docs/design.md`：里面通常有对方的切分策略、blockDim 选择、以及
**他们对性能标杆的解读**（见下）。

### 2c. 交叉核对性能标杆的解读

任务书给的标杆 CSV 往往只说"参考值"，没说是上界还是下界。
不同参与者可能有相反解读，例如：

- 解读 A：`目标 = csv / 0.8`（允许比参考值慢 25%）
- 解读 B：`目标 = csv * 0.8`（必须比参考值快 25%）

两者差 1.56 倍。**读对手的设计文档是确认解读的最便宜方式**——
如果所有人都按同一个式子写，那多半就是官方口径。
仍有分歧时，把两种解读都写进自己的设计文档并说明取了哪个、为什么。

### 3. 判读 PR 状态

标签含义：

| 标签 | 含义 |
|---|---|
| `ci-pipeline-passed` | CI 过了，作者是认真的 |
| `approved` / `lgtm` | 已获评审通过，接近合入 |
| `存在冲突` | 有冲突，可能停滞 |
| `wait-feedback` | 等反馈，可能停滞 |
| `stat/needs-squash` | 只差 squash，非常接近合入 |
| `ai-co-authored` | 对方也在用 AI 协作 |

**竞争度分档**：
- 🔴 已合入 master → 换任务
- 🔴 PR 已 approved / needs-squash → 基本无戏
- 🟡 PR 大但有冲突 / wait-feedback → 有机会，但要能做得更好
- 🟢 无 PR 且 master 无实现 → 可做（仍需确认可报名）

### 4. 报名状态（agent 查不到）

社区任务 IT 系统的报名/认领状态**无法从仓库推断**。
必须让用户去问 CANN 小助手或看讨论帖。**永远不要断言"这个任务可以报名"。**

## 产出格式

```
## 竞争度报告
| 算子 | master 状态 | 代码仓 PR | 设计文档 PR | 已合入 tasklist/ | 结论 |
|---|---|---|---|---|---|
| chemm | arch22 已合入 | — | — | — | 🔴 放弃 |
| csyrk | 仅 arch35 | #348 +3772 ci-passed | 有 | m0_xxx | 🔴 高风险 |
...
推荐：<任务>，理由：<...>
待用户确认：报名是否仍开放
性能标杆解读：csv/0.8 还是 csv*0.8（与竞品文档核对过？）
```
