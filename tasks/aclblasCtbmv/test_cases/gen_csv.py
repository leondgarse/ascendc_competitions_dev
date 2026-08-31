#!/usr/bin/env python3
"""aclblasCtbmv（Atlas 800I A2/A3, complex64 三角带状矩阵-向量乘 x = op(A)*x）用例生成器

复用说明：用例集与 950 版（output/batch3/aclblasCtbmv/）一致（同种子同用例，输出 CSV 逐字节相同），
仅本说明与 SOC 映射按 A2/A3（ascend910b3/arch22）调整。

列格式（以同族 stbsv/stbmv arch35 CSV 工程列约定为基础，增补 a_fill/x_fill 两列用于
表达 NULLPTR 空指针负向用例与特殊值填充；语法按 test/frame/fill.h 约定）：
  case_name,description,uplo,trans,diag,n,k,lda,incx,a_fill,x_fill,
  expect_result,mere_threshold,mare_multiplier,random_seed

tbmv 家族要点（相对 L2 模板 cgemv 的改造）：
  - 无 alpha/beta/y；枚举为 uplo×trans×diag（2×3×2=12 组合全覆盖）；
  - +k 列（半带宽，k∈[0,n-1]）；lda≥k+1；diag=UNIT 时 golden 不读对角；
  - x 原地覆写（x_fill 即输入填充）。

用法（条数可灵活扩展）：
  python gen_csv.py                          # 默认 1000 精度 + 200 性能
  python gen_csv.py --accuracy 1500 --perf 300
  python gen_csv.py --seed 12345

输入分布说明：当前 ops-blas 测试工程仅支持均匀分布（RANDOM_NORM_5_5 = [-5,5] 均匀）。
生态精度标准要求的 50% 正态分布依赖测试工程扩展（另行任务），扩展完成后通过 --dist mixed 切换。
"""
import csv, os, math, random, argparse

# ---- 尺寸/带宽池 ----
SIZES = [1,2,3,5,7,10,12,15,16,24,32,48,64,65,96,128,200,256,400,512,800,1024,2048]
MED = [10,12,15,24,48,96,200,400]
UPLOS = ["UPPER", "LOWER"]
TRANS = ["N", "T", "C"]
DIAGS = ["NON_UNIT", "UNIT"]
INCS = [1,2,3,-1,-2,-3]
FILL_RAND = "RANDOM_NORM_5_5"   # [-5,5] 均匀分布（生态标准值域；正态分布待测试工程扩展）
FILLS = [("RANDOM_NORM_5_5","rand"),("VALUE_NORM_0","zero"),("RANDOM_ALTER","alter"),
         ("RANDOM_EXTREME","extreme"),("VALUE_NORM_INF","inf"),("VALUE_NORM_NAN","nan")]
MERE = "0.00012207031"   # 2^-13，与 ops-blas 仓惯例一致
MARE = "10.0"
HOST_BUDGET = 512*1024**2  # 512MB host 内存预算（complex64 = 8B/elem）

def fits(n, k):
    return (k + 1) * n * 8 + 2 * n * 8 <= HOST_BUDGET

def kpick(rng, n):
    """合法带宽：0 / 1 / 小值 / 中间 / 满带 n-1"""
    if n <= 1:
        return 0
    cands = [0, 1, min(3, n-1), (n-1)//2, n-1]
    return cands[rng.randrange(len(cands))]

class Gen:
    def __init__(self, seed):
        self.rows = []
        self.idx = 0
        self.rng = random.Random(seed)

    def add(self, nm, ds, uplo, trans, diag, n, k, lda=None, incx=1,
            exp="ACLBLAS_STATUS_SUCCESS", af=FILL_RAND, xf=FILL_RAND, mere=MERE, mare=MARE):
        self.idx += 1
        if lda is None:
            lda = k + 1
        self.rows.append({
            "case_name": nm, "description": ds,
            "uplo": uplo, "trans": trans, "diag": diag,
            "n": n, "k": k, "lda": lda, "incx": incx,
            "a_fill": af, "x_fill": xf,
            "expect_result": exp, "mere_threshold": mere, "mare_multiplier": mare,
            "random_seed": 20260000 + self.idx})

    # ---- 精度用例各类别 ----
    def l0_basic(self):
        for u in UPLOS:
            for t in TRANS:
                for d in DIAGS:
                    n, k = 8, 2
                    self.add(f"TC_L0_{self.idx+1:03d}", f"basic_{u}_{t}_{d}", u, t, d, n, k)

    def l1_size(self):
        for sz in SIZES:
            for i, (u, t, d) in enumerate([(u, t, d) for u in UPLOS for t in TRANS for d in DIAGS]):
                if i % 3 == sz % 3:   # 尺寸×枚举组合轮转采样，保证三枚举各自全覆盖
                    self.add(f"TC_SQ_{self.idx+1:03d}", f"sq_n{sz}_{u}_{t}_{d}", u, t, d, sz, min(sz-1, 4) if sz > 0 else 0)

    def l2_band(self):
        """带宽扫描：k=0 / 1 / 小值 / 半带 / 满带 n-1，× 枚举轮转"""
        combos = [(u, t, d) for u in UPLOS for t in TRANS for d in DIAGS]
        i = 0
        for n in [4, 8, 16, 32, 64]:
            for k in sorted({0, 1, min(3, n-1), (n-1)//2, n-1}):
                u, t, d = combos[i % len(combos)]
                self.add(f"TC_BD_{self.idx+1:03d}", f"band_n{n}_k{k}_{u}_{t}_{d}", u, t, d, n, k)
                i += 1

    def l4_ld_inc(self):
        combos = [(u, t, d) for u in UPLOS for t in TRANS for d in DIAGS]
        i = 0
        for sz in [16, 32, 64]:
            for u in UPLOS:
                u_, t, d = combos[i % len(combos)]
                self.add(f"TC_LD_{self.idx+1:03d}", f"ld_{sz}_{u}", u_, t, d, sz, 4, lda=4+1+4)
                i += 1
        for ix in INCS:
            for u in UPLOS:
                u_, t, d = combos[i % len(combos)]
                self.add(f"TC_INC_{self.idx+1:03d}", f"inc_{ix}_{u}", u_, t, d, 64, 4, incx=ix)
                i += 1

    def l5_fill(self):
        for f, l in FILLS:
            self.add(f"TC_FL_{self.idx+1:03d}", f"fill_a_{l}", "UPPER", "N", "NON_UNIT", 32, 4, af=f)
            self.add(f"TC_FL_{self.idx+1:03d}", f"fill_x_{l}", "LOWER", "N", "NON_UNIT", 32, 4, xf=f)
        # diag=UNIT 时 A 对角不读：对角放 NaN/Inf 仍应通过
        self.add(f"TC_FL_{self.idx+1:03d}", "unit_diag_nan", "UPPER", "N", "UNIT", 32, 4, af="VALUE_NORM_NAN")
        self.add(f"TC_FL_{self.idx+1:03d}", "unit_diag_inf", "LOWER", "C", "UNIT", 32, 4, af="VALUE_NORM_INF")

    def l5b_cover(self):
        """中等尺寸 × uplo×trans×diag 全组合（12 组合 × 2 尺寸）"""
        for sz in [24, 96]:
            for u in UPLOS:
                for t in TRANS:
                    for d in DIAGS:
                        self.add(f"TC_CV_{self.idx+1:03d}", f"cv_{sz}_{u}_{t}_{d}", u, t, d, sz, min(sz-1, 5))

    def l6_edge(self):
        # 零维 quick return（合法 no-op）
        self.add(f"TC_ED_{self.idx+1:03d}", "n0", "UPPER", "N", "NON_UNIT", 0, 0, lda=1)
        # 负向：空指针（n>0）
        self.add(f"TC_ED_{self.idx+1:03d}", "null_A", "UPPER", "N", "NON_UNIT", 16, 4, af="NULLPTR", exp="ACLBLAS_STATUS_INVALID_VALUE")
        self.add(f"TC_ED_{self.idx+1:03d}", "null_x", "LOWER", "N", "NON_UNIT", 16, 4, xf="NULLPTR", exp="ACLBLAS_STATUS_INVALID_VALUE")
        # 负向：非法枚举（999 解析为非法值）
        self.add(f"TC_ED_{self.idx+1:03d}", "invalid_uplo", "999", "N", "NON_UNIT", 16, 4, exp="ACLBLAS_STATUS_INVALID_ENUM")
        self.add(f"TC_ED_{self.idx+1:03d}", "invalid_trans", "UPPER", "999", "NON_UNIT", 16, 4, exp="ACLBLAS_STATUS_INVALID_ENUM")
        self.add(f"TC_ED_{self.idx+1:03d}", "invalid_diag", "LOWER", "N", "999", 16, 4, exp="ACLBLAS_STATUS_INVALID_ENUM")
        # 负向：非法前导维 / 负维度 / 负带宽 / 零步长
        self.add(f"TC_ED_{self.idx+1:03d}", "invalid_lda", "UPPER", "N", "NON_UNIT", 16, 4, lda=4, exp="ACLBLAS_STATUS_INVALID_VALUE")
        self.add(f"TC_ED_{self.idx+1:03d}", "neg_n", "UPPER", "N", "NON_UNIT", -1, 0, lda=1, exp="ACLBLAS_STATUS_INVALID_VALUE")
        self.add(f"TC_ED_{self.idx+1:03d}", "neg_k", "UPPER", "N", "NON_UNIT", 16, -1, lda=1, exp="ACLBLAS_STATUS_INVALID_VALUE")
        self.add(f"TC_ED_{self.idx+1:03d}", "incx0", "UPPER", "N", "NON_UNIT", 16, 4, incx=0, exp="ACLBLAS_STATUS_INVALID_VALUE")

    def ex_extend(self, target):
        """TC_EX 扩展精度用例：确定性采样 尺寸×带宽×枚举×步长×padding×填充，补足到 target 条"""
        pool = [1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,1000,1024,1025,2000,2048]
        combos = [(u, t, d) for u in UPLOS for t in TRANS for d in DIAGS]
        i = 0
        while len([r for r in self.rows if not r["case_name"].startswith("TC_PF")]) < target:
            n = pool[self.rng.randrange(len(pool))]
            k = kpick(self.rng, n)
            u, t, d = combos[i % len(combos)]
            ix = INCS[self.rng.randrange(len(INCS))] if i % 3 == 0 else 1
            lda = k + 1 + 8 if i % 5 == 4 else None
            self.add(f"TC_EX_{self.idx+1:04d}", f"ex_n{n}_k{k}_{u}_{t}_{d}", u, t, d, n, k, lda=lda, incx=ix)
            i += 1

    # ---- 性能用例 ----
    def perf(self, target):
        # 任务书 §3.3 典型 case（逐参数精确一致，gpu_baseline 前 4 条）
        for n, k, u, t, d in [(512,8,"UPPER","N","NON_UNIT"),
                              (1024,16,"UPPER","N","NON_UNIT"),
                              (2048,32,"LOWER","T","NON_UNIT"),
                              (4096,64,"LOWER","C","UNIT")]:
            self.add(f"TC_PF_{self.idx+1:04d}", f"pf_doc_n{n}_k{k}_{u}_{t}_{d}", u, t, d, n, k)
        # 小尺寸延迟敏感区
        for sz in [1,2,4,8,16,32,64,128,256,512]:
            self.add(f"TC_PF_{self.idx+1:04d}", f"pf_small_{sz}", "UPPER", "N", "NON_UNIT", sz, min(sz-1, 4))
        # 规模扫描（k 随规模取对数增长，预算闸门内，单边 ≤4096）
        n_sc = min(40, target // 4)
        max_n = 4096
        for i in range(n_sc):
            sz = int(round(1000 * (max_n / 1000) ** (i / max(n_sc - 1, 1))))
            k = max(1, min(64, int(math.log2(max(sz, 2)))))
            if not fits(sz, k):
                break
            u, t, d = UPLOS[i % 2], TRANS[i % 3], DIAGS[i % 2]
            self.add(f"TC_PF_{self.idx+1:04d}", f"pf_sq_{sz}_k{k}", u, t, d, sz, k)
        # 带宽性能网格：固定 n 扫 k（含满带）
        for n in [1024, 4096]:
            for k in [1, 8, 32, 128, min(512, n-1)]:
                if fits(n, k):
                    self.add(f"TC_PF_{self.idx+1:04d}", f"pf_band_{n}_k{k}", "LOWER", "N", "NON_UNIT", n, k)
        # 枚举组合性能
        for u in UPLOS:
            for t in TRANS:
                for d in DIAGS:
                    self.add(f"TC_PF_{self.idx+1:04d}", f"pf_combo_{u}_{t}_{d}", u, t, d, 2048, 16)
        # 补足到 target（纯连续访存：incx=1，lda=k+1 紧凑）
        while len([r for r in self.rows if r["case_name"].startswith("TC_PF")]) < target:
            n = self.rng.randrange(16, 4097)
            k = self.rng.randrange(0, min(n - 1, 512) + 1)
            if not fits(n, k):
                continue
            self.add(f"TC_PF_{self.idx+1:04d}", f"pf_mix_n{n}_k{k}",
                     UPLOS[self.rng.randrange(2)], TRANS[self.rng.randrange(3)], DIAGS[self.rng.randrange(2)], n, k)

FIELDS = ["case_name","description","uplo","trans","diag","n","k","lda","incx",
          "a_fill","x_fill","expect_result","mere_threshold","mare_multiplier","random_seed"]

def main():
    ap = argparse.ArgumentParser(description="aclblasCtbmv 用例生成（条数可扩展）")
    ap.add_argument("--accuracy", type=int, default=1000, help="精度用例条数（默认1000）")
    ap.add_argument("--perf", type=int, default=200, help="性能/内存用例条数（默认200）")
    ap.add_argument("--seed", type=int, default=20260823, help="随机种子（固定可复现）")
    ap.add_argument("--dist", choices=["uniform", "mixed"], default="uniform",
                    help="输入分布；mixed（均匀50%%+正态50%%）依赖测试工程扩展，暂以均匀生成")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline-out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "ctbmv_test.csv")
    bl_out = args.baseline_out or os.path.join(here, "gpu_baseline.csv")
    if args.dist == "mixed":
        print("[提示] 正态分布依赖 ops-blas 测试工程扩展（另行任务），本次以均匀分布生成")

    g = Gen(args.seed)
    g.l0_basic(); g.l1_size(); g.l2_band(); g.l4_ld_inc(); g.l5_fill(); g.l5b_cover(); g.l6_edge()
    if args.accuracy > len(g.rows):
        g.ex_extend(args.accuracy)
    g.perf(args.perf)

    acc_rows = [r for r in g.rows if not r["case_name"].startswith("TC_PF")]
    pf_rows = [r for r in g.rows if r["case_name"].startswith("TC_PF")]
    # 性能用例纯连续校验：非单位步长/非紧凑前导维会污染性能与内存基线
    for r in pf_rows:
        assert r["incx"] == 1, f"性能用例含非单位步长: {r['case_name']}"
        assert r["lda"] == r["k"] + 1,             f"性能用例含非紧凑前导维: {r['case_name']}"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in g.rows: w.writerow(r)

    with open(bl_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","n","k","uplo","trans","diag","incx","gpu_ms"])
        for i, r in enumerate(pf_rows):
            prefix = "ctbmv-base" if i < 4 else "ctbmv-perf"
            w.writerow([f"{prefix}-{i:03d}", r["n"], r["k"], r["uplo"], r["trans"], r["diag"], r["incx"], ""])

    from collections import Counter
    cats = Counter(r["case_name"].split("_")[1] for r in acc_rows)
    print(f"{out}: 共 {len(g.rows)} 条（精度 {len(acc_rows)} + 性能/内存 {len(pf_rows)}）")
    print("精度用例分类:", dict(sorted(cats.items())))
    print(f"{bl_out}: {len(pf_rows)} 条基线占位（gpu_ms 待回填）")

if __name__ == "__main__":
    main()
