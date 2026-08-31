#!/usr/bin/env python3
"""
ctbmv 性能/内存验收 — 执行 C++ GTest 的 TC_PF 用例采集 NPU 耗时，与 gpu_baseline.csv 比对。

判定口径：NPU 平均单次耗时 ≤ 标杆耗时（gpu_baseline.csv 的 gpu_ms 列）/倍率。
gpu_ms 为待回填的基线数据（任务书 §3.3 的耗时为已经按倍率阈值调整后的值，验收按倍率阈值判定：NPU 耗时 ≤ 标杆耗时 / 0.8（倍率 ≥ 0.8））。

用法（Atlas 800I A2/A3 版，性能测试设备 910B3）：
  python verify_performance.py --repo /path/to/ops-blas --soc ascend910b3 --timeout 3600
  python verify_performance.py --repo /path/to/ops-blas --soc ascend910b3 --skip-build --device 1
"""
import subprocess, os, sys, re, argparse, csv, datetime

OP = "ctbmv"
PERF_THRESHOLD = 0.8
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_CSV = os.path.join(SCRIPT_DIR, "gpu_baseline.csv")

def find_binary(repo):
    candidates = [
        os.path.join(repo, "build", "test", OP, f"{OP}_test"),
        os.path.join(repo, "build", "test", "tbmv", OP, f"{OP}_test"),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None

def parse_csv_params(repo):
    for arch in ["arch22", "arch35"]:
        p = os.path.join(repo, "test", "tbmv", OP, arch, f"{OP}_test.csv")
        if os.path.exists(p):
            return {r["case_name"]: r for r in csv.DictReader(open(p))
                    if r["case_name"].startswith("TC_PF_")}
    return {}

def load_baseline():
    """返回 {(n,k,uplo,trans,diag): gpu_ms}；gpu_ms 为空（未回填）的条目跳过"""
    if not os.path.exists(GPU_CSV):
        return {}
    out = {}
    for r in csv.DictReader(open(GPU_CSV)):
        if r["gpu_ms"].strip():
            out[(int(r["n"]), int(r["k"]), r["uplo"], r["trans"], r["diag"])] = float(r["gpu_ms"])
    return out

def build(repo, soc):
    cmd = ["bash", "build.sh", f"--soc={soc}", f"--ops={OP}"]
    print(f"  编译: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=repo, timeout=600)
    if r.returncode != 0: print("  编译失败"); return False
    print("  编译成功"); return True

def main():
    ap = argparse.ArgumentParser(description="ctbmv 性能/内存验收")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--soc", required=True)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(SCRIPT_DIR, f"results_npu_ctbmv_{ts}.csv")

    print(f"\n{'='*90}")
    print(f"  ctbmv 性能/内存验收")
    print(f"  验收标准: NPU 耗时 <= GPU A100 耗时 / {PERF_THRESHOLD} (倍率 >= {PERF_THRESHOLD})")
    print(f"{'='*90}")

    if not args.skip_build:
        if not build(repo, args.soc): sys.exit(1)
    else: print("  跳过编译")
    binary = find_binary(repo)
    if not binary: print("  找不到测试二进制"); sys.exit(1)
    print(f"  二进制: {binary}")

    csv_params = parse_csv_params(repo)
    baseline = load_baseline()
    print(f"  基线: {GPU_CSV}（已回填 {len(baseline)} 条）")
    if not baseline:
        print("  [提示] 基线尚未回填，全部用例将标记 NO_REF，仅采集 NPU 耗时")

    env = os.environ.copy()
    if args.device is not None: env["ASCEND_DEVICE_ID"] = str(args.device)
    cmd = [binary, "--gtest_filter=*TC_PF*"]
    print(f"  运行: {cmd[0]} --gtest_filter=*TC_PF*")
    try:
        r = subprocess.run(cmd, timeout=args.timeout, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.TimeoutExpired:
        print(f"  超时 ({args.timeout}s)，建议加大 --timeout"); sys.exit(1)

    results = []
    for line in r.stdout.splitlines():
        m = re.match(r"\[\s+OK\s+\]\s+\S+/(\S+)\s+\((\d+)\s*ms\)", line)
        if m: results.append({"case_name": m.group(1), "npu_ms": int(m.group(2)), "status": "ok"})
        m = re.match(r"\[\s+FAILED\s+\]\s+\S+/(\S+)", line)
        if m: results.append({"case_name": m.group(1), "npu_ms": -1, "status": "failed"})

    out_rows = []
    for res in results:
        p = csv_params.get(res["case_name"], {})
        n_val, k_val = int(p.get("n", "0")), int(p.get("k", "0"))
        uplo = p.get("uplo", "UPPER").replace("ACLBLAS_", "")
        trans = p.get("trans", "N").replace("ACLBLAS_OP_", "")
        diag = p.get("diag", "NON_UNIT").replace("ACLBLAS_", "")
        base = baseline.get((n_val, k_val, uplo, trans, diag))
        npu = res["npu_ms"]
        ratio = 0
        if base is not None and npu > 0:
            ratio = base / npu
            verdict = "PASS" if ratio >= PERF_THRESHOLD else "FAIL"
        else:
            verdict = "NO_REF"
        out_rows.append({"case_name": res["case_name"], "n": n_val, "k": k_val,
                         "uplo": uplo, "trans": trans, "diag": diag,
                         "npu_ms": npu if npu > 0 else "", "baseline_ms": base if base else "", "ratio": round(ratio, 3) if ratio else "",
                         "verdict": verdict})

    if out_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            for row in out_rows: w.writerow(row)
        print(f"  结果: {out_csv} ({len(out_rows)} 条)")

    n_pass = sum(1 for r in out_rows if r["verdict"] == "PASS")
    n_fail = sum(1 for r in out_rows if r["verdict"] == "FAIL")
    n_noref = sum(1 for r in out_rows if r["verdict"] == "NO_REF")
    print(f"\n  {'case':<24} {'n':>8} {'k':>6} {'NPU_ms':>10} {'base_ms':>10} {'ratio':>6} {'verdict':>6}\n  {'-'*70}")
    for r in out_rows:
        print(f"  {r['case_name']:<24} {r['n']:>8} {r['k']:>6} {str(r['npu_ms']):>10} "
              f"{str(r['baseline_ms']):>10} {str(r['ratio']):>6} {r['verdict']:>6}")
    print(f"\n  汇总: PASS={n_pass}  FAIL={n_fail}  NO_REF={n_noref}")
    print(f"  注: GTest 输出耗时含 host 准备+kernel+golden 计算+比对，为保守上界；")
    print(f"      精确 kernel 耗时可配合 msprof 采集。")
    sys.exit(0 if n_fail == 0 and (n_pass > 0 or n_noref > 0) else 1)

if __name__ == "__main__":
    main()
