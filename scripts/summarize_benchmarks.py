# scripts/summarize_benchmarks.py
from pathlib import Path
import csv, subprocess, sys, re

ART = Path("artifacts")

CURVES = {
    "NeuralLinear (cold)": ART / "offline_ctr_neurallinear_cold.csv",
    "NeuralLinear (warm)": ART / "offline_ctr_neurallinear_warm.csv",
    "LinUCB (baseline)"  : ART / "offline_ctr_linucb.csv",
}

POLICIES = {
    "NeuralLinear (cold)": ART / "heads_neurallinear_cold.json",
    "NeuralLinear (warm)": ART / "heads_neurallinear_warm.json",
    "LinUCB (baseline)"  : ART / "heads_linucb.json",
}

OUT_CSV = ART / "benchmarks_summary.csv"

def final_ctr(csv_path: Path) -> float:
    """
    Robust CSV reader (no pandas). Expects headers: impressions,clicks,ctr
    Returns the last CTR as float, or NaN if file missing/empty.
    """
    try:
        with csv_path.open("r", newline="") as f:
            rdr = csv.DictReader(f)
            last = None
            for row in rdr:
                last = row
            if not last:
                return float("nan")
            return float(last["ctr"])
    except FileNotFoundError:
        return float("nan")

def run_ope(policy_path: Path):
    """
    Call the existing CLI: python scripts/ope_eval_policy.py <policy_json>
    Parse IPS and DR from stdout. Returns (ips, dr) as floats.
    """
    cmd = [sys.executable, "scripts/ope_eval_policy.py", str(policy_path)]
    out = subprocess.check_output(cmd, text=True)
    # Look for lines like: "IPS: 0.0506" and "DR : 0.0503"
    ips = dr = float("nan")
    for line in out.splitlines():
        m1 = re.search(r"IPS:\s*([0-9.]+)", line)
        m2 = re.search(r"DR\s*:\s*([0-9.]+)", line)
        if m1:
            ips = float(m1.group(1))
        if m2:
            dr = float(m2.group(1))
    return ips, dr

def main():
    rows = []
    for name in ["NeuralLinear (cold)", "NeuralLinear (warm)", "LinUCB (baseline)"]:
        ctr = final_ctr(CURVES[name])
        ips, dr = run_ope(POLICIES[name])
        rows.append((name, ctr, ips, dr))

    # Print table
    print("Model, Offline CTR, IPS (OPE), DR (OPE)")
    for name, ctr, ips, dr in rows:
        ctr_pct = "nan" if ctr != ctr else f"{100*ctr:.2f}%"
        print(f"{name}, {ctr_pct}, {ips:.4f}, {dr:.4f}")

    # Save to CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "offline_ctr", "ips", "dr"])
        for name, ctr, ips, dr in rows:
            w.writerow([name, ctr, ips, dr])
    print(f"\n✅ Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
