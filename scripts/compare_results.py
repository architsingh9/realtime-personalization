# scripts/compare_results.py
from __future__ import annotations
import csv
from pathlib import Path

# Force a headless, stable backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)

RESULTS = {
    "NeuralLinear (cold)": ART / "offline_ctr_neurallinear_cold.csv",
    "NeuralLinear (warm)": ART / "offline_ctr_neurallinear_warm.csv",
    "LinUCB (baseline)" : ART / "offline_ctr_linucb.csv",
}

def read_ctr_csv(path: Path):
    xs, ys = [], []
    if not path.exists():
        return xs, ys, None
    with path.open("r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)

        def parse_row(row):
            try:
                imp = int(float(row[0]))
                ctr = float(row[2])
                return imp, ctr
            except Exception:
                return None

        if header:
            p = parse_row(header)
            if p:  # no header, first row is data
                xs.append(int(p[0])); ys.append(float(p[1]))
        for row in r:
            p = parse_row(row)
            if p:
                xs.append(int(p[0])); ys.append(float(p[1]))

    final_ctr = float(ys[-1]) if ys else None
    return xs, ys, final_ctr

def format_pct(x):
    return f"{100.0*x:.2f}%" if x is not None else "—"

def main():
    curves = {}
    lines = ["Model, Final CTR"]

    for name, path in RESULTS.items():
        x, y, final_ctr = read_ctr_csv(path)
        curves[name] = (x, y, final_ctr)
        lines.append(f"{name}, {format_pct(final_ctr)}")

    summary_path = ART / "benchmark_summary.txt"
    summary_path.write_text("\n".join(lines))
    print("\n".join(lines))

    plt.figure(figsize=(8, 5), dpi=140)
    any_plotted = False
    for name, (x, y, _) in curves.items():
        if x and y:
            # ensure plain numeric lists
            x_plot = [int(v) for v in x]
            y_plot = [100.0*float(v) for v in y]
            plt.plot(x_plot, y_plot, label=name)
            any_plotted = True

    plt.xlabel("Impressions")
    plt.ylabel("CTR (%)")
    plt.title("Offline Replay: CTR Curves")
    if any_plotted:
        plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out_png = ART / "ctr_curves.png"
    try:
        plt.savefig(out_png)
        print(f"Saved plot → {out_png}")
    except Exception as e:
        print(f"Plot save failed (skipping image). Reason: {e}")

    cold = curves.get("NeuralLinear (cold)", (None, None, None))[2]
    warm = curves.get("NeuralLinear (warm)", (None, None, None))[2]
    base = curves.get("LinUCB (baseline)", (None, None, None))[2]

    def lift(a,b):
        if a is None or b is None or b == 0: return None
        return (a/b - 1.0)*100.0

    if warm is not None and cold is not None:
        l = lift(warm, cold)
        if l is not None:
            print(f"Warm vs Cold: +{l:.2f}% relative CTR")

    if warm is not None and base is not None:
        l = lift(warm, base)
        if l is not None:
            print(f"Warm vs LinUCB: +{l:.2f}% relative CTR")

if __name__ == "__main__":
    main()
