import os
import json
import argparse
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p, obj):
    ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(path, header, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def bar_compare(labels, a, b, title, ylabel, out_path):
    x = np.arange(len(labels))
    w = 0.35
    plt.figure()
    plt.bar(x - w/2, a, width=w, label="Baseline")
    plt.bar(x + w/2, b, width=w, label="SACU")
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main(baseline_results_json, sacu_results_json, out_root):
    base = read_json(baseline_results_json)
    sacu = read_json(sacu_results_json)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ensure_dir(os.path.join(out_root, f"eval_baseline_vs_sacu_{stamp}"))
    tab_dir = ensure_dir(os.path.join(out_dir, "tables"))
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    # Summary table (paper-ready)
    rows = []
    metrics = [
        ("MAE_mean", "MAE_std"),
        ("RMSE_mean", "RMSE_std"),
        ("Residual_mean", "Residual_std"),
        ("Latency_sec_mean", "Latency_sec_std"),
        ("Latency_sec_p95", None),
        ("Latency_sec_p99", None),
        ("test_count", None),
    ]

    for mean_key, std_key in metrics:
        rows.append([
            mean_key,
            base.get(mean_key, ""),
            base.get(std_key, "") if std_key else "",
            sacu.get(mean_key, ""),
            sacu.get(std_key, "") if std_key else "",
        ])

    save_csv(
        os.path.join(tab_dir, "eval_comparison.csv"),
        header=["metric", "baseline_value", "baseline_std", "sacu_value", "sacu_std"],
        rows=rows,
    )

    # Key comparison plots
    bar_compare(
        ["MAE", "RMSE", "Residual"],
        [base["MAE_mean"], base["RMSE_mean"], base["Residual_mean"]],
        [sacu["MAE_mean"], sacu["RMSE_mean"], sacu["Residual_mean"]],
        "Test Performance Comparison",
        "Value",
        os.path.join(fig_dir, "performance_bar.png"),
    )

    bar_compare(
        ["Latency_mean", "Latency_p95", "Latency_p99"],
        [base["Latency_sec_mean"], base["Latency_sec_p95"], base["Latency_sec_p99"]],
        [sacu["Latency_sec_mean"], sacu["Latency_sec_p95"], sacu["Latency_sec_p99"]],
        "Latency Comparison (sec per sequence)",
        "Seconds",
        os.path.join(fig_dir, "latency_bar.png"),
    )

    out = {
        "baseline_results_json": baseline_results_json,
        "sacu_results_json": sacu_results_json,
        "out_dir": out_dir,
    }
    save_json(os.path.join(out_dir, "results.json"), out)

    print("[DONE] Eval comparison saved to:", out_dir)
    print(" - Tables:", tab_dir)
    print(" - Figures:", fig_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Path to baseline evaluate results.json")
    ap.add_argument("--sacu", required=True, help="Path to sacu evaluate results.json")
    ap.add_argument("--out_root", default=r"D:\47\472\New-Papers\GIS\Codes\outputs\compare")
    args = ap.parse_args()
    main(args.baseline, args.sacu, args.out_root)