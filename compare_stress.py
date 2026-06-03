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


def plot_curves(levels, base_vals, sacu_vals, title, ylabel, out_path):
    plt.figure()
    for mode, vals in base_vals.items():
        plt.plot(levels, vals, marker="o", linestyle="-", label=f"Baseline-{mode}")
    for mode, vals in sacu_vals.items():
        plt.plot(levels, vals, marker="s", linestyle="--", label=f"SACU-{mode}")
    plt.title(title)
    plt.xlabel("Stress level")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def extract_matrix(stress_report, modes, levels, metric_key):
    mat = np.zeros((len(modes), len(levels)), dtype=np.float64)
    for i_m, mode in enumerate(modes):
        for i_l, level in enumerate(levels):
            key = f"{mode}_level_{level}"
            mat[i_m, i_l] = stress_report[key][metric_key]
    return mat


def main(baseline_stress_json, sacu_stress_json, out_root):
    base = read_json(baseline_stress_json)
    sacu = read_json(sacu_stress_json)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ensure_dir(os.path.join(out_root, f"stress_baseline_vs_sacu_{stamp}"))
    tab_dir = ensure_dir(os.path.join(out_dir, "tables"))
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    modes = ["noise", "dropout", "drift"]
    levels = [0.05, 0.10, 0.20]

    # Build compact comparison table: (mode, level, metric) baseline vs sacu
    rows = []
    for mode in modes:
        for level in levels:
            key = f"{mode}_level_{level}"
            rows.append([mode, level, "MAE_mean", base["report"][key]["MAE_mean"], sacu["report"][key]["MAE_mean"]])
            rows.append([mode, level, "RMSE_mean", base["report"][key]["RMSE_mean"], sacu["report"][key]["RMSE_mean"]])
            rows.append([mode, level, "Residual_mean", base["report"][key]["Residual_mean"], sacu["report"][key]["Residual_mean"]])

    save_csv(
        os.path.join(tab_dir, "stress_comparison_long.csv"),
        header=["mode", "level", "metric", "baseline", "sacu"],
        rows=rows,
    )

    # Curves: overlay baseline vs sacu for each metric
    base_mae = extract_matrix(base["report"], modes, levels, "MAE_mean")
    sacu_mae = extract_matrix(sacu["report"], modes, levels, "MAE_mean")
    base_rmse = extract_matrix(base["report"], modes, levels, "RMSE_mean")
    sacu_rmse = extract_matrix(sacu["report"], modes, levels, "RMSE_mean")
    base_res = extract_matrix(base["report"], modes, levels, "Residual_mean")
    sacu_res = extract_matrix(sacu["report"], modes, levels, "Residual_mean")

    base_vals = {modes[i]: base_mae[i].tolist() for i in range(len(modes))}
    sacu_vals = {modes[i]: sacu_mae[i].tolist() for i in range(len(modes))}
    plot_curves(levels, base_vals, sacu_vals, "Stress Test: MAE vs Level (Baseline vs SACU)", "MAE",
                os.path.join(fig_dir, "mae_vs_level_compare.png"))

    base_vals = {modes[i]: base_rmse[i].tolist() for i in range(len(modes))}
    sacu_vals = {modes[i]: sacu_rmse[i].tolist() for i in range(len(modes))}
    plot_curves(levels, base_vals, sacu_vals, "Stress Test: RMSE vs Level (Baseline vs SACU)", "RMSE",
                os.path.join(fig_dir, "rmse_vs_level_compare.png"))

    base_vals = {modes[i]: base_res[i].tolist() for i in range(len(modes))}
    sacu_vals = {modes[i]: sacu_res[i].tolist() for i in range(len(modes))}
    plot_curves(levels, base_vals, sacu_vals, "Stress Test: Residual vs Level (Baseline vs SACU)", "Residual",
                os.path.join(fig_dir, "residual_vs_level_compare.png"))

    out = {
        "baseline_stress_json": baseline_stress_json,
        "sacu_stress_json": sacu_stress_json,
        "out_dir": out_dir,
    }
    save_json(os.path.join(out_dir, "results.json"), out)

    print("[DONE] Stress comparison saved to:", out_dir)
    print(" - Tables:", tab_dir)
    print(" - Figures:", fig_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Path to baseline stress results.json (new run-folder style)")
    ap.add_argument("--sacu", required=True, help="Path to sacu stress results.json (new run-folder style)")
    ap.add_argument("--out_root", default=r"D:\47\472\New-Papers\GIS\Codes\outputs\compare")
    args = ap.parse_args()
    main(args.baseline, args.sacu, args.out_root)