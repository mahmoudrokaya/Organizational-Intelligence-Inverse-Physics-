import os
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Non-GUI backend (prevents Tkinter issues)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import ensure_dir, list_npz_files, save_json
from src.data_loader import make_dataset
from src.train_framework import compute_influence_weights, mae, rmse
from src.physics_metrics import wave_residual_norm

# IMPORTANT: import custom SACU classes so Keras can deserialize them
from src.models_sacu import OrgSACUSolver, SACU, MicroExpert, stitch_patches


ROOT = r"D:\47\472\New-Papers\GIS\Codes"
SEQ_DIR = os.path.join(ROOT, "data", "sim", "sequences")


def save_csv(path, header, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def apply_stress(x, mode="noise", level=0.2, seed=123):
    """
    x: (B,T,H,W,2) channels: [y, mask]
    modes:
      - noise: add Gaussian noise to y where mask=1
      - dropout: drop observations by reducing mask
      - drift: noise + dropout
    """
    rng = tf.random.Generator.from_seed(seed)
    y = x[..., 0:1]
    m = x[..., 1:2]

    if mode in ("noise", "drift"):
        eps = rng.normal(tf.shape(y), stddev=level, dtype=tf.float32)
        y = y + eps * m

    if mode in ("dropout", "drift"):
        keep = 1.0 - level
        drop_mask = tf.cast(rng.uniform(tf.shape(m)) < keep, tf.float32)
        m = m * drop_mask
        y = y * drop_mask

    return tf.concat([y, m], axis=-1)


def predict_sacu(model, x, y_true):
    """
    SACU forward returns aux patch outputs; we compute influence weights and stitch.
    """
    _, aux = model(x, training=False)
    patch_outs = aux["patch_outs"]
    regions = aux["regions"]
    H = tf.shape(x)[2]
    W = tf.shape(x)[3]
    w, _ = compute_influence_weights(patch_outs, regions, y_true, beta=5.0)
    return stitch_patches(patch_outs, regions, w, H, W)


def _safe_std(values):
    v = np.asarray(values, dtype=np.float64)
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def _plot_curves(levels, mode_to_vals, title, ylabel, out_path):
    plt.figure()
    for mode, vals in mode_to_vals.items():
        plt.plot(levels, vals, marker="o", label=mode)
    plt.title(title)
    plt.xlabel("Stress level")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_heatmap(modes, levels, mat, title, out_path):
    plt.figure()
    im = plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xlabel("Stress level")
    plt.ylabel("Mode")
    plt.xticks(range(len(levels)), [str(l) for l in levels])
    plt.yticks(range(len(modes)), modes)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def load_any_model(model_path: str):
    """
    Robust loader for both baseline and SACU models.
    """
    custom_objects = {
        "OrgSACUSolver": OrgSACUSolver,
        "SACU": SACU,
        "MicroExpert": MicroExpert,
    }
    return keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)


def main(model_path: str, is_sacu: int, out_dir: str = None, num_batches: int = 20, seed: int = 7):
    root = ROOT
    out_root = ensure_dir(out_dir or os.path.join(root, "outputs", "stress"))

    tag = "sacu" if is_sacu else "baseline"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(os.path.join(out_root, f"{tag}_stress_{stamp}"))
    fig_dir = ensure_dir(os.path.join(run_dir, "figures"))
    tab_dir = ensure_dir(os.path.join(run_dir, "tables"))

    files = list_npz_files(SEQ_DIR)
    if len(files) == 0:
        raise RuntimeError(f"No .npz sequences found in: {SEQ_DIR}")

    # Test split = last 15%
    test_files = files[int(0.85 * len(files)):]
    ds = make_dataset(test_files, batch_size=1, shuffle=False, repeat=False)

    model = load_any_model(model_path)

    modes = ["noise", "dropout", "drift"]
    levels = [0.05, 0.10, 0.20]

    per_rows = []
    report = {}

    for mode in modes:
        for level in levels:
            maes, rmses, ress = [], [], []

            for batch in ds.take(num_batches):
                features, y_true = batch
                x0 = features["x"]
                c = features["c_field"]
                dt = features["dt"]
                dx = features["dx"]

                x = apply_stress(x0, mode=mode, level=level, seed=seed)

                if is_sacu:
                    y_pred = predict_sacu(model, x, y_true)
                else:
                    y_pred = model(x, training=False)

                maes.append(float(mae(y_true, y_pred).numpy()))
                rmses.append(float(rmse(y_true, y_pred).numpy()))
                ress.append(float(wave_residual_norm(y_pred, c, dt, dx).numpy()))

            key = f"{mode}_level_{level}"
            stats = {
                "MAE_mean": float(np.mean(maes)),
                "MAE_std": _safe_std(maes),
                "RMSE_mean": float(np.mean(rmses)),
                "RMSE_std": _safe_std(rmses),
                "Residual_mean": float(np.mean(ress)),
                "Residual_std": _safe_std(ress),
                "batches": int(len(maes)),
            }
            report[key] = stats

            per_rows.append([
                mode, level,
                stats["MAE_mean"], stats["MAE_std"],
                stats["RMSE_mean"], stats["RMSE_std"],
                stats["Residual_mean"], stats["Residual_std"],
                stats["batches"],
            ])

    # Tables
    per_csv = os.path.join(tab_dir, "per_condition.csv")
    save_csv(
        per_csv,
        header=[
            "mode", "level",
            "MAE_mean", "MAE_std",
            "RMSE_mean", "RMSE_std",
            "Residual_mean", "Residual_std",
            "batches",
        ],
        rows=per_rows,
    )

    summary_rows = []
    for mode in modes:
        mode_rows = [r for r in per_rows if r[0] == mode]
        mae_means = [r[2] for r in mode_rows]
        rmse_means = [r[4] for r in mode_rows]
        res_means = [r[6] for r in mode_rows]
        summary_rows.append([
            mode,
            float(np.mean(mae_means)),
            float(np.mean(rmse_means)),
            float(np.mean(res_means)),
        ])

    summary_csv = os.path.join(tab_dir, "summary_by_mode.csv")
    save_csv(
        summary_csv,
        header=["mode", "MAE_mean_over_levels", "RMSE_mean_over_levels", "Residual_mean_over_levels"],
        rows=summary_rows,
    )

    # JSON
    results = {
        "model_path": model_path,
        "is_sacu": bool(is_sacu),
        "seq_dir": SEQ_DIR,
        "run_dir": run_dir,
        "test_count": int(len(test_files)),
        "num_batches_per_condition": int(num_batches),
        "seed": int(seed),
        "report": report,
    }
    json_path = os.path.join(run_dir, "results.json")
    save_json(json_path, results)

    # Build matrices for plots
    mae_mat = np.zeros((len(modes), len(levels)), dtype=np.float64)
    rmse_mat = np.zeros((len(modes), len(levels)), dtype=np.float64)
    res_mat = np.zeros((len(modes), len(levels)), dtype=np.float64)

    for i_m, mode in enumerate(modes):
        for i_l, level in enumerate(levels):
            key = f"{mode}_level_{level}"
            mae_mat[i_m, i_l] = report[key]["MAE_mean"]
            rmse_mat[i_m, i_l] = report[key]["RMSE_mean"]
            res_mat[i_m, i_l] = report[key]["Residual_mean"]

    _plot_curves(levels, {modes[i]: mae_mat[i].tolist() for i in range(len(modes))},
                 f"{tag.upper()} Stress: MAE vs Level", "MAE", os.path.join(fig_dir, "mae_vs_level.png"))
    _plot_curves(levels, {modes[i]: rmse_mat[i].tolist() for i in range(len(modes))},
                 f"{tag.upper()} Stress: RMSE vs Level", "RMSE", os.path.join(fig_dir, "rmse_vs_level.png"))
    _plot_curves(levels, {modes[i]: res_mat[i].tolist() for i in range(len(modes))},
                 f"{tag.upper()} Stress: Residual vs Level", "Physics residual", os.path.join(fig_dir, "residual_vs_level.png"))

    _plot_heatmap(modes, levels, mae_mat, f"{tag.upper()} Heatmap: MAE (mode x level)", os.path.join(fig_dir, "mae_heatmap.png"))
    _plot_heatmap(modes, levels, rmse_mat, f"{tag.upper()} Heatmap: RMSE (mode x level)", os.path.join(fig_dir, "rmse_heatmap.png"))
    _plot_heatmap(modes, levels, res_mat, f"{tag.upper()} Heatmap: Residual (mode x level)", os.path.join(fig_dir, "residual_heatmap.png"))

    print("[DONE] Stress artifacts saved to:", run_dir)
    print(" - JSON:", json_path)
    print(" - Tables:", tab_dir)
    print(" - Figures:", fig_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--is_sacu", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--num_batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    main(
        model_path=args.model_path,
        is_sacu=args.is_sacu,
        out_dir=args.out_dir,
        num_batches=args.num_batches,
        seed=args.seed,
    )