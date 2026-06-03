import os
import time
import json
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import ensure_dir, list_npz_files, save_json
from src.data_loader import make_dataset
from src.train_framework import compute_influence_weights, mae, rmse
from src.physics_metrics import wave_residual_norm

# IMPORTANT: import custom SACU classes
from src.models_sacu import OrgSACUSolver, SACU, MicroExpert, stitch_patches


ROOT = r"D:\47\472\New-Papers\GIS\Codes"
SEQ_DIR = os.path.join(ROOT, "data", "sim", "sequences")


def load_model_any(path):
    return keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "OrgSACUSolver": OrgSACUSolver,
            "SACU": SACU,
            "MicroExpert": MicroExpert,
        },
    )


def predict_sacu(model, x, y_true):
    _, aux = model(x, training=False)
    patch_outs = aux["patch_outs"]
    regions = aux["regions"]
    H = tf.shape(x)[2]
    W = tf.shape(x)[3]
    w, _ = compute_influence_weights(patch_outs, regions, y_true, beta=5.0)
    return stitch_patches(patch_outs, regions, w, H, W)


def save_csv(path, header, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def main(model_path, is_sacu, batch_size=1, latency_warmup=5, latency_runs=20):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = "sacu" if is_sacu else "baseline"

    out_dir = ensure_dir(os.path.join(ROOT, "outputs", "eval", f"{tag}_eval_{stamp}"))
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))
    tab_dir = ensure_dir(os.path.join(out_dir, "tables"))

    files = list_npz_files(SEQ_DIR)
    test_files = files[int(0.85 * len(files)):]
    ds = make_dataset(test_files, batch_size=batch_size, shuffle=False, repeat=False)

    model = load_model_any(model_path)

    maes, rmses, ress = [], [], []
    preds_all, trues_all = [], []

    # === Accuracy evaluation ===
    for batch in ds:
        features, y_true = batch
        x = features["x"]
        c = features["c_field"]
        dt = features["dt"]
        dx = features["dx"]

        if is_sacu:
            y_pred = predict_sacu(model, x, y_true)
        else:
            y_pred = model(x, training=False)

        maes.append(float(mae(y_true, y_pred).numpy()))
        rmses.append(float(rmse(y_true, y_pred).numpy()))
        ress.append(float(wave_residual_norm(y_pred, c, dt, dx).numpy()))

        preds_all.append(y_pred.numpy().flatten())
        trues_all.append(y_true.numpy().flatten())

    preds_all = np.concatenate(preds_all)
    trues_all = np.concatenate(trues_all)

    # === Latency measurement ===
    ds_latency = make_dataset(test_files[:1], batch_size=1, shuffle=False, repeat=True)
    sample = next(iter(ds_latency))
    features, y_true = sample
    x = features["x"]

    # warmup
    for _ in range(latency_warmup):
        _ = model(x, training=False)

    latencies = []
    for _ in range(latency_runs):
        t0 = time.time()
        _ = model(x, training=False)
        latencies.append(time.time() - t0)

    # === Save tables ===
    summary_rows = [[
        np.mean(maes), np.std(maes),
        np.mean(rmses), np.std(rmses),
        np.mean(ress), np.std(ress),
        np.mean(latencies), np.std(latencies),
        np.percentile(latencies, 95),
        np.percentile(latencies, 99),
        len(test_files)
    ]]

    save_csv(
        os.path.join(tab_dir, "summary.csv"),
        [
            "MAE_mean", "MAE_std",
            "RMSE_mean", "RMSE_std",
            "Residual_mean", "Residual_std",
            "Latency_mean", "Latency_std",
            "Latency_p95", "Latency_p99",
            "Test_count"
        ],
        summary_rows
    )

    # === Save figures ===
    plt.figure()
    plt.hist(np.abs(preds_all - trues_all), bins=50)
    plt.title("Absolute Error Histogram")
    plt.savefig(os.path.join(fig_dir, "error_hist.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.hist(latencies, bins=20)
    plt.title("Latency Histogram (sec)")
    plt.savefig(os.path.join(fig_dir, "latency_hist.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.scatter(trues_all[:5000], preds_all[:5000], s=2)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Prediction vs True (subset)")
    plt.savefig(os.path.join(fig_dir, "pred_vs_true_scatter.png"), dpi=300)
    plt.close()

    results = {
        "MAE_mean": float(np.mean(maes)),
        "RMSE_mean": float(np.mean(rmses)),
        "Residual_mean": float(np.mean(ress)),
        "Latency_sec_mean": float(np.mean(latencies)),
        "Latency_sec_std": float(np.std(latencies)),
        "Latency_sec_p95": float(np.percentile(latencies, 95)),
        "Latency_sec_p99": float(np.percentile(latencies, 99)),
        "test_count": int(len(test_files)),
        "out_dir": out_dir
    }

    save_json(os.path.join(out_dir, "results.json"), results)

    print("[DONE] Evaluation saved to:", out_dir)
    print(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--is_sacu", type=int, default=0)
    args = ap.parse_args()
    main(args.model_path, args.is_sacu)