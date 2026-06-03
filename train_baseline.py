# src/train_baseline.py
import os
import re
import io
import time
import json
import math
import argparse
import contextlib
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Project imports (must run as: python -m src.train_baseline)
from src.utils import ensure_dir, now_str, list_npz_files, save_json
from src.data_loader import make_dataset
from src.models_baseline import build_baseline_conv3d
from src.train_framework import Trainer


# -----------------------------
# Physics residual (lightweight)
# Assumes a diffusion-like PDE: du/dt - c(x,y) * Laplacian(u) ≈ 0
# If your simulator uses a different operator, adjust here.
# -----------------------------
def _laplacian_2d(u, dx):
    """
    u: (B, T, H, W, 1) float32
    dx: scalar float32
    returns: (B, T, H, W, 1)
    """
    # 2D 5-point stencil kernel
    k = tf.constant(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=tf.float32
    )
    k = k[:, :, tf.newaxis, tf.newaxis]  # (3,3,1,1)

    b, t, h, w, c = tf.unstack(tf.shape(u))
    u2 = tf.reshape(u, [b * t, h, w, 1])  # merge (B,T) for conv2d

    # symmetric padding to avoid shrinking
    u2p = tf.pad(u2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
    lap = tf.nn.conv2d(u2p, k, strides=1, padding="VALID")

    lap = tf.reshape(lap, [b, t, h, w, 1])
    lap = lap / (dx * dx + 1e-12)
    return lap


def residual_norm(u_hat, c_field, dt, dx):
    """
    u_hat: (B,T,H,W,1)
    c_field: (B,H,W) or (H,W)
    dt, dx: scalars
    returns scalar residual norm (float)
    """
    u_hat = tf.convert_to_tensor(u_hat, tf.float32)
    dt = tf.cast(dt, tf.float32)
    dx = tf.cast(dx, tf.float32)

    # Ensure c_field broadcastable to (B,T,H,W,1)
    c_field = tf.convert_to_tensor(c_field, tf.float32)
    if len(c_field.shape) == 2:
        c_field = c_field[tf.newaxis, ...]  # (1,H,W)
    c = c_field[:, tf.newaxis, :, :, tf.newaxis]  # (B,1,H,W,1)

    # forward difference in time: du/dt at t=0..T-2
    du = u_hat[:, 1:, :, :, :] - u_hat[:, :-1, :, :, :]
    du_dt = du / (dt + 1e-12)

    # laplacian at matching time indices (use t=0..T-2)
    lap = _laplacian_2d(u_hat, dx)
    lap = lap[:, :-1, :, :, :]

    r = du_dt - c * lap
    # L2 mean norm
    return tf.sqrt(tf.reduce_mean(tf.square(r)) + 1e-12)


# -----------------------------
# Helpers: metrics parsing + plotting
# -----------------------------
_EPOCH_RE = re.compile(
    r"Epoch\s+(?P<ep>\d+)\s*/\s*(?P<tot>\d+)\s*\|\s*"
    r"loss\s+(?P<loss>[0-9.eE+-]+)\s*\|\s*MAE\s+(?P<mae>[0-9.eE+-]+)\s*\|\s*"
    r"RMSE\s+(?P<rmse>[0-9.eE+-]+)\s*\|\s*Res\s+(?P<res>[0-9.eE+-]+)\s*\|\|\s*"
    r"val\s+MAE\s+(?P<vmae>[0-9.eE+-]+)\s*\|\s*val\s+RMSE\s+(?P<vrmse>[0-9.eE+-]+)\s*\|\s*val\s+Res\s+(?P<vres>[0-9.eE+-]+)"
)

def parse_trainer_stdout(text):
    rows = []
    for line in text.splitlines():
        m = _EPOCH_RE.search(line)
        if not m:
            continue
        rows.append({
            "epoch": int(m.group("ep")),
            "loss": float(m.group("loss")),
            "mae": float(m.group("mae")),
            "rmse": float(m.group("rmse")),
            "res": float(m.group("res")),
            "val_mae": float(m.group("vmae")),
            "val_rmse": float(m.group("vrmse")),
            "val_res": float(m.group("vres")),
        })
    return rows


def save_csv(path, rows):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_history(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")  # non-GUI backend, avoids Tkinter entirely
    import matplotlib.pyplot as plt

    if not rows:
        return

    epochs = [r["epoch"] for r in rows]

    def _plot(y_key, y_key_val, title, fname):
        y = [r[y_key] for r in rows]
        yv = [r[y_key_val] for r in rows]
        plt.figure()
        plt.plot(epochs, y, label=y_key)
        plt.plot(epochs, yv, label=y_key_val)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(out_dir) / fname, dpi=200)
        plt.close()

    _plot("loss", "loss", "Loss", "fig_loss.png")  # loss has no val in print; keep single
    # If you want a true val-loss, add it inside Trainer later.
    plt.figure()
    plt.plot(epochs, [r["loss"] for r in rows], label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "fig_loss.png", dpi=200)
    plt.close()

    _plot("mae", "val_mae", "MAE", "fig_mae.png")
    _plot("rmse", "val_rmse", "RMSE", "fig_rmse.png")
    _plot("res", "val_res", "Residual norm", "fig_residual.png")


# -----------------------------
# Benchmarking: latency + throughput + model size
# -----------------------------
def model_size_bytes(model: keras.Model) -> int:
    # assumes float32 weights
    return int(model.count_params()) * 4


def benchmark_inference(model, dataset, num_batches=20, warmup=5):
    times = []
    seen = 0
    for batch_i, (features, _) in enumerate(dataset):
        x = features["x"]
        # Warmup
        if batch_i < warmup:
            _ = model(x, training=False)
            continue

        t0 = time.perf_counter()
        _ = model(x, training=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        seen += 1
        if seen >= num_batches:
            break

    if not times:
        return {"avg_s": None, "p50_s": None, "p95_s": None, "throughput_batches_per_s": None}

    times = np.array(times, dtype=np.float64)
    avg = float(times.mean())
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    thr = float(1.0 / avg) if avg > 0 else None
    return {"avg_s": avg, "p50_s": p50, "p95_s": p95, "throughput_batches_per_s": thr}


def evaluate_on_dataset(model, dataset, max_batches=None):
    maes, rmses, ress = [], [], []
    n_batches = 0

    for features, y_true in dataset:
        y_pred = model(features["x"], training=False)

        # MAE / RMSE over the full tensor
        err = tf.cast(y_pred, tf.float32) - tf.cast(y_true, tf.float32)
        mae = tf.reduce_mean(tf.abs(err))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(err)) + 1e-12)

        # residual norm
        res = residual_norm(
            y_pred,
            features["c_field"],
            features["dt"],
            features["dx"],
        )

        maes.append(float(mae.numpy()))
        rmses.append(float(rmse.numpy()))
        ress.append(float(res.numpy()))

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    if n_batches == 0:
        return None

    return {
        "mae_mean": float(np.mean(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "residual_mean": float(np.mean(ress)),
        "batches": int(n_batches),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="Project root (contains src/, data/)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_phys", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_latency_batches", type=int, default=20)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Robust root resolution:
    # - When run as module: python -m src.train_baseline, __file__ is .../src/train_baseline.py
    # - Default root becomes parent of src/
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    root_str = str(root)

    seq_dir = root / "data" / "sim" / "sequences"
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    out_dir = Path(ensure_dir(str(root / "outputs" / "runs" / f"baseline_{now_str()}")))
    fig_dir = Path(ensure_dir(str(out_dir / "figures")))

    # Split
    files = list_npz_files(str(seq_dir))
    n = len(files)
    if n < 5:
        raise RuntimeError(f"Not enough sequences in {seq_dir} (found {n})")

    train_files = files[: int(0.7 * n)]
    val_files   = files[int(0.7 * n): int(0.85 * n)]
    test_files  = files[int(0.85 * n):]

    train_ds = make_dataset(train_files, batch_size=args.batch_size, shuffle=True, repeat=False)
    val_ds   = make_dataset(val_files, batch_size=args.batch_size, shuffle=False, repeat=False)
    test_ds  = make_dataset(test_files, batch_size=args.batch_size, shuffle=False, repeat=False)

    # Model
    model = build_baseline_conv3d(input_channels=2, base=args.base)
    opt = keras.optimizers.Adam(args.lr)

    # Training with stdout capture (no Trainer modification required)
    trainer = Trainer(
        model,
        opt,
        str(out_dir),
        use_physics_loss=True,
        lambda_phys=args.lambda_phys,
        beta=args.beta,
    )

    print(f"[INFO] ROOT     : {root_str}")
    print(f"[INFO] SEQ_DIR  : {seq_dir}")
    print(f"[INFO] OUT_DIR  : {out_dir}")
    print(f"[INFO] #train/#val/#test = {len(train_files)}/{len(val_files)}/{len(test_files)}")

    t_train0 = time.perf_counter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        trainer.fit(train_ds, val_ds, epochs=args.epochs, is_sacu=False)
    t_train1 = time.perf_counter()

    trainer_stdout = buf.getvalue()
    (out_dir / "trainer_stdout.txt").write_text(trainer_stdout, encoding="utf-8")

    rows = parse_trainer_stdout(trainer_stdout)
    save_csv(out_dir / "history.csv", rows)
    save_json(str(out_dir / "history.json"), rows)

    # Figures
    plot_history(rows, fig_dir)

    # Save model + split
    model.save(str(out_dir / "model.keras"))  # Keras v3 format 
    save_json(str(out_dir / "split.json"), {"train": len(train_files), "val": len(val_files), "test": len(test_files)})

    # Evaluation on test set
    test_metrics = evaluate_on_dataset(model, test_ds)
    save_json(str(out_dir / "test_metrics.json"), test_metrics if test_metrics else {})

    # Latency benchmark
    lat = benchmark_inference(model, test_ds, num_batches=args.num_latency_batches, warmup=5)
    save_json(str(out_dir / "latency.json"), lat)

    # Complexity report (time + space, empirical)
    n_params = int(model.count_params())
    size_b = model_size_bytes(model)
    size_mb = size_b / (1024 ** 2)

    report = {
        "root": root_str,
        "seq_dir": str(seq_dir),
        "out_dir": str(out_dir),
        "params": n_params,
        "approx_model_size_mb_fp32": size_mb,
        "train_time_s_total": float(t_train1 - t_train0),
        "train_time_s_per_epoch_avg": float((t_train1 - t_train0) / max(args.epochs, 1)),
        "test_metrics": test_metrics,
        "latency": lat,
        "notes": [
            "Model size assumes float32 parameters only (≈4 bytes/param).",
            "Residual norm uses a diffusion-like residual: du/dt - c(x,y)*Laplacian(u). Adjust if your operator differs.",
            "For theoretical Big-O, see paper Section 3.5; this report provides empirical timing + parameter scaling.",
        ],
    }
    save_json(str(out_dir / "complexity_report.json"), report)

    # Human-readable text report
    lines = []
    lines.append("Baseline Training Report")
    lines.append("=" * 60)
    lines.append(f"ROOT: {root_str}")
    lines.append(f"SEQ_DIR: {seq_dir}")
    lines.append(f"OUT_DIR: {out_dir}")
    lines.append(f"Splits (#train/#val/#test): {len(train_files)}/{len(val_files)}/{len(test_files)}")
    lines.append("")
    lines.append(f"Params: {n_params:,}")
    lines.append(f"Approx model size (fp32): {size_mb:.2f} MB")
    lines.append("")
    lines.append(f"Train time total: {report['train_time_s_total']:.2f} s")
    lines.append(f"Train time/epoch avg: {report['train_time_s_per_epoch_avg']:.2f} s")
    lines.append("")
    if test_metrics:
        lines.append("Test metrics:")
        lines.append(f"  MAE (mean): {test_metrics['mae_mean']:.6f}")
        lines.append(f"  RMSE(mean): {test_metrics['rmse_mean']:.6f}")
        lines.append(f"  Residual(mean): {test_metrics['residual_mean']:.6f}")
        lines.append(f"  Batches: {test_metrics['batches']}")
    else:
        lines.append("Test metrics: (none computed; check dataset pipeline)")

    lines.append("")
    lines.append("Latency benchmark (per batch):")
    lines.append(f"  avg_s: {lat['avg_s']}")
    lines.append(f"  p50_s: {lat['p50_s']}")
    lines.append(f"  p95_s: {lat['p95_s']}")
    lines.append(f"  throughput_batches_per_s: {lat['throughput_batches_per_s']}")
    lines.append("")
    lines.append("Figures saved under: outputs/.../figures/")
    (out_dir / "REPORT.txt").write_text("\n".join(lines), encoding="utf-8")

    print("[DONE] Saved run artifacts to:", out_dir)


if __name__ == "__main__":
    # Running as a script is supported, but prefer module mode:
    # python -m src.train_baseline  
    main()