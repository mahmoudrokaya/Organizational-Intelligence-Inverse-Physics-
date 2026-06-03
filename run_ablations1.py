import os
import time
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.utils import ensure_dir, now_str, list_npz_files, save_json
from src.data_loader import make_dataset
from src.models_sacu import OrgSACUSolver, stitch_patches
from src.train_framework import Trainer, compute_influence_weights, mae, rmse
from src.physics_metrics import wave_residual_norm


def _safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def evaluate_sacu_model(
    model: keras.Model,
    test_ds,
    beta: float,
    num_latency_batches: int = 15,
) -> Dict[str, Any]:
    """
    Evaluate SACU without re-loading model from disk (avoids custom-class deserialization issues).
    Reproduces SACU stitching:
      - forward pass -> patch_outs, regions
      - compute influence weights using y_true
      - stitch full-field prediction
      - compute MAE/RMSE/physics residual
    Also measures latency per batch (forward + stitch + metrics), for the first num_latency_batches.
    """
    maes, rmses, ress = [], [], []
    latencies = []
    count = 0

    for batch in test_ds:
        features, y_true = batch
        x = features["x"]
        c_field = features["c_field"]
        dt = features["dt"]
        dx = features["dx"]

        t0 = time.perf_counter()

        # SACU forward: returns (_, aux)
        _, aux = model(x, training=False)
        patch_outs = aux["patch_outs"]
        regions = aux["regions"]

        H = tf.shape(x)[2]
        W = tf.shape(x)[3]

        w, _ = compute_influence_weights(patch_outs, regions, y_true, beta=beta)
        y_pred = stitch_patches(patch_outs, regions, w, H, W)

        m = mae(y_true, y_pred)
        r = rmse(y_true, y_pred)
        res = wave_residual_norm(y_pred, c_field, dt, dx)

        t1 = time.perf_counter()

        maes.append(float(m.numpy()))
        rmses.append(float(r.numpy()))
        ress.append(float(res.numpy()))

        if count < num_latency_batches:
            latencies.append(t1 - t0)

        count += 1

    lat_arr = np.array(latencies, dtype=np.float64) if len(latencies) else np.array([0.0], dtype=np.float64)

    return {
        "MAE_mean": float(np.mean(maes)) if maes else None,
        "MAE_std": float(np.std(maes)) if maes else None,
        "RMSE_mean": float(np.mean(rmses)) if rmses else None,
        "RMSE_std": float(np.std(rmses)) if rmses else None,
        "Residual_mean": float(np.mean(ress)) if ress else None,
        "Residual_std": float(np.std(ress)) if ress else None,
        "Latency_sec_mean": float(np.mean(lat_arr)),
        "Latency_sec_std": float(np.std(lat_arr)),
        "Latency_sec_p95": float(np.percentile(lat_arr, 95)),
        "Latency_sec_p99": float(np.percentile(lat_arr, 99)),
        "latency_batches": int(min(num_latency_batches, count)),
        "test_count": int(count),
    }


def run_one(
    root: Path,
    seq_dir: Path,
    out_dir: Path,
    name: str,
    use_role: bool,
    use_comms: bool,
    use_physics_loss: bool,
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 1,
    lambda_phys: float = 0.05,
    beta: float = 5.0,
) -> Path:
    run_dir = _safe_mkdir(out_dir / name)
    _safe_mkdir(run_dir / "figures")
    _safe_mkdir(run_dir / "tables")

    files = list_npz_files(str(seq_dir))
    n = len(files)
    if n < 5:
        raise RuntimeError(f"Not enough sequences in {seq_dir} (found {n})")

    train_files = files[: int(0.7 * n)]
    val_files = files[int(0.7 * n): int(0.85 * n)]
    test_files = files[int(0.85 * n):]

    train_ds = make_dataset(train_files, batch_size=batch_size, shuffle=True, repeat=False)
    val_ds = make_dataset(val_files, batch_size=batch_size, shuffle=False, repeat=False)
    test_ds = make_dataset(test_files, batch_size=batch_size, shuffle=False, repeat=False)

    model = OrgSACUSolver(
        grid=4,
        overlap=8,
        K=4,
        hidden=64,
        msg_dim=16,
        use_role=use_role,
        use_comms=use_comms,
    )

    opt = keras.optimizers.Adam(lr)

    trainer = Trainer(
        model,
        opt,
        str(run_dir),
        use_physics_loss=use_physics_loss,
        lambda_phys=lambda_phys,
        beta=beta,
    )

    # Train
    trainer.fit(train_ds, val_ds, epochs=epochs, is_sacu=True)

    # Save model + config
    model_path = run_dir / "model.keras"
    model.save(str(model_path))

    save_json(str(run_dir / "config.json"), {
        "name": name,
        "use_role": use_role,
        "use_comms": use_comms,
        "use_physics_loss": use_physics_loss,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "lambda_phys": lambda_phys,
        "beta": beta,
        "split": {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
    })

    # Evaluate directly (no load_model -> avoids OrgSACUSolver deserialization error)
    eval_metrics = evaluate_sacu_model(model, test_ds, beta=beta, num_latency_batches=15)
    eval_metrics.update({
        "model_path": str(model_path),
        "run_dir": str(run_dir),
        "is_sacu": True,
    })

    save_json(str(run_dir / "eval_results.json"), eval_metrics)
    (run_dir / "eval_results.txt").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    # Per-run table for quick reporting
    with open(run_dir / "tables" / "eval_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k in [
            "MAE_mean", "MAE_std",
            "RMSE_mean", "RMSE_std",
            "Residual_mean", "Residual_std",
            "Latency_sec_mean", "Latency_sec_std",
            "Latency_sec_p95", "Latency_sec_p99",
            "test_count",
        ]:
            w.writerow([k, eval_metrics.get(k)])

    return run_dir


def write_summary(out_dir: Path, summary: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for name, d in summary.items():
        rows.append({
            "ablation": name,
            "Test_MAE_mean": d.get("MAE_mean"),
            "Test_RMSE_mean": d.get("RMSE_mean"),
            "Test_Residual_mean": d.get("Residual_mean"),
            "Latency_sec_mean": d.get("Latency_sec_mean"),
            "Latency_sec_p95": d.get("Latency_sec_p95"),
            "Latency_sec_p99": d.get("Latency_sec_p99"),
            "test_count": d.get("test_count"),
        })

    csv_path = out_dir / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    save_json(str(out_dir / "ablation_summary.json"), rows)


def main():
    root = Path(__file__).resolve().parents[1]
    seq_dir = root / "data" / "sim" / "sequences"
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    abl_root = Path(ensure_dir(str(root / "outputs" / "ablations" / f"abl_{now_str()}")))

    # Ablations aligned with your current plan
    runs: List[Tuple[str, bool, bool, bool]] = [
        ("full", True, True, True),
        ("no_comms", True, False, True),
        ("no_roles", False, True, True),
        ("no_physics", True, True, False),
    ]

    index: Dict[str, str] = {}
    summary: Dict[str, Dict[str, Any]] = {}

    for name, use_role, use_comms, use_phys in runs:
        run_dir = run_one(
            root=root,
            seq_dir=seq_dir,
            out_dir=abl_root,
            name=name,
            use_role=use_role,
            use_comms=use_comms,
            use_physics_loss=use_phys,
            epochs=3,
            lr=1e-3,
            batch_size=1,
            lambda_phys=0.05,
            beta=5.0,
        )
        index[name] = str(run_dir)

        with open(run_dir / "eval_results.json", "r", encoding="utf-8") as f:
            summary[name] = json.load(f)

    save_json(str(abl_root / "index.json"), index)
    write_summary(abl_root, summary)

    print("[DONE] Ablations saved in:", str(abl_root))
    print(" - index.json:", str(abl_root / "index.json"))
    print(" - summary   :", str(abl_root / "ablation_summary.csv"))


if __name__ == "__main__":
    main()