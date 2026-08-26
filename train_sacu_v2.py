from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ============================================================
# PATHS
# ============================================================

ORIGINAL_ROOT = Path(r"D:\47\472\New-Papers\GIS\Codes")
NEW_ROOT = ORIGINAL_ROOT / "New_Branch"

DATA_ROOT = ORIGINAL_ROOT / "data"
SEQ_DIR = DATA_ROOT / "sim" / "sequences"

OUTPUT_ROOT = NEW_ROOT / "outputs" / "core_validation"

# Make New_Branch imports take priority.
sys.path.insert(0, str(NEW_ROOT))

from src.data_loader import make_dataset
from src.models_sacu import OrgSACUSolver
from src.physics_metrics import wave_residual_norm
from src.training.trainer_v2 import (
    TrainerV2,
    predict_sacu_deployment,
    mae,
    rmse,
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

CONFIG = {
    # Reproducibility
    "seed": 42,

    # Dataset
    "train_fraction": 0.70,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "batch_size": 1,

    # SACU architecture
    "grid": 4,
    "overlap": 8,
    "K": 4,
    "hidden": 64,
    "msg_dim": 16,
    "use_role": True,
    "use_comms": True,

    # Optimization
    "epochs": 5,
    "learning_rate": 1e-3,

    # Physics loss
    "use_physics_loss": True,
    "lambda_phys": 0.05,

    # Deployment-valid influence weighting
    "sensor_weight": 0.50,
    "physics_weight": 0.35,
    "entropy_weight": 0.15,
    "temperature": 5.0,

    # Logging
    "save_every_epoch": True,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_reproducibility(seed: int) -> None:
    """
    Configure Python, NumPy, and TensorFlow reproducibility.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
        print("[INFO] TensorFlow deterministic operations enabled.")
    except Exception as exc:
        print(
            "[WARNING] Could not enable TensorFlow deterministic operations:",
            exc,
        )


# ============================================================
# FILE/DIRECTORY HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def save_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)

    if not rows:
        return

    keys = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# DATASET SPLITTING
# ============================================================

def list_sequence_files() -> list[str]:
    files = sorted(
        str(p)
        for p in SEQ_DIR.glob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No .npz sequence files found in:\n{SEQ_DIR}"
        )

    return files


def split_files(files: list[str]):
    """
    Preserve the existing manuscript/code convention:
        first 70% -> training
        next 15%  -> validation
        last 15%  -> testing

    IMPORTANT:
    This does not itself prove parameter-disjointness.
    That must be confirmed separately by the dataset audit.
    """

    n = len(files)

    train_end = int(
        CONFIG["train_fraction"] * n
    )

    val_end = int(
        (
            CONFIG["train_fraction"]
            + CONFIG["val_fraction"]
        )
        * n
    )

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    return (
        train_files,
        val_files,
        test_files,
    )


def save_split_manifest(
    run_dir: Path,
    train_files,
    val_files,
    test_files,
) -> None:

    save_json(
        run_dir / "split_summary.json",
        {
            "total": (
                len(train_files)
                + len(val_files)
                + len(test_files)
            ),
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
            "train_fraction": CONFIG["train_fraction"],
            "val_fraction": CONFIG["val_fraction"],
            "test_fraction": CONFIG["test_fraction"],
            "split_method":
                "sorted file list using 70/15/15 positional partition",
        },
    )

    save_text(
        run_dir / "splits" / "train_files.txt",
        "\n".join(train_files),
    )

    save_text(
        run_dir / "splits" / "validation_files.txt",
        "\n".join(val_files),
    )

    save_text(
        run_dir / "splits" / "test_files.txt",
        "\n".join(test_files),
    )


# ============================================================
# ENVIRONMENT RECORD
# ============================================================

def get_environment_info() -> dict:

    physical_gpus = tf.config.list_physical_devices(
        "GPU"
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tensorflow_version": tf.__version__,
        "keras_version": (
            getattr(
                tf.keras,
                "__version__",
                "bundled_tf_keras",
            )
        ),
        "numpy_version": np.__version__,
        "gpu_devices": [
            str(x)
            for x in physical_gpus
        ],
        "tensorflow_build": (
            tf.sysconfig.get_build_info()
        ),
    }


# ============================================================
# METRIC HELPERS
# ============================================================

def influence_entropy(weights):
    """
    Entropy of agent-level influence distribution.

    weights shape: (B, N_agents)
    """

    weights = tf.clip_by_value(
        weights,
        1e-8,
        1.0,
    )

    entropy = -tf.reduce_sum(
        weights * tf.math.log(weights),
        axis=1,
    )

    return tf.reduce_mean(entropy)


def gate_entropy(gates):
    """
    gates shape:
        (B, N_agents, K)
    """

    gates = tf.clip_by_value(
        gates,
        1e-8,
        1.0,
    )

    entropy = -tf.reduce_sum(
        gates * tf.math.log(gates),
        axis=-1,
    )

    return tf.reduce_mean(entropy)


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    trainer: TrainerV2,
    train_ds,
    epoch: int,
):
    losses = []
    maes = []
    rmses = []
    residuals = []

    start_time = time.perf_counter()

    for step, batch in enumerate(
        train_ds,
        start=1,
    ):

        features, y_true = batch

        x = features["x"]
        c = features["c_field"]
        dt = features["dt"]
        dx = features["dx"]

        (
            loss,
            m,
            r,
            residual,
        ) = trainer.train_step_sacu(
            x,
            y_true,
            c,
            dt,
            dx,
        )

        losses.append(
            float(loss.numpy())
        )

        maes.append(
            float(m.numpy())
        )

        rmses.append(
            float(r.numpy())
        )

        residuals.append(
            float(residual.numpy())
        )

        if step % 100 == 0:
            print(
                f"Epoch {epoch} | "
                f"step {step} | "
                f"loss={np.mean(losses):.6f} | "
                f"MAE={np.mean(maes):.6f} | "
                f"RMSE={np.mean(rmses):.6f} | "
                f"Residual={np.mean(residuals):.6f}"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return {
        "train_loss": float(
            np.mean(losses)
        ),
        "train_mae": float(
            np.mean(maes)
        ),
        "train_rmse": float(
            np.mean(rmses)
        ),
        "train_residual": float(
            np.mean(residuals)
        ),
        "train_time_sec": float(
            elapsed
        ),
    }


# ============================================================
# VALIDATION
# ============================================================

def evaluate_dataset(
    model,
    dataset,
):
    maes = []
    rmses = []
    residuals = []

    influence_entropies = []
    gate_entropies = []

    sensor_scores = []
    physics_scores = []

    for batch in dataset:

        features, y_true = batch

        x = features["x"]
        c = features["c_field"]
        dt = features["dt"]
        dx = features["dx"]

        # ------------------------------------------------
        # DEPLOYMENT-VALID PREDICTION
        #
        # y_true is deliberately NOT passed here.
        # ------------------------------------------------

        y_pred, diagnostics = (
            predict_sacu_deployment(
                model=model,
                x=x,
                c_field=c,
                dt=dt,
                dx=dx,
                training=False,
                sensor_weight=CONFIG[
                    "sensor_weight"
                ],
                physics_weight=CONFIG[
                    "physics_weight"
                ],
                entropy_weight=CONFIG[
                    "entropy_weight"
                ],
                temperature=CONFIG[
                    "temperature"
                ],
            )
        )

        # ------------------------------------------------
        # Ground truth begins here only.
        # ------------------------------------------------

        m = mae(
            y_true,
            y_pred,
        )

        r = rmse(
            y_true,
            y_pred,
        )

        p = wave_residual_norm(
            y_pred,
            c,
            dt,
            dx,
        )

        maes.append(
            float(m.numpy())
        )

        rmses.append(
            float(r.numpy())
        )

        residuals.append(
            float(p.numpy())
        )

        weights = diagnostics[
            "influence_weights"
        ]

        gates = diagnostics[
            "gates"
        ]

        influence_entropies.append(
            float(
                influence_entropy(
                    weights
                ).numpy()
            )
        )

        gate_entropies.append(
            float(
                gate_entropy(
                    gates
                ).numpy()
            )
        )

        sensor_scores.append(
            float(
                tf.reduce_mean(
                    diagnostics[
                        "sensor_score"
                    ]
                ).numpy()
            )
        )

        physics_scores.append(
            float(
                tf.reduce_mean(
                    diagnostics[
                        "physics_score"
                    ]
                ).numpy()
            )
        )

    return {
        "mae": float(
            np.mean(maes)
        ),
        "mae_std": float(
            np.std(maes)
        ),

        "rmse": float(
            np.mean(rmses)
        ),
        "rmse_std": float(
            np.std(rmses)
        ),

        "residual": float(
            np.mean(residuals)
        ),
        "residual_std": float(
            np.std(residuals)
        ),

        "influence_entropy":
            float(
                np.mean(
                    influence_entropies
                )
            ),

        "gate_entropy":
            float(
                np.mean(
                    gate_entropies
                )
            ),

        "sensor_consistency_score":
            float(
                np.mean(
                    sensor_scores
                )
            ),

        "physics_score":
            float(
                np.mean(
                    physics_scores
                )
            ),
    }


# ============================================================
# TEST-TIME LATENCY
# ============================================================

def measure_latency(
    model,
    test_ds,
    warmup=3,
    runs=10,
):
    """
    Measures complete SACU V2 deployment inference:
        model forward
        + deployment-valid weighting
        + stitching

    Does NOT include MAE/RMSE evaluation against y_true.
    """

    sample = next(
        iter(
            test_ds.take(1)
        )
    )

    features, _ = sample

    x = features["x"]
    c = features["c_field"]
    dt = features["dt"]
    dx = features["dx"]

    # Warm-up
    for _ in range(warmup):

        _ = predict_sacu_deployment(
            model,
            x,
            c,
            dt,
            dx,
            training=False,
            sensor_weight=CONFIG[
                "sensor_weight"
            ],
            physics_weight=CONFIG[
                "physics_weight"
            ],
            entropy_weight=CONFIG[
                "entropy_weight"
            ],
            temperature=CONFIG[
                "temperature"
            ],
        )

    latencies = []

    for _ in range(runs):

        start = time.perf_counter()

        _ = predict_sacu_deployment(
            model,
            x,
            c,
            dt,
            dx,
            training=False,
            sensor_weight=CONFIG[
                "sensor_weight"
            ],
            physics_weight=CONFIG[
                "physics_weight"
            ],
            entropy_weight=CONFIG[
                "entropy_weight"
            ],
            temperature=CONFIG[
                "temperature"
            ],
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        latencies.append(
            elapsed
        )

    return {
        "latency_mean_sec":
            float(
                np.mean(
                    latencies
                )
            ),

        "latency_std_sec":
            float(
                np.std(
                    latencies
                )
            ),

        "latency_p95_sec":
            float(
                np.percentile(
                    latencies,
                    95,
                )
            ),

        "latency_p99_sec":
            float(
                np.percentile(
                    latencies,
                    99,
                )
            ),

        "latency_runs":
            len(
                latencies
            ),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_reproducibility(
        CONFIG["seed"]
    )

    # --------------------------------------------------------
    # Create run directory
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    run_dir = ensure_dir(
        OUTPUT_ROOT
        / f"sacu_v2_{timestamp}"
    )

    ensure_dir(
        run_dir / "models"
    )

    ensure_dir(
        run_dir / "tables"
    )

    ensure_dir(
        run_dir / "splits"
    )

    ensure_dir(
        run_dir / "logs"
    )

    print()
    print("=" * 80)
    print("SACU V2 CORE VALIDATION")
    print("=" * 80)

    print(
        "Run directory:",
        run_dir,
    )

    # --------------------------------------------------------
    # Save configuration and environment
    # --------------------------------------------------------

    save_json(
        run_dir / "config.json",
        CONFIG,
    )

    save_json(
        run_dir / "environment.json",
        get_environment_info(),
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    files = list_sequence_files()

    (
        train_files,
        val_files,
        test_files,
    ) = split_files(files)

    save_split_manifest(
        run_dir,
        train_files,
        val_files,
        test_files,
    )

    print()
    print(
        f"Total sequences : {len(files)}"
    )

    print(
        f"Train sequences : {len(train_files)}"
    )

    print(
        f"Validation      : {len(val_files)}"
    )

    print(
        f"Test            : {len(test_files)}"
    )

    # --------------------------------------------------------
    # TensorFlow datasets
    # --------------------------------------------------------

    train_ds = make_dataset(
        train_files,
        batch_size=CONFIG[
            "batch_size"
        ],
        shuffle=True,
        repeat=False,
        deterministic=True,
    )

    val_ds = make_dataset(
        val_files,
        batch_size=CONFIG[
            "batch_size"
        ],
        shuffle=False,
        repeat=False,
        deterministic=True,
    )

    test_ds = make_dataset(
        test_files,
        batch_size=CONFIG[
            "batch_size"
        ],
        shuffle=False,
        repeat=False,
        deterministic=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = OrgSACUSolver(
        grid=CONFIG["grid"],
        overlap=CONFIG["overlap"],
        K=CONFIG["K"],
        hidden=CONFIG["hidden"],
        msg_dim=CONFIG["msg_dim"],
        use_role=CONFIG[
            "use_role"
        ],
        use_comms=CONFIG[
            "use_comms"
        ],
    )

    # Build model using one real batch.
    first_batch = next(
        iter(
            train_ds.take(1)
        )
    )

    first_features, _ = (
        first_batch
    )

    _ = model(
        first_features["x"],
        training=False,
    )

    print()
    print(
        "Trainable parameters:",
        model.count_params(),
    )

    # --------------------------------------------------------
    # Optimizer / trainer
    # --------------------------------------------------------

    optimizer = keras.optimizers.Adam(
        learning_rate=CONFIG[
            "learning_rate"
        ]
    )

    trainer = TrainerV2(
        model=model,
        optimizer=optimizer,

        use_physics_loss=CONFIG[
            "use_physics_loss"
        ],

        lambda_phys=CONFIG[
            "lambda_phys"
        ],

        sensor_weight=CONFIG[
            "sensor_weight"
        ],

        physics_weight=CONFIG[
            "physics_weight"
        ],

        entropy_weight=CONFIG[
            "entropy_weight"
        ],

        temperature=CONFIG[
            "temperature"
        ],
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    history = []

    best_val_rmse = np.inf

    total_start = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        CONFIG["epochs"] + 1,
    ):

        print()
        print("-" * 80)

        print(
            f"EPOCH {epoch}/"
            f"{CONFIG['epochs']}"
        )

        print("-" * 80)

        train_metrics = (
            train_one_epoch(
                trainer,
                train_ds,
                epoch,
            )
        )

        val_metrics = (
            evaluate_dataset(
                model,
                val_ds,
            )
        )

        epoch_row = {
            "epoch": epoch,

            **train_metrics,

            "val_mae":
                val_metrics["mae"],

            "val_rmse":
                val_metrics["rmse"],

            "val_residual":
                val_metrics[
                    "residual"
                ],

            "val_influence_entropy":
                val_metrics[
                    "influence_entropy"
                ],

            "val_gate_entropy":
                val_metrics[
                    "gate_entropy"
                ],

            "val_sensor_score":
                val_metrics[
                    "sensor_consistency_score"
                ],

            "val_physics_score":
                val_metrics[
                    "physics_score"
                ],
        }

        history.append(
            epoch_row
        )

        print()
        print(
            f"Train loss     : "
            f"{train_metrics['train_loss']:.6f}"
        )

        print(
            f"Train MAE      : "
            f"{train_metrics['train_mae']:.6f}"
        )

        print(
            f"Train RMSE     : "
            f"{train_metrics['train_rmse']:.6f}"
        )

        print(
            f"Train residual : "
            f"{train_metrics['train_residual']:.6f}"
        )

        print(
            f"Validation MAE : "
            f"{val_metrics['mae']:.6f}"
        )

        print(
            f"Validation RMSE: "
            f"{val_metrics['rmse']:.6f}"
        )

        print(
            f"Val residual   : "
            f"{val_metrics['residual']:.6f}"
        )

        print(
            f"Influence entropy: "
            f"{val_metrics['influence_entropy']:.6f}"
        )

        print(
            f"Gate entropy     : "
            f"{val_metrics['gate_entropy']:.6f}"
        )

        # Save current model
        if CONFIG[
            "save_every_epoch"
        ]:

            epoch_model = (
                run_dir
                / "models"
                / f"epoch_{epoch:03d}.keras"
            )

            model.save(
                epoch_model
            )

        # Best model based only on validation RMSE.
        if (
            val_metrics["rmse"]
            < best_val_rmse
        ):

            best_val_rmse = (
                val_metrics[
                    "rmse"
                ]
            )

            best_model_path = (
                run_dir
                / "models"
                / "best_model.keras"
            )

            model.save(
                best_model_path
            )

            save_json(
                run_dir
                / "best_model.json",
                {
                    "epoch": epoch,
                    "validation_rmse":
                        best_val_rmse,
                    "path":
                        str(
                            best_model_path
                        ),
                },
            )

        save_csv(
            run_dir
            / "tables"
            / "training_history.csv",
            history,
        )

    total_training_time = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    final_model_path = (
        run_dir
        / "models"
        / "final_model.keras"
    )

    model.save(
        final_model_path
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    val_metrics = (
        evaluate_dataset(
            model,
            val_ds,
        )
    )

    # --------------------------------------------------------
    # HELD-OUT TEST
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("HELD-OUT TEST")
    print("=" * 80)

    test_metrics = (
        evaluate_dataset(
            model,
            test_ds,
        )
    )

    latency_metrics = (
        measure_latency(
            model,
            test_ds,
        )
    )

    # --------------------------------------------------------
    # SAVE FINAL RESULTS
    # --------------------------------------------------------

    results = {
        "experiment":
            "SACU_V2_core_validation",

        "methodology": {
            "target_used_for_influence":
                False,

            "deployment_influence_signals": [
                "observed_sensor_consistency",
                "local_wave_equation_residual",
                "gate_entropy",
            ],

            "sensor_weight":
                CONFIG[
                    "sensor_weight"
                ],

            "physics_weight":
                CONFIG[
                    "physics_weight"
                ],

            "entropy_weight":
                CONFIG[
                    "entropy_weight"
                ],

            "temperature":
                CONFIG[
                    "temperature"
                ],
        },

        "dataset": {
            "total":
                len(files),

            "train":
                len(train_files),

            "validation":
                len(val_files),

            "test":
                len(test_files),
        },

        "model": {
            "grid":
                CONFIG[
                    "grid"
                ],

            "number_of_sacus":
                CONFIG[
                    "grid"
                ]
                ** 2,

            "micro_experts":
                CONFIG[
                    "K"
                ],

            "hidden":
                CONFIG[
                    "hidden"
                ],

            "msg_dim":
                CONFIG[
                    "msg_dim"
                ],

            "parameter_count":
                int(
                    model.count_params()
                ),
        },

        "training": {
            "epochs":
                CONFIG[
                    "epochs"
                ],

            "learning_rate":
                CONFIG[
                    "learning_rate"
                ],

            "total_training_time_sec":
                float(
                    total_training_time
                ),

            "best_validation_rmse":
                float(
                    best_val_rmse
                ),
        },

        "validation":
            val_metrics,

        "test":
            test_metrics,

        "latency":
            latency_metrics,

        "final_model":
            str(
                final_model_path
            ),
    }

    save_json(
        run_dir / "results.json",
        results,
    )

    # Compact paper-oriented row
    paper_row = [{
        "Model":
            "SACU V2",

        "MAE_mean":
            test_metrics[
                "mae"
            ],

        "MAE_std":
            test_metrics[
                "mae_std"
            ],

        "RMSE_mean":
            test_metrics[
                "rmse"
            ],

        "RMSE_std":
            test_metrics[
                "rmse_std"
            ],

        "Residual_mean":
            test_metrics[
                "residual"
            ],

        "Residual_std":
            test_metrics[
                "residual_std"
            ],

        "Latency_mean_sec":
            latency_metrics[
                "latency_mean_sec"
            ],

        "Latency_p95_sec":
            latency_metrics[
                "latency_p95_sec"
            ],

        "Latency_p99_sec":
            latency_metrics[
                "latency_p99_sec"
            ],

        "Influence_entropy":
            test_metrics[
                "influence_entropy"
            ],

        "Gate_entropy":
            test_metrics[
                "gate_entropy"
            ],

        "Test_count":
            len(
                test_files
            ),
    }]

    save_csv(
        run_dir
        / "tables"
        / "test_summary.csv",
        paper_row,
    )

    print()
    print("=" * 80)
    print("SACU V2 TRAINING COMPLETED")
    print("=" * 80)

    print(
        f"Run directory:\n{run_dir}"
    )

    print()

    print(
        f"Test MAE      : "
        f"{test_metrics['mae']:.6f}"
    )

    print(
        f"Test RMSE     : "
        f"{test_metrics['rmse']:.6f}"
    )

    print(
        f"Test residual : "
        f"{test_metrics['residual']:.6f}"
    )

    print(
        f"Latency mean  : "
        f"{latency_metrics['latency_mean_sec']:.6f} sec"
    )

    print(
        f"Latency p95   : "
        f"{latency_metrics['latency_p95_sec']:.6f} sec"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "These results come from target-free "
        "SACU V2 inference."
    )


if __name__ == "__main__":
    main()