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

SEQ_DIR = ORIGINAL_ROOT / "data" / "sim" / "sequences"

OUTPUT_ROOT = (
    NEW_ROOT
    / "outputs"
    / "organizational_evolution"
)

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
# CONFIGURATION
# ============================================================

CONFIG = {
    "seed": 42,

    "train_fraction": 0.70,
    "val_fraction": 0.15,
    "test_fraction": 0.15,

    "batch_size": 1,
    "epochs": 5,

    "grid": 4,
    "overlap": 8,
    "K": 4,
    "hidden": 64,
    "msg_dim": 16,
    "use_role": True,
    "use_comms": True,

    "learning_rate": 1e-3,

    "use_physics_loss": True,
    "lambda_phys": 0.05,

    "sensor_weight": 0.50,
    "physics_weight": 0.35,
    "entropy_weight": 0.15,
    "temperature": 5.0,

    # How many batches to log deeply per epoch.
    # These are diagnostic organizational logs, not replacements
    # for the manuscript's reported experimental numbers.
    "deep_log_train_batches": 5,
    "deep_log_val_batches": 5,

    "save_model_every_epoch": True,
}


# ============================================================
# GENERAL HELPERS
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


def save_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
        print("[INFO] TensorFlow deterministic operations enabled.")
    except Exception as exc:
        print("[WARNING] Could not enable deterministic ops:", exc)


def get_environment_info() -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu_devices": [
            str(x)
            for x in tf.config.list_physical_devices("GPU")
        ],
        "tensorflow_build":
            tf.sysconfig.get_build_info(),
    }


# ============================================================
# DATASET
# ============================================================

def list_sequence_files() -> list[str]:
    files = sorted(
        str(p)
        for p in SEQ_DIR.glob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No .npz files found in {SEQ_DIR}"
        )

    return files


def split_files(files: list[str]):
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

    return train_files, val_files, test_files


# ============================================================
# ENTROPY / SPECIALIZATION METRICS
# ============================================================

def entropy_rows(probs: tf.Tensor) -> tf.Tensor:
    """
    probs shape (..., K)
    returns entropy over K.
    """
    probs = tf.clip_by_value(
        probs,
        1e-8,
        1.0,
    )

    return -tf.reduce_sum(
        probs * tf.math.log(probs),
        axis=-1,
    )


def normalized_entropy_rows(
    probs: tf.Tensor,
) -> tf.Tensor:
    """
    Entropy normalized by log(K).
    0 = completely concentrated.
    1 = uniform.
    """
    K = tf.cast(
        tf.shape(probs)[-1],
        tf.float32,
    )

    h = entropy_rows(probs)

    return (
        h
        / tf.math.log(
            tf.maximum(K, 2.0)
        )
    )


def effective_number_from_entropy(
    entropy: tf.Tensor,
) -> tf.Tensor:
    """
    exp(H) = effective number of active categories.
    """
    return tf.exp(entropy)


# ============================================================
# INTER-AGENT DISAGREEMENT
# ============================================================

def compute_pairwise_patch_disagreement(
    patch_outs,
    regions,
):
    """
    Measure prediction disagreement only where SACU patches overlap.

    Returns a list of dicts:
        agent_i
        agent_j
        overlap_voxels
        mean_absolute_disagreement

    This is actual model behavior.
    It does not use y_true.
    """

    rows = []

    N = len(patch_outs)

    for i in range(N):
        y0_i, y1_i, x0_i, x1_i = (
            regions[i]
        )

        for j in range(i + 1, N):
            y0_j, y1_j, x0_j, x1_j = (
                regions[j]
            )

            y0 = max(y0_i, y0_j)
            y1 = min(y1_i, y1_j)

            x0 = max(x0_i, x0_j)
            x1 = min(x1_i, x1_j)

            if y1 <= y0 or x1 <= x0:
                continue

            pi = patch_outs[i][
                :,
                :,
                y0 - y0_i:y1 - y0_i,
                x0 - x0_i:x1 - x0_i,
                :
            ]

            pj = patch_outs[j][
                :,
                :,
                y0 - y0_j:y1 - y0_j,
                x0 - x0_j:x1 - x0_j,
                :
            ]

            disagreement = tf.reduce_mean(
                tf.abs(pi - pj)
            )

            rows.append(
                {
                    "agent_i": i,
                    "agent_j": j,
                    "overlap_height":
                        int(y1 - y0),
                    "overlap_width":
                        int(x1 - x0),
                    "mean_absolute_disagreement":
                        float(
                            disagreement.numpy()
                        ),
                }
            )

    return rows


# ============================================================
# ORGANIZATIONAL SNAPSHOT
# ============================================================

def collect_organizational_snapshot(
    model,
    features,
    y_true,
    epoch: int,
    batch_index: int,
    split_name: str,
):
    """
    Collect one detailed organizational snapshot.

    IMPORTANT:
    y_true is used only for final reconstruction metrics.
    It is NOT used to compute influence weights, gates,
    organizational scores, or patch selection.
    """

    x = features["x"]
    c = features["c_field"]
    dt = features["dt"]
    dx = features["dx"]

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    _, aux = model(
        x,
        training=False,
    )

    patch_outs = aux["patch_outs"]
    regions = aux["regions"]
    gates = aux["gates"]

    # --------------------------------------------------------
    # Deployment-valid final prediction
    # --------------------------------------------------------

    y_pred, diagnostics = (
        predict_sacu_deployment(
            model=model,
            x=x,
            c_field=c,
            dt=dt,
            dx=dx,
            training=False,

            sensor_weight=
                CONFIG["sensor_weight"],

            physics_weight=
                CONFIG["physics_weight"],

            entropy_weight=
                CONFIG["entropy_weight"],

            temperature=
                CONFIG["temperature"],
        )
    )

    weights = diagnostics[
        "influence_weights"
    ]

    sensor_score = diagnostics[
        "sensor_score"
    ]

    physics_score = diagnostics[
        "physics_score"
    ]

    composite_score = diagnostics[
        "composite_score"
    ]

    gates_tensor = diagnostics[
        "gates"
    ]
    # shape B,N,K

    B = int(
        tf.shape(
            gates_tensor
        )[0]
    )

    N = int(
        tf.shape(
            gates_tensor
        )[1]
    )

    K = int(
        tf.shape(
            gates_tensor
        )[2]
    )

    # --------------------------------------------------------
    # Final metrics — y_true begins here
    # --------------------------------------------------------

    global_mae = float(
        mae(
            y_true,
            y_pred,
        ).numpy()
    )

    global_rmse = float(
        rmse(
            y_true,
            y_pred,
        ).numpy()
    )

    global_residual = float(
        wave_residual_norm(
            y_pred,
            c,
            dt,
            dx,
        ).numpy()
    )

    # --------------------------------------------------------
    # Agent-level logs
    # --------------------------------------------------------

    agent_rows = []

    for b in range(B):

        for agent_id in range(N):

            g = gates_tensor[
                b,
                agent_id,
                :
            ]

            h = float(
                entropy_rows(
                    g
                ).numpy()
            )

            hn = float(
                normalized_entropy_rows(
                    g
                ).numpy()
            )

            effective_experts = float(
                effective_number_from_entropy(
                    tf.constant(h)
                ).numpy()
            )

            dominant_expert = int(
                tf.argmax(
                    g
                ).numpy()
            )

            max_gate = float(
                tf.reduce_max(
                    g
                ).numpy()
            )

            influence = float(
                weights[
                    b,
                    agent_id
                ].numpy()
            )

            row = {
                "epoch":
                    epoch,

                "split":
                    split_name,

                "batch_index":
                    batch_index,

                "sample_in_batch":
                    b,

                "agent_id":
                    agent_id,

                "dominant_expert":
                    dominant_expert,

                "dominant_expert_probability":
                    max_gate,

                "gate_entropy":
                    h,

                "gate_entropy_normalized":
                    hn,

                "effective_number_of_experts":
                    effective_experts,

                "influence_weight":
                    influence,

                "sensor_consistency_score":
                    float(
                        sensor_score[
                            b,
                            agent_id
                        ].numpy()
                    ),

                "local_physics_score":
                    float(
                        physics_score[
                            b,
                            agent_id
                        ].numpy()
                    ),

                "composite_influence_score":
                    float(
                        composite_score[
                            b,
                            agent_id
                        ].numpy()
                    ),

                "global_mae":
                    global_mae,

                "global_rmse":
                    global_rmse,

                "global_residual":
                    global_residual,
            }

            for k in range(K):
                row[
                    f"gate_expert_{k}"
                ] = float(
                    g[k].numpy()
                )

            agent_rows.append(row)

    # --------------------------------------------------------
    # System-level organizational metrics
    # --------------------------------------------------------

    influence_h = entropy_rows(
        weights
    )

    influence_h_norm = (
        influence_h
        / tf.math.log(
            tf.cast(
                N,
                tf.float32,
            )
        )
    )

    gate_h = entropy_rows(
        gates_tensor
    )

    gate_h_norm = (
        gate_h
        / tf.math.log(
            tf.cast(
                K,
                tf.float32,
            )
        )
    )

    dominant = tf.argmax(
        gates_tensor,
        axis=-1,
    )

    system_row = {
        "epoch":
            epoch,

        "split":
            split_name,

        "batch_index":
            batch_index,

        "global_mae":
            global_mae,

        "global_rmse":
            global_rmse,

        "global_residual":
            global_residual,

        "mean_influence_entropy":
            float(
                tf.reduce_mean(
                    influence_h
                ).numpy()
            ),

        "mean_influence_entropy_normalized":
            float(
                tf.reduce_mean(
                    influence_h_norm
                ).numpy()
            ),

        "mean_gate_entropy":
            float(
                tf.reduce_mean(
                    gate_h
                ).numpy()
            ),

        "mean_gate_entropy_normalized":
            float(
                tf.reduce_mean(
                    gate_h_norm
                ).numpy()
            ),

        "mean_max_influence":
            float(
                tf.reduce_mean(
                    tf.reduce_max(
                        weights,
                        axis=1,
                    )
                ).numpy()
            ),

        "mean_max_gate_probability":
            float(
                tf.reduce_mean(
                    tf.reduce_max(
                        gates_tensor,
                        axis=-1,
                    )
                ).numpy()
            ),

        "number_of_agents":
            N,

        "number_of_micro_experts":
            K,
    }

    # Count dominant-expert usage.
    dominant_np = (
        dominant.numpy()
        .reshape(-1)
    )

    counts = np.bincount(
        dominant_np,
        minlength=K,
    )

    for k in range(K):
        system_row[
            f"dominant_expert_{k}_count"
        ] = int(
            counts[k]
        )

    # --------------------------------------------------------
    # Inter-agent disagreement
    # --------------------------------------------------------

    disagreement_rows = (
        compute_pairwise_patch_disagreement(
            patch_outs,
            regions,
        )
    )

    for row in disagreement_rows:
        row[
            "epoch"
        ] = epoch

        row[
            "split"
        ] = split_name

        row[
            "batch_index"
        ] = batch_index

    return (
        agent_rows,
        system_row,
        disagreement_rows,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

def train_epoch(
    trainer,
    train_ds,
):
    losses = []
    maes = []
    rmses = []
    residuals = []

    start = time.perf_counter()

    for features, y_true in train_ds:

        x = features["x"]
        c = features["c_field"]
        dt = features["dt"]
        dx = features["dx"]

        (
            loss,
            m,
            r,
            res,
        ) = trainer.train_step_sacu(
            x,
            y_true,
            c,
            dt,
            dx,
        )

        losses.append(
            float(
                loss.numpy()
            )
        )

        maes.append(
            float(
                m.numpy()
            )
        )

        rmses.append(
            float(
                r.numpy()
            )
        )

        residuals.append(
            float(
                res.numpy()
            )
        )

    return {
        "train_loss":
            float(
                np.mean(
                    losses
                )
            ),

        "train_mae":
            float(
                np.mean(
                    maes
                )
            ),

        "train_rmse":
            float(
                np.mean(
                    rmses
                )
            ),

        "train_residual":
            float(
                np.mean(
                    residuals
                )
            ),

        "train_time_sec":
            float(
                time.perf_counter()
                - start
            ),
    }


# ============================================================
# STANDARD DATASET EVALUATION
# ============================================================

def evaluate_dataset(
    model,
    dataset,
):
    maes = []
    rmses = []
    residuals = []

    for features, y_true in dataset:

        x = features["x"]
        c = features["c_field"]
        dt = features["dt"]
        dx = features["dx"]

        y_pred, _ = (
            predict_sacu_deployment(
                model,
                x,
                c,
                dt,
                dx,
                training=False,

                sensor_weight=
                    CONFIG[
                        "sensor_weight"
                    ],

                physics_weight=
                    CONFIG[
                        "physics_weight"
                    ],

                entropy_weight=
                    CONFIG[
                        "entropy_weight"
                    ],

                temperature=
                    CONFIG[
                        "temperature"
                    ],
            )
        )

        maes.append(
            float(
                mae(
                    y_true,
                    y_pred,
                ).numpy()
            )
        )

        rmses.append(
            float(
                rmse(
                    y_true,
                    y_pred,
                ).numpy()
            )
        )

        residuals.append(
            float(
                wave_residual_norm(
                    y_pred,
                    c,
                    dt,
                    dx,
                ).numpy()
            )
        )

    return {
        "mae":
            float(
                np.mean(
                    maes
                )
            ),

        "rmse":
            float(
                np.mean(
                    rmses
                )
            ),

        "residual":
            float(
                np.mean(
                    residuals
                )
            ),
    }


# ============================================================
# DEEP LOGGING
# ============================================================

def log_dataset_snapshots(
    model,
    dataset,
    epoch,
    split_name,
    max_batches,
):
    all_agent_rows = []
    all_system_rows = []
    all_disagreement_rows = []

    for batch_index, (
        features,
        y_true,
    ) in enumerate(dataset):

        if batch_index >= max_batches:
            break

        (
            agent_rows,
            system_row,
            disagreement_rows,
        ) = collect_organizational_snapshot(
            model=model,
            features=features,
            y_true=y_true,
            epoch=epoch,
            batch_index=batch_index,
            split_name=split_name,
        )

        all_agent_rows.extend(
            agent_rows
        )

        all_system_rows.append(
            system_row
        )

        all_disagreement_rows.extend(
            disagreement_rows
        )

    return (
        all_agent_rows,
        all_system_rows,
        all_disagreement_rows,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_reproducibility(
        CONFIG["seed"]
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    run_dir = ensure_dir(
        OUTPUT_ROOT
        / f"org_evolution_{timestamp}"
    )

    model_dir = ensure_dir(
        run_dir / "models"
    )

    table_dir = ensure_dir(
        run_dir / "tables"
    )

    split_dir = ensure_dir(
        run_dir / "splits"
    )

    print()
    print("=" * 80)
    print(
        "ORGANIZATIONAL EVOLUTION EXPERIMENT"
    )
    print("=" * 80)

    print(
        "Run directory:",
        run_dir,
    )

    # --------------------------------------------------------
    # Save experiment metadata
    # --------------------------------------------------------

    save_json(
        run_dir
        / "config.json",
        CONFIG,
    )

    save_json(
        run_dir
        / "environment.json",
        get_environment_info(),
    )

    implementation_scope = {
        "measured_dynamically": [
            "influence_weights",
            "gate_probabilities",
            "dominant_micro_expert",
            "gate_entropy",
            "influence_entropy",
            "sensor_consistency",
            "local_physics_score",
            "inter_agent_prediction_disagreement",
        ],

        "not_dynamically_implemented_in_current_solver": [
            "role_reallocation",
            "role_switch_events",
            "learned_communication_edge_weights",
            "communication_topology_changes",
        ],

        "reason": (
            "The current OrgSACUSolver assigns role IDs "
            "deterministically from agent index and uses a "
            "fixed neighbor graph. These quantities are therefore "
            "not reported as dynamic empirical results."
        ),
    }

    save_json(
        run_dir
        / "implementation_scope.json",
        implementation_scope,
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

    save_json(
        run_dir
        / "split_summary.json",
        {
            "total":
                len(files),

            "train":
                len(train_files),

            "validation":
                len(val_files),

            "test":
                len(test_files),

            "note":
                (
                    "These counts belong only to this new "
                    "reviewer-driven organizational-evolution experiment. "
                    "They do not replace any number already reported "
                    "in the current manuscript."
                ),
        },
    )

    (
        split_dir
        / "train_files.txt"
    ).write_text(
        "\n".join(
            train_files
        ),
        encoding="utf-8",
    )

    (
        split_dir
        / "validation_files.txt"
    ).write_text(
        "\n".join(
            val_files
        ),
        encoding="utf-8",
    )

    (
        split_dir
        / "test_files.txt"
    ).write_text(
        "\n".join(
            test_files
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # TensorFlow datasets
    # --------------------------------------------------------

    train_ds = make_dataset(
        train_files,
        batch_size=
            CONFIG["batch_size"],
        shuffle=True,
        repeat=False,
        deterministic=True,
    )

    train_log_ds = make_dataset(
        train_files,
        batch_size=
            CONFIG["batch_size"],
        shuffle=False,
        repeat=False,
        deterministic=True,
    )

    val_ds = make_dataset(
        val_files,
        batch_size=
            CONFIG["batch_size"],
        shuffle=False,
        repeat=False,
        deterministic=True,
    )

    test_ds = make_dataset(
        test_files,
        batch_size=
            CONFIG["batch_size"],
        shuffle=False,
        repeat=False,
        deterministic=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = OrgSACUSolver(
        grid=
            CONFIG["grid"],

        overlap=
            CONFIG["overlap"],

        K=
            CONFIG["K"],

        hidden=
            CONFIG["hidden"],

        msg_dim=
            CONFIG["msg_dim"],

        use_role=
            CONFIG["use_role"],

        use_comms=
            CONFIG["use_comms"],
    )

    first_features, _ = next(
        iter(
            train_ds.take(1)
        )
    )

    _ = model(
        first_features["x"],
        training=False,
    )

    print(
        "Trainable parameters:",
        model.count_params(),
    )

    optimizer = (
        keras.optimizers.Adam(
            CONFIG[
                "learning_rate"
            ]
        )
    )

    trainer = TrainerV2(
        model=model,
        optimizer=optimizer,

        use_physics_loss=
            CONFIG[
                "use_physics_loss"
            ],

        lambda_phys=
            CONFIG[
                "lambda_phys"
            ],

        sensor_weight=
            CONFIG[
                "sensor_weight"
            ],

        physics_weight=
            CONFIG[
                "physics_weight"
            ],

        entropy_weight=
            CONFIG[
                "entropy_weight"
            ],

        temperature=
            CONFIG[
                "temperature"
            ],
    )

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    history_rows = []

    all_agent_rows = []
    all_system_rows = []
    all_disagreement_rows = []

    # --------------------------------------------------------
    # Snapshot before training
    # --------------------------------------------------------

    print()
    print(
        "[INFO] Logging epoch-0 organizational state..."
    )

    (
        agent_rows,
        system_rows,
        disagreement_rows,
    ) = log_dataset_snapshots(
        model,
        val_ds,
        epoch=0,
        split_name="validation",
        max_batches=
            CONFIG[
                "deep_log_val_batches"
            ],
    )

    all_agent_rows.extend(
        agent_rows
    )

    all_system_rows.extend(
        system_rows
    )

    all_disagreement_rows.extend(
        disagreement_rows
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_rmse = np.inf
    best_epoch = None

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
            train_epoch(
                trainer,
                train_ds,
            )
        )

        val_metrics = (
            evaluate_dataset(
                model,
                val_ds,
            )
        )

        history_row = {
            "epoch":
                epoch,

            **train_metrics,

            "val_mae":
                val_metrics["mae"],

            "val_rmse":
                val_metrics["rmse"],

            "val_residual":
                val_metrics[
                    "residual"
                ],
        }

        history_rows.append(
            history_row
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

        # ----------------------------------------------------
        # Deep organizational snapshots
        # ----------------------------------------------------

        (
            tr_agent,
            tr_system,
            tr_disagreement,
        ) = log_dataset_snapshots(
            model,
            train_log_ds,
            epoch=epoch,
            split_name="train",
            max_batches=
                CONFIG[
                    "deep_log_train_batches"
                ],
        )

        (
            va_agent,
            va_system,
            va_disagreement,
        ) = log_dataset_snapshots(
            model,
            val_ds,
            epoch=epoch,
            split_name="validation",
            max_batches=
                CONFIG[
                    "deep_log_val_batches"
                ],
        )

        all_agent_rows.extend(
            tr_agent
        )

        all_agent_rows.extend(
            va_agent
        )

        all_system_rows.extend(
            tr_system
        )

        all_system_rows.extend(
            va_system
        )

        all_disagreement_rows.extend(
            tr_disagreement
        )

        all_disagreement_rows.extend(
            va_disagreement
        )

        # ----------------------------------------------------
        # Model saving
        # ----------------------------------------------------

        if CONFIG[
            "save_model_every_epoch"
        ]:

            model.save(
                model_dir
                / f"epoch_{epoch:03d}.keras"
            )

        if (
            val_metrics["rmse"]
            < best_val_rmse
        ):

            best_val_rmse = (
                val_metrics[
                    "rmse"
                ]
            )

            best_epoch = epoch

            model.save(
                model_dir
                / "best_model.keras"
            )

        # Save continuously.
        save_csv(
            table_dir
            / "training_history.csv",
            history_rows,
        )

        save_csv(
            table_dir
            / "agent_evolution.csv",
            all_agent_rows,
        )

        save_csv(
            table_dir
            / "system_evolution.csv",
            all_system_rows,
        )

        save_csv(
            table_dir
            / "inter_agent_disagreement.csv",
            all_disagreement_rows,
        )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model.save(
        model_dir
        / "final_model.keras"
    )

    # --------------------------------------------------------
    # Final test of current trained state
    # --------------------------------------------------------

    test_metrics = (
        evaluate_dataset(
            model,
            test_ds,
        )
    )

    # Test organizational snapshot.
    (
        te_agent,
        te_system,
        te_disagreement,
    ) = log_dataset_snapshots(
        model,
        test_ds,
        epoch=
            CONFIG["epochs"],
        split_name="test",
        max_batches=
            min(
                5,
                len(
                    test_files
                ),
            ),
    )

    all_agent_rows.extend(
        te_agent
    )

    all_system_rows.extend(
        te_system
    )

    all_disagreement_rows.extend(
        te_disagreement
    )

    save_csv(
        table_dir
        / "agent_evolution.csv",
        all_agent_rows,
    )

    save_csv(
        table_dir
        / "system_evolution.csv",
        all_system_rows,
    )

    save_csv(
        table_dir
        / "inter_agent_disagreement.csv",
        all_disagreement_rows,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    results = {
        "experiment":
            "reviewer_requested_organizational_evolution_analysis",

        "new_experiment":
            True,

        "replaces_existing_manuscript_numbers":
            False,

        "best_validation_epoch":
            best_epoch,

        "best_validation_rmse":
            float(
                best_val_rmse
            ),

        "test_metrics_of_this_new_experiment":
            test_metrics,

        "measured_organizational_quantities":
            implementation_scope[
                "measured_dynamically"
            ],

        "not_claimed_as_dynamic":
            implementation_scope[
                "not_dynamically_implemented_in_current_solver"
            ],

        "important_methodological_note":
            (
                "All influence weights used in this experiment "
                "were computed without y_true. Ground truth was "
                "used only after prediction for supervised training "
                "loss and evaluation metrics."
            ),
    }

    save_json(
        run_dir
        / "results.json",
        results,
    )

    print()
    print("=" * 80)
    print(
        "ORGANIZATIONAL EVOLUTION RUN COMPLETED"
    )
    print("=" * 80)

    print(
        "Run directory:"
    )
    print(
        run_dir
    )

    print()
    print(
        "New-experiment Test MAE:",
        f"{test_metrics['mae']:.6f}",
    )

    print(
        "New-experiment Test RMSE:",
        f"{test_metrics['rmse']:.6f}",
    )

    print(
        "New-experiment Test residual:",
        f"{test_metrics['residual']:.6f}",
    )

    print()
    print(
        "These values belong only to this new "
        "reviewer-requested experiment."
    )

    print(
        "They do not replace numerical values "
        "already reported in the current manuscript."
    )


if __name__ == "__main__":
    main()