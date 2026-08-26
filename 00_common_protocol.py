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
from typing import Any

import numpy as np
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

ORIGINAL_ROOT = Path(
    r"D:\47\472\New-Papers\GIS\Codes"
)

NEW_ROOT = (
    ORIGINAL_ROOT
    / "New_Branch"
)

DATA_ROOT = (
    ORIGINAL_ROOT
    / "data"
)

SEQUENCE_DIR = (
    DATA_ROOT
    / "sim"
    / "sequences"
)

EXPERIMENT_ROOT = (
    NEW_ROOT
    / "experiments"
    / "03_baseline_comparison"
)

OUTPUT_ROOT = (
    NEW_ROOT
    / "outputs"
    / "baseline_comparison"
)


# Make the new branch authoritative.
if str(NEW_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(NEW_ROOT),
    )


# ============================================================
# COMMON EXPERIMENTAL CONFIGURATION
# ============================================================

COMMON_CONFIG = {

    # --------------------------------------------------------
    # Experimental identity
    # --------------------------------------------------------

    "experiment_family":
        "reviewer_requested_baseline_comparison",

    "new_experiment":
        True,

    "replaces_existing_manuscript_numbers":
        False,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    "seed":
        42,

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    "train_fraction":
        0.70,

    "validation_fraction":
        0.15,

    "test_fraction":
        0.15,

    # --------------------------------------------------------
    # Input / output
    # --------------------------------------------------------

    "input_channels":
        2,

    "output_channels":
        1,

    "input_semantics": [
        "observed_field",
        "sensor_mask",
    ],

    "target_semantics":
        "full_propagation_field",

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    "batch_size":
        1,

    "epochs":
        5,

    "learning_rate":
        1e-3,

    "optimizer":
        "Adam",

    # --------------------------------------------------------
    # Physics
    # --------------------------------------------------------

    "physics_operator":
        "second_order_wave_equation",

    "use_common_wave_residual":
        True,

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    "metrics": [
        "MAE",
        "RMSE",
        "wave_residual",
        "inference_latency",
        "parameter_count",
    ],

    # --------------------------------------------------------
    # Latency measurement
    # --------------------------------------------------------

    "latency_warmup_runs":
        3,

    "latency_measurement_runs":
        10,

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    "selection_metric":
        "validation_RMSE",

    "test_policy":
        "evaluate_best_validation_checkpoint_only",

    # --------------------------------------------------------
    # Reviewer-response rule
    # --------------------------------------------------------

    "result_policy":
        (
            "All values produced by this baseline suite are "
            "new reviewer-requested experimental results. "
            "They do not replace or modify numerical values "
            "already reported in the current manuscript."
        ),
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_reproducibility(
    seed: int | None = None,
) -> int:

    if seed is None:
        seed = int(
            COMMON_CONFIG["seed"]
        )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()

        print(
            "[INFO] TensorFlow deterministic "
            "operations enabled."
        )

    except Exception as exc:

        print(
            "[WARNING] TensorFlow deterministic "
            f"operations could not be enabled: {exc}"
        )

    return seed


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def ensure_dir(
    path: Path,
) -> Path:

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def create_run_directory(
    model_name: str,
) -> Path:

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    run_dir = ensure_dir(
        OUTPUT_ROOT
        / safe_name
        / f"{safe_name}_{timestamp}"
    )

    for subfolder in [
        "models",
        "tables",
        "logs",
        "splits",
        "figures",
        "raw",
    ]:

        ensure_dir(
            run_dir
            / subfolder
        )

    return run_dir


# ============================================================
# JSON / CSV HELPERS
# ============================================================

def save_json(
    path: Path,
    obj: Any,
) -> None:

    ensure_dir(
        path.parent
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def save_csv(
    path: Path,
    rows: list[dict],
) -> None:

    ensure_dir(
        path.parent
    )

    if not rows:

        path.write_text(
            "",
            encoding="utf-8",
        )

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

        writer.writerows(
            rows
        )


# ============================================================
# DATASET FILE DISCOVERY
# ============================================================

def list_sequence_files() -> list[str]:

    files = sorted(
        str(path)
        for path
        in SEQUENCE_DIR.glob(
            "*.npz"
        )
    )

    if not files:

        raise FileNotFoundError(
            "No sequence files found in:\n"
            f"{SEQUENCE_DIR}"
        )

    return files


# ============================================================
# FIXED SPLIT
# ============================================================

def split_sequence_files(
    files: list[str],
):

    n = len(files)

    train_end = int(
        COMMON_CONFIG[
            "train_fraction"
        ]
        * n
    )

    validation_end = int(
        (
            COMMON_CONFIG[
                "train_fraction"
            ]
            +
            COMMON_CONFIG[
                "validation_fraction"
            ]
        )
        * n
    )

    train_files = (
        files[
            :train_end
        ]
    )

    validation_files = (
        files[
            train_end:
            validation_end
        ]
    )

    test_files = (
        files[
            validation_end:
        ]
    )

    return (
        train_files,
        validation_files,
        test_files,
    )


# ============================================================
# SPLIT MANIFEST
# ============================================================

def save_split_manifest(
    run_dir: Path,
    train_files: list[str],
    validation_files: list[str],
    test_files: list[str],
) -> None:

    split_dir = (
        run_dir
        / "splits"
    )

    ensure_dir(
        split_dir
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
            validation_files
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

    save_json(
        run_dir
        / "split_summary.json",
        {
            "total":
                (
                    len(train_files)
                    + len(validation_files)
                    + len(test_files)
                ),

            "train":
                len(
                    train_files
                ),

            "validation":
                len(
                    validation_files
                ),

            "test":
                len(
                    test_files
                ),

            "split_method":
                (
                    "sorted sequence filenames using "
                    "70/15/15 positional partition"
                ),

            "important_note":
                (
                    "This split belongs only to the new "
                    "reviewer-requested comparison experiments "
                    "and does not replace the experimental "
                    "numbers already reported in the manuscript."
                ),
        },
    )


# ============================================================
# ENVIRONMENT INFORMATION
# ============================================================

def collect_environment_info() -> dict:

    try:

        keras_version = (
            tf.keras.__version__
        )

    except Exception:

        keras_version = (
            "bundled_tf_keras"
        )

    return {

        "timestamp":
            datetime.now().isoformat(),

        "python_version":
            sys.version,

        "tensorflow_version":
            tf.__version__,

        "keras_version":
            keras_version,

        "numpy_version":
            np.__version__,

        "platform":
            platform.platform(),

        "processor":
            platform.processor(),

        "physical_cpus":
            [
                str(device)
                for device
                in tf.config.list_physical_devices(
                    "CPU"
                )
            ],

        "physical_gpus":
            [
                str(device)
                for device
                in tf.config.list_physical_devices(
                    "GPU"
                )
            ],

        "tensorflow_build_info":
            tf.sysconfig.get_build_info(),
    }


# ============================================================
# COMMON METRICS
# ============================================================

def mae(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:

    return tf.reduce_mean(
        tf.abs(
            y_true
            - y_pred
        )
    )


def rmse(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:

    return tf.sqrt(
        tf.reduce_mean(
            tf.square(
                y_true
                - y_pred
            )
        )
        + 1e-12
    )


# ============================================================
# COMMON WAVE RESIDUAL
# ============================================================

def wave_residual_per_sample(
    y_pred: tf.Tensor,
    c_field: tf.Tensor,
    dt: tf.Tensor,
    dx: tf.Tensor,
) -> tf.Tensor:

    """
    Common second-order wave-equation residual.

    Every new baseline MUST use this exact residual
    implementation so that residual comparisons are fair.

    Equation:
        u[t+1] - 2u[t] + u[t-1]
        - dt^2 c^2 Laplacian(u[t])
    """

    u = tf.squeeze(
        tf.cast(
            y_pred,
            tf.float32,
        ),
        axis=-1,
    )

    B = tf.shape(u)[0]

    H = tf.shape(u)[2]

    W = tf.shape(u)[3]

    c_field = tf.cast(
        c_field,
        tf.float32,
    )

    if c_field.shape.rank == 2:

        c = tf.broadcast_to(
            c_field[
                None,
                ...
            ],
            [
                B,
                H,
                W,
            ],
        )

    else:

        c = c_field

    dt_scalar = tf.reshape(
        tf.cast(
            dt,
            tf.float32,
        ),
        [-1],
    )[0]

    dx_scalar = tf.reshape(
        tf.cast(
            dx,
            tf.float32,
        ),
        [-1],
    )[0]

    u_prev = (
        u[
            :,
            :-2,
            :,
            :
        ]
    )

    u_mid = (
        u[
            :,
            1:-1,
            :,
            :
        ]
    )

    u_next = (
        u[
            :,
            2:,
            :,
            :
        ]
    )

    BT = (
        tf.shape(
            u_mid
        )[0]
        *
        tf.shape(
            u_mid
        )[1]
    )

    u2 = tf.reshape(
        u_mid,
        [
            BT,
            H,
            W,
            1,
        ],
    )

    kernel = tf.constant(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=tf.float32,
    )

    kernel = (
        kernel[
            :,
            :,
            None,
            None,
        ]
    )

    u2_padded = tf.pad(
        u2,
        [
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 0],
        ],
        mode="REFLECT",
    )

    laplacian = tf.nn.conv2d(
        u2_padded,
        kernel,
        strides=1,
        padding="VALID",
    )

    laplacian = tf.reshape(
        laplacian[
            ...,
            0
        ],
        [
            B,
            tf.shape(
                u_mid
            )[1],
            H,
            W,
        ],
    )

    laplacian = (
        laplacian
        /
        (
            dx_scalar
            * dx_scalar
            + 1e-8
        )
    )

    residual = (
        (
            u_next
            - 2.0
            * u_mid
            + u_prev
        )
        -
        (
            dt_scalar
            * dt_scalar
        )
        *
        tf.expand_dims(
            c
            * c,
            axis=1,
        )
        *
        laplacian
    )

    return tf.sqrt(
        tf.reduce_mean(
            tf.square(
                residual
            ),
            axis=[
                1,
                2,
                3,
            ],
        )
        + 1e-8
    )


def wave_residual_mean(
    y_pred: tf.Tensor,
    c_field: tf.Tensor,
    dt: tf.Tensor,
    dx: tf.Tensor,
) -> tf.Tensor:

    return tf.reduce_mean(
        wave_residual_per_sample(
            y_pred,
            c_field,
            dt,
            dx,
        )
    )


# ============================================================
# COMMON MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataset,
    predict_function=None,
) -> dict:

    """
    Evaluate any baseline model using exactly the same metrics.

    predict_function:
        Optional function with signature:
            predict_function(model, features) -> y_pred

        If omitted:
            model(features["x"], training=False)
    """

    mae_values = []

    rmse_values = []

    residual_values = []

    per_sample_rows = []

    sample_index = 0

    for (
        features,
        y_true,
    ) in dataset:

        if predict_function is None:

            y_pred = model(
                features["x"],
                training=False,
            )

        else:

            y_pred = (
                predict_function(
                    model,
                    features,
                )
            )

        m = float(
            mae(
                y_true,
                y_pred,
            ).numpy()
        )

        r = float(
            rmse(
                y_true,
                y_pred,
            ).numpy()
        )

        p = float(
            wave_residual_mean(
                y_pred,
                features[
                    "c_field"
                ],
                features[
                    "dt"
                ],
                features[
                    "dx"
                ],
            ).numpy()
        )

        mae_values.append(
            m
        )

        rmse_values.append(
            r
        )

        residual_values.append(
            p
        )

        path_value = (
            features[
                "path"
            ].numpy()[0]
        )

        if isinstance(
            path_value,
            bytes,
        ):

            path_value = (
                path_value.decode(
                    "utf-8"
                )
            )

        per_sample_rows.append(
            {
                "index":
                    sample_index,

                "path":
                    str(
                        path_value
                    ),

                "mae":
                    m,

                "rmse":
                    r,

                "wave_residual":
                    p,
            }
        )

        sample_index += 1

    result = {

        "count":
            len(
                mae_values
            ),

        "MAE_mean":
            float(
                np.mean(
                    mae_values
                )
            ),

        "MAE_std":
            float(
                np.std(
                    mae_values,
                    ddof=1,
                )
            )
            if len(
                mae_values
            ) > 1
            else 0.0,

        "RMSE_mean":
            float(
                np.mean(
                    rmse_values
                )
            ),

        "RMSE_std":
            float(
                np.std(
                    rmse_values,
                    ddof=1,
                )
            )
            if len(
                rmse_values
            ) > 1
            else 0.0,

        "Residual_mean":
            float(
                np.mean(
                    residual_values
                )
            ),

        "Residual_std":
            float(
                np.std(
                    residual_values,
                    ddof=1,
                )
            )
            if len(
                residual_values
            ) > 1
            else 0.0,

        "per_sample":
            per_sample_rows,
    }

    return result


# ============================================================
# COMMON LATENCY MEASUREMENT
# ============================================================

def measure_inference_latency(
    model,
    dataset,
    predict_function=None,
) -> dict:

    """
    Measures only model inference.

    Ground truth is not used.

    All new models must use this exact timing routine.
    """

    sample = next(
        iter(
            dataset.take(1)
        )
    )

    features, _ = sample

    def predict_once():

        if predict_function is None:

            return model(
                features["x"],
                training=False,
            )

        return predict_function(
            model,
            features,
        )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(
        int(
            COMMON_CONFIG[
                "latency_warmup_runs"
            ]
        )
    ):

        _ = predict_once()

    # --------------------------------------------------------
    # Timed runs
    # --------------------------------------------------------

    latencies = []

    for _ in range(
        int(
            COMMON_CONFIG[
                "latency_measurement_runs"
            ]
        )
    ):

        start = (
            time.perf_counter()
        )

        output = (
            predict_once()
        )

        # Force TensorFlow execution before stopping timer.
        _ = tf.reduce_sum(
            output
        ).numpy()

        elapsed = (
            time.perf_counter()
            - start
        )

        latencies.append(
            elapsed
        )

    return {

        "latency_runs":
            len(
                latencies
            ),

        "latency_mean_sec":
            float(
                np.mean(
                    latencies
                )
            ),

        "latency_std_sec":
            float(
                np.std(
                    latencies,
                    ddof=1,
                )
            )
            if len(
                latencies
            ) > 1
            else 0.0,

        "latency_p50_sec":
            float(
                np.percentile(
                    latencies,
                    50,
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
    }


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_trainable_parameters(
    model,
) -> int:

    return int(
        np.sum(
            [
                np.prod(
                    variable.shape
                )
                for variable
                in model.trainable_variables
            ]
        )
    )


# ============================================================
# COMMON BEST-CHECKPOINT POLICY
# ============================================================

def is_better_checkpoint(
    current_validation_rmse: float,
    best_validation_rmse: float,
) -> bool:

    return (
        current_validation_rmse
        <
        best_validation_rmse
    )


# ============================================================
# COMMON RESULT PACKAGE
# ============================================================

def build_result_record(
    *,
    model_name: str,
    model,
    validation_metrics: dict,
    test_metrics: dict,
    latency_metrics: dict,
    training_time_sec: float,
    best_epoch: int,
    model_specific_config: dict,
) -> dict:

    return {

        "experiment_family":
            COMMON_CONFIG[
                "experiment_family"
            ],

        "model":
            model_name,

        "new_experiment":
            True,

        "replaces_existing_manuscript_numbers":
            False,

        "parameter_count":
            count_trainable_parameters(
                model
            ),

        "training_time_sec":
            float(
                training_time_sec
            ),

        "best_validation_epoch":
            int(
                best_epoch
            ),

        "validation":
            validation_metrics,

        "test":
            test_metrics,

        "latency":
            latency_metrics,

        "common_protocol":
            COMMON_CONFIG,

        "model_specific_config":
            model_specific_config,

        "important_note":
            (
                "These results belong only to the new "
                "reviewer-requested baseline comparison. "
                "They do not replace, alter, or reinterpret "
                "the numerical values already reported in "
                "the current manuscript."
            ),
    }


# ============================================================
# PROTOCOL INITIALIZATION
# ============================================================

def initialize_experiment(
    model_name: str,
    model_specific_config: dict,
):

    seed = set_reproducibility()

    files = (
        list_sequence_files()
    )

    (
        train_files,
        validation_files,
        test_files,
    ) = split_sequence_files(
        files
    )

    run_dir = (
        create_run_directory(
            model_name
        )
    )

    save_split_manifest(
        run_dir,
        train_files,
        validation_files,
        test_files,
    )

    save_json(
        run_dir
        / "common_config.json",
        COMMON_CONFIG,
    )

    save_json(
        run_dir
        / "model_config.json",
        model_specific_config,
    )

    save_json(
        run_dir
        / "environment.json",
        collect_environment_info(),
    )

    save_json(
        run_dir
        / "experiment_identity.json",
        {
            "model":
                model_name,

            "seed":
                seed,

            "run_directory":
                str(
                    run_dir
                ),

            "dataset_directory":
                str(
                    SEQUENCE_DIR
                ),

            "total_sequences_for_this_new_experiment":
                len(
                    files
                ),

            "train_sequences":
                len(
                    train_files
                ),

            "validation_sequences":
                len(
                    validation_files
                ),

            "test_sequences":
                len(
                    test_files
                ),

            "replaces_existing_manuscript_numbers":
                False,
        },
    )

    return {
        "run_dir":
            run_dir,

        "files":
            files,

        "train_files":
            train_files,

        "validation_files":
            validation_files,

        "test_files":
            test_files,
    }


# ============================================================
# PROTOCOL SELF-TEST
# ============================================================

def protocol_self_test():

    print()
    print("=" * 80)
    print(
        "COMMON BASELINE PROTOCOL SELF-TEST"
    )
    print("=" * 80)

    print(
        "Project root:"
    )
    print(
        ORIGINAL_ROOT
    )

    print()

    print(
        "New branch:"
    )
    print(
        NEW_ROOT
    )

    print()

    print(
        "Sequence directory:"
    )
    print(
        SEQUENCE_DIR
    )

    files = (
        list_sequence_files()
    )

    (
        train_files,
        validation_files,
        test_files,
    ) = split_sequence_files(
        files
    )

    print()
    print(
        "Sequences:"
    )

    print(
        f"  Total      : {len(files)}"
    )

    print(
        f"  Train      : {len(train_files)}"
    )

    print(
        f"  Validation : {len(validation_files)}"
    )

    print(
        f"  Test       : {len(test_files)}"
    )

    print()
    print(
        "Common protocol:"
    )

    for key, value in (
        COMMON_CONFIG.items()
    ):

        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "TensorFlow version:",
        tf.__version__,
    )

    print(
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        ),
    )

    print()
    print(
        "PASS: common baseline protocol is ready."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    protocol_self_test()