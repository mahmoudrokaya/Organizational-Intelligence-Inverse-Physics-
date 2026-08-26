from __future__ import annotations

# ============================================================
# SACU DOMAIN-SIZE SCALABILITY
#
# Reviewer-requested computational scalability experiment.
#
# Conditions:
#   64 x 64
#   128 x 128  [native/reference]
#   256 x 256
#
# SCIENTIFIC POLICY
# ------------------------------------------------------------
# The trained grid=4 / 16-agent SACU checkpoint is fixed.
#
# Native 128x128:
#   - uses original held-out test inputs and paired targets
#   - may report MAE, RMSE, and wave residual
#
# Resized 64x64 and 256x256:
#   - are deterministic computational workload transformations
#   - report latency, throughput, memory, and parameter count
#   - DO NOT report predictive metrics because simulator-
#     generated paired targets do not exist for those resized
#     conditions
#
# No retraining occurs.
# No test-driven selection occurs.
# No submitted-manuscript numerical value is replaced.
# ============================================================


# ============================================================
# STANDARD LIBRARY
# ============================================================

import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


# ============================================================
# GPU ENVIRONMENT
#
# Must be set before TensorFlow import.
# ============================================================

os.environ.setdefault(
    "TF_FORCE_GPU_ALLOW_GROWTH",
    "true",
)

os.environ.setdefault(
    "TF_GPU_ALLOCATOR",
    "cuda_malloc_async",
)


# ============================================================
# NUMERICAL LIBRARIES
# ============================================================

import numpy as np
import tensorflow as tf


# ============================================================
# GPU CONFIGURATION
# ============================================================

def configure_gpu() -> None:

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    for gpu in gpus:

        try:

            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )

        except RuntimeError:

            # Device may already be initialized.
            pass


configure_gpu()


# ============================================================
# PATHS
# ============================================================

THIS_FILE = Path(__file__).resolve()

EXPERIMENT_DIR = THIS_FILE.parent

PROTOCOL_PATH = (
    EXPERIMENT_DIR
    / "00_common_scalability_protocol.py"
)


# ============================================================
# LOAD SCALABILITY PROTOCOL
# ============================================================

if not PROTOCOL_PATH.exists():

    raise FileNotFoundError(
        "Scalability protocol not found:\n"
        f"{PROTOCOL_PATH}"
    )


spec = importlib.util.spec_from_file_location(
    "scalability_protocol",
    PROTOCOL_PATH,
)

if spec is None or spec.loader is None:

    raise ImportError(
        "Unable to load scalability protocol."
    )


common = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    common
)


NEW_ROOT = common.NEW_ROOT


if str(NEW_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(NEW_ROOT),
    )


# ============================================================
# ACTUAL SACU IMPLEMENTATION
# ============================================================

from src.models_sacu import (
    OrgSACUSolver,
)

from src.training.trainer_v2 import (
    predict_sacu_deployment,
)

from src.data_loader import (
    make_dataset,
)

from src.physics_metrics import (
    wave_residual_norm,
)


# ============================================================
# FIXED REFERENCE CONDITION
# ============================================================

REFERENCE_GRID = 4
REFERENCE_AGENTS = 16

REFERENCE_HEIGHT = 128
REFERENCE_WIDTH = 128

DOMAIN_SIZES = [
    64,
    128,
    256,
]


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "SACU spatial-domain computational "
            "scalability experiment."
        )
    )


    parser.add_argument(
        "--size",
        type=int,
        required=True,
        choices=DOMAIN_SIZES,
        help=(
            "Spatial side length: 64, 128, or 256."
        ),
    )


    return parser.parse_args()


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def read_json(
    path: Path,
) -> Dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def save_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def save_csv(
    path: Path,
    rows: Sequence[
        Dict[str, Any]
    ],
) -> None:

    rows = list(
        rows
    )


    if not rows:

        return


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    fieldnames: List[str] = []
    seen = set()


    for row in rows:

        for key in row.keys():

            if key not in seen:

                seen.add(
                    key
                )

                fieldnames.append(
                    key
                )


    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# FIND COMPLETED 16-AGENT REFERENCE CHECKPOINT
# ============================================================

def find_reference_checkpoint() -> Tuple[
    Path,
    Dict[str, Any],
    Path,
]:

    scalability_root = (
        NEW_ROOT
        / "outputs"
        / "scalability"
    )


    if not scalability_root.exists():

        raise FileNotFoundError(
            "Scalability output directory does not exist:\n"
            f"{scalability_root}"
        )


    candidates = []


    for result_path in scalability_root.rglob(
        "results.json"
    ):

        try:

            result = read_json(
                result_path
            )

        except Exception:

            continue


        try:

            grid = int(
                result.get(
                    "grid",
                    -1,
                )
            )

            agents = int(
                result.get(
                    "agent_count",
                    -1,
                )
            )

        except Exception:

            continue


        if grid != REFERENCE_GRID:

            continue


        if agents != REFERENCE_AGENTS:

            continue


        reload_difference = result.get(
            "checkpoint_reload_difference",
            result.get(
                "checkpoint_reload_rmse_difference",
                None,
            ),
        )


        if reload_difference is None:

            continue


        try:

            reload_difference = float(
                reload_difference
            )

        except Exception:

            continue


        if reload_difference > 1e-6:

            continue


        checkpoint = (
            result_path.parent
            / "best.weights.h5"
        )


        if not checkpoint.exists():

            continue


        candidates.append(
            (
                result_path.stat().st_mtime,
                checkpoint,
                result,
                result_path,
            )
        )


    if not candidates:

        raise RuntimeError(
            "No valid completed grid=4 / 16-agent "
            "SACU checkpoint was found."
        )


    candidates.sort(
        key=lambda item:
            item[
                0
            ]
    )


    (
        _,
        checkpoint,
        result,
        result_path,
    ) = candidates[
        -1
    ]


    print(
        "Selected 16-agent reference result:"
    )

    print(
        result_path
    )


    print()


    print(
        "Selected checkpoint:"
    )

    print(
        checkpoint
    )


    return (
        checkpoint,
        result,
        result_path,
    )


# ============================================================
# READ REFERENCE CONFIGURATION
# ============================================================

def reference_configuration(
    result: Dict[
        str,
        Any
    ],
) -> Dict[str, Any]:

    required = [
        "overlap",
        "K",
        "hidden",
        "msg_dim",
        "use_role",
        "use_comms",
    ]


    for key in required:

        if key not in result:

            raise RuntimeError(
                "Reference result is missing "
                f"configuration field '{key}'."
            )


    return {

        "overlap":
            int(
                result[
                    "overlap"
                ]
            ),

        "K":
            int(
                result[
                    "K"
                ]
            ),

        "hidden":
            int(
                result[
                    "hidden"
                ]
            ),

        "msg_dim":
            int(
                result[
                    "msg_dim"
                ]
            ),

        "use_role":
            bool(
                result[
                    "use_role"
                ]
            ),

        "use_comms":
            bool(
                result[
                    "use_comms"
                ]
            ),

        "sensor_weight":
            float(
                result.get(
                    "sensor_weight",
                    0.50,
                )
            ),

        "physics_weight":
            float(
                result.get(
                    "physics_weight",
                    0.35,
                )
            ),

        "entropy_weight":
            float(
                result.get(
                    "entropy_weight",
                    0.15,
                )
            ),

        "temperature":
            float(
                result.get(
                    "temperature",
                    5.0,
                )
            ),
    }


# ============================================================
# TEST DATASET
# ============================================================

def build_test_dataset():

    (
        _,
        _,
        test_files,
    ) = common.split_sequence_files()


    test_files = [
        str(
            path
        )
        for path
        in test_files
    ]


    batch_size = int(
        common.COMMON_CONFIG[
            "batch_size"
        ]
    )


    if batch_size != 1:

        raise ValueError(
            "Domain-size scalability "
            "requires batch_size=1."
        )


    test_ds = make_dataset(
        test_files,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        deterministic=True,
    )


    return test_ds


# ============================================================
# PRINT ACTUAL FEATURE SHAPES
# ============================================================

def print_feature_shapes(
    features: Dict[
        str,
        tf.Tensor
    ],
) -> None:

    print()
    print(
        "Native feature shapes:"
    )


    for key, value in (
        features.items()
    ):

        if tf.is_tensor(
            value
        ):

            print(
                f"  {key}: "
                f"shape={value.shape}, "
                f"rank={value.shape.rank}"
            )

        else:

            print(
                f"  {key}: "
                f"type={type(value)}"
            )


# ============================================================
# SPATIAL RESIZE
#
# Supports actual dataset layouts:
#
# rank 5:
#   (B,T,H,W,C)
#
# rank 4:
#   (B,H,W,C)
#
# rank 3:
#   (B,H,W)
#
# rank 2:
#   (H,W)
#
# The original tensor rank is preserved.
# ============================================================

def resize_spatial_tensor(
    tensor: tf.Tensor,
    new_height: int,
    new_width: int,
) -> tf.Tensor:

    tensor = tf.cast(
        tensor,
        tf.float32,
    )


    rank = tensor.shape.rank


    # ========================================================
    # Rank 5:
    # (B,T,H,W,C)
    # ========================================================

    if rank == 5:

        shape = tf.shape(
            tensor
        )


        batch = shape[
            0
        ]

        time_steps = shape[
            1
        ]

        old_height = shape[
            2
        ]

        old_width = shape[
            3
        ]

        channels = shape[
            4
        ]


        flattened = tf.reshape(
            tensor,
            [
                batch
                * time_steps,

                old_height,

                old_width,

                channels,
            ],
        )


        resized = tf.image.resize(
            flattened,
            [
                new_height,
                new_width,
            ],
            method="bilinear",
            antialias=True,
        )


        restored = tf.reshape(
            resized,
            [
                batch,
                time_steps,
                new_height,
                new_width,
                channels,
            ],
        )


        return restored


    # ========================================================
    # Rank 4:
    # (B,H,W,C)
    # ========================================================

    if rank == 4:

        return tf.image.resize(
            tensor,
            [
                new_height,
                new_width,
            ],
            method="bilinear",
            antialias=True,
        )


    # ========================================================
    # Rank 3:
    # (B,H,W)
    #
    # Actual c_field layout in this project.
    # ========================================================

    if rank == 3:

        expanded = tensor[
            ...,
            tf.newaxis
        ]


        resized = tf.image.resize(
            expanded,
            [
                new_height,
                new_width,
            ],
            method="bilinear",
            antialias=True,
        )


        restored = tf.squeeze(
            resized,
            axis=-1,
        )


        return restored


    # ========================================================
    # Rank 2:
    # (H,W)
    # ========================================================

    if rank == 2:

        expanded = tensor[
            tf.newaxis,
            ...,
            tf.newaxis
        ]


        resized = tf.image.resize(
            expanded,
            [
                new_height,
                new_width,
            ],
            method="bilinear",
            antialias=True,
        )


        restored = tf.squeeze(
            resized,
            axis=[
                0,
                3,
            ],
        )


        return restored


    raise ValueError(
        "Unsupported spatial tensor rank.\n"
        f"Rank: {rank}\n"
        f"Shape: {tensor.shape}"
    )


# ============================================================
# TRANSFORM FEATURES FOR COMPUTATIONAL WORKLOAD
#
# Only spatial tensors are resized.
#
# x:
#   (B,T,H,W,C)
#
# c_field:
#   currently (B,H,W)
#
# dt and dx remain unchanged because resized conditions are
# computational benchmarks only.
# ============================================================

def transform_features(
    features: Dict[
        str,
        tf.Tensor
    ],
    size: int,
) -> Dict[
    str,
    tf.Tensor
]:

    transformed = dict(
        features
    )


    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    if "x" not in features:

        raise KeyError(
            "Feature dictionary has no 'x'."
        )


    transformed[
        "x"
    ] = resize_spatial_tensor(
        features[
            "x"
        ],
        size,
        size,
    )


    # --------------------------------------------------------
    # Propagation coefficient field
    # --------------------------------------------------------

    if "c_field" in features:

        transformed[
            "c_field"
        ] = resize_spatial_tensor(
            features[
                "c_field"
            ],
            size,
            size,
        )


    return transformed


# ============================================================
# VERIFY TRANSFORMATION
# ============================================================

def verify_transformed_features(
    native_features,
    transformed_features,
    size: int,
) -> None:

    x = transformed_features[
        "x"
    ]


    if int(
        tf.shape(
            x
        )[2].numpy()
    ) != size:

        raise RuntimeError(
            "Transformed x height is incorrect."
        )


    if int(
        tf.shape(
            x
        )[3].numpy()
    ) != size:

        raise RuntimeError(
            "Transformed x width is incorrect."
        )


    # Rank must remain identical.

    if (
        transformed_features[
            "x"
        ].shape.rank
        !=
        native_features[
            "x"
        ].shape.rank
    ):

        raise RuntimeError(
            "x rank changed during transformation."
        )


    if "c_field" in native_features:

        if (
            transformed_features[
                "c_field"
            ].shape.rank
            !=
            native_features[
                "c_field"
            ].shape.rank
        ):

            raise RuntimeError(
                "c_field rank changed during transformation."
            )


    print()
    print(
        "Transformed feature shapes:"
    )


    for key, value in (
        transformed_features.items()
    ):

        if tf.is_tensor(
            value
        ):

            print(
                f"  {key}: "
                f"shape={value.shape}, "
                f"rank={value.shape.rank}"
            )


# ============================================================
# BUILD FIXED 16-AGENT MODEL
# ============================================================

def build_model(
    sample_x: tf.Tensor,
    reference_config: Dict[
        str,
        Any
    ],
):

    model = OrgSACUSolver(

        grid=
            REFERENCE_GRID,

        overlap=
            reference_config[
                "overlap"
            ],

        K=
            reference_config[
                "K"
            ],

        hidden=
            reference_config[
                "hidden"
            ],

        msg_dim=
            reference_config[
                "msg_dim"
            ],

        use_role=
            reference_config[
                "use_role"
            ],

        use_comms=
            reference_config[
                "use_comms"
            ],

        name=
            "org_sacu_domain_scaling_reference",
    )


    # Explicitly build all spatially dependent layers.

    _ = model(
        sample_x,
        training=False,
    )


    if not model.built:

        raise RuntimeError(
            "OrgSACUSolver did not build."
        )


    if model.N != REFERENCE_AGENTS:

        raise RuntimeError(
            "Domain scaling must use "
            "exactly 16 SACU agents."
        )


    return model


# ============================================================
# LOAD FIXED CHECKPOINT
# ============================================================

def load_reference_weights(
    model,
    checkpoint: Path,
) -> None:

    try:

        model.load_weights(
            str(
                checkpoint
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Unable to restore the fixed 16-agent "
            "checkpoint into this spatial condition.\n"
            "No alternative weights will be substituted."
        ) from exc


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    features,
    reference_config,
):

    (
        y_pred,
        _,
    ) = predict_sacu_deployment(

        model,

        features[
            "x"
        ],

        features[
            "c_field"
        ],

        features[
            "dt"
        ],

        features[
            "dx"
        ],

        training=False,

        sensor_weight=
            reference_config[
                "sensor_weight"
            ],

        physics_weight=
            reference_config[
                "physics_weight"
            ],

        entropy_weight=
            reference_config[
                "entropy_weight"
            ],

        temperature=
            reference_config[
                "temperature"
            ],
    )


    return y_pred


# ============================================================
# NATIVE 128x128 HELD-OUT EVALUATION
#
# Predictive metrics are scientifically valid here because
# the paired target comes from the original simulator.
# ============================================================

def evaluate_native_test(
    model,
    test_ds,
    reference_config,
):

    maes = []
    rmses = []
    residuals = []


    for (
        features,
        y_true,
    ) in test_ds:

        y_pred = predict(
            model,
            features,
            reference_config,
        )


        batch_mae = tf.reduce_mean(
            tf.abs(
                y_true
                - y_pred
            )
        )


        batch_rmse = tf.sqrt(
            tf.reduce_mean(
                tf.square(
                    y_true
                    - y_pred
                )
            )
            + 1e-12
        )


        batch_residual = wave_residual_norm(

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
        )


        maes.append(
            float(
                batch_mae.numpy()
            )
        )


        rmses.append(
            float(
                batch_rmse.numpy()
            )
        )


        residuals.append(
            float(
                batch_residual.numpy()
            )
        )


        del y_pred


    if not maes:

        raise RuntimeError(
            "Native held-out test dataset is empty."
        )


    return {

        "test_mae":
            float(
                np.mean(
                    maes
                )
            ),

        "test_rmse":
            float(
                np.mean(
                    rmses
                )
            ),

        "test_wave_residual":
            float(
                np.mean(
                    residuals
                )
            ),
    }


# ============================================================
# COMPUTATIONAL DOMAIN BENCHMARK
#
# Preprocessing resize is NOT timed.
#
# Only SACU deployment time is measured.
# ============================================================

def benchmark_domain(
    model,
    benchmark_features,
    reference_config,
):

    warmup_runs = int(
        common.SCALABILITY_CONFIG[
            "latency_warmup_runs"
        ]
    )


    measurement_runs = int(
        common.SCALABILITY_CONFIG[
            "latency_measurement_runs"
        ]
    )


    # --------------------------------------------------------
    # Warm-up / compilation / autotuning
    # --------------------------------------------------------

    for _ in range(
        warmup_runs
    ):

        y_pred = predict(
            model,
            benchmark_features,
            reference_config,
        )


        _ = tf.reduce_sum(
            y_pred
        ).numpy()


    # --------------------------------------------------------
    # Reset memory statistics AFTER warmup.
    #
    # This is important because initial graph compilation and
    # cuDNN autotuning should not dominate steady-state memory.
    # --------------------------------------------------------

    common.reset_gpu_memory_stats()


    timings = []


    for _ in range(
        measurement_runs
    ):

        start = (
            time.perf_counter()
        )


        y_pred = predict(
            model,
            benchmark_features,
            reference_config,
        )


        # Synchronize GPU before stopping timer.
        _ = tf.reduce_sum(
            y_pred
        ).numpy()


        elapsed = (
            time.perf_counter()
            - start
        )


        timings.append(
            elapsed
        )


    timing = common.summarize_timings(
        timings
    )


    memory = common.gpu_memory_info()


    input_shape = tuple(
        int(
            value
        )
        for value
        in tf.shape(
            benchmark_features[
                "x"
            ]
        ).numpy()
    )


    return {

        "input_shape":
            input_shape,

        "latency_runs":
            int(
                timing[
                    "runs"
                ]
            ),

        "latency_mean_sec":
            float(
                timing[
                    "mean_sec"
                ]
            ),

        "latency_p50_sec":
            float(
                timing[
                    "p50_sec"
                ]
            ),

        "latency_p95_sec":
            float(
                timing[
                    "p95_sec"
                ]
            ),

        "latency_p99_sec":
            float(
                timing[
                    "p99_sec"
                ]
            ),

        "throughput_sequences_per_sec":
            float(
                1.0
                /
                timing[
                    "mean_sec"
                ]
            ),

        "peak_gpu_memory_bytes":
            memory.get(
                "peak_bytes"
            ),

        "current_gpu_memory_bytes":
            memory.get(
                "current_bytes"
            ),
    }


# ============================================================
# FINITE-PREDICTION CHECK
# ============================================================

def assert_finite_prediction(
    y_pred: tf.Tensor,
) -> None:

    finite = bool(
        tf.reduce_all(
            tf.math.is_finite(
                y_pred
            )
        ).numpy()
    )


    if not finite:

        raise RuntimeError(
            "Non-finite SACU prediction detected."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    size = int(
        args.size
    )


    common.set_determinism()


    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 80)

    print(
        "SACU DOMAIN-SIZE SCALABILITY"
    )

    print("=" * 80)


    print(
        "Selected spatial size:",
        f"{size} x {size}",
    )


    print(
        "Native/reference size:",
        "128 x 128",
    )


    print(
        "Fixed SACU architecture:",
        "grid=4 -> 16 agents",
    )


    print(
        "Execution policy:",
        "fixed trained checkpoint; no retraining",
    )


    print(
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        ),
    )


    print(
        "TF_FORCE_GPU_ALLOW_GROWTH:",
        os.environ.get(
            "TF_FORCE_GPU_ALLOW_GROWTH"
        ),
    )


    print(
        "TF_GPU_ALLOCATOR:",
        os.environ.get(
            "TF_GPU_ALLOCATOR"
        ),
    )


    # ========================================================
    # FIND VALIDATED 16-AGENT CHECKPOINT
    # ========================================================

    (
        checkpoint,
        reference_result,
        reference_result_path,
    ) = find_reference_checkpoint()


    config = reference_configuration(
        reference_result
    )


    print()
    print(
        "Reference architecture:"
    )


    for key, value in (
        config.items()
    ):

        print(
            f"  {key}: {value}"
        )


    # ========================================================
    # BUILD HELD-OUT DATASET
    # ========================================================

    test_ds = build_test_dataset()


    (
        native_features,
        native_target,
    ) = next(
        iter(
            test_ds.take(1)
        )
    )


    print_feature_shapes(
        native_features
    )


    print(
        "  target:",
        f"shape={native_target.shape}, "
        f"rank={native_target.shape.rank}",
    )


    # ========================================================
    # CREATE COMPUTATIONAL CONDITION
    # ========================================================

    if size == REFERENCE_HEIGHT:

        benchmark_features = (
            native_features
        )


        input_transformation = (
            "none"
        )


    else:

        benchmark_features = (
            transform_features(
                native_features,
                size,
            )
        )


        verify_transformed_features(

            native_features,

            benchmark_features,

            size,
        )


        input_transformation = (
            "deterministic_bilinear_resize_"
            "for_computational_scaling"
        )


    # ========================================================
    # BUILD MODEL FOR TARGET SPATIAL SHAPE
    # ========================================================

    model = build_model(

        benchmark_features[
            "x"
        ],

        config,
    )


    parameter_count = int(
        model.count_params()
    )


    print()
    print(
        "Parameter count:",
        parameter_count,
    )


    if (
        "parameter_count"
        in reference_result
    ):

        reference_parameter_count = int(
            reference_result[
                "parameter_count"
            ]
        )


        if (
            parameter_count
            !=
            reference_parameter_count
        ):

            raise RuntimeError(
                "Parameter count changed with domain size.\n"
                f"Reference: "
                f"{reference_parameter_count}\n"
                f"Current: {parameter_count}"
            )


    # ========================================================
    # RESTORE EXACT SAME TRAINED WEIGHTS
    # ========================================================

    load_reference_weights(
        model,
        checkpoint,
    )


    print(
        "PASS: reference checkpoint restored."
    )


    # ========================================================
    # SANITY INFERENCE
    # ========================================================

    sanity_prediction = predict(

        model,

        benchmark_features,

        config,
    )


    assert_finite_prediction(
        sanity_prediction
    )


    print(
        "PASS: finite inference obtained "
        f"at {size}x{size}."
    )


    print(
        "Prediction shape:",
        sanity_prediction.shape,
    )


    del sanity_prediction

    gc.collect()


    # ========================================================
    # COMPUTATIONAL BENCHMARK
    # ========================================================

    benchmark = benchmark_domain(

        model,

        benchmark_features,

        config,
    )


    # ========================================================
    # PREDICTIVE METRICS
    #
    # Only native 128x128 condition is scientifically valid.
    # ========================================================

    if size == REFERENCE_HEIGHT:

        predictive_metrics = (
            evaluate_native_test(

                model,

                test_ds,

                config,
            )
        )


        predictive_metrics_valid = True


        predictive_metric_reason = (
            "Native 128x128 held-out samples with "
            "simulator-generated paired targets."
        )


    else:

        predictive_metrics = {

            "test_mae":
                None,

            "test_rmse":
                None,

            "test_wave_residual":
                None,
        }


        predictive_metrics_valid = False


        predictive_metric_reason = (
            "The input was deterministically resized only "
            "to create a computational workload. No "
            "simulator-generated paired target exists at "
            f"{size}x{size}; predictive metrics are therefore "
            "intentionally omitted."
        )


    # ========================================================
    # CREATE AUDITABLE RUN DIRECTORY
    # ========================================================

    experiment = (
        common.initialize_scalability_run(

            experiment_name=
                (
                    "domain_size_scaling_"
                    f"{size}x{size}"
                ),

            experiment_config=
                {

                    "experiment":
                        "domain_size_scaling",

                    "domain_height":
                        size,

                    "domain_width":
                        size,

                    "reference_height":
                        REFERENCE_HEIGHT,

                    "reference_width":
                        REFERENCE_WIDTH,

                    "grid":
                        REFERENCE_GRID,

                    "agent_count":
                        REFERENCE_AGENTS,

                    "weights_policy":
                        (
                            "fixed_completed_16_agent_"
                            "checkpoint"
                        ),

                    "checkpoint_source":
                        str(
                            checkpoint
                        ),

                    "reference_result_source":
                        str(
                            reference_result_path
                        ),

                    "input_transformation":
                        input_transformation,

                    "predictive_metrics_valid":
                        predictive_metrics_valid,

                    "new_reviewer_requested_experiment":
                        True,

                    "replaces_existing_manuscript_numbers":
                        False,
                },
        )
    )


    condition_dir = (
        experiment[
            "run_dir"
        ]
        / f"domain_{size}x{size}"
    )


    condition_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # RESULT RECORD
    # ========================================================

    result = {

        "domain_height":
            size,

        "domain_width":
            size,

        "domain_pixels":
            int(
                size
                * size
            ),

        "domain_pixel_ratio_vs_128":
            float(
                (
                    size
                    * size
                )
                /
                (
                    REFERENCE_HEIGHT
                    * REFERENCE_WIDTH
                )
            ),

        "native_reference_condition":
            bool(
                size
                == REFERENCE_HEIGHT
            ),

        "grid":
            REFERENCE_GRID,

        "agent_count":
            REFERENCE_AGENTS,

        "parameter_count":
            parameter_count,

        "weights_source":
            str(
                checkpoint
            ),

        "reference_result_source":
            str(
                reference_result_path
            ),

        "input_transformation":
            input_transformation,

        "predictive_metrics_valid":
            predictive_metrics_valid,

        "predictive_metric_reason":
            predictive_metric_reason,

        "test_mae":
            predictive_metrics[
                "test_mae"
            ],

        "test_rmse":
            predictive_metrics[
                "test_rmse"
            ],

        "test_wave_residual":
            predictive_metrics[
                "test_wave_residual"
            ],

        "input_shape":
            benchmark[
                "input_shape"
            ],

        "latency_runs":
            benchmark[
                "latency_runs"
            ],

        "latency_mean_sec":
            benchmark[
                "latency_mean_sec"
            ],

        "latency_p50_sec":
            benchmark[
                "latency_p50_sec"
            ],

        "latency_p95_sec":
            benchmark[
                "latency_p95_sec"
            ],

        "latency_p99_sec":
            benchmark[
                "latency_p99_sec"
            ],

        "throughput_sequences_per_sec":
            benchmark[
                "throughput_sequences_per_sec"
            ],

        "peak_gpu_memory_bytes":
            benchmark[
                "peak_gpu_memory_bytes"
            ],

        "current_gpu_memory_bytes":
            benchmark[
                "current_gpu_memory_bytes"
            ],

        "hardware_mode":
            (
                "GPU"
                if tf.config.list_physical_devices(
                    "GPU"
                )
                else "CPU"
            ),

        "tensorflow_version":
            tf.__version__,

        "test_used_for_selection":
            False,

        "new_reviewer_requested_experiment":
            True,

        "replaces_existing_manuscript_numbers":
            False,
    }


    # ========================================================
    # SAVE
    # ========================================================

    save_json(
        condition_dir
        / "results.json",
        result,
    )


    save_csv(
        condition_dir
        / "summary.csv",
        [
            result
        ],
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 80)

    print(
        "DOMAIN-SIZE CONDITION COMPLETED"
    )

    print("=" * 80)


    print(
        "Spatial size:",
        f"{size} x {size}",
    )


    print(
        "Spatial workload ratio vs 128x128:",
        (
            f"{result['domain_pixel_ratio_vs_128']:.3f}x"
        ),
    )


    print(
        "Agents:",
        REFERENCE_AGENTS,
    )


    print(
        "Parameters:",
        parameter_count,
    )


    print(
        "Mean latency:",
        (
            f"{result['latency_mean_sec']:.6f} sec"
        ),
    )


    print(
        "P50 latency:",
        (
            f"{result['latency_p50_sec']:.6f} sec"
        ),
    )


    print(
        "P95 latency:",
        (
            f"{result['latency_p95_sec']:.6f} sec"
        ),
    )


    print(
        "P99 latency:",
        (
            f"{result['latency_p99_sec']:.6f} sec"
        ),
    )


    print(
        "Throughput:",
        (
            f"{result['throughput_sequences_per_sec']:.3f} "
            "sequences/sec"
        ),
    )


    print(
        "Peak GPU memory:",
        result[
            "peak_gpu_memory_bytes"
        ],
        "bytes",
    )


    # ========================================================
    # NATIVE PREDICTIVE RESULTS
    # ========================================================

    if predictive_metrics_valid:

        print()

        print(
            "Native held-out predictive metrics:"
        )


        print(
            "  Test MAE:",
            f"{result['test_mae']:.6f}",
        )


        print(
            "  Test RMSE:",
            f"{result['test_rmse']:.6f}",
        )


        print(
            "  Wave residual:",
            (
                f"{result['test_wave_residual']:.6f}"
            ),
        )


    else:

        print()

        print(
            "Predictive metrics: intentionally omitted."
        )


        print(
            "Reason:"
        )


        print(
            predictive_metric_reason
        )


    print()

    print(
        "PASS: fixed trained 16-agent checkpoint "
        "was used."
    )


    print(
        "PASS: SACU architecture was not modified."
    )


    print(
        "PASS: resized conditions were treated only "
        "as computational workloads."
    )


    print(
        "PASS: no unsupported predictive metric was "
        "fabricated."
    )


    print(
        "No submitted-manuscript numerical value "
        "was replaced."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()