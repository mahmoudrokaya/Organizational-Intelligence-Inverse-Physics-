from __future__ import annotations

# ============================================================
# SACU COMMUNICATION-COST SCALABILITY
#
# Reviewer-requested scalability experiment.
#
# Controlled communication-delay scenarios:
#
#     0 ms   [reference]
#     1 ms
#     5 ms
#    10 ms
#
# SCIENTIFIC POLICY
# ------------------------------------------------------------
# - Fixed trained grid=4 / 16-agent SACU checkpoint.
# - Fixed native 128x128 held-out input.
# - No retraining.
# - No test-based selection.
#
# Communication delay is injected exactly between:
#
#   Pass 1:
#       SACUs produce local predictions + outgoing messages
#
#   synchronization barrier:
#       controlled delay
#
#   Pass 2:
#       SACUs consume neighborhood-aggregated messages
#
# The injected delay is a CONTROLLED SCENARIO.
# It is NOT claimed to be measured physical network latency.
#
# The prediction must remain numerically unchanged across
# all delay settings.
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
    make_regions,
    stitch_patches,
)

from src.inference_weights import (
    compute_deployment_influence_weights,
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
# FIXED EXPERIMENTAL CONDITIONS
# ============================================================

REFERENCE_GRID = 4
REFERENCE_AGENTS = 16

REFERENCE_HEIGHT = 128
REFERENCE_WIDTH = 128

COMMUNICATION_DELAYS_MS = [
    0.0,
    1.0,
    5.0,
    10.0,
]


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "SACU controlled communication-cost "
            "scalability experiment."
        )
    )


    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        choices=COMMUNICATION_DELAYS_MS,
        help=(
            "Run one communication delay condition "
            "(0, 1, 5, or 10 ms). "
            "If omitted, all four conditions are run."
        ),
    )


    return parser.parse_args()


# ============================================================
# SERIALIZATION
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
# FIND VALIDATED 16-AGENT CHECKPOINT
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
            "Scalability output root not found:\n"
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
            "No valid completed 16-agent "
            "SACU checkpoint was found."
        )


    candidates.sort(
        key=lambda item:
            item[0]
    )


    (
        _,
        checkpoint,
        result,
        result_path,
    ) = candidates[-1]


    return (
        checkpoint,
        result,
        result_path,
    )


# ============================================================
# REFERENCE CONFIG
# ============================================================

def reference_configuration(
    result: Dict[str, Any],
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
                f"Reference result missing '{key}'."
            )


    config = {

        "overlap":
            int(
                result["overlap"]
            ),

        "K":
            int(
                result["K"]
            ),

        "hidden":
            int(
                result["hidden"]
            ),

        "msg_dim":
            int(
                result["msg_dim"]
            ),

        "use_role":
            bool(
                result["use_role"]
            ),

        "use_comms":
            bool(
                result["use_comms"]
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


    if not config[
        "use_comms"
    ]:

        raise RuntimeError(
            "Reference SACU model has communications disabled. "
            "Communication-cost scaling would therefore "
            "not be meaningful."
        )


    return config


# ============================================================
# TEST DATA
# ============================================================

def build_test_dataset():

    (
        _,
        _,
        test_files,
    ) = common.split_sequence_files()


    test_files = [
        str(path)
        for path in test_files
    ]


    batch_size = int(
        common.COMMON_CONFIG[
            "batch_size"
        ]
    )


    if batch_size != 1:

        raise ValueError(
            "Communication-cost scaling "
            "requires batch_size=1."
        )


    return make_dataset(
        test_files,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        deterministic=True,
    )


# ============================================================
# MODEL
# ============================================================

def build_model(
    sample_x: tf.Tensor,
    config: Dict[str, Any],
):

    model = OrgSACUSolver(

        grid=
            REFERENCE_GRID,

        overlap=
            config[
                "overlap"
            ],

        K=
            config[
                "K"
            ],

        hidden=
            config[
                "hidden"
            ],

        msg_dim=
            config[
                "msg_dim"
            ],

        use_role=
            config[
                "use_role"
            ],

        use_comms=
            config[
                "use_comms"
            ],

        name=
            "org_sacu_communication_scaling",
    )


    _ = model(
        sample_x,
        training=False,
    )


    if model.N != REFERENCE_AGENTS:

        raise RuntimeError(
            "Communication-cost experiment must "
            "use exactly 16 SACUs."
        )


    return model


# ============================================================
# LOAD FIXED WEIGHTS
# ============================================================

def load_reference_weights(
    model,
    checkpoint: Path,
) -> None:

    model.load_weights(
        str(
            checkpoint
        )
    )


# ============================================================
# SYNCHRONIZE GPU
# ============================================================

def synchronize_tensor(
    tensor: tf.Tensor,
) -> None:

    _ = tf.reduce_sum(
        tensor
    ).numpy()


# ============================================================
# CONTROLLED COMMUNICATION DELAY
#
# The delay is inserted once at the global communication
# synchronization barrier.
#
# This represents a controlled per-round communication delay,
# NOT measured physical network latency.
# ============================================================

def inject_communication_delay(
    delay_ms: float,
) -> None:

    if delay_ms <= 0.0:

        return


    time.sleep(
        delay_ms
        / 1000.0
    )


# ============================================================
# EXPLICIT SACU DEPLOYMENT WITH INJECTED COMMUNICATION DELAY
#
# This duplicates the implemented OrgSACUSolver.call logic,
# but exposes the communication synchronization point so the
# controlled delay can be inserted.
#
# Prediction mathematics remain unchanged.
# ============================================================

def predict_with_communication_delay(
    model,
    x,
    c_field,
    dt,
    dx,
    delay_ms: float,
    config: Dict[str, Any],
):

    batch_size = tf.shape(
        x
    )[0]

    time_steps = tf.shape(
        x
    )[1]


    height = int(
        x.shape[
            2
        ]
    )

    width = int(
        x.shape[
            3
        ]
    )


    regions = make_regions(
        height,
        width,
        grid=model.grid,
        overlap=model.overlap,
    )


    # --------------------------------------------------------
    # Role identifiers
    # --------------------------------------------------------

    role_ids = (
        tf.range(
            model.N
        )
        % 8
    )


    role_ids = tf.broadcast_to(
        role_ids[
            None,
            :
        ],
        [
            batch_size,
            model.N,
        ],
    )


    zero_message = tf.zeros(
        [
            batch_size,
            model.msg_dim,
        ],
        dtype=x.dtype,
    )


    # ========================================================
    # PASS 1
    #
    # Local prediction + outgoing communication message.
    # ========================================================

    messages = []
    first_patch_outputs = []
    first_gates = []


    for i, (
        y0,
        y1,
        x0,
        x1,
    ) in enumerate(
        regions
    ):

        xp = x[
            :,
            :,
            y0:y1,
            x0:x1,
            :
        ]


        role_id = (
            role_ids[
                :,
                i
            ]
            if model.use_role
            else None
        )


        message_in = (
            zero_message
            if model.use_comms
            else None
        )


        (
            y_hat,
            gate,
            message_out,
        ) = model.sacus[
            i
        ](
            xp,
            role_id=role_id,
            msg_in=message_in,
            training=False,
        )


        first_patch_outputs.append(
            y_hat
        )

        first_gates.append(
            gate
        )

        messages.append(
            message_out
        )


    # --------------------------------------------------------
    # If communication disabled, no barrier is meaningful.
    # --------------------------------------------------------

    if not model.use_comms:

        patch_outputs = (
            first_patch_outputs
        )

        gates = (
            first_gates
        )


    else:

        messages_tensor = tf.stack(
            messages,
            axis=1,
        )


        # ====================================================
        # CONTROLLED COMMUNICATION SYNCHRONIZATION BARRIER
        # ====================================================

        # Ensure pass-1 GPU work has completed before delay.
        synchronize_tensor(
            messages_tensor
        )


        barrier_start = (
            time.perf_counter()
        )


        inject_communication_delay(
            delay_ms
        )


        realized_barrier_sec = (
            time.perf_counter()
            - barrier_start
        )


        # ====================================================
        # PASS 2
        #
        # Neighborhood-aggregated messages condition the
        # second SACU pass.
        # ====================================================

        patch_outputs = []
        gates = []


        for i, (
            y0,
            y1,
            x0,
            x1,
        ) in enumerate(
            regions
        ):

            xp = x[
                :,
                :,
                y0:y1,
                x0:x1,
                :
            ]


            role_id = (
                role_ids[
                    :,
                    i
                ]
                if model.use_role
                else None
            )


            neighbors = model.neigh[
                i
            ]


            if len(
                neighbors
            ) == 0:

                message_in = (
                    zero_message
                )


            else:

                message_in = tf.reduce_mean(

                    tf.gather(
                        messages_tensor,
                        neighbors,
                        axis=1,
                    ),

                    axis=1,
                )


            (
                y_hat,
                gate,
                _,
            ) = model.sacus[
                i
            ](
                xp,
                role_id=role_id,
                msg_in=message_in,
                training=False,
            )


            patch_outputs.append(
                y_hat
            )

            gates.append(
                gate
            )


    # ========================================================
    # DEPLOYMENT INFLUENCE WEIGHTS
    # ========================================================

    (
        weights,
        diagnostics,
    ) = compute_deployment_influence_weights(

        patch_outputs,
        gates,
        regions,

        x,
        c_field,
        dt,
        dx,

        config[
            "sensor_weight"
        ],

        config[
            "physics_weight"
        ],

        config[
            "entropy_weight"
        ],

        config[
            "temperature"
        ],
    )


    # ========================================================
    # GLOBAL STITCHING
    # ========================================================

    y_pred = stitch_patches(

        patch_outputs,
        regions,
        weights,

        tf.shape(
            x
        )[2],

        tf.shape(
            x
        )[3],
    )


    diagnostics[
        "influence_weights"
    ] = weights


    diagnostics[
        "gates"
    ] = tf.stack(
        gates,
        axis=1,
    )


    diagnostics[
        "requested_delay_ms"
    ] = float(
        delay_ms
    )


    diagnostics[
        "realized_barrier_sec"
    ] = (
        float(
            realized_barrier_sec
        )
        if model.use_comms
        else 0.0
    )


    return (
        y_pred,
        diagnostics,
    )


# ============================================================
# STANDARD REFERENCE PREDICTION
# ============================================================

def standard_predict(
    model,
    features,
    config,
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
            config[
                "sensor_weight"
            ],

        physics_weight=
            config[
                "physics_weight"
            ],

        entropy_weight=
            config[
                "entropy_weight"
            ],

        temperature=
            config[
                "temperature"
            ],
    )


    return y_pred


# ============================================================
# NUMERICAL EQUIVALENCE CHECK
#
# At delay = 0, the explicit implementation must reproduce
# the standard validated SACU deployment path.
# ============================================================

def verify_zero_delay_equivalence(
    model,
    features,
    config,
) -> Dict[str, float]:

    reference_prediction = (
        standard_predict(
            model,
            features,
            config,
        )
    )


    (
        explicit_prediction,
        _,
    ) = predict_with_communication_delay(

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

        delay_ms=
            0.0,

        config=
            config,
    )


    synchronize_tensor(
        reference_prediction
    )

    synchronize_tensor(
        explicit_prediction
    )


    difference = (
        explicit_prediction
        - reference_prediction
    )


    max_abs_difference = float(
        tf.reduce_max(
            tf.abs(
                difference
            )
        ).numpy()
    )


    mean_abs_difference = float(
        tf.reduce_mean(
            tf.abs(
                difference
            )
        ).numpy()
    )


    print()
    print(
        "Zero-delay deployment equivalence:"
    )


    print(
        "  Max absolute difference:",
        max_abs_difference,
    )


    print(
        "  Mean absolute difference:",
        mean_abs_difference,
    )


    if max_abs_difference > 1e-6:

        raise RuntimeError(
            "Explicit communication-delay deployment "
            "does not reproduce the standard SACU "
            "deployment at zero injected delay."
        )


    return {

        "zero_delay_max_abs_difference":
            max_abs_difference,

        "zero_delay_mean_abs_difference":
            mean_abs_difference,
    }


# ============================================================
# DELAY-INVARIANCE CHECK
#
# Communication delay must alter wall-clock timing only.
# It must not alter numerical predictions.
# ============================================================

def prediction_difference(
    reference_prediction,
    candidate_prediction,
):

    difference = (
        candidate_prediction
        - reference_prediction
    )


    return {

        "prediction_max_abs_difference":
            float(
                tf.reduce_max(
                    tf.abs(
                        difference
                    )
                ).numpy()
            ),

        "prediction_mean_abs_difference":
            float(
                tf.reduce_mean(
                    tf.abs(
                        difference
                    )
                ).numpy()
            ),
    }


# ============================================================
# BENCHMARK ONE COMMUNICATION DELAY
# ============================================================

def benchmark_delay(
    model,
    features,
    delay_ms: float,
    config,
    zero_delay_prediction,
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


    # ========================================================
    # WARMUP
    # ========================================================

    for _ in range(
        warmup_runs
    ):

        (
            y_pred,
            _,
        ) = predict_with_communication_delay(

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

            delay_ms=
                delay_ms,

            config=
                config,
        )


        synchronize_tensor(
            y_pred
        )


    # --------------------------------------------------------
    # Steady-state memory statistics.
    # --------------------------------------------------------

    common.reset_gpu_memory_stats()


    timings = []
    realized_barrier_times = []

    final_prediction = None


    # ========================================================
    # MEASURE
    # ========================================================

    for _ in range(
        measurement_runs
    ):

        start = (
            time.perf_counter()
        )


        (
            y_pred,
            diagnostics,
        ) = predict_with_communication_delay(

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

            delay_ms=
                delay_ms,

            config=
                config,
        )


        # Full GPU synchronization before stopping timer.
        synchronize_tensor(
            y_pred
        )


        elapsed = (
            time.perf_counter()
            - start
        )


        timings.append(
            elapsed
        )


        realized_barrier_times.append(
            diagnostics[
                "realized_barrier_sec"
            ]
        )


        final_prediction = (
            y_pred
        )


    timing = common.summarize_timings(
        timings
    )


    barrier_timing = common.summarize_timings(
        realized_barrier_times
    )


    memory = common.gpu_memory_info()


    prediction_check = (
        prediction_difference(

            zero_delay_prediction,

            final_prediction,
        )
    )


    if (
        prediction_check[
            "prediction_max_abs_difference"
        ]
        > 1e-6
    ):

        raise RuntimeError(
            "Injected communication delay altered "
            "the SACU numerical prediction."
        )


    return {

        "requested_delay_ms":
            float(
                delay_ms
            ),

        "requested_delay_sec":
            float(
                delay_ms
                / 1000.0
            ),

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

        "realized_barrier_mean_sec":
            float(
                barrier_timing[
                    "mean_sec"
                ]
            ),

        "realized_barrier_p95_sec":
            float(
                barrier_timing[
                    "p95_sec"
                ]
            ),

        "peak_gpu_memory_bytes":
            memory.get(
                "peak_bytes"
            ),

        **prediction_check,
    }


# ============================================================
# NATIVE TEST METRICS
#
# These are computed once only. Communication delay does not
# change the prediction, so there is no scientific reason to
# recompute or imply accuracy variation across delay settings.
# ============================================================

def evaluate_reference_accuracy(
    model,
    test_ds,
    config,
):

    maes = []
    rmses = []
    residuals = []


    for (
        features,
        y_true,
    ) in test_ds:

        y_pred = standard_predict(
            model,
            features,
            config,
        )


        mae_value = tf.reduce_mean(
            tf.abs(
                y_true
                - y_pred
            )
        )


        rmse_value = tf.sqrt(
            tf.reduce_mean(
                tf.square(
                    y_true
                    - y_pred
                )
            )
            + 1e-12
        )


        residual_value = wave_residual_norm(

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
                mae_value.numpy()
            )
        )


        rmses.append(
            float(
                rmse_value.numpy()
            )
        )


        residuals.append(
            float(
                residual_value.numpy()
            )
        )


    return {

        "reference_test_mae":
            float(
                np.mean(
                    maes
                )
            ),

        "reference_test_rmse":
            float(
                np.mean(
                    rmses
                )
            ),

        "reference_wave_residual":
            float(
                np.mean(
                    residuals
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    common.set_determinism()


    if args.delay is None:

        delays_to_run = (
            COMMUNICATION_DELAYS_MS
        )

    else:

        delays_to_run = [
            float(
                args.delay
            )
        ]


    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 80)

    print(
        "SACU COMMUNICATION-COST SCALABILITY"
    )

    print("=" * 80)


    print(
        "Fixed architecture:",
        "grid=4 -> 16 SACUs",
    )


    print(
        "Fixed domain:",
        "128 x 128",
    )


    print(
        "Communication delay conditions:",
        delays_to_run,
    )


    print(
        "Delay interpretation:",
        "controlled per-round synchronization delay",
    )


    print(
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        ),
    )


    # ========================================================
    # REFERENCE CHECKPOINT
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
        "Reference result:"
    )

    print(
        reference_result_path
    )


    print(
        "Reference checkpoint:"
    )

    print(
        checkpoint
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
    # DATA
    # ========================================================

    test_ds = (
        build_test_dataset()
    )


    (
        benchmark_features,
        _,
    ) = next(
        iter(
            test_ds.take(1)
        )
    )


    print()
    print(
        "Benchmark input shape:",
        benchmark_features[
            "x"
        ].shape,
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = build_model(

        benchmark_features[
            "x"
        ],

        config,
    )


    load_reference_weights(
        model,
        checkpoint,
    )


    parameter_count = int(
        model.count_params()
    )


    if parameter_count != int(
        reference_result[
            "parameter_count"
        ]
    ):

        raise RuntimeError(
            "Parameter count differs from "
            "the validated reference model."
        )


    print(
        "Parameters:",
        parameter_count,
    )


    print(
        "PASS: fixed checkpoint restored."
    )


    # ========================================================
    # VERIFY EXPLICIT COMMUNICATION IMPLEMENTATION
    # ========================================================

    equivalence = (
        verify_zero_delay_equivalence(

            model,

            benchmark_features,

            config,
        )
    )


    # ========================================================
    # ZERO-DELAY REFERENCE PREDICTION
    # ========================================================

    (
        zero_delay_prediction,
        _,
    ) = predict_with_communication_delay(

        model,

        benchmark_features[
            "x"
        ],

        benchmark_features[
            "c_field"
        ],

        benchmark_features[
            "dt"
        ],

        benchmark_features[
            "dx"
        ],

        delay_ms=
            0.0,

        config=
            config,
    )


    synchronize_tensor(
        zero_delay_prediction
    )


    # ========================================================
    # REFERENCE ACCURACY
    # ========================================================

    accuracy = (
        evaluate_reference_accuracy(

            model,

            test_ds,

            config,
        )
    )


    print()
    print(
        "Native reference predictive metrics:"
    )


    print(
        "  MAE:",
        f"{accuracy['reference_test_mae']:.6f}",
    )


    print(
        "  RMSE:",
        f"{accuracy['reference_test_rmse']:.6f}",
    )


    print(
        "  Wave residual:",
        f"{accuracy['reference_wave_residual']:.6f}",
    )


    # ========================================================
    # CREATE RUN DIRECTORY
    # ========================================================

    experiment = (
        common.initialize_scalability_run(

            experiment_name=
                "communication_cost_scaling",

            experiment_config=
                {

                    "experiment":
                        "communication_cost_scaling",

                    "grid":
                        REFERENCE_GRID,

                    "agent_count":
                        REFERENCE_AGENTS,

                    "domain_height":
                        REFERENCE_HEIGHT,

                    "domain_width":
                        REFERENCE_WIDTH,

                    "delay_values_ms":
                        delays_to_run,

                    "delay_injection_location":
                        (
                            "between_pass1_message_generation_"
                            "and_pass2_message_conditioning"
                        ),

                    "delay_semantics":
                        (
                            "controlled_per_round_"
                            "synchronization_delay"
                        ),

                    "measured_network_latency":
                        False,

                    "retraining":
                        False,

                    "checkpoint_source":
                        str(
                            checkpoint
                        ),

                    "new_reviewer_requested_experiment":
                        True,

                    "replaces_existing_manuscript_numbers":
                        False,
                },
        )
    )


    output_dir = (
        experiment[
            "run_dir"
        ]
    )


    # ========================================================
    # RUN DELAY CONDITIONS
    # ========================================================

    results = []


    for delay_ms in delays_to_run:

        print()
        print("-" * 80)

        print(
            f"COMMUNICATION DELAY: "
            f"{delay_ms:.1f} ms"
        )

        print("-" * 80)


        condition = benchmark_delay(

            model,

            benchmark_features,

            delay_ms,

            config,

            zero_delay_prediction,
        )


        condition.update(
            {

                "grid":
                    REFERENCE_GRID,

                "agent_count":
                    REFERENCE_AGENTS,

                "domain_height":
                    REFERENCE_HEIGHT,

                "domain_width":
                    REFERENCE_WIDTH,

                "parameter_count":
                    parameter_count,

                "delay_type":
                    (
                        "controlled_injected_"
                        "synchronization_delay"
                    ),

                "measured_physical_network_latency":
                    False,

                **accuracy,

                **equivalence,
            }
        )


        results.append(
            condition
        )


        print(
            "Mean end-to-end latency:",
            (
                f"{condition['latency_mean_sec']:.6f} sec"
            ),
        )


        print(
            "P95 end-to-end latency:",
            (
                f"{condition['latency_p95_sec']:.6f} sec"
            ),
        )


        print(
            "Throughput:",
            (
                f"{condition['throughput_sequences_per_sec']:.3f} "
                "sequences/sec"
            ),
        )


        print(
            "Realized barrier:",
            (
                f"{condition['realized_barrier_mean_sec'] * 1000:.3f} ms"
            ),
        )


        print(
            "Prediction max difference:",
            condition[
                "prediction_max_abs_difference"
            ],
        )


        if (
            condition[
                "prediction_max_abs_difference"
            ]
            > 1e-6
        ):

            raise RuntimeError(
                "Delay changed model prediction."
            )


    # ========================================================
    # ADD RELATIVE SCALING
    # ========================================================

    zero_row = None


    for row in results:

        if (
            abs(
                row[
                    "requested_delay_ms"
                ]
            )
            < 1e-12
        ):

            zero_row = row
            break


    # If user ran a single nonzero condition, obtain a clean
    # zero-delay timing reference separately.
    if zero_row is None:

        zero_row = benchmark_delay(

            model,

            benchmark_features,

            0.0,

            config,

            zero_delay_prediction,
        )


    base_latency = float(
        zero_row[
            "latency_mean_sec"
        ]
    )


    base_throughput = float(
        zero_row[
            "throughput_sequences_per_sec"
        ]
    )


    for row in results:

        row[
            "latency_increase_sec_vs_zero"
        ] = (
            float(
                row[
                    "latency_mean_sec"
                ]
            )
            -
            base_latency
        )


        row[
            "latency_ratio_vs_zero"
        ] = (
            float(
                row[
                    "latency_mean_sec"
                ]
            )
            /
            base_latency
        )


        row[
            "throughput_ratio_vs_zero"
        ] = (
            float(
                row[
                    "throughput_sequences_per_sec"
                ]
            )
            /
            base_throughput
        )


        row[
            "throughput_reduction_percent_vs_zero"
        ] = (
            (
                1.0
                -
                row[
                    "throughput_ratio_vs_zero"
                ]
            )
            * 100.0
        )


    # ========================================================
    # SAVE
    # ========================================================

    save_csv(
        output_dir
        / "communication_cost_scaling.csv",
        results,
    )


    save_json(
        output_dir
        / "communication_cost_scaling.json",
        {

            "experiment":
                "communication_cost_scaling",

            "delay_semantics":
                (
                    "controlled synchronization delay "
                    "inserted between SACU communication passes"
                ),

            "physical_network_latency_measured":
                False,

            "reference_checkpoint":
                str(
                    checkpoint
                ),

            "reference_accuracy":
                accuracy,

            "zero_delay_equivalence":
                equivalence,

            "results":
                results,
        },
    )


    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 100)

    print(
        "COMMUNICATION-COST SCALING SUMMARY"
    )

    print("=" * 100)


    print(
        f"{'Delay ms':>10}"
        f"{'Latency s':>14}"
        f"{'P95 s':>14}"
        f"{'Throughput':>14}"
        f"{'Barrier ms':>14}"
        f"{'Pred diff':>14}"
    )


    print(
        "-" * 80
    )


    for row in results:

        print(
            f"{row['requested_delay_ms']:>10.1f}"
            f"{row['latency_mean_sec']:>14.6f}"
            f"{row['latency_p95_sec']:>14.6f}"
            f"{row['throughput_sequences_per_sec']:>14.3f}"
            f"{row['realized_barrier_mean_sec'] * 1000:>14.3f}"
            f"{row['prediction_max_abs_difference']:>14.3e}"
        )


    print()
    print(
        "PASS: communication-cost scaling completed."
    )


    print(
        "PASS: the fixed trained 16-agent SACU "
        "checkpoint was preserved."
    )


    print(
        "PASS: communication delay changed timing only; "
        "predictions remained unchanged."
    )


    print(
        "IMPORTANT: injected delays are controlled "
        "communication scenarios, not measured network latency."
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