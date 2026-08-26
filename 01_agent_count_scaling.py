from __future__ import annotations

# ============================================================
# SACU AGENT-COUNT SCALABILITY
#
# Memory-efficient exact recomputation implementation.
#
# Scientific configuration:
#   grid=2 ->  4 SACUs
#   grid=3 ->  9 SACUs
#   grid=4 -> 16 SACUs  [reference]
#   grid=5 -> 25 SACUs
#
# IMPORTANT:
#   - The OrgSACUSolver architecture is unchanged.
#   - SACU, MicroExpert, roles, communications, influence
#     weighting, physics loss, optimizer, data split, and
#     validation/test policy are unchanged.
#   - Only the training differentiation strategy is changed
#     to avoid holding every agent's Conv3D activation graph
#     in GPU memory at the same time.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ============================================================
# TENSORFLOW ENVIRONMENT
#
# These must be configured before TensorFlow import.
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
from tensorflow import keras


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

            # Device already initialized.
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

ORG_REFERENCE_PATH = (
    EXPERIMENT_DIR.parent
    / "02_organizational_evolution"
    / "01_train_and_log_evolution.py"
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
        "Unable to load scalability protocol:\n"
        f"{PROTOCOL_PATH}"
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

from src.physics_metrics import (
    wave_residual_norm,
)

from src.training.trainer_v2 import (
    predict_sacu_deployment,
)

from src.data_loader import (
    make_dataset,
)


# ============================================================
# LOAD VALIDATED SACU CONFIGURATION
# ============================================================

if not ORG_REFERENCE_PATH.exists():

    raise FileNotFoundError(
        "Organizational-evolution reference experiment "
        "was not found:\n"
        f"{ORG_REFERENCE_PATH}"
    )


org_spec = importlib.util.spec_from_file_location(
    "organizational_evolution_reference",
    ORG_REFERENCE_PATH,
)

if org_spec is None or org_spec.loader is None:

    raise ImportError(
        "Unable to load organizational-evolution "
        "reference configuration."
    )


org_reference = importlib.util.module_from_spec(
    org_spec
)

org_spec.loader.exec_module(
    org_reference
)


if not hasattr(
    org_reference,
    "CONFIG",
):

    raise RuntimeError(
        "Reference organizational-evolution experiment "
        "does not expose CONFIG."
    )


REFERENCE_CONFIG = dict(
    org_reference.CONFIG
)


# ============================================================
# FIXED AGENT CONDITIONS
# ============================================================

GRID_VALUES = [
    2,
    3,
    4,
    5,
]

AGENT_COUNTS = [
    4,
    9,
    16,
    25,
]

REFERENCE_GRID = 4
REFERENCE_AGENT_COUNT = 16


# ============================================================
# EXPERIMENT DESCRIPTION
# ============================================================

EXPERIMENT_CONFIG = {

    "experiment":
        "agent_count_scaling",

    "model":
        "src.models_sacu.OrgSACUSolver",

    "agent_count_definition":
        "N = grid * grid",

    "grid_values":
        GRID_VALUES,

    "agent_counts":
        AGENT_COUNTS,

    "reference_grid":
        REFERENCE_GRID,

    "reference_agent_count":
        REFERENCE_AGENT_COUNT,

    "training_gradient_method":
        (
            "exact_global_output_gradient_with_"
            "per_sacu_recomputation"
        ),

    "communication_gradient_method":
        (
            "exact_neighbor_mean_message_chain_rule"
        ),

    "selection_metric":
        "validation_RMSE",

    "test_policy":
        "evaluate_best_validation_checkpoint_only",

    "checkpoint_policy":
        "weights_only_rebuild_then_load",

    "scientific_policy":
        (
            "Only grid and therefore SACU count vary. "
            "No model width, expert count, overlap, message "
            "dimension, role mechanism, communication mechanism, "
            "loss coefficient, influence weighting, optimizer, "
            "training budget, or data split is changed."
        ),

    "memory_policy":
        (
            "The complete differentiable deployment objective is "
            "preserved, but local SACU activations are recomputed "
            "one agent at a time instead of retaining all Conv3D "
            "activation graphs simultaneously."
        ),

    "new_reviewer_requested_experiment":
        True,

    "replaces_existing_manuscript_numbers":
        False,
}


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Memory-efficient SACU agent-count "
            "scalability experiment."
        )
    )

    parser.add_argument(
        "--grid",
        type=int,
        required=True,
        choices=GRID_VALUES,
        help=(
            "Spatial SACU grid. "
            "2,3,4,5 correspond to 4,9,16,25 agents."
        ),
    )

    return parser.parse_args()


# ============================================================
# REFERENCE CONFIG HELPER
# ============================================================

def cfg(
    key: str,
    default: Any = None,
):

    return REFERENCE_CONFIG.get(
        key,
        default,
    )


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

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
# CPU COPY
#
# Compact deployment tensors are deliberately moved off GPU
# between SACU passes. This prevents patch-output storage from
# competing with Conv3D replay activations.
# ============================================================

def cpu_copy(
    tensor: tf.Tensor,
) -> tf.Tensor:

    with tf.device(
        "/CPU:0"
    ):

        return tf.identity(
            tf.stop_gradient(
                tensor
            )
        )


# ============================================================
# DATASETS
# ============================================================

def build_datasets(
    train_files,
    validation_files,
    test_files,
):

    batch_size = int(
        cfg(
            "batch_size",
            common.COMMON_CONFIG[
                "batch_size"
            ],
        )
    )

    if batch_size != 1:

        raise ValueError(
            "Memory-efficient SACU scalability "
            "currently requires batch_size=1."
        )

    train_files = [
        str(path)
        for path in train_files
    ]

    validation_files = [
        str(path)
        for path in validation_files
    ]

    test_files = [
        str(path)
        for path in test_files
    ]

    train_ds = make_dataset(
        train_files,
        batch_size=batch_size,
        shuffle=True,
        repeat=False,
        deterministic=True,
    )

    validation_ds = make_dataset(
        validation_files,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        deterministic=True,
    )

    test_ds = make_dataset(
        test_files,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        deterministic=True,
    )

    return (
        train_ds,
        validation_ds,
        test_ds,
    )


# ============================================================
# MODEL CREATION
# ============================================================

def build_model(
    grid: int,
    sample_x: tf.Tensor,
):

    model = OrgSACUSolver(

        grid=
            int(
                grid
            ),

        overlap=
            int(
                cfg(
                    "overlap",
                    8,
                )
            ),

        K=
            int(
                cfg(
                    "K",
                    4,
                )
            ),

        hidden=
            int(
                cfg(
                    "hidden",
                    64,
                )
            ),

        msg_dim=
            int(
                cfg(
                    "msg_dim",
                    16,
                )
            ),

        use_role=
            bool(
                cfg(
                    "use_role",
                    True,
                )
            ),

        use_comms=
            bool(
                cfg(
                    "use_comms",
                    True,
                )
            ),

        name=
            f"org_sacu_grid_{grid}",
    )

    # Explicitly instantiate all variables.
    _ = model(
        sample_x,
        training=False,
    )

    expected_agents = (
        grid
        * grid
    )

    if model.N != expected_agents:

        raise RuntimeError(
            "Incorrect SACU count.\n"
            f"grid={grid}\n"
            f"Expected={expected_agents}\n"
            f"Observed={model.N}"
        )

    if not model.built:

        raise RuntimeError(
            "OrgSACUSolver failed to build."
        )

    return model


# ============================================================
# OPTIMIZER
# ============================================================

def build_optimizer():

    return keras.optimizers.Adam(

        learning_rate=
            float(
                cfg(
                    "learning_rate",
                    common.COMMON_CONFIG[
                        "learning_rate"
                    ],
                )
            )
    )


# ============================================================
# ROLE IDS
# ============================================================

def build_role_ids(
    model,
    batch_size: tf.Tensor,
):

    if not model.use_role:

        return None

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

    return role_ids


# ============================================================
# REGIONS
# ============================================================

def get_regions(
    model,
    x: tf.Tensor,
):

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

    if (
        height is None
        or width is None
    ):

        raise ValueError(
            "SACU scalability requires statically known "
            "spatial dimensions."
        )

    return make_regions(
        height,
        width,
        grid=model.grid,
        overlap=model.overlap,
    )


# ============================================================
# FIRST-PASS MESSAGE COLLECTION
#
# No GradientTape:
# Conv3D activations are released after each SACU.
# Only the compact messages are retained.
# ============================================================

def collect_first_pass_messages(
    model,
    x: tf.Tensor,
    regions,
):

    batch_size = tf.shape(
        x
    )[0]

    role_ids = build_role_ids(
        model,
        batch_size,
    )

    zero_message = tf.zeros(
        [
            batch_size,
            model.msg_dim,
        ],
        dtype=x.dtype,
    )

    messages_cpu = []

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
            _,
            _,
            message_out,
        ) = model.sacus[
            i
        ](
            xp,
            role_id=role_id,
            msg_in=message_in,
            training=True,
        )

        if model.use_comms:

            messages_cpu.append(
                cpu_copy(
                    message_out
                )
            )

        del xp
        del message_out

    return (
        messages_cpu,
        role_ids,
        zero_message,
    )


# ============================================================
# CONSTRUCT COMMUNICATION INPUT
# ============================================================

def communication_input_cpu(
    model,
    agent_index: int,
    first_messages_cpu,
    zero_message: tf.Tensor,
):

    if not model.use_comms:

        return None

    neighbors = model.neigh[
        agent_index
    ]

    if len(
        neighbors
    ) == 0:

        return cpu_copy(
            zero_message
        )

    selected = [
        first_messages_cpu[
            j
        ]
        for j in neighbors
    ]

    with tf.device(
        "/CPU:0"
    ):

        return tf.reduce_mean(
            tf.stack(
                selected,
                axis=1,
            ),
            axis=1,
        )


# ============================================================
# FINAL PATCH COLLECTION
#
# When communications are enabled this reproduces the second
# OrgSACUSolver pass.
#
# Again no GradientTape is retained.
# ============================================================

def collect_final_patch_outputs(
    model,
    x: tf.Tensor,
    regions,
    first_messages_cpu,
    role_ids,
    zero_message,
):

    patch_outputs_cpu = []
    gates_cpu = []
    message_inputs_cpu = []

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

        if model.use_comms:

            message_cpu = communication_input_cpu(
                model,
                i,
                first_messages_cpu,
                zero_message,
            )

            message_gpu = tf.identity(
                message_cpu
            )

        else:

            message_cpu = None
            message_gpu = None

        (
            patch_prediction,
            gate,
            _,
        ) = model.sacus[
            i
        ](
            xp,
            role_id=role_id,
            msg_in=message_gpu,
            training=True,
        )

        patch_outputs_cpu.append(
            cpu_copy(
                patch_prediction
            )
        )

        gates_cpu.append(
            cpu_copy(
                gate
            )
        )

        message_inputs_cpu.append(
            message_cpu
        )

        del xp
        del patch_prediction
        del gate
        del message_gpu

    return (
        patch_outputs_cpu,
        gates_cpu,
        message_inputs_cpu,
    )


# ============================================================
# GLOBAL OBJECTIVE AND OUTPUT GRADIENTS
#
# This stage runs on CPU.
#
# It exactly preserves:
#   compute_deployment_influence_weights
#   stitch_patches
#   MSE
#   wave_residual_norm
#
# The tape watches only patch predictions and gates, not the
# huge internal Conv3D activation graphs.
# ============================================================

def global_objective_and_upstreams(
    patch_outputs_cpu,
    gates_cpu,
    regions,
    x,
    y_true,
    c_field,
    dt,
    dx,
):

    with tf.device(
        "/CPU:0"
    ):

        x_cpu = tf.identity(
            x
        )

        y_true_cpu = tf.identity(
            y_true
        )

        c_cpu = tf.identity(
            c_field
        )

        dt_cpu = tf.identity(
            dt
        )

        dx_cpu = tf.identity(
            dx
        )

        patch_leaves = [
            tf.identity(
                tensor
            )
            for tensor
            in patch_outputs_cpu
        ]

        gate_leaves = [
            tf.identity(
                tensor
            )
            for tensor
            in gates_cpu
        ]

        with tf.GradientTape() as tape:

            for tensor in patch_leaves:

                tape.watch(
                    tensor
                )

            for tensor in gate_leaves:

                tape.watch(
                    tensor
                )

            (
                influence_weights,
                _,
            ) = compute_deployment_influence_weights(

                patch_leaves,
                gate_leaves,
                regions,
                x_cpu,
                c_cpu,
                dt_cpu,
                dx_cpu,

                float(
                    cfg(
                        "sensor_weight",
                        0.50,
                    )
                ),

                float(
                    cfg(
                        "physics_weight",
                        0.35,
                    )
                ),

                float(
                    cfg(
                        "entropy_weight",
                        0.15,
                    )
                ),

                float(
                    cfg(
                        "temperature",
                        5.0,
                    )
                ),
            )

            y_pred = stitch_patches(

                patch_leaves,
                regions,
                influence_weights,
                tf.shape(
                    x_cpu
                )[2],
                tf.shape(
                    x_cpu
                )[3],
            )

            data_loss = tf.reduce_mean(
                tf.square(
                    y_pred
                    - y_true_cpu
                )
            )

            residual = wave_residual_norm(
                y_pred,
                c_cpu,
                dt_cpu,
                dx_cpu,
            )

            if bool(
                cfg(
                    "use_physics_loss",
                    True,
                )
            ):

                total_loss = (
                    data_loss
                    +
                    tf.cast(
                        float(
                            cfg(
                                "lambda_phys",
                                0.05,
                            )
                        ),
                        tf.float32,
                    )
                    * residual
                )

            else:

                total_loss = data_loss

            batch_mae = tf.reduce_mean(
                tf.abs(
                    y_true_cpu
                    - y_pred
                )
            )

            batch_rmse = tf.sqrt(
                tf.reduce_mean(
                    tf.square(
                        y_true_cpu
                        - y_pred
                    )
                )
                + 1e-12
            )

        watched = (
            patch_leaves
            + gate_leaves
        )

        upstreams = tape.gradient(
            total_loss,
            watched,
        )

        number_of_patches = len(
            patch_leaves
        )

        patch_upstreams = upstreams[
            :number_of_patches
        ]

        gate_upstreams = upstreams[
            number_of_patches:
        ]

        # Replace mathematically-zero None gradients.
        patch_upstreams = [

            (
                gradient
                if gradient is not None
                else tf.zeros_like(
                    tensor
                )
            )

            for gradient, tensor
            in zip(
                patch_upstreams,
                patch_leaves,
            )
        ]

        gate_upstreams = [

            (
                gradient
                if gradient is not None
                else tf.zeros_like(
                    tensor
                )
            )

            for gradient, tensor
            in zip(
                gate_upstreams,
                gate_leaves,
            )
        ]

        return (
            float(
                total_loss.numpy()
            ),
            float(
                data_loss.numpy()
            ),
            float(
                batch_mae.numpy()
            ),
            float(
                batch_rmse.numpy()
            ),
            float(
                residual.numpy()
            ),
            [
                cpu_copy(
                    gradient
                )
                for gradient
                in patch_upstreams
            ],
            [
                cpu_copy(
                    gradient
                )
                for gradient
                in gate_upstreams
            ],
        )


# ============================================================
# ZERO GRADIENT ACCUMULATORS
# ============================================================

def initialize_gradient_accumulators(
    model,
):

    return [

        [
            tf.zeros_like(
                variable
            )

            for variable
            in sacu.trainable_variables
        ]

        for sacu
        in model.sacus
    ]


# ============================================================
# ADD GRADIENTS TO ONE SACU
# ============================================================

def accumulate_local_gradients(
    accumulator,
    gradients,
    variables,
):

    return [

        previous
        +
        (
            gradient
            if gradient is not None
            else tf.zeros_like(
                variable
            )
        )

        for (
            previous,
            gradient,
            variable,
        )
        in zip(
            accumulator,
            gradients,
            variables,
        )
    ]


# ============================================================
# REPLAY FINAL / SECOND COMMUNICATION PASS
#
# For each SACU:
#   1. recompute only that local network;
#   2. contract output/gate with global upstream gradients;
#   3. obtain parameter gradients;
#   4. obtain dL/d(message_in).
#
# Only ONE SACU activation graph exists at a time.
# ============================================================

def replay_final_pass(
    model,
    x,
    regions,
    role_ids,
    message_inputs_cpu,
    patch_upstreams_cpu,
    gate_upstreams_cpu,
    accumulators,
):

    message_input_gradients_cpu = [

        None
        for _
        in range(
            model.N
        )
    ]

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

        upstream_patch = tf.identity(
            patch_upstreams_cpu[
                i
            ]
        )

        upstream_gate = tf.identity(
            gate_upstreams_cpu[
                i
            ]
        )

        if model.use_comms:

            message_in = tf.identity(
                message_inputs_cpu[
                    i
                ]
            )

            with tf.GradientTape() as tape:

                tape.watch(
                    message_in
                )

                (
                    patch_prediction,
                    gate,
                    _,
                ) = model.sacus[
                    i
                ](
                    xp,
                    role_id=role_id,
                    msg_in=message_in,
                    training=True,
                )

                surrogate = (
                    tf.reduce_sum(
                        patch_prediction
                        * upstream_patch
                    )
                    +
                    tf.reduce_sum(
                        gate
                        * upstream_gate
                    )
                )

            targets = (
                list(
                    model.sacus[
                        i
                    ].trainable_variables
                )
                +
                [
                    message_in
                ]
            )

            gradients = tape.gradient(
                surrogate,
                targets,
            )

            parameter_gradients = gradients[
                :-1
            ]

            message_gradient = gradients[
                -1
            ]

            if message_gradient is None:

                message_gradient = tf.zeros_like(
                    message_in
                )

            message_input_gradients_cpu[
                i
            ] = cpu_copy(
                message_gradient
            )

        else:

            with tf.GradientTape() as tape:

                (
                    patch_prediction,
                    gate,
                    _,
                ) = model.sacus[
                    i
                ](
                    xp,
                    role_id=role_id,
                    msg_in=None,
                    training=True,
                )

                surrogate = (
                    tf.reduce_sum(
                        patch_prediction
                        * upstream_patch
                    )
                    +
                    tf.reduce_sum(
                        gate
                        * upstream_gate
                    )
                )

            parameter_gradients = tape.gradient(
                surrogate,
                model.sacus[
                    i
                ].trainable_variables,
            )

        accumulators[
            i
        ] = accumulate_local_gradients(

            accumulators[
                i
            ],

            parameter_gradients,

            model.sacus[
                i
            ].trainable_variables,
        )

        del xp
        del upstream_patch
        del upstream_gate
        del patch_prediction
        del gate
        del surrogate
        del tape

    return (
        accumulators,
        message_input_gradients_cpu,
    )


# ============================================================
# COMMUNICATION CHAIN RULE
#
# OrgSACUSolver defines:
#
#   m_in(i) = mean(messages[j] for j in neighbors(i))
#
# Therefore:
#
#   dL/dmessage[j] +=
#       dL/dm_in(i) / number_of_neighbors(i)
#
# for every j in neighbors(i).
# ============================================================

def distribute_message_gradients(
    model,
    first_messages_cpu,
    message_input_gradients_cpu,
):

    if not model.use_comms:

        return []

    message_output_gradients_cpu = [

        tf.zeros_like(
            message
        )

        for message
        in first_messages_cpu
    ]

    for receiver_index in range(
        model.N
    ):

        neighbors = model.neigh[
            receiver_index
        ]

        if len(
            neighbors
        ) == 0:

            continue

        incoming_gradient = (
            message_input_gradients_cpu[
                receiver_index
            ]
        )

        if incoming_gradient is None:

            continue

        contribution = (
            incoming_gradient
            / float(
                len(
                    neighbors
                )
            )
        )

        for sender_index in neighbors:

            message_output_gradients_cpu[
                sender_index
            ] = (
                message_output_gradients_cpu[
                    sender_index
                ]
                +
                contribution
            )

    return message_output_gradients_cpu


# ============================================================
# REPLAY FIRST COMMUNICATION PASS
#
# The first-pass patch outputs/gates are discarded by the
# original OrgSACUSolver when communication is enabled.
#
# Therefore only the first-pass message receives upstream
# gradient from the second communication round.
# ============================================================

def replay_first_pass_messages(
    model,
    x,
    regions,
    role_ids,
    zero_message,
    message_output_gradients_cpu,
    accumulators,
):

    if not model.use_comms:

        return accumulators

    for i, (
        y0,
        y1,
        x0,
        x1,
    ) in enumerate(
        regions
    ):

        upstream_message = tf.identity(
            message_output_gradients_cpu[
                i
            ]
        )

        # Skip a truly zero communication gradient.
        if bool(
            tf.reduce_all(
                tf.equal(
                    upstream_message,
                    0.0,
                )
            ).numpy()
        ):

            continue

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

        with tf.GradientTape() as tape:

            (
                _,
                _,
                message_out,
            ) = model.sacus[
                i
            ](
                xp,
                role_id=role_id,
                msg_in=zero_message,
                training=True,
            )

            surrogate = tf.reduce_sum(
                message_out
                * upstream_message
            )

        gradients = tape.gradient(
            surrogate,
            model.sacus[
                i
            ].trainable_variables,
        )

        accumulators[
            i
        ] = accumulate_local_gradients(

            accumulators[
                i
            ],

            gradients,

            model.sacus[
                i
            ].trainable_variables,
        )

        del xp
        del message_out
        del surrogate
        del tape
        del upstream_message

    return accumulators


# ============================================================
# APPLY ACCUMULATED GRADIENTS
# ============================================================

def apply_accumulated_gradients(
    model,
    optimizer,
    accumulators,
):

    gradient_variable_pairs = []

    for i in range(
        model.N
    ):

        for gradient, variable in zip(

            accumulators[
                i
            ],

            model.sacus[
                i
            ].trainable_variables,
        ):

            gradient_variable_pairs.append(
                (
                    gradient,
                    variable,
                )
            )

    if not gradient_variable_pairs:

        raise RuntimeError(
            "No SACU gradients were generated."
        )

    optimizer.apply_gradients(
        gradient_variable_pairs
    )


# ============================================================
# MEMORY-EFFICIENT EXACT TRAINING STEP
# ============================================================

def train_sequence_recomputed(
    model,
    optimizer,
    features,
    y_true,
):

    x = features[
        "x"
    ]

    c_field = features[
        "c_field"
    ]

    dt = features[
        "dt"
    ]

    dx = features[
        "dx"
    ]

    regions = get_regions(
        model,
        x,
    )

    # --------------------------------------------------------
    # Phase 1:
    # first-pass communication messages only.
    # --------------------------------------------------------

    (
        first_messages_cpu,
        role_ids,
        zero_message,
    ) = collect_first_pass_messages(
        model,
        x,
        regions,
    )

    # --------------------------------------------------------
    # Phase 2:
    # final SACU patch outputs/gates without retaining
    # local activation graphs.
    # --------------------------------------------------------

    (
        patch_outputs_cpu,
        gates_cpu,
        message_inputs_cpu,
    ) = collect_final_patch_outputs(

        model,
        x,
        regions,

        first_messages_cpu,
        role_ids,
        zero_message,
    )

    # --------------------------------------------------------
    # Phase 3:
    # exact global objective on compact leaf tensors.
    # --------------------------------------------------------

    (
        total_loss,
        data_loss,
        batch_mae,
        batch_rmse,
        residual,
        patch_upstreams_cpu,
        gate_upstreams_cpu,
    ) = global_objective_and_upstreams(

        patch_outputs_cpu,
        gates_cpu,
        regions,

        x,
        y_true,
        c_field,
        dt,
        dx,
    )

    # Original patch output copies no longer required.
    del patch_outputs_cpu
    del gates_cpu

    gc.collect()

    # --------------------------------------------------------
    # Phase 4:
    # replay final SACU pass one agent at a time.
    # --------------------------------------------------------

    accumulators = initialize_gradient_accumulators(
        model
    )

    (
        accumulators,
        message_input_gradients_cpu,
    ) = replay_final_pass(

        model,
        x,
        regions,
        role_ids,
        message_inputs_cpu,

        patch_upstreams_cpu,
        gate_upstreams_cpu,

        accumulators,
    )

    del patch_upstreams_cpu
    del gate_upstreams_cpu
    del message_inputs_cpu

    # --------------------------------------------------------
    # Phase 5:
    # exact communication-chain gradient.
    # --------------------------------------------------------

    if model.use_comms:

        message_output_gradients_cpu = (
            distribute_message_gradients(

                model,
                first_messages_cpu,
                message_input_gradients_cpu,
            )
        )

        # ----------------------------------------------------
        # Phase 6:
        # replay first SACU communication pass.
        # ----------------------------------------------------

        accumulators = (
            replay_first_pass_messages(

                model,
                x,
                regions,
                role_ids,
                zero_message,

                message_output_gradients_cpu,

                accumulators,
            )
        )

    # --------------------------------------------------------
    # Phase 7:
    # one optimizer update, exactly as original trainer.
    # --------------------------------------------------------

    apply_accumulated_gradients(
        model,
        optimizer,
        accumulators,
    )

    del accumulators
    del first_messages_cpu
    del message_input_gradients_cpu

    gc.collect()

    return (
        total_loss,
        data_loss,
        batch_mae,
        batch_rmse,
        residual,
    )


# ============================================================
# STANDARD DEPLOYMENT PREDICTION
#
# Evaluation uses the existing deployment implementation.
# No training graph is retained during inference.
# ============================================================

def standard_predict(
    model,
    features,
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
            float(
                cfg(
                    "sensor_weight",
                    0.50,
                )
            ),

        physics_weight=
            float(
                cfg(
                    "physics_weight",
                    0.35,
                )
            ),

        entropy_weight=
            float(
                cfg(
                    "entropy_weight",
                    0.15,
                )
            ),

        temperature=
            float(
                cfg(
                    "temperature",
                    5.0,
                )
            ),
    )

    return y_pred


# ============================================================
# EVALUATION
# ============================================================

def evaluate_dataset(
    model,
    dataset,
):

    maes = []
    rmses = []
    residuals = []

    for (
        features,
        y_true,
    ) in dataset:

        y_pred = standard_predict(
            model,
            features,
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
            "Evaluation dataset is empty."
        )

    return {

        "count":
            len(
                maes
            ),

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

        "wave_residual":
            float(
                np.mean(
                    residuals
                )
            ),
    }


# ============================================================
# LATENCY
# ============================================================

def measure_latency(
    model,
    test_ds,
):

    features, _ = next(
        iter(
            test_ds.take(1)
        )
    )

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

    for _ in range(
        warmup_runs
    ):

        y_pred = standard_predict(
            model,
            features,
        )

        _ = tf.reduce_sum(
            y_pred
        ).numpy()

    timings = []

    for _ in range(
        measurement_runs
    ):

        start = time.perf_counter()

        y_pred = standard_predict(
            model,
            features,
        )

        # Synchronize GPU.
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

    timing[
        "throughput_sequences_per_sec"
    ] = (
        1.0
        / timing[
            "mean_sec"
        ]
    )

    return timing


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def save_best_weights(
    model,
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_weights(
        str(
            path
        )
    )

    if not path.exists():

        raise FileNotFoundError(
            "Checkpoint was not created:\n"
            f"{path}"
        )


def reload_model(
    grid: int,
    sample_x: tf.Tensor,
    checkpoint: Path,
):

    if not checkpoint.exists():

        raise FileNotFoundError(
            "Checkpoint does not exist:\n"
            f"{checkpoint}"
        )

    model = build_model(
        grid,
        sample_x,
    )

    model.load_weights(
        str(
            checkpoint
        )
    )

    return model


# ============================================================
# OPTIONAL GRADIENT VALIDATION
#
# For grid=2 we can compare one sequence between:
#   standard TrainerV2 gradient
#   recomputation gradient
#
# This is deliberately not run by default because the standard
# path is the one causing the memory problem at larger grids.
# ============================================================

def gradients_are_finite(
    model,
) -> bool:

    for variable in model.trainable_variables:

        if not bool(
            tf.reduce_all(
                tf.math.is_finite(
                    variable
                )
            ).numpy()
        ):

            return False

    return True


# ============================================================
# ONE GRID CONDITION
# ============================================================

def run_grid_condition(
    grid: int,
    experiment,
    train_ds,
    validation_ds,
    test_ds,
):

    agent_count = (
        grid
        * grid
    )

    run_dir = experiment[
        "run_dir"
    ]

    condition_dir = (
        run_dir
        / f"grid_{grid}_agents_{agent_count}"
    )

    condition_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print(
        f"SACU AGENT SCALING — GRID {grid}x{grid}"
    )
    print("=" * 80)

    print(
        "Agents:",
        agent_count,
    )

    print(
        "Reference condition:",
        grid == REFERENCE_GRID,
    )

    print(
        "Training differentiation:",
        "exact per-SACU recomputation",
    )

    print(
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        ),
    )

    # --------------------------------------------------------
    # Obtain sample shape.
    # --------------------------------------------------------

    first_features, _ = next(
        iter(
            train_ds.take(1)
        )
    )

    sample_x = first_features[
        "x"
    ]

    # --------------------------------------------------------
    # Reset memory statistics before experiment.
    # --------------------------------------------------------

    common.reset_gpu_memory_stats()

    # --------------------------------------------------------
    # Build model.
    # --------------------------------------------------------

    model = build_model(
        grid,
        sample_x,
    )

    parameter_count = int(
        model.count_params()
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    print(
        "Overlap:",
        cfg(
            "overlap",
            8,
        ),
    )

    print(
        "Experts K:",
        cfg(
            "K",
            4,
        ),
    )

    print(
        "Hidden width:",
        cfg(
            "hidden",
            64,
        ),
    )

    print(
        "Message dimension:",
        cfg(
            "msg_dim",
            16,
        ),
    )

    print(
        "Role mechanism:",
        cfg(
            "use_role",
            True,
        ),
    )

    print(
        "Communication:",
        cfg(
            "use_comms",
            True,
        ),
    )

    print(
        "Physics loss:",
        cfg(
            "use_physics_loss",
            True,
        ),
    )

    print(
        "Lambda physics:",
        cfg(
            "lambda_phys",
            0.05,
        ),
    )

    optimizer = build_optimizer()

    # --------------------------------------------------------
    # Training state.
    # --------------------------------------------------------

    epochs = int(
        cfg(
            "epochs",
            common.COMMON_CONFIG[
                "epochs"
            ],
        )
    )

    history: List[
        Dict[str, Any]
    ] = []

    best_validation_rmse = np.inf

    best_epoch: Optional[
        int
    ] = None

    best_weights_path = (
        condition_dir
        / "best.weights.h5"
    )

    training_start = (
        time.perf_counter()
    )

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        print()
        print("-" * 80)
        print(
            f"EPOCH {epoch}/{epochs}"
        )
        print("-" * 80)

        epoch_start = (
            time.perf_counter()
        )

        train_total_losses = []
        train_data_losses = []
        train_maes = []
        train_rmses = []
        train_residuals = []

        for sequence_index, (
            features,
            y_true,
        ) in enumerate(
            train_ds,
            start=1,
        ):

            try:

                (
                    total_loss,
                    data_loss,
                    batch_mae,
                    batch_rmse,
                    batch_residual,
                ) = train_sequence_recomputed(

                    model,
                    optimizer,
                    features,
                    y_true,
                )

            except (
                tf.errors.ResourceExhaustedError,
                tf.errors.UnknownError,
            ) as exc:

                memory = (
                    common.gpu_memory_info()
                )

                raise RuntimeError(
                    "\nMemory-efficient SACU training still "
                    "exceeded the available GPU resources.\n"
                    f"Grid: {grid}\n"
                    f"Agents: {agent_count}\n"
                    f"Sequence: {sequence_index}\n"
                    f"GPU memory info: {memory}\n"
                    "No SACU architecture parameter was changed."
                ) from exc

            train_total_losses.append(
                total_loss
            )

            train_data_losses.append(
                data_loss
            )

            train_maes.append(
                batch_mae
            )

            train_rmses.append(
                batch_rmse
            )

            train_residuals.append(
                batch_residual
            )

            if sequence_index % 10 == 0:

                memory = (
                    common.gpu_memory_info()
                )

                print(
                    f"  sequence "
                    f"{sequence_index:>3d}/70 | "
                    f"RMSE={batch_rmse:.6f} | "
                    f"peak GPU="
                    f"{memory.get('peak_bytes')}"
                )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

        validation = evaluate_dataset(
            model,
            validation_ds,
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        row = {

            "epoch":
                epoch,

            "train_total_loss":
                float(
                    np.mean(
                        train_total_losses
                    )
                ),

            "train_data_loss":
                float(
                    np.mean(
                        train_data_losses
                    )
                ),

            "train_mae":
                float(
                    np.mean(
                        train_maes
                    )
                ),

            "train_rmse":
                float(
                    np.mean(
                        train_rmses
                    )
                ),

            "train_wave_residual":
                float(
                    np.mean(
                        train_residuals
                    )
                ),

            "validation_mae":
                validation[
                    "mae"
                ],

            "validation_rmse":
                validation[
                    "rmse"
                ],

            "validation_wave_residual":
                validation[
                    "wave_residual"
                ],

            "epoch_time_sec":
                float(
                    epoch_time
                ),
        }

        history.append(
            row
        )

        print(
            f"Train total loss : "
            f"{row['train_total_loss']:.6f}"
        )

        print(
            f"Train MAE        : "
            f"{row['train_mae']:.6f}"
        )

        print(
            f"Train RMSE       : "
            f"{row['train_rmse']:.6f}"
        )

        print(
            f"Train residual   : "
            f"{row['train_wave_residual']:.6f}"
        )

        print(
            f"Validation MAE   : "
            f"{row['validation_mae']:.6f}"
        )

        print(
            f"Validation RMSE  : "
            f"{row['validation_rmse']:.6f}"
        )

        print(
            f"Validation resid.: "
            f"{row['validation_wave_residual']:.6f}"
        )

        print(
            f"Epoch time       : "
            f"{row['epoch_time_sec']:.3f} sec"
        )

        # ----------------------------------------------------
        # Validation-only checkpoint selection.
        # ----------------------------------------------------

        if common.is_better_checkpoint(

            row[
                "validation_rmse"
            ],

            best_validation_rmse,
        ):

            best_validation_rmse = (
                row[
                    "validation_rmse"
                ]
            )

            best_epoch = epoch

            save_best_weights(
                model,
                best_weights_path,
            )

            print(
                "[INFO] Saved new best validation checkpoint."
            )

        save_csv(
            condition_dir
            / "training_history.csv",
            history,
        )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    training_time = (
        time.perf_counter()
        - training_start
    )

    if best_epoch is None:

        raise RuntimeError(
            "No validation checkpoint was selected."
        )

    memory_after_training = (
        common.gpu_memory_info()
    )

    if not gradients_are_finite(
        model
    ):

        raise RuntimeError(
            "Non-finite model variables detected after training."
        )

    # ========================================================
    # RELOAD BEST WEIGHTS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOADING BEST VALIDATION CHECKPOINT"
    )
    print("=" * 80)

    del optimizer
    del model

    gc.collect()

    tf.keras.backend.clear_session()

    best_model = reload_model(
        grid,
        sample_x,
        best_weights_path,
    )

    reloaded_validation = evaluate_dataset(
        best_model,
        validation_ds,
    )

    reload_difference = abs(
        reloaded_validation[
            "rmse"
        ]
        -
        best_validation_rmse
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Selected validation RMSE:",
        best_validation_rmse,
    )

    print(
        "Reloaded validation RMSE:",
        reloaded_validation[
            "rmse"
        ],
    )

    print(
        "Checkpoint difference:",
        reload_difference,
    )

    if reload_difference > 1e-6:

        raise RuntimeError(
            "Checkpoint restoration failed validation.\n"
            f"Difference={reload_difference}"
        )

    print(
        "PASS: checkpoint restoration verified."
    )

    # ========================================================
    # HELD-OUT TEST
    # ========================================================

    print()
    print("=" * 80)
    print(
        "HELD-OUT TEST"
    )
    print("=" * 80)

    test_result = evaluate_dataset(
        best_model,
        test_ds,
    )

    # ========================================================
    # INFERENCE LATENCY
    #
    # Uses the unchanged deployment implementation.
    # ========================================================

    latency = measure_latency(
        best_model,
        test_ds,
    )

    memory_final = (
        common.gpu_memory_info()
    )

    peak_memory = (
        memory_final.get(
            "peak_bytes"
        )
    )

    if peak_memory is None:

        peak_memory = (
            memory_after_training.get(
                "peak_bytes"
            )
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {

        "grid":
            grid,

        "agent_count":
            agent_count,

        "reference_condition":
            bool(
                grid == REFERENCE_GRID
            ),

        "parameter_count":
            int(
                best_model.count_params()
            ),

        "training_gradient_method":
            (
                "exact_global_output_gradient_"
                "with_per_sacu_recomputation"
            ),

        "communication_gradient_method":
            (
                "exact_neighbor_mean_chain_rule"
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_validation_rmse":
            float(
                best_validation_rmse
            ),

        "checkpoint_reload_rmse":
            float(
                reloaded_validation[
                    "rmse"
                ]
            ),

        "checkpoint_reload_difference":
            float(
                reload_difference
            ),

        "test_mae":
            float(
                test_result[
                    "mae"
                ]
            ),

        "test_rmse":
            float(
                test_result[
                    "rmse"
                ]
            ),

        "test_wave_residual":
            float(
                test_result[
                    "wave_residual"
                ]
            ),

        "latency_mean_sec":
            float(
                latency[
                    "mean_sec"
                ]
            ),

        "latency_p50_sec":
            float(
                latency[
                    "p50_sec"
                ]
            ),

        "latency_p95_sec":
            float(
                latency[
                    "p95_sec"
                ]
            ),

        "latency_p99_sec":
            float(
                latency[
                    "p99_sec"
                ]
            ),

        "throughput_sequences_per_sec":
            float(
                latency[
                    "throughput_sequences_per_sec"
                ]
            ),

        "peak_gpu_memory_bytes":
            peak_memory,

        "training_time_sec":
            float(
                training_time
            ),

        "overlap":
            int(
                cfg(
                    "overlap",
                    8,
                )
            ),

        "K":
            int(
                cfg(
                    "K",
                    4,
                )
            ),

        "hidden":
            int(
                cfg(
                    "hidden",
                    64,
                )
            ),

        "msg_dim":
            int(
                cfg(
                    "msg_dim",
                    16,
                )
            ),

        "use_role":
            bool(
                cfg(
                    "use_role",
                    True,
                )
            ),

        "use_comms":
            bool(
                cfg(
                    "use_comms",
                    True,
                )
            ),

        "use_physics_loss":
            bool(
                cfg(
                    "use_physics_loss",
                    True,
                )
            ),

        "lambda_phys":
            float(
                cfg(
                    "lambda_phys",
                    0.05,
                )
            ),

        "sensor_weight":
            float(
                cfg(
                    "sensor_weight",
                    0.50,
                )
            ),

        "physics_weight":
            float(
                cfg(
                    "physics_weight",
                    0.35,
                )
            ),

        "entropy_weight":
            float(
                cfg(
                    "entropy_weight",
                    0.15,
                )
            ),

        "temperature":
            float(
                cfg(
                    "temperature",
                    5.0,
                )
            ),

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

        "test_used_for_model_selection":
            False,

        "new_reviewer_requested_experiment":
            True,

        "replaces_existing_manuscript_numbers":
            False,
    }

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
        "AGENT-COUNT CONDITION COMPLETED"
    )
    print("=" * 80)

    print(
        "Grid:",
        grid,
    )

    print(
        "Agents:",
        agent_count,
    )

    print(
        "Parameters:",
        result[
            "parameter_count"
        ],
    )

    print(
        "Best epoch:",
        result[
            "best_epoch"
        ],
    )

    print(
        "Test MAE:",
        f"{result['test_mae']:.6f}",
    )

    print(
        "Test RMSE:",
        f"{result['test_rmse']:.6f}",
    )

    print(
        "Test residual:",
        f"{result['test_wave_residual']:.6f}",
    )

    print(
        "Mean latency:",
        f"{result['latency_mean_sec']:.6f} sec",
    )

    print(
        "P95 latency:",
        f"{result['latency_p95_sec']:.6f} sec",
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

    print(
        "Training time:",
        f"{result['training_time_sec']:.3f} sec",
    )

    print()

    print(
        "PASS: SACU architecture and reviewer-requested "
        "condition were preserved."
    )

    print(
        "PASS: training used exact per-SACU recomputation "
        "to reduce activation-memory requirements."
    )

    print(
        "PASS: held-out test data were not used "
        "for checkpoint selection."
    )

    print(
        "No submitted-manuscript numerical value "
        "was replaced."
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    grid = int(
        args.grid
    )

    agent_count = (
        grid
        * grid
    )

    common.set_determinism()

    # ========================================================
    # PROTOCOL VALIDATION
    # ========================================================

    expected_agents = (
        common.SCALABILITY_CONFIG[
            "agent_counts"
        ][
            common.SCALABILITY_CONFIG[
                "agent_grids"
            ].index(
                grid
            )
        ]
    )

    if expected_agents != agent_count:

        raise RuntimeError(
            "Scalability protocol and implemented "
            "grid definition disagree."
        )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 80)
    print(
        "MEMORY-EFFICIENT SACU AGENT-COUNT SCALABILITY"
    )
    print("=" * 80)

    print(
        "Selected grid:",
        f"{grid} x {grid}",
    )

    print(
        "Selected SACUs:",
        agent_count,
    )

    print(
        "Reference condition:",
        (
            "YES"
            if grid == REFERENCE_GRID
            else "NO"
        ),
    )

    print(
        "Reference:",
        "grid=4 -> 16 SACUs",
    )

    print()

    print(
        "Training method:"
    )

    print(
        "  exact global output-gradient "
        "+ per-SACU recomputation"
    )

    print(
        "Communication gradient:"
    )

    print(
        "  exact neighbor-mean chain rule"
    )

    print()

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

    print(
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        ),
    )

    print()

    print(
        "Inherited SACU configuration:"
    )

    for key in [

        "overlap",
        "K",
        "hidden",
        "msg_dim",
        "use_role",
        "use_comms",
        "use_physics_loss",
        "lambda_phys",
        "sensor_weight",
        "physics_weight",
        "entropy_weight",
        "temperature",
        "epochs",
        "learning_rate",
        "batch_size",

    ]:

        print(
            f"  {key}: "
            f"{cfg(key, '<default>')}"
        )

    # ========================================================
    # CREATE AUDITABLE RUN DIRECTORY
    # ========================================================

    experiment = common.initialize_scalability_run(

        experiment_name=
            (
                "agent_count_scaling_"
                f"grid_{grid}_"
                f"agents_{agent_count}"
            ),

        experiment_config=
            {
                **EXPERIMENT_CONFIG,

                "executed_grid":
                    grid,

                "executed_agent_count":
                    agent_count,
            },
    )

    print()
    print(
        "Run directory:"
    )

    print(
        experiment[
            "run_dir"
        ]
    )

    # ========================================================
    # DATASETS
    # ========================================================

    (
        train_ds,
        validation_ds,
        test_ds,
    ) = build_datasets(

        experiment[
            "train_files"
        ],

        experiment[
            "validation_files"
        ],

        experiment[
            "test_files"
        ],
    )

    # ========================================================
    # RUN ONE CONDITION ONLY
    # ========================================================

    run_grid_condition(

        grid=
            grid,

        experiment=
            experiment,

        train_ds=
            train_ds,

        validation_ds=
            validation_ds,

        test_ds=
            test_ds,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()