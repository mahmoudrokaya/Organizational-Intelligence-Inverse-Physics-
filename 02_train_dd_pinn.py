from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ============================================================
# SCRIPT / PROTOCOL
# ============================================================

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_DIR = THIS_FILE.parent
PROTOCOL_PATH = EXPERIMENT_DIR / "00_common_operator_protocol.py"

if not PROTOCOL_PATH.exists():
    raise FileNotFoundError(
        f"Modern-baseline protocol not found:\n{PROTOCOL_PATH}"
    )

spec = importlib.util.spec_from_file_location(
    "operator_protocol",
    PROTOCOL_PATH,
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Unable to load protocol:\n{PROTOCOL_PATH}"
    )

common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

NEW_ROOT = common.NEW_ROOT

if str(NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(NEW_ROOT))

from src.data_loader import make_dataset


# ============================================================
# FIXED SCIENTIFIC CONFIGURATION
# ============================================================

MODEL_NAME = "DD_PINN_4Subdomains"

MODEL_CONFIG = {

    "architecture":
        "overlapping_domain_decomposed_physics_informed_neural_network",

    "spatial_partition":
        "2x2",

    "number_of_subdomains":
        4,

    "subdomain_network_hidden_layers":
        2,

    "subdomain_network_width":
        220,

    "activation":
        "relu",

    "output_activation":
        "linear",

    "overlap_fraction_of_half_domain":
        0.10,

    "blending":
        "linear_partition_of_unity",

    "training_objective":
        "full_sequence_MSE_plus_global_wave_residual",

    "lambda_phys":
        0.05,

    "uses_physics_loss":
        True,

    "physics_scope":
        "globally_reconstructed_blended_field",

    "point_chunk_size":
        65536,

    "gradient_method":
        "exact_two_pass_output_gradient_then_local_VJP",

    "optimizer_updates_per_sequence":
        1,

    "expected_parameter_count":
        198004,

    "reference_parameter_count":
        198401,

    "checkpoint_policy":
        "weights_only_rebuild_then_load",

    "important_note":
        (
            "New reviewer-requested DD-PINN experiment. "
            "Four overlapping 2x2 spatial subdomains are used. "
            "Each local estimator has two hidden layers of width 220. "
            "The complete partition-of-unity blended field is used "
            "for the common wave-equation residual. "
            "Checkpointing stores weights only; the identical "
            "architecture is explicitly rebuilt before loading. "
            "No submitted-manuscript number is replaced."
        ),
}


NUMBER_OF_SUBDOMAINS = 4
LOCAL_WIDTH = 220
LOCAL_LAYERS = 2
OVERLAP_FRACTION = 0.10
LAMBDA_PHYS = 0.05
CHUNK_SIZE = 65536
EXPECTED_PARAMETERS = 198004

_PARTITION_CACHE: Dict[
    Tuple[int, int, int],
    tf.Tensor
] = {}


# ============================================================
# LOCAL PINN
# ============================================================

@keras.utils.register_keras_serializable(
    package="ReviewerModernBaselines"
)
class LocalPINN(keras.layers.Layer):

    def __init__(
        self,
        width=220,
        hidden_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.width = int(width)
        self.hidden_layers_count = int(hidden_layers)

        self.hidden_layers_list = [
            keras.layers.Dense(
                self.width,
                activation="relu",
                name=f"hidden_{i + 1}",
            )
            for i in range(self.hidden_layers_count)
        ]

        self.output_layer = keras.layers.Dense(
            1,
            activation=None,
            name="output",
        )


    def call(
        self,
        x,
        training=False,
    ):

        z = tf.cast(x, tf.float32)

        for layer in self.hidden_layers_list:
            z = layer(
                z,
                training=training,
            )

        return self.output_layer(
            z,
            training=training,
        )


    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "width":
                    self.width,

                "hidden_layers":
                    self.hidden_layers_count,
            }
        )

        return config


# ============================================================
# DOMAIN-DECOMPOSED MODEL
# ============================================================

@keras.utils.register_keras_serializable(
    package="ReviewerModernBaselines"
)
class DomainDecomposedPINN(keras.Model):

    def __init__(
        self,
        number_of_subdomains=4,
        local_width=220,
        local_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.number_of_subdomains = int(
            number_of_subdomains
        )

        self.local_width = int(
            local_width
        )

        self.local_layers = int(
            local_layers
        )

        if self.number_of_subdomains != 4:
            raise ValueError(
                "This experiment requires exactly four subdomains."
            )

        self.subdomains = [
            LocalPINN(
                width=self.local_width,
                hidden_layers=self.local_layers,
                name=f"subdomain_{i}",
            )
            for i in range(
                self.number_of_subdomains
            )
        ]


    def call(
        self,
        inputs,
        training=False,
    ):

        x_points, partition_weights = inputs

        x_points = tf.cast(
            x_points,
            tf.float32,
        )

        partition_weights = tf.cast(
            partition_weights,
            tf.float32,
        )

        local_outputs = [
            local_model(
                x_points,
                training=training,
            )
            for local_model
            in self.subdomains
        ]

        stacked = tf.concat(
            local_outputs,
            axis=-1,
        )

        return tf.reduce_sum(
            stacked
            * partition_weights,
            axis=-1,
            keepdims=True,
        )


    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "number_of_subdomains":
                    self.number_of_subdomains,

                "local_width":
                    self.local_width,

                "local_layers":
                    self.local_layers,
            }
        )

        return config


# ============================================================
# RELIABLE MODEL CONSTRUCTION
# ============================================================

def build_dd_pinn_model():

    """
    Construct and explicitly build the COMPLETE composite model.

    This function is used BOTH:
        1. before training;
        2. before loading the best checkpoint.

    Therefore checkpoint restoration never depends on Keras
    reconstructing unbuilt subclassed Dense layers.
    """

    model = DomainDecomposedPINN(
        number_of_subdomains=
            NUMBER_OF_SUBDOMAINS,

        local_width=
            LOCAL_WIDTH,

        local_layers=
            LOCAL_LAYERS,

        name=
            "domain_decomposed_pinn_4subdomains",
    )


    dummy_x = tf.zeros(
        shape=(
            1,
            int(
                common.COMMON_CONFIG[
                    "input_channels"
                ]
            ),
        ),
        dtype=tf.float32,
    )


    # Any valid partition-of-unity vector is sufficient
    # to instantiate every local subnetwork.
    dummy_partition = tf.constant(
        [
            [
                0.25,
                0.25,
                0.25,
                0.25,
            ]
        ],
        dtype=tf.float32,
    )


    # IMPORTANT:
    # Call the TOP-LEVEL model rather than only local layers.
    _ = model(
        (
            dummy_x,
            dummy_partition,
        ),
        training=False,
    )


    if not model.built:
        raise RuntimeError(
            "DomainDecomposedPINN failed to build."
        )


    # Explicit verification that every local output layer
    # actually owns kernel + bias.
    for index, local_model in enumerate(
        model.subdomains
    ):

        if len(
            local_model.output_layer.weights
        ) != 2:

            raise RuntimeError(
                f"Subdomain {index} output layer was not "
                f"properly built. Expected 2 variables, "
                f"observed "
                f"{len(local_model.output_layer.weights)}."
            )


    parameter_count = (
        common.count_trainable_parameters(
            model
        )
    )


    if parameter_count != EXPECTED_PARAMETERS:

        raise RuntimeError(
            "Unexpected DD-PINN parameter count.\n"
            f"Expected: {EXPECTED_PARAMETERS}\n"
            f"Observed: {parameter_count}"
        )


    return model


# ============================================================
# FLATTEN / RESTORE
# ============================================================

def flatten_features(x):

    return tf.reshape(
        tf.cast(
            x,
            tf.float32,
        ),
        [
            -1,
            int(
                common.COMMON_CONFIG[
                    "input_channels"
                ]
            ),
        ],
    )


def restore_prediction_shape(
    flat_prediction,
    reference_x,
):

    output_shape = tf.concat(
        [
            tf.shape(
                reference_x
            )[:-1],

            [
                int(
                    common.COMMON_CONFIG[
                        "output_channels"
                    ]
                )
            ],
        ],
        axis=0,
    )

    return tf.reshape(
        flat_prediction,
        output_shape,
    )


# ============================================================
# PARTITION OF UNITY
# ============================================================

def one_dimensional_blend(
    size: int,
):

    half = size // 2

    overlap = max(
        1,
        int(
            round(
                half
                * OVERLAP_FRACTION
            )
        ),
    )

    left_edge = half - overlap
    right_edge = half + overlap

    coordinates = np.arange(
        size,
        dtype=np.float32,
    )

    left = np.ones(
        size,
        dtype=np.float32,
    )

    left[
        coordinates >= right_edge
    ] = 0.0

    transition = (
        (coordinates > left_edge)
        &
        (coordinates < right_edge)
    )

    left[
        transition
    ] = (
        right_edge
        - coordinates[
            transition
        ]
    ) / float(
        right_edge
        - left_edge
    )

    right = 1.0 - left

    return (
        left,
        right,
        overlap,
    )


def get_partition_weights(x):

    shape = tf.shape(x).numpy()

    batch = int(shape[0])
    t_steps = int(shape[1])
    height = int(shape[2])
    width = int(shape[3])

    if batch != 1:
        raise ValueError(
            "DD-PINN requires batch_size=1."
        )

    key = (
        t_steps,
        height,
        width,
    )

    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]


    x_left, x_right, overlap_x = (
        one_dimensional_blend(
            width
        )
    )

    y_top, y_bottom, overlap_y = (
        one_dimensional_blend(
            height
        )
    )


    spatial = np.stack(
        [
            np.outer(
                y_top,
                x_left,
            ),

            np.outer(
                y_top,
                x_right,
            ),

            np.outer(
                y_bottom,
                x_left,
            ),

            np.outer(
                y_bottom,
                x_right,
            ),
        ],
        axis=-1,
    )


    spatial /= np.maximum(
        np.sum(
            spatial,
            axis=-1,
            keepdims=True,
        ),
        1e-12,
    )


    temporal = np.broadcast_to(
        spatial[
            None,
            ...
        ],
        (
            t_steps,
            height,
            width,
            NUMBER_OF_SUBDOMAINS,
        ),
    )


    flat = np.reshape(
        temporal,
        (
            -1,
            NUMBER_OF_SUBDOMAINS,
        ),
    ).astype(
        np.float32
    )


    result = tf.constant(
        flat,
        dtype=tf.float32,
    )


    _PARTITION_CACHE[
        key
    ] = result


    print(
        "[INFO] Created spatial partition:",
        f"{height}x{width},",
        f"overlap_x={overlap_x},",
        f"overlap_y={overlap_y}",
    )


    return result


# ============================================================
# CHUNKED GLOBAL PREDICTION
# ============================================================

def predict_flat_chunked(
    model,
    x,
):

    flat_x = flatten_features(x)

    weights = get_partition_weights(x)

    total_points = int(
        tf.shape(flat_x)[0].numpy()
    )

    output_chunks = []


    for start in range(
        0,
        total_points,
        CHUNK_SIZE,
    ):

        end = min(
            start + CHUNK_SIZE,
            total_points,
        )

        x_chunk = flat_x[
            start:end
        ]

        w_chunk = weights[
            start:end
        ]

        n_chunk = int(
            tf.shape(
                x_chunk
            )[0].numpy()
        )


        chunk_prediction = tf.zeros(
            (
                n_chunk,
                1,
            ),
            dtype=tf.float32,
        )


        for subdomain_index in range(
            NUMBER_OF_SUBDOMAINS
        ):

            local_weight = (
                w_chunk[
                    :,
                    subdomain_index
                ]
            )


            local_indices = tf.where(
                local_weight > 0.0
            )[:, 0]


            if int(
                tf.size(
                    local_indices
                ).numpy()
            ) == 0:

                continue


            local_x = tf.gather(
                x_chunk,
                local_indices,
            )


            local_prediction = (
                model.subdomains[
                    subdomain_index
                ](
                    local_x,
                    training=False,
                )
            )


            local_weight_values = tf.gather(
                local_weight,
                local_indices,
            )[:, None]


            weighted_prediction = (
                local_prediction
                * local_weight_values
            )


            chunk_prediction = (
                tf.tensor_scatter_nd_add(
                    chunk_prediction,
                    local_indices[:, None],
                    weighted_prediction,
                )
            )


        output_chunks.append(
            chunk_prediction
        )


    return tf.concat(
        output_chunks,
        axis=0,
    )


def predict_field(
    model,
    x,
):

    return restore_prediction_shape(

        predict_flat_chunked(
            model,
            x,
        ),

        x,
    )


# ============================================================
# COMPLETE OBJECTIVE
# ============================================================

def full_loss_and_output_gradient(
    flat_prediction,
    x_reference,
    y_true,
    c_field,
    dt,
    dx,
):

    flat_prediction = tf.cast(
        flat_prediction,
        tf.float32,
    )

    y_true = tf.cast(
        y_true,
        tf.float32,
    )


    with tf.GradientTape() as tape:

        tape.watch(
            flat_prediction
        )


        y_pred = restore_prediction_shape(
            flat_prediction,
            x_reference,
        )


        data_loss = tf.reduce_mean(
            tf.square(
                y_pred
                - y_true
            )
        )


        physics_loss = (
            common.wave_residual_mean(
                y_pred,
                c_field,
                dt,
                dx,
            )
        )


        total_loss = (
            data_loss
            +
            tf.cast(
                LAMBDA_PHYS,
                tf.float32,
            )
            * physics_loss
        )


    output_gradient = tape.gradient(
        total_loss,
        flat_prediction,
    )


    if output_gradient is None:

        raise RuntimeError(
            "Unable to compute dL/dPrediction."
        )


    return (
        total_loss,
        data_loss,
        physics_loss,
        common.mae(
            y_true,
            y_pred,
        ),
        common.rmse(
            y_true,
            y_pred,
        ),
        output_gradient,
    )


# ============================================================
# LOCAL VECTOR-JACOBIAN PRODUCT
# ============================================================

@tf.function(
    reduce_retracing=True
)
def local_parameter_vjp(
    local_model,
    x_local,
    upstream_local,
):

    upstream_local = tf.stop_gradient(
        tf.cast(
            upstream_local,
            tf.float32,
        )
    )


    with tf.GradientTape() as tape:

        prediction = local_model(
            x_local,
            training=True,
        )


        surrogate = tf.reduce_sum(
            prediction
            * upstream_local
        )


    return tape.gradient(
        surrogate,
        local_model.trainable_variables,
    )


# ============================================================
# TRAIN COMPLETE SEQUENCE
# ============================================================

def train_one_sequence(
    model,
    optimizer,
    features,
    y_true,
):

    x = features[
        "x"
    ]


    # --------------------------------------------------------
    # PASS 1
    # --------------------------------------------------------

    flat_prediction = tf.stop_gradient(
        predict_flat_chunked(
            model,
            x,
        )
    )


    (
        total_loss,
        data_loss,
        physics_loss,
        sequence_mae,
        sequence_rmse,
        output_gradient,
    ) = full_loss_and_output_gradient(

        flat_prediction,
        x,
        y_true,

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


    # --------------------------------------------------------
    # PASS 2
    # --------------------------------------------------------

    flat_x = flatten_features(x)
    weights = get_partition_weights(x)

    total_points = int(
        tf.shape(flat_x)[0].numpy()
    )


    accumulated = [

        [
            tf.zeros_like(variable)

            for variable
            in local_model.trainable_variables
        ]

        for local_model
        in model.subdomains
    ]


    for start in range(
        0,
        total_points,
        CHUNK_SIZE,
    ):

        end = min(
            start + CHUNK_SIZE,
            total_points,
        )


        x_chunk = flat_x[
            start:end
        ]

        w_chunk = weights[
            start:end
        ]

        upstream_chunk = output_gradient[
            start:end
        ]


        for subdomain_index in range(
            NUMBER_OF_SUBDOMAINS
        ):

            local_weight = (
                w_chunk[
                    :,
                    subdomain_index
                ]
            )


            local_indices = tf.where(
                local_weight > 0.0
            )[:, 0]


            if int(
                tf.size(
                    local_indices
                ).numpy()
            ) == 0:

                continue


            local_x = tf.gather(
                x_chunk,
                local_indices,
            )


            local_upstream = tf.gather(
                upstream_chunk,
                local_indices,
            )


            local_weight_values = tf.gather(
                local_weight,
                local_indices,
            )[:, None]


            weighted_upstream = (
                local_upstream
                * local_weight_values
            )


            gradients = local_parameter_vjp(

                model.subdomains[
                    subdomain_index
                ],

                local_x,

                weighted_upstream,
            )


            accumulated[
                subdomain_index
            ] = [

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

                    accumulated[
                        subdomain_index
                    ],

                    gradients,

                    model.subdomains[
                        subdomain_index
                    ].trainable_variables,
                )
            ]


    gradient_variable_pairs = []


    for subdomain_index in range(
        NUMBER_OF_SUBDOMAINS
    ):

        gradient_variable_pairs.extend(
            zip(

                accumulated[
                    subdomain_index
                ],

                model.subdomains[
                    subdomain_index
                ].trainable_variables,
            )
        )


    optimizer.apply_gradients(
        gradient_variable_pairs
    )


    return (
        total_loss,
        data_loss,
        physics_loss,
        sequence_mae,
        sequence_rmse,
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
        common.COMMON_CONFIG[
            "batch_size"
        ]
    )

    if batch_size != 1:
        raise ValueError(
            "DD-PINN requires batch_size=1."
        )


    # Path -> TensorFlow-safe string

    train_files = [
        str(p)
        for p in train_files
    ]

    validation_files = [
        str(p)
        for p in validation_files
    ]

    test_files = [
        str(p)
        for p in test_files
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
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataset,
):

    maes = []
    rmses = []
    residuals = []
    rows = []


    for index, (
        features,
        y_true,
    ) in enumerate(dataset):


        y_pred = predict_field(
            model,
            features[
                "x"
            ],
        )


        sample_mae = float(
            common.mae(
                y_true,
                y_pred,
            ).numpy()
        )


        sample_rmse = float(
            common.rmse(
                y_true,
                y_pred,
            ).numpy()
        )


        sample_residual = float(
            common.wave_residual_mean(

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


        maes.append(
            sample_mae
        )

        rmses.append(
            sample_rmse
        )

        residuals.append(
            sample_residual
        )


        rows.append(
            {
                "index":
                    index,

                "mae":
                    sample_mae,

                "rmse":
                    sample_rmse,

                "wave_residual":
                    sample_residual,
            }
        )


    return {

        "count":
            len(maes),

        "MAE_mean":
            float(
                np.mean(maes)
            ),

        "MAE_std":
            float(
                np.std(
                    maes,
                    ddof=1,
                )
            ),

        "RMSE_mean":
            float(
                np.mean(rmses)
            ),

        "RMSE_std":
            float(
                np.std(
                    rmses,
                    ddof=1,
                )
            ),

        "Residual_mean":
            float(
                np.mean(
                    residuals
                )
            ),

        "Residual_std":
            float(
                np.std(
                    residuals,
                    ddof=1,
                )
            ),

        "per_sample":
            rows,
    }


# ============================================================
# LATENCY
# ============================================================

def measure_inference_latency(
    model,
    dataset,
):

    features, _ = next(
        iter(
            dataset.take(1)
        )
    )

    x = features["x"]

    _ = get_partition_weights(x)


    warmup_runs = int(
        common.COMMON_CONFIG[
            "latency_warmup_runs"
        ]
    )

    measurement_runs = int(
        common.COMMON_CONFIG[
            "latency_measurement_runs"
        ]
    )


    for _ in range(warmup_runs):

        prediction = predict_field(
            model,
            x,
        )

        _ = tf.reduce_sum(
            prediction
        ).numpy()


    timings = []


    for _ in range(
        measurement_runs
    ):

        start = time.perf_counter()


        prediction = predict_field(
            model,
            x,
        )


        _ = tf.reduce_sum(
            prediction
        ).numpy()


        timings.append(
            time.perf_counter()
            - start
        )


    return {

        "latency_runs":
            len(timings),

        "latency_mean_sec":
            float(
                np.mean(timings)
            ),

        "latency_std_sec":
            float(
                np.std(
                    timings,
                    ddof=1,
                )
            ),

        "latency_p50_sec":
            float(
                np.percentile(
                    timings,
                    50,
                )
            ),

        "latency_p95_sec":
            float(
                np.percentile(
                    timings,
                    95,
                )
            ),

        "latency_p99_sec":
            float(
                np.percentile(
                    timings,
                    99,
                )
            ),
    }


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def save_best_weights(
    model,
    path: Path,
):

    """
    Store weights only.

    Avoid complete subclass-model serialization because
    nested Dense layers can be reconstructed as unbuilt during
    keras.models.load_model(), producing:

        expected 0 variables, received 2 variables.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    model.save_weights(
        str(path)
    )


    if not path.exists():

        raise FileNotFoundError(
            "Weights checkpoint was not created:\n"
            f"{path}"
        )


def load_best_weights(
    path: Path,
):

    """
    Rebuild the identical scientific architecture,
    instantiate every variable, and ONLY THEN restore weights.
    """

    if not path.exists():

        raise FileNotFoundError(
            "Best weights checkpoint not found:\n"
            f"{path}"
        )


    restored_model = (
        build_dd_pinn_model()
    )


    print(
        "Restored architecture built:",
        restored_model.built,
    )


    print(
        "Restored architecture parameters:",
        common.count_trainable_parameters(
            restored_model
        ),
    )


    restored_model.load_weights(
        str(path)
    )


    # --------------------------------------------------------
    # Verify all restored parameters are finite.
    # --------------------------------------------------------

    for variable in (
        restored_model.trainable_variables
    ):

        if not bool(
            tf.reduce_all(
                tf.math.is_finite(
                    variable
                )
            ).numpy()
        ):

            raise RuntimeError(
                "Non-finite variable detected "
                "after checkpoint restoration."
            )


    return restored_model


# ============================================================
# MAIN
# ============================================================

def main():

    experiment = (
        common.initialize_operator_experiment(
            MODEL_NAME,
            MODEL_CONFIG,
        )
    )


    run_dir = experiment[
        "run_dir"
    ]

    train_files = experiment[
        "train_files"
    ]

    validation_files = experiment[
        "validation_files"
    ]

    test_files = experiment[
        "test_files"
    ]


    print()
    print("=" * 80)
    print(
        "DOMAIN-DECOMPOSED PINN BASELINE TRAINING"
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
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        ),
    )

    print(
        "Spatial partition: 2 x 2"
    )

    print(
        "Subdomains:",
        NUMBER_OF_SUBDOMAINS,
    )

    print(
        "Local architecture:",
        f"{LOCAL_LAYERS} x {LOCAL_WIDTH}",
    )

    print(
        "Overlap fraction:",
        OVERLAP_FRACTION,
    )

    print(
        "Lambda physics:",
        LAMBDA_PHYS,
    )

    print(
        "Point chunk size:",
        CHUNK_SIZE,
    )


    (
        train_ds,
        validation_ds,
        test_ds,
    ) = build_datasets(
        train_files,
        validation_files,
        test_files,
    )


    # ========================================================
    # BUILD COMPLETE MODEL
    # ========================================================

    model = build_dd_pinn_model()


    parameter_count = (
        common.count_trainable_parameters(
            model
        )
    )


    budget_report = (
        common.parameter_budget_report(
            parameter_count
        )
    )


    print()
    print(
        "Top-level model built:",
        model.built,
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    print(
        "Reference parameters:",
        budget_report[
            "reference_parameter_count"
        ],
    )

    print(
        "Parameter difference:",
        f"{budget_report['relative_difference_percent']:+.3f}%",
    )

    print(
        "Within comparable budget:",
        budget_report[
            "within_default_comparable_budget"
        ],
    )


    common.assert_comparable_parameter_budget(
        model
    )


    optimizer = keras.optimizers.Adam(
        learning_rate=float(
            common.COMMON_CONFIG[
                "learning_rate"
            ]
        )
    )


    history = []

    best_validation_rmse = np.inf
    best_epoch = None


    # ========================================================
    # IMPORTANT CHANGE:
    #
    # weights-only checkpoint.
    #
    # Do NOT use:
    #
    #   best_model.keras
    #   model.save(...)
    #   keras.models.load_model(...)
    #
    # ========================================================

    best_weights_path = (
        run_dir
        / "models"
        / "best.weights.h5"
    )


    epochs = int(
        common.COMMON_CONFIG[
            "epochs"
        ]
    )


    total_start = (
        time.perf_counter()
    )


    # ========================================================
    # TRAINING LOOP
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


        total_losses = []
        data_losses = []
        physics_losses = []
        train_maes = []
        train_rmses = []


        epoch_start = (
            time.perf_counter()
        )


        for (
            features,
            y_true,
        ) in train_ds:


            (
                total_loss,
                data_loss,
                physics_loss,
                sequence_mae,
                sequence_rmse,
            ) = train_one_sequence(

                model,
                optimizer,
                features,
                y_true,
            )


            total_losses.append(
                float(
                    total_loss.numpy()
                )
            )

            data_losses.append(
                float(
                    data_loss.numpy()
                )
            )

            physics_losses.append(
                float(
                    physics_loss.numpy()
                )
            )

            train_maes.append(
                float(
                    sequence_mae.numpy()
                )
            )

            train_rmses.append(
                float(
                    sequence_rmse.numpy()
                )
            )


        epoch_time = (
            time.perf_counter()
            - epoch_start
        )


        validation_metrics = (
            evaluate_model(
                model,
                validation_ds,
            )
        )


        validation_metrics.pop(
            "per_sample"
        )


        row = {

            "epoch":
                epoch,

            "train_total_loss":
                float(
                    np.mean(
                        total_losses
                    )
                ),

            "train_data_loss":
                float(
                    np.mean(
                        data_losses
                    )
                ),

            "train_physics_loss":
                float(
                    np.mean(
                        physics_losses
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

            "validation_mae":
                validation_metrics[
                    "MAE_mean"
                ],

            "validation_rmse":
                validation_metrics[
                    "RMSE_mean"
                ],

            "validation_residual":
                validation_metrics[
                    "Residual_mean"
                ],

            "epoch_time_sec":
                epoch_time,
        }


        history.append(
            row
        )


        print(
            f"Total loss     : "
            f"{row['train_total_loss']:.6f}"
        )

        print(
            f"Data loss      : "
            f"{row['train_data_loss']:.6f}"
        )

        print(
            f"Physics loss   : "
            f"{row['train_physics_loss']:.6f}"
        )

        print(
            f"Train MAE      : "
            f"{row['train_mae']:.6f}"
        )

        print(
            f"Train RMSE     : "
            f"{row['train_rmse']:.6f}"
        )

        print(
            f"Validation MAE : "
            f"{row['validation_mae']:.6f}"
        )

        print(
            f"Validation RMSE: "
            f"{row['validation_rmse']:.6f}"
        )

        print(
            f"Val residual   : "
            f"{row['validation_residual']:.6f}"
        )

        print(
            f"Epoch time     : "
            f"{row['epoch_time_sec']:.3f} sec"
        )


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
                "[INFO] Saved new best WEIGHTS checkpoint."
            )


        common.save_csv(
            run_dir
            / "tables"
            / "training_history.csv",
            history,
        )


    training_time = (
        time.perf_counter()
        - total_start
    )


    if best_epoch is None:

        raise RuntimeError(
            "No best validation checkpoint selected."
        )


    # ========================================================
    # REBUILD + LOAD WEIGHTS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOADING BEST VALIDATION CHECKPOINT"
    )
    print("=" * 80)


    print(
        "Checkpoint type: weights only"
    )

    print(
        "Checkpoint path:",
        best_weights_path,
    )


    best_model = (
        load_best_weights(
            best_weights_path
        )
    )


    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation RMSE:",
        best_validation_rmse,
    )


    # ========================================================
    # RESTORATION CONSISTENCY CHECK
    # ========================================================

    restored_validation = (
        evaluate_model(
            best_model,
            validation_ds,
        )
    )


    restored_validation_rmse = (
        restored_validation[
            "RMSE_mean"
        ]
    )


    print(
        "Reloaded validation RMSE:",
        restored_validation_rmse,
    )


    reload_difference = abs(
        restored_validation_rmse
        - best_validation_rmse
    )


    print(
        "Checkpoint reload RMSE difference:",
        reload_difference,
    )


    # Numerical roundoff only.
    if reload_difference > 1e-6:

        raise RuntimeError(
            "Reloaded checkpoint does not reproduce "
            "the selected validation RMSE.\n"
            f"Before save: {best_validation_rmse}\n"
            f"After load : {restored_validation_rmse}\n"
            f"Difference : {reload_difference}"
        )


    print(
        "PASS: checkpoint restoration verified."
    )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    validation_rows = (
        restored_validation.pop(
            "per_sample"
        )
    )


    common.save_csv(
        run_dir
        / "tables"
        / "validation_per_sample.csv",
        validation_rows,
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


    test_result = (
        evaluate_model(
            best_model,
            test_ds,
        )
    )


    test_rows = (
        test_result.pop(
            "per_sample"
        )
    )


    common.save_csv(
        run_dir
        / "tables"
        / "test_per_sample.csv",
        test_rows,
    )


    # ========================================================
    # LATENCY
    # ========================================================

    latency_result = (
        measure_inference_latency(
            best_model,
            test_ds,
        )
    )


    # ========================================================
    # RESULTS
    # ========================================================

    extra_results = {

        "domain_decomposition":
            True,

        "subdomain_layout":
            "2x2",

        "number_of_subdomains":
            NUMBER_OF_SUBDOMAINS,

        "overlap_fraction":
            OVERLAP_FRACTION,

        "partition_of_unity":
            True,

        "local_hidden_layers":
            LOCAL_LAYERS,

        "local_width":
            LOCAL_WIDTH,

        "lambda_phys":
            LAMBDA_PHYS,

        "physics_scope":
            "globally_blended_complete_field",

        "checkpoint_type":
            "weights_only",

        "checkpoint_reload_verified":
            True,

        "checkpoint_reload_rmse_difference":
            reload_difference,

        "test_data_used_for_selection":
            False,
    }


    result_record = (
        common.build_result_record(

            model_name=
                MODEL_NAME,

            model=
                best_model,

            validation_metrics=
                restored_validation,

            test_metrics=
                test_result,

            latency_metrics=
                latency_result,

            training_time_sec=
                training_time,

            best_epoch=
                best_epoch,

            model_specific_config=
                MODEL_CONFIG,

            extra_results=
                extra_results,
        )
    )


    common.save_json(
        run_dir
        / "results.json",
        result_record,
    )


    summary_row = [

        {

            "model":
                MODEL_NAME,

            "parameter_count":
                common.count_trainable_parameters(
                    best_model
                ),

            "reference_parameter_count":
                common.REFERENCE_PARAMETER_COUNT,

            "parameter_difference_percent":
                budget_report[
                    "relative_difference_percent"
                ],

            "subdomains":
                NUMBER_OF_SUBDOMAINS,

            "local_hidden_layers":
                LOCAL_LAYERS,

            "local_width":
                LOCAL_WIDTH,

            "overlap_fraction":
                OVERLAP_FRACTION,

            "lambda_phys":
                LAMBDA_PHYS,

            "best_epoch":
                best_epoch,

            "best_validation_rmse":
                best_validation_rmse,

            "checkpoint_reload_rmse":
                restored_validation_rmse,

            "checkpoint_reload_difference":
                reload_difference,

            "test_MAE_mean":
                test_result[
                    "MAE_mean"
                ],

            "test_MAE_std":
                test_result[
                    "MAE_std"
                ],

            "test_RMSE_mean":
                test_result[
                    "RMSE_mean"
                ],

            "test_RMSE_std":
                test_result[
                    "RMSE_std"
                ],

            "test_residual_mean":
                test_result[
                    "Residual_mean"
                ],

            "test_residual_std":
                test_result[
                    "Residual_std"
                ],

            "latency_mean_sec":
                latency_result[
                    "latency_mean_sec"
                ],

            "latency_p95_sec":
                latency_result[
                    "latency_p95_sec"
                ],

            "training_time_sec":
                training_time,

            "hardware_mode":
                "GPU"
                if tf.config.list_physical_devices(
                    "GPU"
                )
                else "CPU",

            "tensorflow_version":
                tf.__version__,

            "new_reviewer_requested_experiment":
                True,

            "replaces_existing_manuscript_numbers":
                False,
        }
    ]


    common.save_csv(
        run_dir
        / "tables"
        / "summary.csv",
        summary_row,
    )


    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "DOMAIN-DECOMPOSED PINN BASELINE COMPLETED"
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
        "Parameter count:",
        common.count_trainable_parameters(
            best_model
        ),
    )

    print(
        "Reference parameter count:",
        common.REFERENCE_PARAMETER_COUNT,
    )

    print(
        "Parameter difference:",
        f"{budget_report['relative_difference_percent']:+.3f}%",
    )

    print(
        "Subdomains:",
        NUMBER_OF_SUBDOMAINS,
    )

    print(
        "Local architecture:",
        f"{LOCAL_LAYERS} x {LOCAL_WIDTH}",
    )

    print(
        "Overlap fraction:",
        OVERLAP_FRACTION,
    )

    print(
        "Lambda physics:",
        LAMBDA_PHYS,
    )

    print()


    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation RMSE:",
        f"{best_validation_rmse:.8f}",
    )

    print(
        "Reloaded validation RMSE:",
        f"{restored_validation_rmse:.8f}",
    )

    print(
        "Reload difference:",
        f"{reload_difference:.12f}",
    )

    print()


    print(
        "Test MAE:",
        f"{test_result['MAE_mean']:.6f}",
    )

    print(
        "Test RMSE:",
        f"{test_result['RMSE_mean']:.6f}",
    )

    print(
        "Test residual:",
        f"{test_result['Residual_mean']:.6f}",
    )

    print()


    print(
        "Latency mean:",
        f"{latency_result['latency_mean_sec']:.6f} sec",
    )

    print(
        "Latency p95:",
        f"{latency_result['latency_p95_sec']:.6f} sec",
    )

    print(
        "Training time:",
        f"{training_time:.3f} sec",
    )

    print()

    print(
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        ),
    )

    print()

    print(
        "PASS: best checkpoint was rebuilt, "
        "restored, and numerically verified."
    )

    print(
        "These are new reviewer-requested "
        "experimental results only."
    )

    print(
        "No submitted-manuscript numerical "
        "value has been replaced."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()