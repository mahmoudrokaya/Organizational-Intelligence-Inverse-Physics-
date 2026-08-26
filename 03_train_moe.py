from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ============================================================
# SCRIPT LOCATION
# ============================================================

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_DIR = THIS_FILE.parent

PROTOCOL_PATH = (
    EXPERIMENT_DIR
    / "00_common_protocol.py"
)


# ============================================================
# LOAD COMMON PROTOCOL FIRST
# ============================================================

if not PROTOCOL_PATH.exists():

    raise FileNotFoundError(
        "Common protocol not found:\n"
        f"{PROTOCOL_PATH}"
    )


spec = importlib.util.spec_from_file_location(
    "common_protocol",
    PROTOCOL_PATH,
)

if spec is None or spec.loader is None:

    raise ImportError(
        "Unable to load common protocol:\n"
        f"{PROTOCOL_PATH}"
    )


common = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    common
)


# ============================================================
# PROJECT ROOT
# ============================================================

NEW_ROOT = common.NEW_ROOT

if not NEW_ROOT.exists():

    raise FileNotFoundError(
        "Resolved New_Branch does not exist:\n"
        f"{NEW_ROOT}"
    )


if str(NEW_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(NEW_ROOT),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

try:

    from src.data_loader import make_dataset

except ModuleNotFoundError as exc:

    raise ModuleNotFoundError(
        "Could not import src.data_loader.\n"
        f"Resolved NEW_ROOT: {NEW_ROOT}\n"
        f"Expected: {NEW_ROOT / 'src' / 'data_loader.py'}"
    ) from exc


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "MoE_4Experts"

MODEL_CONFIG = {

    "architecture":
        "centralized_dense_mixture_of_experts",

    "number_of_experts":
        4,

    "expert_hidden_layers":
        2,

    "expert_width":
        220,

    "expert_activation":
        "relu",

    "gate_hidden_layers":
        1,

    "gate_width":
        128,

    "gate_activation":
        "relu",

    "gate_output":
        "softmax",

    "routing":
        "dense_soft_mixture",

    "training_objective":
        "full_sequence_supervised_reconstruction_MSE",

    "uses_physics_loss":
        False,

    "uses_dynamic_roles":
        False,

    "uses_inter_agent_communication":
        False,

    "uses_spatial_domain_decomposition":
        False,

    # --------------------------------------------------------
    # Computational implementation only.
    #
    # 200 x 128 x 128 = 3,276,800 pointwise positions.
    # 65,536 gives exactly 50 chunks per complete sequence.
    # --------------------------------------------------------

    "point_chunk_size":
        65536,

    "gradient_accumulation":
        "exact_full_sequence_gradient",

    "optimizer_updates_per_sequence":
        1,

    "important_note":
        (
            "This is the new reviewer-requested centralized "
            "dense mixture-of-experts baseline. It contains four "
            "experts, each with two hidden layers of width 220, "
            "and a one-hidden-layer width-128 softmax gating "
            "network. All four experts are evaluated at each "
            "point and combined through dense soft routing. "
            "Chunking is used only to avoid simultaneously "
            "materializing expert hidden activations over the "
            "complete spatiotemporal field. Gradients are "
            "accumulated across all chunks and normalized by "
            "the complete sequence target-element count before "
            "one optimizer update. The model definition and "
            "full-sequence MSE objective are unchanged. "
            "These results do not replace any numerical value "
            "already reported in the manuscript."
        ),
}


CHUNK_SIZE = int(
    MODEL_CONFIG[
        "point_chunk_size"
    ]
)

NUMBER_OF_EXPERTS = int(
    MODEL_CONFIG[
        "number_of_experts"
    ]
)


# ============================================================
# EXPERT
# ============================================================

@keras.utils.register_keras_serializable(
    package="ReviewerBaselines"
)
class MoEExpert(keras.layers.Layer):

    def __init__(
        self,
        width: int = 220,
        hidden_layers: int = 2,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )

        self.width = int(
            width
        )

        self.hidden_layers_count = int(
            hidden_layers
        )

        self.hidden_layers_list = [

            keras.layers.Dense(
                units=self.width,
                activation="relu",
                name=f"hidden_{i + 1}",
            )

            for i in range(
                self.hidden_layers_count
            )
        ]

        self.output_layer = (
            keras.layers.Dense(
                units=1,
                activation=None,
                name="output",
            )
        )


    def call(
        self,
        x,
        training: bool = False,
    ):

        z = tf.cast(
            x,
            tf.float32,
        )

        for layer in self.hidden_layers_list:

            z = layer(
                z,
                training=training,
            )

        return self.output_layer(
            z,
            training=training,
        )


    def get_config(
        self,
    ):

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
# MIXTURE OF EXPERTS
# ============================================================

@keras.utils.register_keras_serializable(
    package="ReviewerBaselines"
)
class CentralizedMoE(keras.Model):

    def __init__(
        self,
        number_of_experts: int = 4,
        expert_width: int = 220,
        expert_hidden_layers: int = 2,
        gate_width: int = 128,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )

        self.number_of_experts = int(
            number_of_experts
        )

        self.expert_width = int(
            expert_width
        )

        self.expert_hidden_layers = int(
            expert_hidden_layers
        )

        self.gate_width = int(
            gate_width
        )


        self.experts = [

            MoEExpert(
                width=
                    self.expert_width,

                hidden_layers=
                    self.expert_hidden_layers,

                name=
                    f"expert_{i}",
            )

            for i in range(
                self.number_of_experts
            )
        ]


        self.gate_hidden = (
            keras.layers.Dense(
                units=self.gate_width,
                activation="relu",
                name="gate_hidden",
            )
        )


        self.gate_output = (
            keras.layers.Dense(
                units=self.number_of_experts,
                activation="softmax",
                name="gate_softmax",
            )
        )


    def call(
        self,
        x,
        training: bool = False,
        return_gate: bool = False,
    ):

        x = tf.cast(
            x,
            tf.float32,
        )


        # ----------------------------------------------------
        # Gating network
        # ----------------------------------------------------

        gate_features = self.gate_hidden(
            x,
            training=training,
        )


        gate_weights = self.gate_output(
            gate_features,
            training=training,
        )


        # ----------------------------------------------------
        # All experts are evaluated.
        # ----------------------------------------------------

        expert_outputs = [

            expert(
                x,
                training=training,
            )

            for expert in self.experts
        ]


        # Shape:
        #
        #   (N, K, 1)
        #
        # where K = number of experts.
        expert_stack = tf.stack(
            expert_outputs,
            axis=1,
        )


        # Shape:
        #
        #   (N, K, 1)
        expanded_gate = tf.expand_dims(
            gate_weights,
            axis=-1,
        )


        # Dense soft mixture.
        y_pred = tf.reduce_sum(
            expert_stack
            * expanded_gate,
            axis=1,
        )


        if return_gate:

            return (
                y_pred,
                gate_weights,
            )


        return y_pred


    def get_config(
        self,
    ):

        config = super().get_config()

        config.update(
            {
                "number_of_experts":
                    self.number_of_experts,

                "expert_width":
                    self.expert_width,

                "expert_hidden_layers":
                    self.expert_hidden_layers,

                "gate_width":
                    self.gate_width,
            }
        )

        return config


# ============================================================
# FLATTEN / RESTORE HELPERS
# ============================================================

def flatten_features(
    x: tf.Tensor,
) -> tf.Tensor:

    x = tf.cast(
        x,
        tf.float32,
    )

    return tf.reshape(
        x,
        [
            -1,
            int(
                common.COMMON_CONFIG[
                    "input_channels"
                ]
            ),
        ],
    )


def flatten_targets(
    y_true: tf.Tensor,
) -> tf.Tensor:

    y_true = tf.cast(
        y_true,
        tf.float32,
    )

    return tf.reshape(
        y_true,
        [
            -1,
            int(
                common.COMMON_CONFIG[
                    "output_channels"
                ]
            ),
        ],
    )


def restore_prediction_shape(
    flat_prediction: tf.Tensor,
    reference_x: tf.Tensor,
) -> tf.Tensor:

    output_channels = int(
        common.COMMON_CONFIG[
            "output_channels"
        ]
    )

    output_shape = tf.concat(
        [
            tf.shape(
                reference_x
            )[:-1],

            [
                output_channels
            ],
        ],
        axis=0,
    )

    return tf.reshape(
        flat_prediction,
        output_shape,
    )


# ============================================================
# TRAIN ONE CHUNK
# ============================================================

@tf.function(
    reduce_retracing=True
)
def chunk_gradient_sums(
    model,
    x_chunk,
    y_chunk,
):

    """
    Compute gradients of SUM squared error for one chunk.

    Gradients are accumulated over all chunks and divided
    once by the complete sequence target-element count.

    This gives the same gradient as complete-sequence MSE.
    """

    with tf.GradientTape() as tape:

        (
            y_pred,
            gate_weights,
        ) = model(
            x_chunk,
            training=True,
            return_gate=True,
        )


        error = (
            y_pred
            - y_chunk
        )


        squared_error_sum = tf.reduce_sum(
            tf.square(
                error
            )
        )


    gradients = tape.gradient(
        squared_error_sum,
        model.trainable_variables,
    )


    absolute_error_sum = tf.reduce_sum(
        tf.abs(
            error
        )
    )


    target_element_count = tf.cast(
        tf.size(
            y_chunk
        ),
        tf.float32,
    )


    # --------------------------------------------------------
    # Gate statistics.
    # These are observational only.
    # They do not alter the loss.
    # --------------------------------------------------------

    clipped_gate = tf.clip_by_value(
        gate_weights,
        1e-8,
        1.0,
    )


    gate_entropy_per_position = (
        -tf.reduce_sum(
            clipped_gate
            * tf.math.log(
                clipped_gate
            ),
            axis=-1,
        )
    )


    gate_entropy_sum = tf.reduce_sum(
        gate_entropy_per_position
    )


    max_gate_sum = tf.reduce_sum(
        tf.reduce_max(
            gate_weights,
            axis=-1,
        )
    )


    position_count = tf.cast(
        tf.shape(
            gate_weights
        )[0],
        tf.float32,
    )


    dominant_expert = tf.argmax(
        gate_weights,
        axis=-1,
        output_type=tf.int32,
    )


    dominant_counts = tf.math.bincount(
        dominant_expert,
        minlength=NUMBER_OF_EXPERTS,
        maxlength=NUMBER_OF_EXPERTS,
        dtype=tf.int64,
    )


    return (
        squared_error_sum,
        absolute_error_sum,
        target_element_count,
        gate_entropy_sum,
        max_gate_sum,
        position_count,
        dominant_counts,
        gradients,
    )


# ============================================================
# TRAIN ONE COMPLETE SEQUENCE
# ============================================================

def train_one_sequence(
    model,
    optimizer,
    x,
    y_true,
):

    """
    Train on one complete sequence using chunked exact
    full-sequence gradient accumulation.

    Exactly one optimizer update is applied per sequence.
    """

    flat_x = flatten_features(
        x
    )

    flat_y = flatten_targets(
        y_true
    )


    total_points = int(
        tf.shape(
            flat_x
        )[0].numpy()
    )


    accumulated_gradients = [

        tf.zeros_like(
            variable
        )

        for variable
        in model.trainable_variables
    ]


    total_squared_error = tf.constant(
        0.0,
        dtype=tf.float32,
    )

    total_absolute_error = tf.constant(
        0.0,
        dtype=tf.float32,
    )

    total_target_elements = tf.constant(
        0.0,
        dtype=tf.float32,
    )


    total_gate_entropy = tf.constant(
        0.0,
        dtype=tf.float32,
    )

    total_max_gate = tf.constant(
        0.0,
        dtype=tf.float32,
    )

    total_positions = tf.constant(
        0.0,
        dtype=tf.float32,
    )


    dominant_counts = tf.zeros(
        shape=(
            NUMBER_OF_EXPERTS,
        ),
        dtype=tf.int64,
    )


    for start in range(
        0,
        total_points,
        CHUNK_SIZE,
    ):

        end = min(
            start + CHUNK_SIZE,
            total_points,
        )


        x_chunk = (
            flat_x[
                start:end
            ]
        )

        y_chunk = (
            flat_y[
                start:end
            ]
        )


        (
            squared_error_sum,
            absolute_error_sum,
            target_element_count,
            gate_entropy_sum,
            max_gate_sum,
            position_count,
            chunk_dominant_counts,
            gradients,
        ) = chunk_gradient_sums(

            model,
            x_chunk,
            y_chunk,
        )


        total_squared_error += (
            squared_error_sum
        )

        total_absolute_error += (
            absolute_error_sum
        )

        total_target_elements += (
            target_element_count
        )


        total_gate_entropy += (
            gate_entropy_sum
        )

        total_max_gate += (
            max_gate_sum
        )

        total_positions += (
            position_count
        )


        dominant_counts += (
            chunk_dominant_counts
        )


        accumulated_gradients = [

            accumulated
            +
            (
                gradient
                if gradient is not None
                else tf.zeros_like(
                    variable
                )
            )

            for (
                accumulated,
                gradient,
                variable,
            )
            in zip(
                accumulated_gradients,
                gradients,
                model.trainable_variables,
            )
        ]


    # --------------------------------------------------------
    # Exact complete-sequence MSE gradient.
    # --------------------------------------------------------

    normalized_gradients = [

        gradient_sum
        /
        total_target_elements

        for gradient_sum
        in accumulated_gradients
    ]


    optimizer.apply_gradients(
        zip(
            normalized_gradients,
            model.trainable_variables,
        )
    )


    sequence_mse = (
        total_squared_error
        /
        total_target_elements
    )


    sequence_mae = (
        total_absolute_error
        /
        total_target_elements
    )


    sequence_rmse = tf.sqrt(
        sequence_mse
        + 1e-12
    )


    mean_gate_entropy = (
        total_gate_entropy
        /
        total_positions
    )


    normalized_gate_entropy = (
        mean_gate_entropy
        /
        tf.math.log(
            tf.cast(
                NUMBER_OF_EXPERTS,
                tf.float32,
            )
        )
    )


    mean_max_gate_probability = (
        total_max_gate
        /
        total_positions
    )


    return (
        sequence_mse,
        sequence_mae,
        sequence_rmse,
        mean_gate_entropy,
        normalized_gate_entropy,
        mean_max_gate_probability,
        dominant_counts,
    )


# ============================================================
# CHUNKED INFERENCE
# ============================================================

def predict_flat_chunked(
    model,
    x,
) -> tf.Tensor:

    flat_x = flatten_features(
        x
    )


    total_points = int(
        tf.shape(
            flat_x
        )[0].numpy()
    )


    outputs = []


    for start in range(
        0,
        total_points,
        CHUNK_SIZE,
    ):

        end = min(
            start + CHUNK_SIZE,
            total_points,
        )


        y_chunk = model(
            flat_x[
                start:end
            ],
            training=False,
        )


        outputs.append(
            y_chunk
        )


    return tf.concat(
        outputs,
        axis=0,
    )


def predict_chunked(
    model,
    x,
) -> tf.Tensor:

    flat_prediction = (
        predict_flat_chunked(
            model,
            x,
        )
    )


    return restore_prediction_shape(
        flat_prediction,
        x,
    )


# ============================================================
# SAMPLE PATH
# ============================================================

def sample_path(
    features,
    index,
):

    if (
        isinstance(
            features,
            dict,
        )
        and "path" in features
    ):

        try:

            value = (
                features[
                    "path"
                ]
                .numpy()[0]
            )

            if isinstance(
                value,
                bytes,
            ):

                return value.decode(
                    "utf-8"
                )

            return str(
                value
            )

        except Exception:

            pass


    return (
        f"sample_{index:05d}"
    )


# ============================================================
# CHUNKED MODEL EVALUATION
# ============================================================

def evaluate_model_chunked(
    model,
    dataset,
):

    mae_values = []
    rmse_values = []
    residual_values = []

    per_sample_rows = []

    index = 0


    for (
        features,
        y_true,
    ) in dataset:

        y_pred = predict_chunked(
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


        mae_values.append(
            sample_mae
        )

        rmse_values.append(
            sample_rmse
        )

        residual_values.append(
            sample_residual
        )


        per_sample_rows.append(
            {

                "index":
                    index,

                "path":
                    sample_path(
                        features,
                        index,
                    ),

                "mae":
                    sample_mae,

                "rmse":
                    sample_rmse,

                "wave_residual":
                    sample_residual,
            }
        )


        index += 1


    if not mae_values:

        raise RuntimeError(
            "Evaluation dataset produced no samples."
        )


    return {

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
            (
                float(
                    np.std(
                        mae_values,
                        ddof=1,
                    )
                )
                if len(
                    mae_values
                ) > 1
                else 0.0
            ),

        "RMSE_mean":
            float(
                np.mean(
                    rmse_values
                )
            ),

        "RMSE_std":
            (
                float(
                    np.std(
                        rmse_values,
                        ddof=1,
                    )
                )
                if len(
                    rmse_values
                ) > 1
                else 0.0
            ),

        "Residual_mean":
            float(
                np.mean(
                    residual_values
                )
            ),

        "Residual_std":
            (
                float(
                    np.std(
                        residual_values,
                        ddof=1,
                    )
                )
                if len(
                    residual_values
                ) > 1
                else 0.0
            ),

        "per_sample":
            per_sample_rows,
    }


# ============================================================
# GATE BEHAVIOR ANALYSIS
# ============================================================

def evaluate_gate_behavior_chunked(
    model,
    dataset,
):

    total_entropy = 0.0
    total_max_gate = 0.0
    total_positions = 0


    dominant_counts = np.zeros(
        NUMBER_OF_EXPERTS,
        dtype=np.int64,
    )


    for (
        features,
        _
    ) in dataset:

        flat_x = flatten_features(
            features[
                "x"
            ]
        )


        number_of_points = int(
            tf.shape(
                flat_x
            )[0].numpy()
        )


        for start in range(
            0,
            number_of_points,
            CHUNK_SIZE,
        ):

            end = min(
                start + CHUNK_SIZE,
                number_of_points,
            )


            (
                _,
                gate_weights,
            ) = model(

                flat_x[
                    start:end
                ],

                training=False,

                return_gate=True,
            )


            clipped_gate = tf.clip_by_value(
                gate_weights,
                1e-8,
                1.0,
            )


            entropy = (
                -tf.reduce_sum(
                    clipped_gate
                    * tf.math.log(
                        clipped_gate
                    ),
                    axis=-1,
                )
            )


            max_gate = tf.reduce_max(
                gate_weights,
                axis=-1,
            )


            dominant = (
                tf.argmax(
                    gate_weights,
                    axis=-1,
                )
                .numpy()
                .reshape(-1)
            )


            positions = int(
                gate_weights.shape[0]
            )


            total_entropy += float(
                tf.reduce_sum(
                    entropy
                ).numpy()
            )


            total_max_gate += float(
                tf.reduce_sum(
                    max_gate
                ).numpy()
            )


            total_positions += (
                positions
            )


            dominant_counts += (
                np.bincount(
                    dominant,
                    minlength=
                        NUMBER_OF_EXPERTS,
                )
            )


    if total_positions == 0:

        raise RuntimeError(
            "No gate positions were evaluated."
        )


    mean_entropy = (
        total_entropy
        /
        total_positions
    )


    normalized_entropy = (
        mean_entropy
        /
        np.log(
            NUMBER_OF_EXPERTS
        )
    )


    mean_max_gate = (
        total_max_gate
        /
        total_positions
    )


    result = {

        "mean_gate_entropy":
            float(
                mean_entropy
            ),

        "normalized_gate_entropy":
            float(
                normalized_entropy
            ),

        "mean_max_gate_probability":
            float(
                mean_max_gate
            ),

        "total_gate_positions":
            int(
                total_positions
            ),
    }


    for expert_index in range(
        NUMBER_OF_EXPERTS
    ):

        count = int(
            dominant_counts[
                expert_index
            ]
        )


        result[
            f"expert_{expert_index}_dominant_count"
        ] = count


        result[
            f"expert_{expert_index}_dominant_fraction"
        ] = float(
            count
            /
            total_positions
        )


    return result


# ============================================================
# CHUNKED LATENCY
# ============================================================

def measure_chunked_latency(
    model,
    dataset,
):

    sample = next(
        iter(
            dataset.take(1)
        )
    )


    features, _ = sample


    x = features[
        "x"
    ]


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


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(
        warmup_runs
    ):

        prediction = predict_chunked(
            model,
            x,
        )


        _ = tf.reduce_sum(
            prediction
        ).numpy()


    # --------------------------------------------------------
    # Timed runs
    # --------------------------------------------------------

    latencies = []


    for _ in range(
        measurement_runs
    ):

        start = (
            time.perf_counter()
        )


        prediction = predict_chunked(
            model,
            x,
        )


        # Explicit synchronization.
        _ = tf.reduce_sum(
            prediction
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
            (
                float(
                    np.std(
                        latencies,
                        ddof=1,
                    )
                )
                if len(
                    latencies
                ) > 1
                else 0.0
            ),

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
# BUILD DATASETS
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
            "The chunked exact MoE implementation requires "
            "batch_size=1 so that exactly one optimizer "
            "update is preserved per complete sequence."
        )


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
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Initialize experiment
    # --------------------------------------------------------

    experiment = (
        common.initialize_experiment(
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


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "MIXTURE-OF-EXPERTS BASELINE TRAINING - "
        "CHUNKED EXACT FULL-SEQUENCE MODE"
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
        "Resolved NEW_ROOT:"
    )

    print(
        NEW_ROOT
    )


    print()


    print(
        "Dataset directory:"
    )

    print(
        common.SEQUENCE_DIR
    )


    print()


    print(
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        ),
    )


    print(
        "Point chunk size:",
        CHUNK_SIZE,
    )


    print(
        "Number of experts:",
        NUMBER_OF_EXPERTS,
    )


    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    (
        train_ds,
        validation_ds,
        test_ds,
    ) = build_datasets(
        train_files,
        validation_files,
        test_files,
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CentralizedMoE(

        number_of_experts=
            MODEL_CONFIG[
                "number_of_experts"
            ],

        expert_width=
            MODEL_CONFIG[
                "expert_width"
            ],

        expert_hidden_layers=
            MODEL_CONFIG[
                "expert_hidden_layers"
            ],

        gate_width=
            MODEL_CONFIG[
                "gate_width"
            ],

        name=
            "centralized_moe_4experts",
    )


    # --------------------------------------------------------
    # Build using one point only.
    #
    # Never materialize all expert hidden fields at once.
    # --------------------------------------------------------

    dummy_input = tf.zeros(
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


    _ = model(
        dummy_input,
        training=False,
    )


    parameter_count = (
        common.count_trainable_parameters(
            model
        )
    )


    print()
    print(
        "Trainable parameters:",
        parameter_count,
    )


    print(
        "MLP/PINN reference parameters:",
        198401,
    )


    print(
        "Parameter-count difference:",
        parameter_count
        - 198401,
    )


    # Based on:
    #
    # 4 experts:
    #   each = 49,501 parameters
    #
    # gate:
    #   900 parameters
    #
    # total = 198,904.
    expected_parameter_count = (
        198904
    )


    if parameter_count != expected_parameter_count:

        raise RuntimeError(
            "Unexpected MoE parameter count.\n"
            f"Expected: {expected_parameter_count}\n"
            f"Observed: {parameter_count}"
        )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = keras.optimizers.Adam(
        learning_rate=
            float(
                common.COMMON_CONFIG[
                    "learning_rate"
                ]
            )
    )


    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    history = []

    best_validation_rmse = (
        np.inf
    )

    best_epoch = None


    best_model_path = (
        run_dir
        / "models"
        / "best_model.keras"
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


        train_losses = []
        train_maes = []
        train_rmses = []

        train_gate_entropy = []
        train_gate_entropy_normalized = []
        train_max_gate = []


        epoch_dominant_counts = np.zeros(
            NUMBER_OF_EXPERTS,
            dtype=np.int64,
        )


        epoch_start = (
            time.perf_counter()
        )


        for (
            features,
            y_true,
        ) in train_ds:

            (
                sequence_mse,
                sequence_mae,
                sequence_rmse,
                gate_entropy,
                normalized_gate_entropy,
                max_gate_probability,
                dominant_counts,
            ) = train_one_sequence(

                model,
                optimizer,

                features[
                    "x"
                ],

                y_true,
            )


            train_losses.append(
                float(
                    sequence_mse.numpy()
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


            train_gate_entropy.append(
                float(
                    gate_entropy.numpy()
                )
            )


            train_gate_entropy_normalized.append(
                float(
                    normalized_gate_entropy.numpy()
                )
            )


            train_max_gate.append(
                float(
                    max_gate_probability.numpy()
                )
            )


            epoch_dominant_counts += (
                dominant_counts.numpy()
            )


        epoch_time = (
            time.perf_counter()
            - epoch_start
        )


        # ----------------------------------------------------
        # Validation prediction metrics
        # ----------------------------------------------------

        validation_metrics = (
            evaluate_model_chunked(
                model,
                validation_ds,
            )
        )


        validation_metrics.pop(
            "per_sample"
        )


        # ----------------------------------------------------
        # Validation gate behavior
        # ----------------------------------------------------

        validation_gate = (
            evaluate_gate_behavior_chunked(
                model,
                validation_ds,
            )
        )


        row = {

            "epoch":
                epoch,

            "train_loss":
                float(
                    np.mean(
                        train_losses
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

            "train_gate_entropy":
                float(
                    np.mean(
                        train_gate_entropy
                    )
                ),

            "train_gate_entropy_normalized":
                float(
                    np.mean(
                        train_gate_entropy_normalized
                    )
                ),

            "train_mean_max_gate_probability":
                float(
                    np.mean(
                        train_max_gate
                    )
                ),

            "validation_mae":
                float(
                    validation_metrics[
                        "MAE_mean"
                    ]
                ),

            "validation_rmse":
                float(
                    validation_metrics[
                        "RMSE_mean"
                    ]
                ),

            "validation_residual":
                float(
                    validation_metrics[
                        "Residual_mean"
                    ]
                ),

            "validation_gate_entropy":
                float(
                    validation_gate[
                        "mean_gate_entropy"
                    ]
                ),

            "validation_gate_entropy_normalized":
                float(
                    validation_gate[
                        "normalized_gate_entropy"
                    ]
                ),

            "validation_mean_max_gate_probability":
                float(
                    validation_gate[
                        "mean_max_gate_probability"
                    ]
                ),

            "epoch_time_sec":
                float(
                    epoch_time
                ),

            "point_chunk_size":
                CHUNK_SIZE,
        }


        for expert_index in range(
            NUMBER_OF_EXPERTS
        ):

            row[
                f"train_expert_{expert_index}_dominant_count"
            ] = int(
                epoch_dominant_counts[
                    expert_index
                ]
            )


            row[
                f"validation_expert_{expert_index}_dominant_fraction"
            ] = float(
                validation_gate[
                    f"expert_{expert_index}_dominant_fraction"
                ]
            )


        history.append(
            row
        )


        print(
            f"Train loss     : "
            f"{row['train_loss']:.6f}"
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
            f"Train gate H   : "
            f"{row['train_gate_entropy']:.6f}"
        )


        print(
            f"Train H norm   : "
            f"{row['train_gate_entropy_normalized']:.6f}"
        )


        print(
            f"Train max gate : "
            f"{row['train_mean_max_gate_probability']:.6f}"
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
            f"Val gate H norm: "
            f"{row['validation_gate_entropy_normalized']:.6f}"
        )


        print(
            f"Val max gate   : "
            f"{row['validation_mean_max_gate_probability']:.6f}"
        )


        print(
            f"Epoch time     : "
            f"{row['epoch_time_sec']:.3f} sec"
        )


        # ----------------------------------------------------
        # Best validation checkpoint
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


            model.save(
                best_model_path
            )


            print(
                "[INFO] Saved new best model."
            )


        common.save_csv(

            run_dir
            / "tables"
            / "training_history.csv",

            history,
        )


    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    training_time = (
        time.perf_counter()
        - total_start
    )


    if best_epoch is None:

        raise RuntimeError(
            "No best validation checkpoint was selected."
        )


    if not best_model_path.exists():

        raise FileNotFoundError(
            "Best MoE model checkpoint was not created:\n"
            f"{best_model_path}"
        )


    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    print()
    print("=" * 80)

    print(
        "LOADING BEST VALIDATION CHECKPOINT"
    )

    print("=" * 80)


    best_model = keras.models.load_model(

        best_model_path,

        custom_objects={
            "MoEExpert":
                MoEExpert,

            "CentralizedMoE":
                CentralizedMoE,
        },

        compile=False,
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
    # FINAL VALIDATION
    # ========================================================

    validation_result = (
        evaluate_model_chunked(
            best_model,
            validation_ds,
        )
    )


    validation_rows = (
        validation_result.pop(
            "per_sample"
        )
    )


    common.save_csv(

        run_dir
        / "tables"
        / "validation_per_sample.csv",

        validation_rows,
    )


    validation_gate = (
        evaluate_gate_behavior_chunked(
            best_model,
            validation_ds,
        )
    )


    common.save_json(

        run_dir
        / "validation_gate_behavior.json",

        validation_gate,
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
        evaluate_model_chunked(
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


    test_gate = (
        evaluate_gate_behavior_chunked(
            best_model,
            test_ds,
        )
    )


    common.save_json(

        run_dir
        / "test_gate_behavior.json",

        test_gate,
    )


    # ========================================================
    # LATENCY
    # ========================================================

    latency_result = (
        measure_chunked_latency(
            best_model,
            test_ds,
        )
    )


    # ========================================================
    # RESULT PACKAGE
    # ========================================================

    result_record = (
        common.build_result_record(

            model_name=
                MODEL_NAME,

            model=
                best_model,

            validation_metrics=
                validation_result,

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
        )
    )


    result_record[
        "validation_gate_behavior"
    ] = validation_gate


    result_record[
        "test_gate_behavior"
    ] = test_gate


    result_record[
        "computational_execution"
    ] = {

        "mode":
            "pointwise_chunked_dense_moe",

        "chunk_size":
            CHUNK_SIZE,

        "optimizer_updates_per_sequence":
            1,

        "gradient_accumulation":
            (
                "sum_squared_error_gradients_over_all_chunks_"
                "then_normalize_by_complete_target_element_count"
            ),

        "routing":
            "dense_soft_mixture",

        "all_experts_evaluated":
            True,

        "mathematical_objective_changed":
            False,

        "model_architecture_changed":
            False,

        "gate_regularizer_added":
            False,
    }


    common.save_json(

        run_dir
        / "results.json",

        result_record,
    )


    # ========================================================
    # SUMMARY CSV
    # ========================================================

    summary_row = [

        {

            "model":
                MODEL_NAME,

            "parameter_count":
                common.count_trainable_parameters(
                    best_model
                ),

            "number_of_experts":
                NUMBER_OF_EXPERTS,

            "expert_width":
                MODEL_CONFIG[
                    "expert_width"
                ],

            "expert_hidden_layers":
                MODEL_CONFIG[
                    "expert_hidden_layers"
                ],

            "gate_width":
                MODEL_CONFIG[
                    "gate_width"
                ],

            "point_chunk_size":
                CHUNK_SIZE,

            "best_epoch":
                best_epoch,

            "best_validation_rmse":
                best_validation_rmse,

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

            "test_gate_entropy":
                test_gate[
                    "mean_gate_entropy"
                ],

            "test_gate_entropy_normalized":
                test_gate[
                    "normalized_gate_entropy"
                ],

            "test_mean_max_gate_probability":
                test_gate[
                    "mean_max_gate_probability"
                ],

            "latency_mean_sec":
                latency_result[
                    "latency_mean_sec"
                ],

            "latency_p50_sec":
                latency_result[
                    "latency_p50_sec"
                ],

            "latency_p95_sec":
                latency_result[
                    "latency_p95_sec"
                ],

            "latency_p99_sec":
                latency_result[
                    "latency_p99_sec"
                ],

            "training_time_sec":
                training_time,

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
        "MIXTURE-OF-EXPERTS BASELINE COMPLETED"
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
        "Number of experts:",
        NUMBER_OF_EXPERTS,
    )


    print(
        "Point chunk size:",
        CHUNK_SIZE,
    )


    print(
        "Best epoch:",
        best_epoch,
    )


    print(
        "Training time:",
        f"{training_time:.3f} sec",
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
        "Test normalized gate entropy:",
        f"{test_gate['normalized_gate_entropy']:.6f}",
    )


    print(
        "Test mean max gate probability:",
        f"{test_gate['mean_max_gate_probability']:.6f}",
    )


    for expert_index in range(
        NUMBER_OF_EXPERTS
    ):

        print(
            f"Expert {expert_index} dominant fraction:",
            f"{test_gate[f'expert_{expert_index}_dominant_fraction']:.6f}",
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


    print()


    print(
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        ),
    )


    print()


    print(
        "These are new reviewer-requested "
        "experimental results only."
    )


    print(
        "The original four-expert dense MoE architecture "
        "and complete-sequence MSE objective were preserved."
    )


    print(
        "Chunking changes only computational materialization "
        "and gradient execution, not the defined MoE."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()