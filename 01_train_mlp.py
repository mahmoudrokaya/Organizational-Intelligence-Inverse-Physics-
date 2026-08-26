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

MODEL_NAME = "MLP_4x256"

MODEL_CONFIG = {

    "architecture":
        "centralized_pointwise_multilayer_perceptron",

    "hidden_layers":
        4,

    "hidden_width":
        256,

    "activation":
        "relu",

    "output_activation":
        "linear",

    "training_objective":
        "full_sequence_supervised_reconstruction_MSE",

    "uses_physics_loss":
        False,

    # --------------------------------------------------------
    # Computational implementation only.
    #
    # The 200x128x128 field contains exactly:
    #
    #   3,276,800 pointwise feature vectors.
    #
    # 65,536 divides this exactly into 50 chunks.
    #
    # This does NOT change the mathematical MLP.
    # --------------------------------------------------------

    "point_chunk_size":
        65536,

    "gradient_accumulation":
        "exact_full_sequence_gradient",

    "optimizer_updates_per_sequence":
        1,

    "important_note":
        (
            "This is the new reviewer-requested 4x256 MLP baseline. "
            "The MLP is shared pointwise across the spatiotemporal "
            "field. Chunking is used only to avoid materializing "
            "the complete width-256 field simultaneously on the GPU. "
            "Gradients are accumulated as sums over all chunks and "
            "normalized by the complete sequence element count before "
            "one optimizer update, preserving the full-sequence MSE "
            "training objective. These results do not replace any "
            "numerical value already reported in the manuscript."
        ),
}


CHUNK_SIZE = int(
    MODEL_CONFIG[
        "point_chunk_size"
    ]
)


# ============================================================
# MODEL
# ============================================================

@keras.utils.register_keras_serializable(
    package="ReviewerBaselines"
)
class CentralizedMLP(keras.Model):

    def __init__(
        self,
        width: int = 256,
        layers: int = 4,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )

        self.width = int(
            width
        )

        self.layers_count = int(
            layers
        )

        self.hidden_layers_list = [

            keras.layers.Dense(
                units=self.width,
                activation="relu",
                name=f"dense_{i + 1}",
            )

            for i in range(
                self.layers_count
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

        # x is normally:
        #
        #   (N_points, 2)
        #
        # during chunked execution.

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

                "layers":
                    self.layers_count,
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
    Return gradient of the SUM of squared errors.

    We deliberately use SUM here.

    Gradients from every chunk are accumulated first and
    divided by the complete number of target elements only
    once at the end of the sequence.

    Therefore:

        accumulated_gradient / N

    equals the gradient of the complete-sequence MSE.
    """

    with tf.GradientTape() as tape:

        prediction = model(
            x_chunk,
            training=True,
        )

        error = (
            prediction
            - y_chunk
        )

        squared_error_sum = (
            tf.reduce_sum(
                tf.square(
                    error
                )
            )
        )

    gradients = tape.gradient(
        squared_error_sum,
        model.trainable_variables,
    )

    absolute_error_sum = (
        tf.reduce_sum(
            tf.abs(
                error
            )
        )
    )

    element_count = tf.cast(
        tf.size(
            y_chunk
        ),
        tf.float32,
    )

    return (
        squared_error_sum,
        absolute_error_sum,
        element_count,
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
    Train on one complete sequence while materializing only
    CHUNK_SIZE pointwise vectors at once.

    Exactly one optimizer update is made for the sequence.
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

    gradient_sums = [

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

    total_element_count = tf.constant(
        0.0,
        dtype=tf.float32,
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
            element_count,
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

        total_element_count += (
            element_count
        )

        gradient_sums = [

            accumulated
            + (
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
                gradient_sums,
                gradients,
                model.trainable_variables,
            )
        ]


    # --------------------------------------------------------
    # Convert accumulated SUM gradients to the gradient of
    # complete-sequence MSE.
    # --------------------------------------------------------

    normalized_gradients = [

        gradient_sum
        /
        total_element_count

        for gradient_sum
        in gradient_sums
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
        total_element_count
    )

    sequence_mae = (
        total_absolute_error
        /
        total_element_count
    )

    sequence_rmse = tf.sqrt(
        sequence_mse
        + 1e-12
    )


    return (
        sequence_mse,
        sequence_mae,
        sequence_rmse,
    )


# ============================================================
# CHUNKED INFERENCE
# ============================================================

def predict_chunked(
    model,
    x,
):

    """
    Equivalent to model(x) pointwise, but does not materialize
    width-256 activations over the complete field.
    """

    flat_x = flatten_features(
        x
    )

    total_points = int(
        tf.shape(
            flat_x
        )[0].numpy()
    )

    predictions = []

    for start in range(
        0,
        total_points,
        CHUNK_SIZE,
    ):

        end = min(
            start + CHUNK_SIZE,
            total_points,
        )

        pred_chunk = model(
            flat_x[
                start:end
            ],
            training=False,
        )

        predictions.append(
            pred_chunk
        )

    flat_prediction = tf.concat(
        predictions,
        axis=0,
    )

    return restore_prediction_shape(
        flat_prediction,
        x,
    )


# ============================================================
# SAMPLE PATH HELPER
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
# CHUNKED EVALUATION
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

        # Explicit synchronization.
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

        # Force completion of GPU operations before stopping.
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
            "The chunked full-sequence baseline currently "
            "requires batch_size=1 to preserve one optimizer "
            "update per sequence."
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
        "MLP BASELINE TRAINING - CHUNKED EXACT FULL-SEQUENCE MODE"
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

    model = CentralizedMLP(

        width=
            MODEL_CONFIG[
                "hidden_width"
            ],

        layers=
            MODEL_CONFIG[
                "hidden_layers"
            ],

        name=
            "centralized_mlp_4x256",
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Build using ONE pointwise feature vector, not the full
    # 200x128x128 field.
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


    if parameter_count != 198401:

        raise RuntimeError(
            "Unexpected MLP parameter count.\n"
            f"Expected: 198401\n"
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


        epoch_time = (
            time.perf_counter()
            - epoch_start
        )


        # ----------------------------------------------------
        # Validation
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

            "epoch_time_sec":
                float(
                    epoch_time
                ),

            "point_chunk_size":
                CHUNK_SIZE,
        }


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


        # ----------------------------------------------------
        # Best checkpoint
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
            "CentralizedMLP":
                CentralizedMLP,
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
        "computational_execution"
    ] = {

        "mode":
            "pointwise_chunked",

        "chunk_size":
            CHUNK_SIZE,

        "optimizer_updates_per_sequence":
            1,

        "gradient_accumulation":
            "sum_then_normalize_by_complete_target_element_count",

        "mathematical_objective_changed":
            False,

        "model_architecture_changed":
            False,
    }


    common.save_json(
        run_dir
        / "results.json",
        result_record,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_row = [

        {

            "model":
                MODEL_NAME,

            "parameter_count":
                common.count_trainable_parameters(
                    best_model
                ),

            "best_epoch":
                best_epoch,

            "best_validation_rmse":
                best_validation_rmse,

            "point_chunk_size":
                CHUNK_SIZE,

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
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "MLP BASELINE COMPLETED"
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
        "The 4x256 architecture and complete-sequence "
        "MSE objective were preserved."
    )

    print(
        "Chunking changes only computational "
        "materialization, not the defined model."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()