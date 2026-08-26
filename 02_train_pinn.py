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

MODEL_NAME = "PINN_4x256"

MODEL_CONFIG = {

    "architecture":
        "centralized_pointwise_physics_informed_neural_network",

    "hidden_layers":
        4,

    "hidden_width":
        256,

    "activation":
        "relu",

    "output_activation":
        "linear",

    "training_objective":
        "full_sequence_MSE_plus_full_field_wave_residual",

    "uses_physics_loss":
        True,

    "lambda_phys":
        0.05,

    # --------------------------------------------------------
    # Computational materialization only.
    #
    # Each complete sequence contains:
    #
    # 200 x 128 x 128 = 3,276,800 positions.
    #
    # 65,536 divides this exactly into 50 chunks.
    # --------------------------------------------------------

    "point_chunk_size":
        65536,

    "gradient_method":
        "exact_two_pass_full_field_chain_rule",

    "optimizer_updates_per_sequence":
        1,

    "important_note":
        (
            "This is the new reviewer-requested 4x256 PINN "
            "baseline. The network is shared pointwise over "
            "the spatiotemporal field, while the wave-equation "
            "physics term is evaluated over the reconstructed "
            "complete prediction field. A two-pass chain-rule "
            "procedure is used only to avoid simultaneous "
            "materialization of all width-256 hidden activations. "
            "The complete-sequence MSE, complete-field physics "
            "residual, lambda_phys=0.05, model architecture, "
            "and one optimizer update per sequence are preserved. "
            "These results do not replace any numerical value "
            "already reported in the manuscript."
        ),
}


CHUNK_SIZE = int(
    MODEL_CONFIG[
        "point_chunk_size"
    ]
)

LAMBDA_PHYS = float(
    MODEL_CONFIG[
        "lambda_phys"
    ]
)


# ============================================================
# MODEL
# ============================================================

@keras.utils.register_keras_serializable(
    package="ReviewerBaselines"
)
class CentralizedPINN(keras.Model):

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
# CHUNKED FORWARD INFERENCE
# ============================================================

def predict_flat_chunked(
    model,
    x,
) -> tf.Tensor:

    """
    Execute the pointwise 4x256 PINN in manageable chunks.

    Returns:
        flat predictions of shape (N_points, 1)
    """

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

        output = model(
            flat_x[
                start:end
            ],
            training=False,
        )

        outputs.append(
            output
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
# FULL-FIELD LOSS AND OUTPUT GRADIENT
# ============================================================

def compute_full_field_loss_and_output_gradient(
    flat_prediction,
    x_reference,
    y_true,
    c_field,
    dt,
    dx,
):

    """
    Compute the exact objective on the complete predicted field:

        total_loss
            = full-sequence MSE
              + lambda_phys * full-field wave residual

    Then obtain:

        d(total_loss) / d(flat_prediction)

    No model hidden activations are retained in this pass.
    """

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
            "Could not compute the full-field "
            "loss gradient with respect to predictions."
        )


    full_mae = common.mae(
        y_true,
        y_pred,
    )

    full_rmse = common.rmse(
        y_true,
        y_pred,
    )


    return (
        total_loss,
        data_loss,
        physics_loss,
        full_mae,
        full_rmse,
        output_gradient,
    )


# ============================================================
# PARAMETER GRADIENT FOR ONE CHUNK
# ============================================================

@tf.function(
    reduce_retracing=True
)
def parameter_gradient_from_upstream(
    model,
    x_chunk,
    upstream_gradient,
):

    """
    Compute the vector-Jacobian product:

        J_model^T * upstream_gradient

    for one pointwise chunk.

    Summing these VJPs over all chunks gives the exact
    parameter gradient of the previously computed complete
    data + physics objective.
    """

    upstream_gradient = tf.stop_gradient(
        tf.cast(
            upstream_gradient,
            tf.float32,
        )
    )

    with tf.GradientTape() as tape:

        prediction = model(
            x_chunk,
            training=True,
        )

        surrogate = tf.reduce_sum(
            prediction
            * upstream_gradient
        )

    gradients = tape.gradient(
        surrogate,
        model.trainable_variables,
    )

    return gradients


# ============================================================
# TRAIN ONE COMPLETE SEQUENCE
# ============================================================

def train_one_sequence(
    model,
    optimizer,
    features,
    y_true,
):

    """
    Exact two-pass training procedure.

    Pass 1
    ------
    - Generate complete prediction pointwise in chunks.
    - Reconstruct complete field.
    - Compute complete MSE + complete wave residual.
    - Obtain exact dL/dPrediction.

    Pass 2
    ------
    - Re-run each pointwise chunk.
    - Backpropagate the appropriate slice of dL/dPrediction.
    - Accumulate parameter gradients.
    - Apply exactly one Adam update for the sequence.
    """

    x = features[
        "x"
    ]

    # --------------------------------------------------------
    # PASS 1:
    # full prediction without retaining network activations
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
    ) = compute_full_field_loss_and_output_gradient(

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
    # PASS 2:
    # exact parameter VJP in chunks
    # --------------------------------------------------------

    flat_x = flatten_features(
        x
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


    for start in range(
        0,
        total_points,
        CHUNK_SIZE,
    ):

        end = min(
            start + CHUNK_SIZE,
            total_points,
        )

        chunk_gradients = (
            parameter_gradient_from_upstream(

                model,

                flat_x[
                    start:end
                ],

                output_gradient[
                    start:end
                ],
            )
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
                chunk_gradients,
                model.trainable_variables,
            )
        ]


    optimizer.apply_gradients(
        zip(
            accumulated_gradients,
            model.trainable_variables,
        )
    )


    return (
        total_loss,
        data_loss,
        physics_loss,
        sequence_mae,
        sequence_rmse,
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
    # Timed inference
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

        # Force GPU synchronization.
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
            "The exact chunked PINN implementation requires "
            "batch_size=1 so that one optimizer update is "
            "preserved per complete sequence."
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
    # Experiment initialization
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
        "PINN BASELINE TRAINING - "
        "CHUNKED EXACT FULL-FIELD PHYSICS MODE"
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
        "Lambda physics:",
        LAMBDA_PHYS,
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

    model = CentralizedPINN(

        width=
            MODEL_CONFIG[
                "hidden_width"
            ],

        layers=
            MODEL_CONFIG[
                "hidden_layers"
            ],

        name=
            "centralized_pinn_4x256",
    )


    # --------------------------------------------------------
    # Build model with one point only.
    #
    # Do NOT send the complete field through Dense(256).
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
            "Unexpected PINN parameter count.\n"
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

            "lambda_phys":
                LAMBDA_PHYS,
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


    if not best_model_path.exists():

        raise FileNotFoundError(
            "Best PINN model checkpoint was not created:\n"
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
            "CentralizedPINN":
                CentralizedPINN,
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
            "two_pass_pointwise_chunked_full_field_physics",

        "chunk_size":
            CHUNK_SIZE,

        "optimizer_updates_per_sequence":
            1,

        "lambda_phys":
            LAMBDA_PHYS,

        "physics_loss_scope":
            "complete_reconstructed_spatiotemporal_field",

        "gradient_procedure":
            (
                "full loss gradient with respect to complete "
                "prediction followed by chunked model VJP "
                "accumulation"
            ),

        "mathematical_objective_changed":
            False,

        "model_architecture_changed":
            False,

        "physics_operator_changed":
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

            "lambda_phys":
                LAMBDA_PHYS,

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
        "PINN BASELINE COMPLETED"
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
        "Lambda physics:",
        LAMBDA_PHYS,
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
        "The original 4x256 architecture, lambda_phys=0.05, "
        "complete-sequence MSE, and complete-field wave "
        "residual objective were preserved."
    )


    print(
        "Chunking changes only computational materialization "
        "and gradient execution, not the defined PINN."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()