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
    / "00_common_operator_protocol.py"
)


# ============================================================
# LOAD COMMON MODERN-BASELINE PROTOCOL
# ============================================================

if not PROTOCOL_PATH.exists():

    raise FileNotFoundError(
        "Modern-baseline protocol not found:\n"
        f"{PROTOCOL_PATH}"
    )


spec = importlib.util.spec_from_file_location(
    "operator_protocol",
    PROTOCOL_PATH,
)

if spec is None or spec.loader is None:

    raise ImportError(
        "Unable to load modern-baseline protocol:\n"
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

MODEL_NAME = "FNO_3D_4x10"

MODEL_CONFIG = {

    "architecture":
        "three_dimensional_fourier_neural_operator",

    "operator_dimensions":
        [
            "time",
            "space_y",
            "space_x",
        ],

    "width":
        10,

    "fourier_layers":
        4,

    "modes_time":
        4,

    "modes_y":
        4,

    "modes_x":
        4,

    "spectral_quadrants":
        4,

    "activation":
        "gelu",

    "projection_hidden_width":
        32,

    "training_objective":
        "full_sequence_supervised_reconstruction_MSE",

    "uses_physics_loss":
        False,

    "uses_wave_residual_for_evaluation":
        True,

    "routing":
        None,

    "domain_decomposition":
        False,

    "expected_parameter_count":
        205655,

    "reference_parameter_count":
        198401,

    "parameter_difference_percent":
        (
            (205655 - 198401)
            / 198401
            * 100.0
        ),

    "important_note":
        (
            "This is a new reviewer-requested Fourier Neural "
            "Operator comparator. The model performs global "
            "spectral convolution jointly over the temporal "
            "and two spatial dimensions. Four low-frequency "
            "mode regions are retained to represent positive "
            "and negative temporal/spatial frequency sectors "
            "while using a real-valued inverse transform. "
            "No physics residual is included in the FNO "
            "training objective; the common second-order wave "
            "residual is used only as an evaluation metric, "
            "consistent with the baseline comparison protocol. "
            "No submitted-manuscript numerical value is replaced."
        ),
}


WIDTH = int(
    MODEL_CONFIG[
        "width"
    ]
)

N_LAYERS = int(
    MODEL_CONFIG[
        "fourier_layers"
    ]
)

MODES_T = int(
    MODEL_CONFIG[
        "modes_time"
    ]
)

MODES_Y = int(
    MODEL_CONFIG[
        "modes_y"
    ]
)

MODES_X = int(
    MODEL_CONFIG[
        "modes_x"
    ]
)

PROJECTION_WIDTH = int(
    MODEL_CONFIG[
        "projection_hidden_width"
    ]
)


# ============================================================
# COMPLEX WEIGHT HELPER
# ============================================================

def make_complex(
    real_part: tf.Tensor,
    imag_part: tf.Tensor,
) -> tf.Tensor:

    return tf.complex(
        tf.cast(
            real_part,
            tf.float32,
        ),
        tf.cast(
            imag_part,
            tf.float32,
        ),
    )


# ============================================================
# 3D SPECTRAL CONVOLUTION
# ============================================================

@keras.utils.register_keras_serializable(
    package="ReviewerModernBaselines"
)
class SpectralConv3D(keras.layers.Layer):

    """
    Genuine 3D Fourier spectral convolution.

    Input:
        (B, T, H, W, Cin)

    Output:
        (B, T, H, W, Cout)

    TensorFlow rfft3d retains the positive half-spectrum
    of the final spatial dimension. Low positive and negative
    frequency sectors are explicitly retained along T and H,
    giving four spectral regions:

        (+T, +H)
        (+T, -H)
        (-T, +H)
        (-T, -H)

    The final W-frequency dimension uses the low positive
    modes supplied by rfft3d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_t: int,
        modes_y: int,
        modes_x: int,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )

        self.in_channels = int(
            in_channels
        )

        self.out_channels = int(
            out_channels
        )

        self.modes_t = int(
            modes_t
        )

        self.modes_y = int(
            modes_y
        )

        self.modes_x = int(
            modes_x
        )


    def build(
        self,
        input_shape,
    ):

        weight_shape = (
            self.in_channels,
            self.out_channels,
            self.modes_t,
            self.modes_y,
            self.modes_x,
        )


        # ----------------------------------------------------
        # Standard FNO-style scale.
        # ----------------------------------------------------

        scale = (
            1.0
            /
            (
                self.in_channels
                * self.out_channels
            )
        )


        self.weight_real = []
        self.weight_imag = []


        for quadrant in range(
            4
        ):

            real_weight = self.add_weight(

                name=
                    f"spectral_real_q{quadrant}",

                shape=
                    weight_shape,

                initializer=
                    keras.initializers.RandomNormal(
                        mean=0.0,
                        stddev=scale,
                    ),

                trainable=True,
            )


            imag_weight = self.add_weight(

                name=
                    f"spectral_imag_q{quadrant}",

                shape=
                    weight_shape,

                initializer=
                    keras.initializers.RandomNormal(
                        mean=0.0,
                        stddev=scale,
                    ),

                trainable=True,
            )


            self.weight_real.append(
                real_weight
            )

            self.weight_imag.append(
                imag_weight
            )


        super().build(
            input_shape
        )


    def multiply_modes(
        self,
        x_modes,
        quadrant: int,
    ):

        weight = make_complex(

            self.weight_real[
                quadrant
            ],

            self.weight_imag[
                quadrant
            ],
        )


        # ----------------------------------------------------
        # x_modes:
        #
        #   B, Cin, Mt, My, Mx
        #
        # weight:
        #
        #   Cin, Cout, Mt, My, Mx
        #
        # output:
        #
        #   B, Cout, Mt, My, Mx
        # ----------------------------------------------------

        return tf.einsum(
            "bctyx,cotyx->botyx",
            x_modes,
            weight,
        )


    def call(
        self,
        x,
    ):

        x = tf.cast(
            x,
            tf.float32,
        )

        # ----------------------------------------------------
        # B,T,H,W,C -> B,C,T,H,W
        # ----------------------------------------------------

        x_channels_first = tf.transpose(
            x,
            perm=[
                0,
                4,
                1,
                2,
                3,
            ],
        )

        input_shape = tf.shape(
            x_channels_first
        )

        n_t = input_shape[2]
        n_y = input_shape[3]
        n_x = input_shape[4]

        # ====================================================
        # DIFFERENTIABLE COMPLEX 3D FOURIER TRANSFORM
        #
        # FFT3D/IFFT3D are used because the NVIDIA TensorFlow
        # 2.17 build used for this experiment does not provide
        # a registered gradient for IRFFT3D.
        #
        # This changes only Fourier-transform implementation;
        # model width, modes, spectral weights, number of
        # blocks, and parameter count remain unchanged.
        # ====================================================

        x_complex = tf.cast(
            x_channels_first,
            tf.complex64,
        )

        x_ft = tf.signal.fft3d(
            x_complex
        )

        tf.debugging.assert_greater_equal(
            n_t,
            2 * self.modes_t,
            message=(
                "Temporal dimension is too small "
                "for configured FNO modes."
            ),
        )

        tf.debugging.assert_greater_equal(
            n_y,
            2 * self.modes_y,
            message=(
                "Spatial Y dimension is too small "
                "for configured FNO modes."
            ),
        )

        tf.debugging.assert_greater_equal(
            n_x,
            self.modes_x,
            message=(
                "Spatial X Fourier dimension is too small "
                "for configured FNO modes."
            ),
        )

        mt = self.modes_t
        my = self.modes_y
        mx = self.modes_x

        # ====================================================
        # SAME FOUR TRAINABLE LOW-FREQUENCY SECTORS
        # ====================================================

        x_pp = x_ft[
            :,
            :,
            :mt,
            :my,
            :mx
        ]

        x_pn = x_ft[
            :,
            :,
            :mt,
            -my:,
            :mx
        ]

        x_np = x_ft[
            :,
            :,
            -mt:,
            :my,
            :mx
        ]

        x_nn = x_ft[
            :,
            :,
            -mt:,
            -my:,
            :mx
        ]

        y_pp = self.multiply_modes(
            x_pp,
            0,
        )

        y_pn = self.multiply_modes(
            x_pn,
            1,
        )

        y_np = self.multiply_modes(
            x_np,
            2,
        )

        y_nn = self.multiply_modes(
            x_nn,
            3,
        )

        # ====================================================
        # PLACE THE FOUR RETAINED REGIONS INTO THE FULL
        # COMPLEX FOURIER GRID.
        #
        # All other coefficients remain zero.
        # ====================================================

        pad_t = n_t - mt
        pad_y = n_y - my
        pad_x = n_x - mx

        out_pp = tf.pad(
            y_pp,
            paddings=[
                [0, 0],
                [0, 0],
                [0, pad_t],
                [0, pad_y],
                [0, pad_x],
            ],
        )

        out_pn = tf.pad(
            y_pn,
            paddings=[
                [0, 0],
                [0, 0],
                [0, pad_t],
                [pad_y, 0],
                [0, pad_x],
            ],
        )

        out_np = tf.pad(
            y_np,
            paddings=[
                [0, 0],
                [0, 0],
                [pad_t, 0],
                [0, pad_y],
                [0, pad_x],
            ],
        )

        out_nn = tf.pad(
            y_nn,
            paddings=[
                [0, 0],
                [0, 0],
                [pad_t, 0],
                [pad_y, 0],
                [0, pad_x],
            ],
        )

        out_ft = (
            out_pp
            + out_pn
            + out_np
            + out_nn
        )

        # ====================================================
        # DIFFERENTIABLE COMPLEX INVERSE FOURIER TRANSFORM
        #
        # The propagation target is real-valued, so the real
        # component is explicitly retained.
        # ====================================================

        y_complex = tf.signal.ifft3d(
            out_ft
        )

        y_channels_first = tf.math.real(
            y_complex
        )

        # ----------------------------------------------------
        # B,C,T,H,W -> B,T,H,W,C
        # ----------------------------------------------------

        y = tf.transpose(
            y_channels_first,
            perm=[
                0,
                2,
                3,
                4,
                1,
            ],
        )

        return y

    def get_config(
        self,
    ):

        config = super().get_config()

        config.update(
            {
                "in_channels":
                    self.in_channels,

                "out_channels":
                    self.out_channels,

                "modes_t":
                    self.modes_t,

                "modes_y":
                    self.modes_y,

                "modes_x":
                    self.modes_x,
            }
        )

        return config


# ============================================================
# FNO BLOCK
# ============================================================

@keras.utils.register_keras_serializable(
    package="ReviewerModernBaselines"
)
class FNOBlock3D(keras.layers.Layer):

    def __init__(
        self,
        width: int,
        modes_t: int,
        modes_y: int,
        modes_x: int,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )

        self.width = int(
            width
        )

        self.modes_t = int(
            modes_t
        )

        self.modes_y = int(
            modes_y
        )

        self.modes_x = int(
            modes_x
        )


        self.spectral = SpectralConv3D(

            in_channels=
                self.width,

            out_channels=
                self.width,

            modes_t=
                self.modes_t,

            modes_y=
                self.modes_y,

            modes_x=
                self.modes_x,

            name=
                "spectral",
        )


        # ----------------------------------------------------
        # Pointwise linear branch Wv.
        #
        # Dense over final channel dimension is equivalent
        # to a 1x1x1 channel projection at every position.
        # ----------------------------------------------------

        self.pointwise = (
            keras.layers.Dense(
                units=self.width,
                activation=None,
                name="pointwise",
            )
        )


    def call(
        self,
        x,
        training: bool = False,
    ):

        spectral_part = (
            self.spectral(
                x
            )
        )


        pointwise_part = (
            self.pointwise(
                x,
                training=training,
            )
        )


        z = (
            spectral_part
            + pointwise_part
        )


        return tf.nn.gelu(
            z
        )


    def get_config(
        self,
    ):

        config = super().get_config()

        config.update(
            {
                "width":
                    self.width,

                "modes_t":
                    self.modes_t,

                "modes_y":
                    self.modes_y,

                "modes_x":
                    self.modes_x,
            }
        )

        return config


# ============================================================
# COMPLETE 3D FOURIER NEURAL OPERATOR
# ============================================================

@keras.utils.register_keras_serializable(
    package="ReviewerModernBaselines"
)
class FourierNeuralOperator3D(keras.Model):

    def __init__(
        self,
        width: int = 10,
        fourier_layers: int = 4,
        modes_t: int = 4,
        modes_y: int = 4,
        modes_x: int = 4,
        projection_width: int = 32,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )


        self.width = int(
            width
        )

        self.fourier_layers_count = int(
            fourier_layers
        )

        self.modes_t = int(
            modes_t
        )

        self.modes_y = int(
            modes_y
        )

        self.modes_x = int(
            modes_x
        )

        self.projection_width = int(
            projection_width
        )


        # ====================================================
        # LIFT INPUT FEATURES
        #
        # 2 -> width
        # ====================================================

        self.lifting = (
            keras.layers.Dense(
                units=self.width,
                activation=None,
                name="lifting",
            )
        )


        # ====================================================
        # FOURIER BLOCKS
        # ====================================================

        self.fourier_blocks = [

            FNOBlock3D(

                width=
                    self.width,

                modes_t=
                    self.modes_t,

                modes_y=
                    self.modes_y,

                modes_x=
                    self.modes_x,

                name=
                    f"fno_block_{i + 1}",
            )

            for i in range(
                self.fourier_layers_count
            )
        ]


        # ====================================================
        # PROJECTION
        #
        # width -> 32 -> 1
        # ====================================================

        self.projection_hidden = (
            keras.layers.Dense(

                units=
                    self.projection_width,

                activation=
                    "gelu",

                name=
                    "projection_hidden",
            )
        )


        self.projection_output = (
            keras.layers.Dense(

                units=1,

                activation=None,

                name=
                    "projection_output",
            )
        )


    def call(
        self,
        x,
        training: bool = False,
    ):

        x = tf.cast(
            x,
            tf.float32,
        )


        z = self.lifting(
            x,
            training=training,
        )


        for block in self.fourier_blocks:

            z = block(
                z,
                training=training,
            )


        z = self.projection_hidden(
            z,
            training=training,
        )


        y_pred = self.projection_output(
            z,
            training=training,
        )


        return y_pred


    def get_config(
        self,
    ):

        config = super().get_config()

        config.update(
            {
                "width":
                    self.width,

                "fourier_layers":
                    self.fourier_layers_count,

                "modes_t":
                    self.modes_t,

                "modes_y":
                    self.modes_y,

                "modes_x":
                    self.modes_x,

                "projection_width":
                    self.projection_width,
            }
        )

        return config


# ============================================================
# TRAINING STEP
# ============================================================

@tf.function(
    reduce_retracing=True
)
def train_step(
    model,
    optimizer,
    x,
    y_true,
):

    with tf.GradientTape() as tape:

        y_pred = model(
            x,
            training=True,
        )


        error = (
            tf.cast(
                y_pred,
                tf.float32,
            )
            -
            tf.cast(
                y_true,
                tf.float32,
            )
        )


        loss = tf.reduce_mean(
            tf.square(
                error
            )
        )


    gradients = tape.gradient(
        loss,
        model.trainable_variables,
    )


    gradient_variable_pairs = [

        (
            gradient,
            variable,
        )

        for (
            gradient,
            variable,
        )
        in zip(
            gradients,
            model.trainable_variables,
        )

        if gradient is not None
    ]


    optimizer.apply_gradients(
        gradient_variable_pairs
    )


    batch_mae = common.mae(
        y_true,
        y_pred,
    )


    batch_rmse = common.rmse(
        y_true,
        y_pred,
    )


    return (
        loss,
        batch_mae,
        batch_rmse,
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
            "The full-field 3D FNO experiment currently "
            "requires batch_size=1 under the unified protocol."
        )


    # ========================================================
    # TensorFlow dataset loader expects string paths.
    #
    # The modern-baseline protocol intentionally stores split
    # entries as pathlib.Path objects for auditability.
    # Convert them here without changing the actual split.
    # ========================================================

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
# EVALUATION
# ============================================================

def evaluate_model(
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

        y_pred = model(
            features[
                "x"
            ],
            training=False,
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
# INFERENCE LATENCY
# ============================================================

def measure_inference_latency(
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
    # Warm-up.
    # --------------------------------------------------------

    for _ in range(
        warmup_runs
    ):

        prediction = model(
            x,
            training=False,
        )


        # Explicit synchronization.
        _ = tf.reduce_sum(
            prediction
        ).numpy()


    # --------------------------------------------------------
    # Timed executions.
    # --------------------------------------------------------

    latencies = []


    for _ in range(
        measurement_runs
    ):

        start = (
            time.perf_counter()
        )


        prediction = model(
            x,
            training=False,
        )


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
# MAIN
# ============================================================

def main():

    # ========================================================
    # INITIALIZE EXPERIMENT
    # ========================================================

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


    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 80)

    print(
        "3D FOURIER NEURAL OPERATOR BASELINE TRAINING"
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
        "FNO width:",
        WIDTH,
    )


    print(
        "Fourier blocks:",
        N_LAYERS,
    )


    print(
        "Spectral modes:",
        f"{MODES_T} x {MODES_Y} x {MODES_X}",
    )


    # ========================================================
    # DATASETS
    # ========================================================

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
    # MODEL
    # ========================================================

    model = FourierNeuralOperator3D(

        width=
            WIDTH,

        fourier_layers=
            N_LAYERS,

        modes_t=
            MODES_T,

        modes_y=
            MODES_Y,

        modes_x=
            MODES_X,

        projection_width=
            PROJECTION_WIDTH,

        name=
            "fno_3d_4x10",
    )


    # --------------------------------------------------------
    # Build on a small synthetic volume.
    #
    # Parameter count does not depend on spatial dimensions.
    # Using a small valid volume avoids allocating the entire
    # dataset field merely to initialize variables.
    # --------------------------------------------------------

    dummy_input = tf.zeros(

        shape=(
            1,
            8,
            16,
            16,
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


    budget_report = (
        common.parameter_budget_report(
            parameter_count
        )
    )


    print()
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


    expected_count = int(
        MODEL_CONFIG[
            "expected_parameter_count"
        ]
    )


    if parameter_count != expected_count:

        raise RuntimeError(
            "Unexpected FNO parameter count.\n"
            f"Expected: {expected_count}\n"
            f"Observed: {parameter_count}"
        )


    common.assert_comparable_parameter_budget(
        model
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = keras.optimizers.Adam(
        learning_rate=
            float(
                common.COMMON_CONFIG[
                    "learning_rate"
                ]
            )
    )


    # ========================================================
    # TRAINING STATE
    # ========================================================

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

            try:

                (
                    loss,
                    batch_mae,
                    batch_rmse,
                ) = train_step(

                    model,
                    optimizer,

                    features[
                        "x"
                    ],

                    y_true,
                )


            except tf.errors.ResourceExhaustedError as exc:

                raise RuntimeError(
                    "\nFNO GPU memory exhaustion occurred.\n"
                    "The architecture has NOT been automatically "
                    "changed because doing so after execution "
                    "would alter the predefined reviewer baseline.\n"
                    "Review the memory profile before changing "
                    "the scientific configuration."
                ) from exc


            train_losses.append(
                float(
                    loss.numpy()
                )
            )


            train_maes.append(
                float(
                    batch_mae.numpy()
                )
            )


            train_rmses.append(
                float(
                    batch_rmse.numpy()
                )
            )


        epoch_time = (
            time.perf_counter()
            - epoch_start
        )


        # ====================================================
        # VALIDATION
        # ====================================================

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


        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

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
            "Best FNO checkpoint was not created:\n"
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
            "SpectralConv3D":
                SpectralConv3D,

            "FNOBlock3D":
                FNOBlock3D,

            "FourierNeuralOperator3D":
                FourierNeuralOperator3D,
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
        evaluate_model(
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
    # RESULT RECORD
    # ========================================================

    extra_results = {

        "operator_type":
            "3D Fourier Neural Operator",

        "global_operator":
            True,

        "spectral_dimensions":
            [
                "time",
                "space_y",
                "space_x",
            ],

        "width":
            WIDTH,

        "fourier_layers":
            N_LAYERS,

        "modes_time":
            MODES_T,

        "modes_y":
            MODES_Y,

        "modes_x":
            MODES_X,

        "parameter_budget":
            budget_report,

        "physics_loss_used_for_training":
            False,

        "common_wave_residual_used_for_evaluation":
            True,

        "test_data_used_for_model_selection":
            False,
    }


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

            extra_results=
                extra_results,
        )
    )


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

            "architecture":
                "3D_FNO",

            "parameter_count":
                common.count_trainable_parameters(
                    best_model
                ),

            "reference_parameter_count":
                budget_report[
                    "reference_parameter_count"
                ],

            "parameter_difference_percent":
                budget_report[
                    "relative_difference_percent"
                ],

            "width":
                WIDTH,

            "fourier_layers":
                N_LAYERS,

            "modes_time":
                MODES_T,

            "modes_y":
                MODES_Y,

            "modes_x":
                MODES_X,

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
        "3D FOURIER NEURAL OPERATOR BASELINE COMPLETED"
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
        "FNO width:",
        WIDTH,
    )


    print(
        "Fourier layers:",
        N_LAYERS,
    )


    print(
        "Spectral modes:",
        f"{MODES_T} x {MODES_Y} x {MODES_X}",
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
        "No submitted-manuscript numerical value "
        "has been replaced."
    )


    print(
        "The FNO architecture was fixed before training "
        "from the comparable parameter-budget requirement."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()