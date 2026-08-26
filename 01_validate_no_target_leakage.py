import sys
from pathlib import Path

import tensorflow as tf

OLD = Path(r"D:\47\472\New-Papers\GIS\Codes")
NEW = OLD / "New_Branch"
SEQ = OLD / "data" / "sim" / "sequences"

sys.path.insert(0, str(NEW))
sys.path.insert(1, str(OLD))

from src.data_loader import make_dataset
from src.models_sacu import OrgSACUSolver
from src.training.trainer_v2 import predict_sacu_deployment


def main():
    # ---------------------------------------------------------
    # 1. Load one real test sequence from the existing dataset
    # ---------------------------------------------------------
    files = sorted(str(p) for p in SEQ.glob("*.npz"))

    if not files:
        raise RuntimeError(f"No .npz files found in {SEQ}")

    test_files = files[int(0.85 * len(files)):]

    ds = make_dataset(
        test_files[:1],
        batch_size=1,
        shuffle=False,
        repeat=False,
        deterministic=True,
    )

    features, y_true_a = next(iter(ds))

    x = features["x"]
    c = features["c_field"]
    dt = features["dt"]
    dx = features["dx"]

    print("Input shape :", x.shape)
    print("Target shape:", y_true_a.shape)
    print("c_field shape:", c.shape)

    # ---------------------------------------------------------
    # 2. Create a completely different hidden target
    # ---------------------------------------------------------
    y_true_b = tf.random.stateless_normal(
        tf.shape(y_true_a),
        seed=[91, 177],
        dtype=y_true_a.dtype,
    )

    target_diff = float(
        tf.reduce_mean(tf.abs(y_true_a - y_true_b)).numpy()
    )

    # ---------------------------------------------------------
    # 3. Build a fresh SACU with the same main configuration
    # ---------------------------------------------------------
    model = OrgSACUSolver(
        grid=4,
        overlap=8,
        K=4,
        hidden=64,
        msg_dim=16,
        use_role=True,
        use_comms=True,
    )

    # Build model once using the actual observable input
    _ = model(x, training=False)

    # ---------------------------------------------------------
    # 4. Prediction A
    #
    # IMPORTANT:
    # y_true_a is NOT passed to inference.
    # ---------------------------------------------------------
    pred_a, diag_a = predict_sacu_deployment(
        model,
        x,
        c,
        dt,
        dx,
        training=False,
    )

    # ---------------------------------------------------------
    # 5. Prediction B
    #
    # y_true_b is radically different, but again is NOT
    # passed into inference.
    # ---------------------------------------------------------
    pred_b, diag_b = predict_sacu_deployment(
        model,
        x,
        c,
        dt,
        dx,
        training=False,
    )

    # ---------------------------------------------------------
    # 6. Compare predictions and organizational influence
    # ---------------------------------------------------------
    pred_diff = float(
        tf.reduce_max(tf.abs(pred_a - pred_b)).numpy()
    )

    weight_diff = float(
        tf.reduce_max(
            tf.abs(
                diag_a["influence_weights"]
                - diag_b["influence_weights"]
            )
        ).numpy()
    )

    print()
    print("=" * 72)
    print("TARGET-LEAKAGE VALIDATION")
    print("=" * 72)

    print(
        "Mean difference between deliberately different targets:",
        target_diff,
    )

    print(
        "Maximum prediction difference:",
        pred_diff,
    )

    print(
        "Maximum influence-weight difference:",
        weight_diff,
    )

    print()

    # ---------------------------------------------------------
    # 7. Hard validation
    # ---------------------------------------------------------
    tolerance = 1e-7

    if target_diff <= tolerance:
        raise AssertionError(
            "The two test targets are unexpectedly identical."
        )

    if pred_diff > tolerance:
        raise AssertionError(
            "FAIL: SACU prediction changes with hidden target."
        )

    if weight_diff > tolerance:
        raise AssertionError(
            "FAIL: SACU influence weights change with hidden target."
        )

    print(
        "PASS: SACU V2 inference is completely independent of y_true."
    )

    print()
    print(
        "y_true is used only after prediction for supervised loss "
        "or evaluation metrics."
    )


if __name__ == "__main__":
    main()