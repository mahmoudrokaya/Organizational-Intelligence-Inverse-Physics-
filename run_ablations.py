import os
from tensorflow import keras

from src.utils import ensure_dir, now_str, list_npz_files, save_json
from src.data_loader import make_dataset
from src.models_sacu import OrgSACUSolver
from src.train_framework import Trainer

ROOT = r"D:\47\472\New-Papers\GIS\Codes"
SEQ_DIR = os.path.join(ROOT, "data", "sim", "sequences")
ABL_DIR = ensure_dir(os.path.join(ROOT, "outputs", "ablations", f"abl_{now_str()}"))


def _force_build_once(model, ds):
    # Run one forward pass so all variables exist before saving
    for batch in ds.take(1):
        x = batch[0]["x"]
        _ = model(x, training=False)
    return model


def run_one(name, use_role, use_comms, use_physics_loss):
    out = ensure_dir(os.path.join(ABL_DIR, name))

    files = list_npz_files(SEQ_DIR)
    n = len(files)
    train_files = files[: int(0.7 * n)]
    val_files = files[int(0.7 * n): int(0.85 * n)]

    train_ds = make_dataset(train_files, batch_size=1, shuffle=True, repeat=False)
    val_ds = make_dataset(val_files, batch_size=1, shuffle=False, repeat=False)

    model = OrgSACUSolver(
        grid=4, overlap=8, K=4, hidden=64, msg_dim=16,
        use_role=use_role, use_comms=use_comms
    )
    opt = keras.optimizers.Adam(1e-3)

    trainer = Trainer(model, opt, out, use_physics_loss=use_physics_loss, lambda_phys=0.05, beta=5.0)
    trainer.fit(train_ds, val_ds, epochs=3, is_sacu=True)

    _force_build_once(model, val_ds)
    model.save(os.path.join(out, "model.keras"))

    save_json(os.path.join(out, "config.json"), {
        "use_role": use_role,
        "use_comms": use_comms,
        "use_physics_loss": use_physics_loss
    })
    return out


def main():
    runs = [
        ("full", True, True, True),
        ("no_comms", True, False, True),
        ("no_roles", False, True, True),
        ("no_physics", True, True, False),
    ]

    out_paths = {}
    for name, r, c, p in runs:
        out_paths[name] = run_one(name, r, c, p)

    save_json(os.path.join(ABL_DIR, "index.json"), out_paths)
    print("All ablations saved in:", ABL_DIR)


if __name__ == "__main__":
    main()