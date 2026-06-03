import os
import tensorflow as tf
from tensorflow import keras

from src.utils import ensure_dir, now_str, list_npz_files, save_json
from src.data_loader import make_dataset
from src.models_sacu import OrgSACUSolver
from src.train_framework import Trainer

ROOT = r"D:\47\472\New-Papers\GIS\Codes"
SEQ_DIR = os.path.join(ROOT, "data", "sim", "sequences")
OUT_DIR = ensure_dir(os.path.join(ROOT, "outputs", "runs", f"sacu_{now_str()}"))

def main():
    files = list_npz_files(SEQ_DIR)
    n = len(files)
    train_files = files[: int(0.7*n)]
    val_files   = files[int(0.7*n): int(0.85*n)]
    test_files  = files[int(0.85*n):]

    train_ds = make_dataset(train_files, batch_size=1, shuffle=True, repeat=False)
    val_ds   = make_dataset(val_files, batch_size=1, shuffle=False, repeat=False)

    model = OrgSACUSolver(grid=4, overlap=8, K=4, hidden=64, msg_dim=16, use_role=True, use_comms=True)
    opt = keras.optimizers.Adam(1e-3)

    trainer = Trainer(model, opt, OUT_DIR, use_physics_loss=True, lambda_phys=0.05, beta=5.0)
    trainer.fit(train_ds, val_ds, epochs=5, is_sacu=True)

    model.save(os.path.join(OUT_DIR, "model.keras"))
    save_json(os.path.join(OUT_DIR, "split.json"), {"train": len(train_files), "val": len(val_files), "test": len(test_files)})
    print("Saved:", OUT_DIR)

if __name__ == "__main__":
    main()
