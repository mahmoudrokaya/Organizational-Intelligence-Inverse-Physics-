import os
import json
import time
from pathlib import Path

def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)

def save_json(path: str, obj: dict):
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def now_str():
    return time.strftime("%Y%m%d-%H%M%S")

def list_npz_files(sequences_dir: str):
    sequences_dir = Path(sequences_dir)
    files = sorted([str(p) for p in sequences_dir.glob("*.npz")])
    if not files:
        raise FileNotFoundError(f"No .npz found in: {sequences_dir}")
    return files
