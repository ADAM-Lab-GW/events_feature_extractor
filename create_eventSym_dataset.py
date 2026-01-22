"""
Convert nested CSV event dataset into a PyTorch-friendly folder structure.

Input:
  <input_root>/<class_name>/<sub_class_name>/<sample>.csv
CSV columns: t, x, y, p
  - t in microseconds
  - p in {0,1}

Output:
  ./data/EventSym/
    training/<class_name>/<sub_class_name>/<sample>.npy
    testing/<class_name>/<sub_class_name>/<sample>.npy

Split:
  - 10 samples per (class/subclass) go to testing (if available)
  - remaining go to training
"""

import os
import random
from pathlib import Path

import numpy as np


# ---------------------------
# CONFIG
# ---------------------------
INPUT_ROOT = "./data/eventsym"   # <-- change this
OUTPUT_ROOT = "./data/eventSym"
N_TEST_SAMPLES_PER_SUBCLASS = 10
RANDOM_SEED = 8932857495889437

TRAIN_FOLDER_LL = "training"
TEST_FOLDER_LL = "testing"


def write_file(data: np.ndarray, filename_stem: str, output_folder: Path) -> None:
    """
    Save as .npy under output_folder with filename_stem (no extension).
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / filename_stem
    np.save(out_path, data)


def load_events_csv(csv_path: Path) -> np.ndarray:
    """
    Load CSV with columns: t,x,y,p
    Returns Nx4 float array: [x, y, t_seconds, p_signed]
    """
    # Robust loading: handles headers and arbitrary column order if names match.
    # If your CSV has extra columns, it will ignore them.
    raw = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)

    # Ensure required columns exist
    required = {"t", "x", "y", "p"}
    if not required.issubset(raw.dtype.names):
        raise ValueError(
            f"{csv_path} is missing required columns. "
            f"Found: {raw.dtype.names}, required: {sorted(required)}"
        )

    t = raw["t"].astype(np.float64) 
    x = raw["x"].astype(np.float32)
    y = raw["y"].astype(np.float32)
    p = raw["p"].astype(np.int16)
    p = np.where(p == 0, -1, 1).astype(np.int8)  # 0->-1, 1->1

    data = np.stack([x, y, t.astype(np.float32), p.astype(np.float32)], axis=1)
    return data


def convert_dataset(input_root: Path, output_root: Path) -> None:
    random.seed(RANDOM_SEED)

    input_root = input_root.resolve()
    output_root = output_root.resolve()

    output_root.mkdir(parents=True, exist_ok=True)


    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    # Expect 2-level hierarchy: class/subclass/*.csv
    class_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])

    if not class_dirs:
        raise ValueError(f"No class folders found under: {input_root}")

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Found {len(class_dirs)} class folders.")

    for class_dir in class_dirs:
        class_name = class_dir.name

        subclass_dirs = sorted([p for p in class_dir.iterdir() if p.is_dir()])
        if not subclass_dirs:
            print(f"[WARN] No subclass folders under class '{class_name}', skipping.")
            continue

        for subclass_dir in subclass_dirs:
            subclass_name = subclass_dir.name

            # Collect CSV files directly under subclass folder
            csv_files = sorted([p for p in subclass_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
            if not csv_files:
                print(f"[WARN] No CSV files in {class_name}/{subclass_name}, skipping.")
                continue

            # Choose test samples
            n_test = min(N_TEST_SAMPLES_PER_SUBCLASS, len(csv_files))
            # Use a deterministic per-subclass seed so ordering elsewhere doesn't change splits
            local_seed = hash((RANDOM_SEED, class_name, subclass_name)) & 0xFFFFFFFF
            rng = random.Random(local_seed)
            test_files = set(rng.sample(csv_files, n_test))
            train_files = [p for p in csv_files if p not in test_files]

            print(f"Processing {class_name}/{subclass_name}: total={len(csv_files)}, test={len(test_files)}, train={len(train_files)}")

            # Write test
            for csv_path in sorted(test_files):
                data = load_events_csv(csv_path)
                sample_stem = csv_path.stem  # e.g., "1_events"
                out_folder = output_root / TEST_FOLDER_LL / class_name / subclass_name
                write_file(data, sample_stem, out_folder)

            # Write train
            for csv_path in train_files:
                data = load_events_csv(csv_path)
                sample_stem = csv_path.stem
                out_folder = output_root / TRAIN_FOLDER_LL / class_name / subclass_name
                write_file(data, sample_stem, out_folder)


if __name__ == "__main__":
    convert_dataset(Path(INPUT_ROOT), Path(OUTPUT_ROOT))
