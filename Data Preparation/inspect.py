#!/usr/bin/env python3
import argparse
import csv
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_NAMES = {
    "0": "train",
    "1": "val",
    "2": "test",
    "-1": "unassigned",
}


def load_metadata(dataset_path):
    metadata_path = dataset_path.parent / "metadata.json"
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(part, total):
    if total == 0:
        return "0.00%"
    return f"{(part / total) * 100:6.2f}%"


def print_counter(title, counter, total, class_to_windows=None):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'class':>7} {'rows':>12} {'row %':>9}", end="")
    if class_to_windows is not None:
        print(f" {'windows':>12} {'window %':>9}")
    else:
        print()

    total_windows = sum(class_to_windows.values()) if class_to_windows else 0
    for cls in sorted(counter, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        print(f"{str(cls):>7} {counter[cls]:12,d} {pct(counter[cls], total):>9}", end="")
        if class_to_windows is not None:
            windows = class_to_windows.get(cls, 0)
            print(f" {windows:12,d} {pct(windows, total_windows):>9}")
        else:
            print()


def inspect_csv(dataset_path, metadata):
    row_counts = Counter()
    split_row_counts = defaultdict(Counter)
    split_totals = Counter()
    total_rows = 0

    required = {"class", "split"}
    with dataset_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{dataset_path} is missing required columns: {sorted(missing)}")

        for row in reader:
            cls = row["class"]
            split = row["split"]
            row_counts[cls] += 1
            split_row_counts[split][cls] += 1
            split_totals[split] += 1
            total_rows += 1

    window_size = metadata.get("window_size")
    class_windows = None
    total_windows = None
    leftover_rows = None

    if window_size:
        class_windows = Counter()
        leftover_rows = Counter()
        for cls, count in row_counts.items():
            class_windows[cls] = count // window_size
            leftover_rows[cls] = count % window_size
        total_windows = total_rows // window_size

    print(f"Dataset: {dataset_path}")
    print(f"Format : CSV dataframe")
    print(f"Rows   : {total_rows:,}")
    if window_size:
        print(f"Window size from metadata: {window_size}")
        print(f"Inferred windows         : {total_windows:,}")
        if any(leftover_rows.values()):
            print(f"Rows not divisible by window size: {dict(leftover_rows)}")

    print_counter("Class distribution", row_counts, total_rows, class_windows)

    print("\nSplit distribution")
    print("------------------")
    for split in sorted(split_totals, key=lambda x: int(x) if str(x).lstrip("-").isdigit() else str(x)):
        label = SPLIT_NAMES.get(str(split), str(split))
        split_windows = None
        if window_size:
            split_windows = sum(split_row_counts[split].values()) // window_size
        print(f"{label:>10}: {split_totals[split]:12,d} rows", end="")
        if split_windows is not None:
            print(f" | {split_windows:8,d} windows")
        else:
            print()

    for split in sorted(split_row_counts, key=lambda x: int(x) if str(x).lstrip("-").isdigit() else str(x)):
        label = SPLIT_NAMES.get(str(split), str(split))
        split_class_windows = None
        if window_size:
            split_class_windows = Counter({
                cls: count // window_size
                for cls, count in split_row_counts[split].items()
            })
        print_counter(f"Class distribution in {label}", split_row_counts[split], split_totals[split], split_class_windows)


def inspect_pickle(dataset_path, metadata):
    tmp_csv = dataset_path.with_suffix(".csv")
    if tmp_csv.exists():
        print(f"{dataset_path} is a pandas dataframe pickle. Inspecting matching CSV for streaming counts: {tmp_csv}\n")
        inspect_csv(tmp_csv, metadata)
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Reading dataset.pkl requires pandas in the active Python environment. "
            "Run this script on dataset.csv, or install/use the same environment used to build the dataset."
        ) from exc

    df = pd.read_pickle(dataset_path)
    if "class" not in df.columns:
        raise ValueError(f"{dataset_path} does not contain a 'class' column")

    total_rows = len(df)
    row_counts = Counter({
        str(cls): int(count)
        for cls, count in df["class"].value_counts().sort_index().items()
    })
    window_size = metadata.get("window_size")
    class_windows = None
    if window_size:
        class_windows = Counter({cls: count // window_size for cls, count in row_counts.items()})

    print(f"Dataset: {dataset_path}")
    print(f"Format : pandas pickle")
    print(f"Shape  : {df.shape}")
    print_counter("Class distribution", row_counts, total_rows, class_windows)


def inspect_numpy_splits(directory):
    script_dir = str(Path(__file__).resolve().parent)
    original_sys_path = list(sys.path)
    shadowed_inspect = sys.modules.get("inspect")

    try:
        sys.path = [
            path for path in sys.path
            if str(Path(path or ".").resolve()) != script_dir
        ]
        if getattr(shadowed_inspect, "__file__", None) == __file__:
            del sys.modules["inspect"]
        np = importlib.import_module("numpy")
    except ImportError:
        print("\nNumPy split files exist, but numpy is not installed or cannot be imported in this Python environment.")
        return
    finally:
        sys.path = original_sys_path

    split_files = [
        ("train", "X_train.npy", "y_train.npy"),
        ("val", "X_val.npy", "y_val.npy"),
        ("test", "X_test.npy", "y_test.npy"),
    ]

    if not any((directory / x_name).exists() for _, x_name, _ in split_files):
        return

    print("\nSaved tensor splits")
    print("-------------------")
    total = 0
    all_counts = Counter()
    for split, x_name, y_name in split_files:
        x_path = directory / x_name
        y_path = directory / y_name
        if not x_path.exists() or not y_path.exists():
            continue

        X = np.load(x_path, allow_pickle=False, mmap_mode="r")
        y = np.load(y_path, allow_pickle=False)
        counts = Counter(str(int(cls)) for cls in y)
        total += len(y)
        all_counts.update(counts)
        print(f"{split:>5}: X shape {tuple(X.shape)} | y shape {tuple(y.shape)} | windows {len(y):,}")
        for cls in sorted(counts, key=lambda x: int(x)):
            print(f"       class {int(cls):2d}: {counts[cls]:6,d} windows")

    print_counter("Combined tensor class distribution", all_counts, total)


def explain_windowing(metadata):
    print("\nWindowing interpretation")
    print("------------------------")
    input_shape = metadata.get("input_shape")
    window_size = metadata.get("window_size")
    overlap = metadata.get("overlap")

    if input_shape:
        print(f"metadata input_shape={input_shape}: each tensor datapoint is already one window.")
    if window_size:
        print(f"dataset.csv/dataset.pkl are flattened rows from windows: {window_size} timestep rows per window.")
    if overlap is not None:
        print(f"metadata overlap={overlap}")
    print("In this project, raw segments are split by subject after window creation, and the saved X_*.npy files are the already-windowed train/val/test tensors.")


def resolve_dataset_path(path):
    path = Path(path)
    if path.is_dir():
        csv_path = path / "dataset.csv"
        pkl_path = path / "dataset.pkl"
        if csv_path.exists():
            return csv_path
        if pkl_path.exists():
            return pkl_path
        raise FileNotFoundError(f"No dataset.csv or dataset.pkl found in {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Inspect class distribution and windowing for dataset.csv/dataset.pkl."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="Output",
        help="Path to a dataset directory, dataset.csv, or dataset.pkl. Default: Output",
    )
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.path)
    metadata = load_metadata(dataset_path)

    if dataset_path.suffix == ".csv":
        inspect_csv(dataset_path, metadata)
    elif dataset_path.suffix == ".pkl":
        inspect_pickle(dataset_path, metadata)
    else:
        raise ValueError(f"Unsupported dataset file: {dataset_path}")

    inspect_numpy_splits(dataset_path.parent)
    explain_windowing(metadata)


if __name__ == "__main__":
    main()
