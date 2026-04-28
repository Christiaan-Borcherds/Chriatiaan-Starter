import numpy as np
import pandas as pd
from collections import Counter


SPLIT_NAMES = {
    0: "Train",
    1: "Validation",
    2: "Test",
    -1: "Unassigned"
}


def print_subject_split_report(train_subjects, val_subjects, test_subjects):
    print("\n" + "=" * 80)
    print("SUBJECT-WISE SPLIT")
    print("=" * 80)

    print(f"Train      ({len(train_subjects):2d} subjects): {sorted(train_subjects)}")
    print(f"Validation ({len(val_subjects):2d} subjects): {sorted(val_subjects)}")
    print(f"Test       ({len(test_subjects):2d} subjects): {sorted(test_subjects)}")

    overlap_train_val = set(train_subjects).intersection(val_subjects)
    overlap_train_test = set(train_subjects).intersection(test_subjects)
    overlap_val_test = set(val_subjects).intersection(test_subjects)

    print("\nLeakage check:")
    print(f"  Train ∩ Validation: {sorted(overlap_train_val)}")
    print(f"  Train ∩ Test      : {sorted(overlap_train_test)}")
    print(f"  Validation ∩ Test : {sorted(overlap_val_test)}")


def print_segment_report(segments):
    print("\n" + "=" * 80)
    print("EXTRACTED ACTIVITY SEGMENTS")
    print("=" * 80)

    print(f"Total segments: {len(segments)}")

    activity_counts = Counter(seg["activity"] for seg in segments)
    user_counts = Counter(seg["user"] for seg in segments)

    print("\nSegments per activity:")
    for activity, count in sorted(activity_counts.items()):
        print(f"  Activity {activity:2d}: {count:5d}")

    print("\nSegments per user:")
    for user, count in sorted(user_counts.items()):
        print(f"  User {user:2d}: {count:5d}")


def print_window_report(train_windows, val_windows, test_windows):
    print("\n" + "=" * 80)
    print("WINDOWED DATASET")
    print("=" * 80)

    split_windows = {
        "Train": train_windows,
        "Validation": val_windows,
        "Test": test_windows,
    }

    for split_name, windows in split_windows.items():
        print(f"\n{split_name}:")
        print(f"  Windows: {len(windows)}")

        if len(windows) == 0:
            continue

        users = sorted(set(w["user"] for w in windows))
        activity_counts = Counter(w["activity"] for w in windows)

        print(f"  Users: {users}")
        print("  Windows per activity:")
        for activity, count in sorted(activity_counts.items()):
            print(f"    Activity {activity:2d}: {count:5d}")


def print_tensor_dataset_report(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 80)
    print("TENSOR DATASETS")
    print("=" * 80)

    datasets = {
        "Train": (X_train, y_train),
        "Validation": (X_val, y_val),
        "Test": (X_test, y_test),
    }

    for name, (X, y) in datasets.items():
        print(f"\n{name}:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")

        if len(y) > 0:
            print(f"  Classes present: {sorted(np.unique(y).astype(int).tolist())}")
            print("  Class counts:")
            for cls, count in sorted(Counter(y.astype(int)).items()):
                print(f"    Class {cls:2d}: {count:5d}")


def print_dataframe_report(df):
    print("\n" + "=" * 80)
    print("DATAFRAME DATASET")
    print("=" * 80)

    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nRows per split:")
    for split_id, count in df["split"].value_counts().sort_index().items():
        print(f"  {SPLIT_NAMES.get(split_id, split_id):<10}: {count}")

    print("\nRows per class:")
    for cls, count in df["class"].value_counts().sort_index().items():
        print(f"  Class {int(cls):2d}: {count}")

    print("\nFirst 5 rows:")
    print(df.head())