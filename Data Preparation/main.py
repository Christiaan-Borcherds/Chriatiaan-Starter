import numpy as np
import pandas as pd
import os
from config import *

from processing import extract_segments, create_windows_from_segment, fit_scaler, apply_scaler, get_split, save_train_val_test_numpy, save_dataset_scalar_metadata

from dataset_builder import build_dataframe_dataset, build_tensor_dataset

from dataset_report import (print_subject_split_report, print_segment_report, print_window_report, print_tensor_dataset_report, print_dataframe_report,)


# 1. Load labels
labels = np.loadtxt(os.path.join(DatasetDir, LABELS_FILE))

print_subject_split_report(
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    TEST_SUBJECTS,
)

# 2. Extract segments
segments = extract_segments(DatasetDir, labels)
print_segment_report(segments)

# 3. Create windows
all_windows = []
for seg in segments:
    all_windows.extend(create_windows_from_segment(seg, WINDOW_SIZE, OVERLAP))

# 4. Split windows
train_windows = [w for w in all_windows if get_split(w["user"]) == 0]
val_windows   = [w for w in all_windows if get_split(w["user"]) == 1]
test_windows  = [w for w in all_windows if get_split(w["user"]) == 2]

print_window_report(train_windows, val_windows, test_windows)

# 5. Build tensor datasets
X_train, y_train, _ = build_tensor_dataset(train_windows)
X_val, y_val, _     = build_tensor_dataset(val_windows)
X_test, y_test, _   = build_tensor_dataset(test_windows)

print_tensor_dataset_report(X_train, y_train, X_val, y_val, X_test, y_test)

# 6. Scaling (train only)
scaler = fit_scaler(X_train)

X_train = apply_scaler(X_train, scaler)
X_val   = apply_scaler(X_val, scaler)
X_test  = apply_scaler(X_test, scaler)

print("\nScaling complete:")
print(f"  X_train min/max: {X_train.min():.4f} / {X_train.max():.4f}")
print(f"  X_val min/max  : {X_val.min():.4f} / {X_val.max():.4f}")
print(f"  X_test min/max : {X_test.min():.4f} / {X_test.max():.4f}")

# 7. Build dataframe
df = build_dataframe_dataset(all_windows, get_split)

print_dataframe_report(df)


# 8. Save
save_train_val_test_numpy(X_train, y_train, X_val, y_val, X_test, y_test, SaveDir)
save_dataset_scalar_metadata(X_train, df, scaler, SaveDir)

# np.save("X_train.npy", X_train)
# df.to_csv("dataset.csv", index=False)

print(" ")

