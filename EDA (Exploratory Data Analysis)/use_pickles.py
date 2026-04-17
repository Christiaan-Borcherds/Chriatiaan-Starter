import os
import pickle
import pandas as pd
import numpy as np


# =========================================================
# CONFIG
# =========================================================
# CACHE_DIR = "/path/to/your/cache_dir"
CACHE_DIR = '/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/EDA (Exploratory Data Analysis)/Data Cache'
# Example:
# CACHE_DIR = "/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/output"


# =========================================================
# PICKLE HELPERS
# =========================================================
def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def print_divider(title="", width=90):
    print("\n" + "=" * width)
    if title:
        print(title)
        print("=" * width)


# =========================================================
# STRUCTURE INSPECTION
# =========================================================
def describe_dataframe(df, name, preview_rows=3):
    print_divider(f"{name}  [type: pandas.DataFrame]")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nDtypes:")
    print(df.dtypes)

    print(f"\nFirst {preview_rows} rows:")
    print(df.head(preview_rows))


def describe_ndarray(arr, name):
    print_divider(f"{name}  [type: numpy.ndarray]")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")

    if arr.ndim == 2 and arr.shape[0] > 0:
        print("\nFirst 3 rows:")
        print(arr[:3])


def describe_dict(d, name, preview_items=3):
    print_divider(f"{name}  [type: dict]")
    print(f"Number of keys: {len(d)}")

    if len(d) == 0:
        print("Dictionary is empty.")
        return

    print("\nFirst keys:")
    keys = list(d.keys())[:preview_items]
    for k in keys:
        value = d[k]
        print(f"  Key: {k!r}")
        print(f"    Value type: {type(value)}")

        if isinstance(value, dict):
            print(f"    Nested keys: {list(value.keys())}")
        elif isinstance(value, np.ndarray):
            print(f"    Shape: {value.shape}")
            print(f"    Dtype: {value.dtype}")
        elif isinstance(value, list):
            print(f"    List length: {len(value)}")
            print(f"    First few items: {value[:5]}")
        else:
            print(f"    Value preview: {value}")


def print_cache_structure(segment_manifest, segment_data_dict, segment_indexes, raw_recordings=None):
    describe_dataframe(segment_manifest, "segment_manifest")

    describe_dict(segment_data_dict, "segment_data_dict")
    print("\nExpected usage:")
    print("  segment_data_dict[segment_id] -> ndarray of shape (T, 6)")
    print("  Channels are:")
    print("    [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]")

    describe_dict(segment_indexes, "segment_indexes")
    print("\nExpected usage:")
    print("  segment_indexes['by_user'][user_id] -> list of segment_ids")
    print("  segment_indexes['by_activity'][activity_id] -> list of segment_ids")
    print("  segment_indexes['by_user_activity'][(user_id, activity_id)] -> list of segment_ids")

    if raw_recordings is not None:
        describe_dict(raw_recordings, "raw_recordings")
        print("\nExpected usage:")
        print("  raw_recordings[(exp_id, user_id)]['acc']  -> full accelerometer recording")
        print("  raw_recordings[(exp_id, user_id)]['gyro'] -> full gyroscope recording")


# =========================================================
# EXAMPLE USAGE
# =========================================================
def show_example_queries(segment_manifest, segment_data_dict, segment_indexes):
    print_divider("EXAMPLE CACHE QUERIES")

    # 1) Find all candidate outliers
    if "candidate_outlier" in segment_manifest.columns:
        flagged = segment_manifest[segment_manifest["candidate_outlier"] == True]
        print(f"Number of candidate outliers: {len(flagged)}")
    else:
        flagged = pd.DataFrame()
        print("No 'candidate_outlier' column found in segment_manifest.")

    # 2) Get one segment's raw data using segment_id
    if len(segment_manifest) > 0:
        example_segment_id = segment_manifest.iloc[0]["segment_id"]
        print(f"\nExample segment_id: {example_segment_id}")

        if example_segment_id in segment_data_dict:
            arr = segment_data_dict[example_segment_id]
            describe_ndarray(arr, f"segment_data_dict['{example_segment_id}']")
        else:
            print("Segment ID not found in segment_data_dict.")

    # 3) Get all segments for one activity
    if "by_activity" in segment_indexes and len(segment_indexes["by_activity"]) > 0:
        first_activity = sorted(segment_indexes["by_activity"].keys())[0]
        activity_segment_ids = segment_indexes["by_activity"][first_activity]
        print(f"\nActivity {first_activity} has {len(activity_segment_ids)} segment(s)")
        print(f"First 5 segment_ids: {activity_segment_ids[:5]}")

    # 4) Get all segments for one user
    if "by_user" in segment_indexes and len(segment_indexes["by_user"]) > 0:
        first_user = sorted(segment_indexes["by_user"].keys())[0]
        user_segment_ids = segment_indexes["by_user"][first_user]
        print(f"\nUser {first_user} has {len(user_segment_ids)} segment(s)")
        print(f"First 5 segment_ids: {user_segment_ids[:5]}")


# =========================================================
# TOP OUTLIERS DISPLAY
# =========================================================
def print_top_candidate_outliers(segment_manifest, top_n=20):
    required_cols = [
        "experiment",
        "user",
        "activity",
        "activity_name",
        "start",
        "end",
        "avg_distance_to_class",
        "outlier_zscore",
        "candidate_outlier",
    ]

    missing = [col for col in required_cols if col not in segment_manifest.columns]
    if missing:
        raise ValueError(f"segment_manifest is missing required columns: {missing}")

    flagged = segment_manifest[segment_manifest["candidate_outlier"] == True].copy()

    if flagged.empty:
        print("\nNo candidate outliers found.")
        return flagged

    ranked = flagged.sort_values("outlier_zscore", ascending=False).head(top_n)

    print_divider("Top candidate outliers", width=100)
    print(
        f"{'Exp':>4} | {'User':>4} | {'Act':>3} | {'Activity':<20} | "
        f"{'Start':>7} | {'End':>7} | {'Avg Dist':>10} | {'z-score':>10} | {'Flag':>5}"
    )
    print("-" * 100)

    for _, row in ranked.iterrows():
        print(
            f"{int(row['experiment']):>4} | "
            f"{int(row['user']):>4} | "
            f"{int(row['activity']):>3} | "
            f"{str(row['activity_name']):<20} | "
            f"{int(row['start']):>7} | "
            f"{int(row['end']):>7} | "
            f"{float(row['avg_distance_to_class']):>10.4f} | "
            f"{float(row['outlier_zscore']):>10.2f} | "
            f"{str(bool(row['candidate_outlier'])):>5}"
        )

    return ranked


# =========================================================
# MAIN
# =========================================================

segment_manifest_path = os.path.join(CACHE_DIR, "segment_manifest.pkl")
segment_data_dict_path = os.path.join(CACHE_DIR, "segment_data_dict.pkl")
segment_indexes_path = os.path.join(CACHE_DIR, "segment_indexes.pkl")
raw_recordings_path = os.path.join(CACHE_DIR, "raw_recordings.pkl")

print_divider("LOADING PICKLES")
print(f"segment_manifest:  {segment_manifest_path}")
print(f"segment_data_dict: {segment_data_dict_path}")
print(f"segment_indexes:   {segment_indexes_path}")
print(f"raw_recordings:    {raw_recordings_path}")

segment_manifest = load_pickle(segment_manifest_path)
segment_data_dict = load_pickle(segment_data_dict_path)
segment_indexes = load_pickle(segment_indexes_path)
raw_recordings = load_pickle(raw_recordings_path) if os.path.exists(raw_recordings_path) else None

print_cache_structure(
    segment_manifest=segment_manifest,
    segment_data_dict=segment_data_dict,
    segment_indexes=segment_indexes,
    raw_recordings=raw_recordings,
)

show_example_queries(
    segment_manifest=segment_manifest,
    segment_data_dict=segment_data_dict,
    segment_indexes=segment_indexes,
)

ranked = print_top_candidate_outliers(segment_manifest, top_n=20)

# Example: print the exact first row like your earlier output
if not ranked.empty:
    top = ranked.iloc[0]
    print_divider("Top 1 candidate outlier")
    print(
        f"{int(top['experiment']):>4} | "
        f"{int(top['user']):>4} | "
        f"{int(top['activity']):>3} | "
        f"{str(top['activity_name']):<20} | "
        f"{int(top['start']):>7} | "
        f"{int(top['end']):>7} | "
        f"{float(top['avg_distance_to_class']):>10.4f} | "
        f"{float(top['outlier_zscore']):>10.4f} | "
        f"{str(bool(top['candidate_outlier'])):>5}"
    )


