import os
import glob
import pickle
import numpy as np
import pandas as pd

from config import *
from data_overview import parse_exp_user_from_filename, label_to_class


channel_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

activity_groups = {
    1: "Dynamic",
    2: "Dynamic",
    3: "Dynamic",
    4: "Static",
    5: "Static",
    6: "Static",
    7: "Transition",
    8: "Transition",
    9: "Transition",
    10: "Transition",
    11: "Transition",
    12: "Transition",
}


def make_segment_id(exp_id, user_id, activity_id, start, end):
    return f"exp{int(exp_id):02d}_user{int(user_id):02d}_act{int(activity_id):02d}_{int(start)}_{int(end)}"


def load_raw_recordings(dataset_dir, acc_pattern, gyro_pattern):
    raw_recordings = {}

    acc_files = sorted(glob.glob(os.path.join(dataset_dir, acc_pattern)))
    gyro_files = sorted(glob.glob(os.path.join(dataset_dir, gyro_pattern)))

    for acc_file, gyro_file in zip(acc_files, gyro_files):
        acc = np.loadtxt(acc_file)
        gyro = np.loadtxt(gyro_file)

        exp_id, user_id = parse_exp_user_from_filename(acc_file)

        raw_recordings[(exp_id, user_id)] = {
            "acc": acc,
            "gyro": gyro,
        }

    return raw_recordings


def build_segment_cache_from_manifest(segment_manifest_df, raw_recordings):
    """
    Requires a dataframe with:
    experiment, user, activity, start, end
    and ideally the rest of the metadata/stats already attached.
    """
    segment_data_dict = {}
    by_user = {}
    by_activity = {}
    by_user_activity = {}

    manifest_df = segment_manifest_df.copy()

    if "segment_id" not in manifest_df.columns:
        manifest_df["segment_id"] = manifest_df.apply(
            lambda row: make_segment_id(
                row["experiment"], row["user"], row["activity"], row["start"], row["end"]
            ),
            axis=1,
        )

    for _, row in manifest_df.iterrows():
        exp_id = int(row["experiment"])
        user_id = int(row["user"])
        activity_id = int(row["activity"])
        start = int(row["start"])
        end = int(row["end"])
        segment_id = row["segment_id"]

        recording = raw_recordings[(exp_id, user_id)]
        acc_seg = recording["acc"][start:end]
        gyro_seg = recording["gyro"][start:end]
        segment_data = np.concatenate((acc_seg, gyro_seg), axis=1)

        segment_data_dict[segment_id] = segment_data

        by_user.setdefault(user_id, []).append(segment_id)
        by_activity.setdefault(activity_id, []).append(segment_id)
        by_user_activity.setdefault((user_id, activity_id), []).append(segment_id)

    indexes = {
        "by_user": by_user,
        "by_activity": by_activity,
        "by_user_activity": by_user_activity,
    }

    return manifest_df, segment_data_dict, indexes


def save_pickle(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_cache_bundle(
    cache_dir,
    manifest_df,
    segment_data_dict,
    indexes,
    raw_recordings=None,
):
    os.makedirs(cache_dir, exist_ok=True)

    save_pickle(manifest_df, os.path.join(cache_dir, "segment_manifest.pkl"))
    save_pickle(segment_data_dict, os.path.join(cache_dir, "segment_data_dict.pkl"))
    save_pickle(indexes, os.path.join(cache_dir, "segment_indexes.pkl"))

    if raw_recordings is not None:
        save_pickle(raw_recordings, os.path.join(cache_dir, "raw_recordings.pkl"))

    print(f"\nSaved cache bundle to: {cache_dir}")
    print("  - segment_manifest.pkl")
    print("  - segment_data_dict.pkl")
    print("  - segment_indexes.pkl")
    if raw_recordings is not None:
        print("  - raw_recordings.pkl")