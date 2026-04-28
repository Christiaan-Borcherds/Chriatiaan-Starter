import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from data_overview import label_to_class
from outlier_investigation import extract_activity_segments_with_metadata


channel_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# --------------------------------------------------
# 2.2.5 Per-Subject Signals
# --------------------------------------------------
def summarize_subject_contribution(segment_manifest_df):
    """
    Basic per-subject contribution summary using the cached segment manifest.
    """
    summary = (
        segment_manifest_df
        .groupby("user")
        .agg(
            segments=("activity", "count"),
            activities=("activity", "nunique"),
            total_samples=("num_samples", "sum"),
            total_seconds=("duration_seconds", "sum"),
            avg_segment_seconds=("duration_seconds", "mean"),
        )
        .reset_index()
        .sort_values("user")
    )

    total_segments = summary["segments"].sum()
    total_samples = summary["total_samples"].sum()

    summary["percent_segments"] = np.where(
        total_segments > 0,
        (summary["segments"] / total_segments) * 100,
        0.0,
    )

    summary["percent_samples"] = np.where(
        total_samples > 0,
        (summary["total_samples"] / total_samples) * 100,
        0.0,
    )

    numeric_cols = [
        "total_seconds",
        "avg_segment_seconds",
        "percent_segments",
        "percent_samples",
    ]
    summary[numeric_cols] = summary[numeric_cols].round(2)

    return summary


def build_per_subject_channel_summary(segment_manifest_df):
    """
    Compute per-subject average channel statistics from the segment feature table.
    """
    channel_feature_cols = []
    for ch in channel_names:
        channel_feature_cols.extend([
            f"{ch}_mean",
            f"{ch}_std",
            f"{ch}_rms",
        ])

    subject_channel_df = (
        segment_manifest_df
        .groupby("user")[channel_feature_cols]
        .mean()
        .reset_index()
        .sort_values("user")
    )

    subject_channel_df[channel_feature_cols] = subject_channel_df[channel_feature_cols].round(6)

    return subject_channel_df


def print_subject_contribution_summary(subject_summary_df):
    print("\n" + "=" * 110)
    print("2.2.5 PER-SUBJECT SIGNALS")
    print("=" * 110)

    print("\nSubject contribution summary:")
    print(
        f"{'User':>4} | {'Segments':>8} | {'% Seg':>7} | {'Samples':>10} | {'% Samp':>7} | "
        f"{'Activities':>10} | {'Seconds':>10} | {'Avg seg s':>10}"
    )
    print("-" * 110)

    for _, row in subject_summary_df.iterrows():
        print(
            f"{int(row['user']):>4} | "
            f"{int(row['segments']):>8} | "
            f"{row['percent_segments']:>6.2f}% | "
            f"{int(row['total_samples']):>10} | "
            f"{row['percent_samples']:>6.2f}% | "
            f"{int(row['activities']):>10} | "
            f"{row['total_seconds']:>10.2f} | "
            f"{row['avg_segment_seconds']:>10.2f}"
        )


def export_per_subject_outputs(subject_summary_df, subject_channel_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    subject_summary_path = os.path.join(output_dir, "per_subject_summary.csv")
    subject_channel_path = os.path.join(output_dir, "per_subject_channel_summary.csv")

    subject_summary_df.to_csv(subject_summary_path, index=False)
    subject_channel_df.to_csv(subject_channel_path, index=False)

    print(f"\nSaved per-subject summary to: {subject_summary_path}")
    print(f"Saved per-subject channel summary to: {subject_channel_path}")


# --------------------------------------------------
# Plotting
# --------------------------------------------------
def plot_subject_variability_bars(subject_channel_df, channel_stat="acc_x_std", save_dir=None):
    """
    Bar plot of one chosen channel statistic across subjects.
    Example channel_stat:
        acc_x_std, acc_y_std, gyro_z_rms, ...
    """
    if channel_stat not in subject_channel_df.columns:
        print(f"Column '{channel_stat}' not found in subject_channel_df")
        return

    plt.figure(figsize=(12, 5))
    plt.bar(subject_channel_df["user"].astype(str), subject_channel_df[channel_stat])
    plt.xlabel("Subject")
    plt.ylabel(channel_stat)
    plt.title(f"Per-subject comparison: {channel_stat}")
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"per_subject_{channel_stat}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_same_activity_across_subjects(
    dataset_dir,
    labels_file,
    acc_pattern,
    gyro_pattern,
    activity_id,
    subject_ids=None,
    max_subjects=5,
    channel_indices=(0, 1, 2),
    save_dir=None,
):
    """
    Plot one representative segment of the same activity across multiple subjects.
    Default shows acceleration axes.
    """
    segment_records = extract_activity_segments_with_metadata(
        dataset_dir=dataset_dir,
        labels_file=labels_file,
        acc_pattern=acc_pattern,
        gyro_pattern=gyro_pattern,
        target_activity=activity_id,
    )

    if len(segment_records) == 0:
        print(f"No segments found for activity {activity_id} - {label_to_class[activity_id]}")
        return

    # Keep first segment per subject
    selected = {}
    for seg in segment_records:
        user_id = int(seg["user"])
        if subject_ids is not None and user_id not in subject_ids:
            continue
        if user_id not in selected:
            selected[user_id] = seg
        if len(selected) >= max_subjects:
            break

    if len(selected) == 0:
        print(f"No matching subjects found for activity {activity_id} - {label_to_class[activity_id]}")
        return

    fig, axes = plt.subplots(len(channel_indices), 1, figsize=(12, 3.5 * len(channel_indices)), squeeze=False)
    axes = axes.flatten()

    for user_id, seg in selected.items():
        time = np.arange(seg["data"].shape[0]) / SAMPLING_RATE
        for ax, ch in zip(axes, channel_indices):
            ax.plot(time, seg["data"][:, ch], label=f"User {user_id}")

    for ax, ch in zip(axes, channel_indices):
        ax.set_title(f"{label_to_class[activity_id]} across subjects - {channel_names[ch]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(channel_names[ch])
        ax.legend()

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"activity_{activity_id:02d}_{label_to_class[activity_id]}_across_subjects.png"
        )
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_same_activity_across_subjects_gyro(
    dataset_dir,
    labels_file,
    acc_pattern,
    gyro_pattern,
    activity_id,
    subject_ids=None,
    max_subjects=5,
    channel_indices=(3, 4, 5),
    save_dir=None,
):
    """
    Same as above, but default shows gyroscope axes.
    """
    plot_same_activity_across_subjects(
        dataset_dir=dataset_dir,
        labels_file=labels_file,
        acc_pattern=acc_pattern,
        gyro_pattern=gyro_pattern,
        activity_id=activity_id,
        subject_ids=subject_ids,
        max_subjects=max_subjects,
        channel_indices=channel_indices,
        save_dir=save_dir,
    )


def plot_per_subject_multibar(subject_channel_df, stat_type="std", save_dir=None):
    """
    Create one figure with 2 subplots:
        1) accelerometer xyz
        2) gyroscope xyz

    stat_type options:
        "mean", "std", "rms"
    """
    valid_stats = {"mean", "std", "rms"}
    if stat_type not in valid_stats:
        raise ValueError(f"stat_type must be one of {valid_stats}")

    subjects = subject_channel_df["user"].astype(str).tolist()
    x = np.arange(len(subjects))
    width = 0.25

    acc_cols = [f"acc_x_{stat_type}", f"acc_y_{stat_type}", f"acc_z_{stat_type}"]
    gyro_cols = [f"gyro_x_{stat_type}", f"gyro_y_{stat_type}", f"gyro_z_{stat_type}"]

    for col in acc_cols + gyro_cols:
        if col not in subject_channel_df.columns:
            raise ValueError(f"Column '{col}' not found in subject_channel_df")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax_acc, ax_gyro = axes

    # -------------------------
    # Accelerometer subplot
    # -------------------------
    ax_acc.bar(x - width, subject_channel_df[acc_cols[0]], width, label="acc_x")
    ax_acc.bar(x,         subject_channel_df[acc_cols[1]], width, label="acc_y")
    ax_acc.bar(x + width, subject_channel_df[acc_cols[2]], width, label="acc_z")
    ax_acc.set_title(f"Per-subject Accelerometer {stat_type.upper()}")
    ax_acc.set_ylabel(stat_type)
    ax_acc.legend()

    # -------------------------
    # Gyroscope subplot
    # -------------------------
    ax_gyro.bar(x - width, subject_channel_df[gyro_cols[0]], width, label="gyro_x")
    ax_gyro.bar(x,         subject_channel_df[gyro_cols[1]], width, label="gyro_y")
    ax_gyro.bar(x + width, subject_channel_df[gyro_cols[2]], width, label="gyro_z")
    ax_gyro.set_title(f"Per-subject Gyroscope {stat_type.upper()}")
    ax_gyro.set_ylabel(stat_type)
    ax_gyro.set_xlabel("Subject")
    ax_gyro.set_xticks(x)
    ax_gyro.set_xticklabels(subjects)
    ax_gyro.legend()

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"per_subject_multibar_{stat_type}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()