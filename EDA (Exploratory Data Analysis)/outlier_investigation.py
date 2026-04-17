import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from data_overview import label_to_class
from signal_behaviour import extract_activity_segments
from statistical_properties import compute_activity_similarity


channel_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def extract_activity_segments_with_metadata(dataset_dir, labels_file, acc_pattern, gyro_pattern, target_activity):
    """
    Extract all raw segments for one activity, with metadata.
    Returns a list of dicts:
        {
            'experiment': int,
            'user': int,
            'activity': int,
            'activity_name': str,
            'start': int,
            'end': int,
            'num_samples': int,
            'duration_seconds': float,
            'data': ndarray of shape (T, 6)
        }
    """
    import glob
    from data_overview import parse_exp_user_from_filename

    acc_files = sorted(glob.glob(os.path.join(dataset_dir, acc_pattern)))
    gyro_files = sorted(glob.glob(os.path.join(dataset_dir, gyro_pattern)))
    labels = np.loadtxt(os.path.join(dataset_dir, labels_file))

    segments = []

    for acc_file, gyro_file in zip(acc_files, gyro_files):
        acc = np.loadtxt(acc_file)
        gyro = np.loadtxt(gyro_file)

        exp_id, user_id = parse_exp_user_from_filename(acc_file)

        file_labels = labels[
            (labels[:, 0].astype(int) == exp_id) &
            (labels[:, 1].astype(int) == user_id) &
            (labels[:, 2].astype(int) == target_activity)
        ]

        for row in file_labels:
            _, _, activity, start, end = row
            start = int(start)
            end = int(end)
            activity = int(activity)

            acc_seg = acc[start:end]
            gyro_seg = gyro[start:end]
            segment_data = np.concatenate((acc_seg, gyro_seg), axis=1)

            segments.append({
                "experiment": exp_id,
                "user": user_id,
                "activity": activity,
                "activity_name": label_to_class[activity],
                "start": start,
                "end": end,
                "num_samples": end - start,
                "duration_seconds": (end - start) / SAMPLING_RATE,
                "data": segment_data,
            })

    return segments







def get_top_outliers_for_activity(similarity_df, activity_id, top_n=3):
    """
    Return the top ranked candidate outliers for one activity.
    """
    activity_df = similarity_df[similarity_df["activity"] == activity_id].copy()
    activity_df = activity_df.sort_values("outlier_zscore", ascending=False)
    return activity_df.head(top_n)


def get_most_typical_for_activity(similarity_df, activity_id, top_n=3):
    """
    Return the most typical segments for one activity
    (lowest avg distance to class).
    """
    activity_df = similarity_df[similarity_df["activity"] == activity_id].copy()
    activity_df = activity_df.sort_values("avg_distance_to_class", ascending=True)
    return activity_df.head(top_n)


def find_matching_segment_record(segment_records, experiment, user, start, end):
    for seg in segment_records:
        if (
            int(seg["experiment"]) == int(experiment) and
            int(seg["user"]) == int(user) and
            int(seg["start"]) == int(start) and
            int(seg["end"]) == int(end)
        ):
            return seg
    return None


def plot_outlier_vs_typical(segment_records, outlier_row, typical_rows, channels=(0, 1, 2)):
    """
    Plot one outlier segment against a few typical segments from the same activity.
    Default uses acc_x, acc_y, acc_z for easier interpretation.
    """
    outlier_seg = find_matching_segment_record(
        segment_records,
        outlier_row["experiment"],
        outlier_row["user"],
        outlier_row["start"],
        outlier_row["end"],
    )

    if outlier_seg is None:
        print("Could not find matching outlier segment in raw records.")
        return

    nrows = len(channels)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3.5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, ch in zip(axes, channels):
        outlier_time = np.arange(outlier_seg["data"].shape[0]) / SAMPLING_RATE
        ax.plot(outlier_time, outlier_seg["data"][:, ch], linewidth=2.5, label="Outlier")

        for i, (_, typical_row) in enumerate(typical_rows.iterrows()):
            typical_seg = find_matching_segment_record(
                segment_records,
                typical_row["experiment"],
                typical_row["user"],
                typical_row["start"],
                typical_row["end"],
            )
            if typical_seg is None:
                continue

            typical_time = np.arange(typical_seg["data"].shape[0]) / SAMPLING_RATE
            ax.plot(typical_time, typical_seg["data"][:, ch], alpha=0.8, label=f"Typical {i+1}")

        ax.set_title(f"{outlier_seg['activity_name']} - {channel_names[ch]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(channel_names[ch])
        ax.legend()

    plt.tight_layout()
    plt.show()


def print_outlier_investigation_summary(similarity_df, activity_id, top_n=5):
    activity_name = label_to_class[activity_id]
    activity_df = similarity_df[similarity_df["activity"] == activity_id].copy()

    if activity_df.empty:
        print(f"\nNo similarity data found for activity {activity_id} - {activity_name}")
        return

    ranked = activity_df.sort_values("outlier_zscore", ascending=False).head(top_n)

    print("\n" + "=" * 100)
    print(f"2.3 OUTLIER INVESTIGATION - {activity_name} (Activity {activity_id})")
    print("=" * 100)
    print(
        f"{'Exp':>4} | {'User':>4} | {'Start':>7} | {'End':>7} | "
        f"{'Samples':>8} | {'Avg Dist':>10} | {'z-score':>10} | {'Flag':>5}"
    )
    print("-" * 100)

    for _, row in ranked.iterrows():
        print(
            f"{int(row['experiment']):>4} | "
            f"{int(row['user']):>4} | "
            f"{int(row['start']):>7} | "
            f"{int(row['end']):>7} | "
            f"{int(row['num_samples']):>8} | "
            f"{row['avg_distance_to_class']:>10.4f} | "
            f"{row['outlier_zscore']:>10.4f} | "
            f"{str(bool(row['candidate_outlier'])):>5}"
        )

    print("\nNote:")
    print("  These are candidate outliers for inspection, not confirmed label errors.")
    print("  Correlation heatmaps below compare segment shape after length-normalisation for visual analysis only.")









def get_flagged_outliers_for_activity(similarity_df, activity_id):
    activity_df = similarity_df[similarity_df["activity"] == activity_id].copy()
    activity_df = activity_df[activity_df["candidate_outlier"] == True]
    activity_df = activity_df.sort_values("outlier_zscore", ascending=False)
    return activity_df


def get_typical_segments_for_activity(similarity_df, activity_id, top_n=3):
    activity_df = similarity_df[similarity_df["activity"] == activity_id].copy()
    activity_df = activity_df.sort_values("avg_distance_to_class", ascending=True)
    return activity_df.head(top_n)


def find_matching_segment_record(segment_records, experiment, user, start, end):
    for seg in segment_records:
        if (
            int(seg["experiment"]) == int(experiment)
            and int(seg["user"]) == int(user)
            and int(seg["start"]) == int(start)
            and int(seg["end"]) == int(end)
        ):
            return seg
    return None


def _plot_overlay_group(
    axes,
    segment_records,
    outlier_rows,
    typical_rows,
    channel_indices,
    title_prefix,
    typical_alpha=0.35,
    outlier_alpha=0.85,
):
    """
    Internal helper for acc/gyro overlay plots.
    """
    # Plot typical segments first
    for i, (_, typical_row) in enumerate(typical_rows.iterrows()):
        seg = find_matching_segment_record(
            segment_records,
            typical_row["experiment"],
            typical_row["user"],
            typical_row["start"],
            typical_row["end"],
        )
        if seg is None:
            continue

        time = np.arange(seg["data"].shape[0]) / SAMPLING_RATE

        for ax, ch in zip(axes, channel_indices):
            ax.plot(
                time,
                seg["data"][:, ch],
                color="gray",
                alpha=typical_alpha,
                linewidth=1.2,
                label="Typical" if i == 0 else None,
            )

    # Plot outliers on top
    for j, (_, outlier_row) in enumerate(outlier_rows.iterrows()):
        seg = find_matching_segment_record(
            segment_records,
            outlier_row["experiment"],
            outlier_row["user"],
            outlier_row["start"],
            outlier_row["end"],
        )
        if seg is None:
            continue

        time = np.arange(seg["data"].shape[0]) / SAMPLING_RATE
        outlier_label = (
            f"Outlier {j+1} "
            f"(u{int(outlier_row['user'])}, "
            f"{int(outlier_row['start'])}-{int(outlier_row['end'])}, "
            f"z={outlier_row['outlier_zscore']:.2f})"
        )

        for ax, ch in zip(axes, channel_indices):
            ax.plot(
                time,
                seg["data"][:, ch],
                alpha=outlier_alpha,
                linewidth=2.0,
                label=outlier_label,
            )

    for ax, ch in zip(axes, channel_indices):
        ax.set_title(f"{title_prefix} - {channel_names[ch]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(channel_names[ch])
        ax.legend(loc="upper right", fontsize=8)


def plot_activity_outliers_vs_typicals_acc(
    activity_id,
    segment_records,
    similarity_df,
    n_typical=3,
    save_dir=None,
):
    outlier_rows = get_flagged_outliers_for_activity(similarity_df, activity_id)
    typical_rows = get_typical_segments_for_activity(similarity_df, activity_id, top_n=n_typical)

    if outlier_rows.empty:
        print(f"No flagged outliers found for activity {activity_id} - {label_to_class[activity_id]}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)

    _plot_overlay_group(
        axes=axes,
        segment_records=segment_records,
        outlier_rows=outlier_rows,
        typical_rows=typical_rows,
        channel_indices=[0, 1, 2],
        title_prefix=f"{label_to_class[activity_id]} (Acceleration)",
    )

    fig.suptitle(
        f"{label_to_class[activity_id]} - Accelerometer Outliers vs Typical Segments",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"activity_{activity_id:02d}_{label_to_class[activity_id]}_outliers_acc.png"
        )
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_activity_outliers_vs_typicals_gyro(
    activity_id,
    segment_records,
    similarity_df,
    n_typical=3,
    save_dir=None,
):
    outlier_rows = get_flagged_outliers_for_activity(similarity_df, activity_id)
    typical_rows = get_typical_segments_for_activity(similarity_df, activity_id, top_n=n_typical)

    if outlier_rows.empty:
        print(f"No flagged outliers found for activity {activity_id} - {label_to_class[activity_id]}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)

    _plot_overlay_group(
        axes=axes,
        segment_records=segment_records,
        outlier_rows=outlier_rows,
        typical_rows=typical_rows,
        channel_indices=[3, 4, 5],
        title_prefix=f"{label_to_class[activity_id]} (Gyroscope)",
    )

    fig.suptitle(
        f"{label_to_class[activity_id]} - Gyroscope Outliers vs Typical Segments",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"activity_{activity_id:02d}_{label_to_class[activity_id]}_outliers_gyro.png"
        )
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_top_global_outliers(
    activity_id,
    segment_records,
    similarity_df,
    n_top_outliers=3,
    n_typical=3,
    save_dir=None,
):
    """
    Detailed figure(s) for the most extreme outliers in one activity.
    One figure per outlier, showing all 6 channels.
    """
    outlier_rows = get_flagged_outliers_for_activity(similarity_df, activity_id).head(n_top_outliers)
    typical_rows = get_typical_segments_for_activity(similarity_df, activity_id, top_n=n_typical)

    if outlier_rows.empty:
        print(f"No flagged outliers found for activity {activity_id} - {label_to_class[activity_id]}")
        return

    for rank, (_, outlier_row) in enumerate(outlier_rows.iterrows(), start=1):
        outlier_seg = find_matching_segment_record(
            segment_records,
            outlier_row["experiment"],
            outlier_row["user"],
            outlier_row["start"],
            outlier_row["end"],
        )

        if outlier_seg is None:
            continue

        fig, axes = plt.subplots(6, 1, figsize=(13, 16), sharex=False)
        outlier_time = np.arange(outlier_seg["data"].shape[0]) / SAMPLING_RATE

        for ch, ax in enumerate(axes):
            # Typical references first
            for i, (_, typical_row) in enumerate(typical_rows.iterrows()):
                typical_seg = find_matching_segment_record(
                    segment_records,
                    typical_row["experiment"],
                    typical_row["user"],
                    typical_row["start"],
                    typical_row["end"],
                )
                if typical_seg is None:
                    continue

                typical_time = np.arange(typical_seg["data"].shape[0]) / SAMPLING_RATE
                ax.plot(
                    typical_time,
                    typical_seg["data"][:, ch],
                    color="gray",
                    alpha=0.35,
                    linewidth=1.2,
                    label="Typical" if i == 0 else None,
                )

            # Outlier on top
            ax.plot(
                outlier_time,
                outlier_seg["data"][:, ch],
                linewidth=2.2,
                alpha=0.9,
                label=(
                    f"Outlier rank {rank} "
                    f"(u{int(outlier_row['user'])}, "
                    f"{int(outlier_row['start'])}-{int(outlier_row['end'])}, "
                    f"z={outlier_row['outlier_zscore']:.2f})"
                ),
            )

            ax.set_title(f"{label_to_class[activity_id]} - {channel_names[ch]}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(channel_names[ch])
            ax.legend(loc="upper right", fontsize=8)

        fig.suptitle(
            f"{label_to_class[activity_id]} - Top Global Outlier {rank} vs Typical Segments",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"activity_{activity_id:02d}_{label_to_class[activity_id]}_top_outlier_{rank}.png"
            )
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()