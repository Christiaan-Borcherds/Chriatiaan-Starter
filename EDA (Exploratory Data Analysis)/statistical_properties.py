import os
import glob
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


def compute_rms(x):
    return np.sqrt(np.mean(np.square(x)))


def extract_all_labelled_segments(dataset_dir, labels_file, acc_pattern, gyro_pattern):
    """
    Extract all labelled segments and retain metadata.
    Each returned item is a dictionary:
    {
        'experiment': int,
        'user': int,
        'activity': int,
        'activity_name': str,
        'group': str,
        'start': int,
        'end': int,
        'num_samples': int,
        'duration_seconds': float,
        'data': ndarray of shape (T, 6)
    }
    """
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
            (labels[:, 1].astype(int) == user_id)
        ]

        for row in file_labels:
            _, _, activity, start, end = row

            activity = int(activity)
            start = int(start)
            end = int(end)

            acc_seg = acc[start:end]
            gyro_seg = gyro[start:end]
            segment_data = np.concatenate((acc_seg, gyro_seg), axis=1)

            num_samples = end - start
            duration_seconds = num_samples / SAMPLING_RATE

            segments.append({
                "experiment": exp_id,
                "user": user_id,
                "activity": activity,
                "activity_name": label_to_class[activity],
                "group": activity_groups[activity],
                "start": start,
                "end": end,
                "num_samples": num_samples,
                "duration_seconds": duration_seconds,
                "data": segment_data,
            })

    return segments


def compute_segment_feature_row(segment):
    """
    Convert one labelled segment into a feature row.
    """
    row = {
        "experiment": segment["experiment"],
        "user": segment["user"],
        "activity": segment["activity"],
        "activity_name": segment["activity_name"],
        "group": segment["group"],
        "start": segment["start"],
        "end": segment["end"],
        "num_samples": segment["num_samples"],
        "duration_seconds": round(segment["duration_seconds"], 4),
    }

    data = segment["data"]

    for i, ch in enumerate(channel_names):
        signal = data[:, i]

        row[f"{ch}_mean"] = round(float(np.mean(signal)), 6)
        row[f"{ch}_std"] = round(float(np.std(signal)), 6)
        row[f"{ch}_min"] = round(float(np.min(signal)), 6)
        row[f"{ch}_max"] = round(float(np.max(signal)), 6)
        row[f"{ch}_rms"] = round(float(compute_rms(signal)), 6)

    return row


def build_segment_feature_table(dataset_dir, labels_file, acc_pattern, gyro_pattern):
    """
    Build one dataframe with one row per labelled segment.
    """
    segments = extract_all_labelled_segments(
        dataset_dir=dataset_dir,
        labels_file=labels_file,
        acc_pattern=acc_pattern,
        gyro_pattern=gyro_pattern,
    )

    rows = [compute_segment_feature_row(segment) for segment in segments]
    df = pd.DataFrame(rows)

    return df


def print_segment_feature_overview(segment_df):
    print("\n" + "=" * 70)
    print("2.4 PHASE 1 - SEGMENT FEATURE TABLE")
    print("=" * 70)

    print(f"Number of labelled segments : {len(segment_df)}")
    print(f"Number of feature columns   : {segment_df.shape[1]}")
    print(f"Groups present              : {sorted(segment_df['group'].unique())}")
    print(f"Activities present          : {sorted(segment_df['activity'].unique())}")
    print(f"Subjects present            : {segment_df['user'].nunique()}")
    print(f"Experiments present         : {segment_df['experiment'].nunique()}")

    print("\nFirst 5 rows:")
    print(segment_df.head())


def summarize_by_group(segment_df):
    """
    Summary table by activity group.
    """
    summary = (
        segment_df
        .groupby("group")
        .agg(
            segments=("activity", "count"),
            subjects=("user", "nunique"),
            avg_duration_seconds=("duration_seconds", "mean"),
            min_duration_seconds=("duration_seconds", "min"),
            max_duration_seconds=("duration_seconds", "max"),
            avg_samples=("num_samples", "mean"),
        )
        .reset_index()
    )

    numeric_cols = [
        "avg_duration_seconds",
        "min_duration_seconds",
        "max_duration_seconds",
        "avg_samples",
    ]
    summary[numeric_cols] = summary[numeric_cols].round(4)

    return summary


def summarize_by_activity(segment_df):
    """
    Summary table by activity.
    """
    summary = (
        segment_df
        .groupby(["activity", "activity_name", "group"])
        .agg(
            segments=("activity", "count"),
            subjects=("user", "nunique"),
            avg_duration_seconds=("duration_seconds", "mean"),
            min_duration_seconds=("duration_seconds", "min"),
            max_duration_seconds=("duration_seconds", "max"),
            avg_samples=("num_samples", "mean"),
        )
        .reset_index()
        .sort_values("activity")
    )

    numeric_cols = [
        "avg_duration_seconds",
        "min_duration_seconds",
        "max_duration_seconds",
        "avg_samples",
    ]
    summary[numeric_cols] = summary[numeric_cols].round(4)

    return summary


def print_group_summary(group_summary_df):
    print("\n" + "-" * 70)
    print("GROUP-LEVEL SUMMARY")
    print("-" * 70)
    print(
        f"{'Group':<12} | {'Segments':>8} | {'Subjects':>8} | "
        f"{'Avg sec':>8} | {'Min sec':>8} | {'Max sec':>8} | {'Avg samp':>10}"
    )
    print("-" * 80)

    for _, row in group_summary_df.iterrows():
        print(
            f"{row['group']:<12} | "
            f"{int(row['segments']):>8} | "
            f"{int(row['subjects']):>8} | "
            f"{row['avg_duration_seconds']:>8.2f} | "
            f"{row['min_duration_seconds']:>8.2f} | "
            f"{row['max_duration_seconds']:>8.2f} | "
            f"{row['avg_samples']:>10.2f}"
        )


def print_activity_summary(activity_summary_df):
    print("\n" + "-" * 110)
    print("ACTIVITY-LEVEL SUMMARY")
    print("-" * 110)
    print(
        f"{'ID':>3} | {'Activity':<20} | {'Group':<10} | {'Segments':>8} | {'Subjects':>8} | "
        f"{'Avg sec':>8} | {'Min sec':>8} | {'Max sec':>8} | {'Avg samp':>10}"
    )
    print("-" * 120)

    for _, row in activity_summary_df.iterrows():
        print(
            f"{int(row['activity']):>3} | "
            f"{row['activity_name']:<20} | "
            f"{row['group']:<10} | "
            f"{int(row['segments']):>8} | "
            f"{int(row['subjects']):>8} | "
            f"{row['avg_duration_seconds']:>8.2f} | "
            f"{row['min_duration_seconds']:>8.2f} | "
            f"{row['max_duration_seconds']:>8.2f} | "
            f"{row['avg_samples']:>10.2f}"
        )


def export_segment_feature_outputs(segment_df, group_summary_df, activity_summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    segment_path = os.path.join(output_dir, "segment_feature_table.csv")
    group_path = os.path.join(output_dir, "segment_group_summary.csv")
    activity_path = os.path.join(output_dir, "segment_activity_summary.csv")

    segment_df.to_csv(segment_path, index=False)
    group_summary_df.to_csv(group_path, index=False)
    activity_summary_df.to_csv(activity_path, index=False)

    print(f"\nSaved segment feature table to: {segment_path}")
    print(f"Saved group summary to: {group_path}")
    print(f"Saved activity summary to: {activity_path}")





# --------------------------------------------------
# Subject-specific statistics
# --------------------------------------------------
def summarize_by_subject(segment_df):
    """
    Subject-level summary based on the segment feature table.
    """
    summary = (
        segment_df
        .groupby("user")
        .agg(
            segments=("activity", "count"),
            activities=("activity", "nunique"),
            groups=("group", "nunique"),
            avg_duration_seconds=("duration_seconds", "mean"),
            min_duration_seconds=("duration_seconds", "min"),
            max_duration_seconds=("duration_seconds", "max"),
            avg_samples=("num_samples", "mean"),
        )
        .reset_index()
        .sort_values("user")
    )

    numeric_cols = [
        "avg_duration_seconds",
        "min_duration_seconds",
        "max_duration_seconds",
        "avg_samples",
    ]
    summary[numeric_cols] = summary[numeric_cols].round(4)

    return summary


def build_subject_channel_summary(segment_df):
    """
    Per-subject channel summary using the segment-level feature columns.
    Uses means of segment summary features per subject.
    """
    channel_feature_cols = []

    for ch in channel_names:
        channel_feature_cols.extend([
            f"{ch}_mean",
            f"{ch}_std",
            f"{ch}_min",
            f"{ch}_max",
            f"{ch}_rms",
        ])

    summary = (
        segment_df
        .groupby("user")[channel_feature_cols]
        .mean()
        .reset_index()
        .sort_values("user")
    )

    summary[channel_feature_cols] = summary[channel_feature_cols].round(6)

    return summary


def print_subject_summary(subject_summary_df):
    print("\n" + "-" * 90)
    print("SUBJECT-LEVEL SUMMARY")
    print("-" * 90)
    print(
        f"{'User':>4} | {'Segments':>8} | {'Acts':>4} | {'Groups':>6} | "
        f"{'Avg sec':>8} | {'Min sec':>8} | {'Max sec':>8} | {'Avg samp':>10}"
    )
    print("-" * 90)

    for _, row in subject_summary_df.iterrows():
        print(
            f"{int(row['user']):>4} | "
            f"{int(row['segments']):>8} | "
            f"{int(row['activities']):>4} | "
            f"{int(row['groups']):>6} | "
            f"{row['avg_duration_seconds']:>8.2f} | "
            f"{row['min_duration_seconds']:>8.2f} | "
            f"{row['max_duration_seconds']:>8.2f} | "
            f"{row['avg_samples']:>10.2f}"
        )


def export_subject_outputs(subject_summary_df, subject_channel_summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    subject_summary_path = os.path.join(output_dir, "subject_summary.csv")
    subject_channel_path = os.path.join(output_dir, "subject_channel_summary.csv")

    subject_summary_df.to_csv(subject_summary_path, index=False)
    subject_channel_summary_df.to_csv(subject_channel_path, index=False)

    print(f"\nSaved subject summary to: {subject_summary_path}")
    print(f"Saved subject channel summary to: {subject_channel_path}")


# --------------------------------------------------
# Similarity and candidate outlier detection
# --------------------------------------------------
def get_feature_columns_for_similarity():
    """
    Returns the numeric feature columns used for segment similarity.
    """
    cols = ["duration_seconds", "num_samples"]

    for ch in channel_names:
        cols.extend([
            f"{ch}_mean",
            f"{ch}_std",
            f"{ch}_min",
            f"{ch}_max",
            f"{ch}_rms",
        ])

    return cols


def zscore_dataframe(df):
    """
    Standardize numeric columns column-wise.
    """
    return (df - df.mean()) / df.std(ddof=0).replace(0, 1)


def compute_pairwise_euclidean(feature_matrix):
    """
    Compute full pairwise Euclidean distance matrix without sklearn.
    """
    X = feature_matrix.to_numpy(dtype=float)
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist


def compute_activity_similarity(segment_df, min_segments=3):
    """
    For each activity:
    - standardize feature vectors within the activity
    - compute pairwise Euclidean distances
    - compute average distance of each segment to all others
    - compute z-score of avg distance within activity

    Returns:
    - similarity_df: one row per segment with outlier scores
    - distance_matrices: dict[activity_id] -> distance DataFrame
    """
    feature_cols = get_feature_columns_for_similarity()

    rows = []
    distance_matrices = {}

    for activity_id in sorted(segment_df["activity"].unique()):
        activity_data = (
            segment_df[segment_df["activity"] == activity_id]
            .copy()
            .reset_index(drop=True)
        )

        if len(activity_data) < min_segments:
            continue

        X = activity_data[feature_cols].copy()
        Xz = zscore_dataframe(X)

        dist = compute_pairwise_euclidean(Xz)
        np.fill_diagonal(dist, np.nan)

        avg_distance = np.nanmean(dist, axis=1)

        mean_dist = np.nanmean(avg_distance)
        std_dist = np.nanstd(avg_distance)
        if std_dist == 0:
            outlier_z = np.zeros_like(avg_distance)
        else:
            outlier_z = (avg_distance - mean_dist) / std_dist

        activity_result = activity_data[[
            "experiment",
            "user",
            "activity",
            "activity_name",
            "group",
            "start",
            "end",
            "num_samples",
            "duration_seconds",
        ]].copy()

        activity_result["avg_distance_to_class"] = np.round(avg_distance, 6)
        activity_result["outlier_zscore"] = np.round(outlier_z, 6)
        activity_result["candidate_outlier"] = activity_result["outlier_zscore"] >= 2.0

        rows.append(activity_result)

        segment_labels = [
            f"exp{int(r['experiment']):02d}_u{int(r['user']):02d}_{int(r['start'])}-{int(r['end'])}"
            for _, r in activity_data.iterrows()
        ]
        distance_matrices[activity_id] = pd.DataFrame(dist, index=segment_labels, columns=segment_labels)

    similarity_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    return similarity_df, distance_matrices


def summarize_candidate_outliers(similarity_df):
    """
    Summarize candidate outliers by activity.
    """
    if similarity_df.empty:
        return pd.DataFrame()

    summary = (
        similarity_df
        .groupby(["activity", "activity_name", "group"])
        .agg(
            segments=("activity", "count"),
            candidate_outliers=("candidate_outlier", "sum"),
            avg_distance_to_class=("avg_distance_to_class", "mean"),
            max_outlier_zscore=("outlier_zscore", "max"),
        )
        .reset_index()
        .sort_values("activity")
    )

    summary["avg_distance_to_class"] = summary["avg_distance_to_class"].round(6)
    summary["max_outlier_zscore"] = summary["max_outlier_zscore"].round(6)

    return summary


def print_similarity_summary(outlier_summary_df):
    print("\n" + "-" * 110)
    print("SEGMENT SIMILARITY AND CANDIDATE OUTLIERS")
    print("-" * 110)
    print(
        f"{'ID':>3} | {'Activity':<20} | {'Group':<10} | "
        f"{'Segments':>8} | {'Cand. Outliers':>14} | {'Avg Dist':>10} | {'Max z':>10}"
    )
    print("-" * 110)

    for _, row in outlier_summary_df.iterrows():
        print(
            f"{int(row['activity']):>3} | "
            f"{row['activity_name']:<20} | "
            f"{row['group']:<10} | "
            f"{int(row['segments']):>8} | "
            f"{int(row['candidate_outliers']):>14} | "
            f"{row['avg_distance_to_class']:>10.4f} | "
            f"{row['max_outlier_zscore']:>10.4f}"
        )

    print("\nInterpretation:")
    print("  - Similarity is computed using segment feature vectors, not raw resampled signals.")
    print("  - Larger average distance means the segment is less similar to others in the same activity.")
    print("  - Segments flagged here are candidate outliers / segments for inspection, not confirmed errors.")


def print_top_candidate_outliers(similarity_df, top_n=20):
    if similarity_df.empty:
        print("\nNo similarity results available.")
        return

    ranked = similarity_df.sort_values("outlier_zscore", ascending=False).head(top_n)

    print("\nTop candidate outliers:")
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
            f"{row['activity_name']:<20} | "
            f"{int(row['start']):>7} | "
            f"{int(row['end']):>7} | "
            f"{row['avg_distance_to_class']:>10.4f} | "
            f"{row['outlier_zscore']:>10.4f} | "
            f"{str(bool(row['candidate_outlier'])):>5}"
        )


def export_similarity_outputs(similarity_df, outlier_summary_df, output_dir, top_n=20):
    os.makedirs(output_dir, exist_ok=True)

    similarity_path = os.path.join(output_dir, "segment_similarity_scores.csv")
    outlier_summary_path = os.path.join(output_dir, "segment_outlier_summary.csv")
    top_outliers_path = os.path.join(output_dir, "top_candidate_outliers.csv")

    similarity_df.to_csv(similarity_path, index=False)
    outlier_summary_df.to_csv(outlier_summary_path, index=False)

    # --- Top outliers (same logic as print) ---
    if not similarity_df.empty:
        ranked = similarity_df.sort_values("outlier_zscore", ascending=False).head(top_n)

        top_outliers_df = ranked[[
            "experiment",
            "user",
            "activity",
            "activity_name",
            "start",
            "end",
            "avg_distance_to_class",
            "outlier_zscore",
            "candidate_outlier"
        ]].copy()

        # Rename columns for readability
        top_outliers_df.columns = [
            "Exp", "User", "Act", "Activity",
            "Start", "End", "Avg Dist", "z-score", "Flag"
        ]

        top_outliers_df.to_csv(top_outliers_path, index=False,  float_format="%.2f")

    print(f"\nSaved segment similarity scores to: {similarity_path}")
    print(f"Saved outlier summary to: {outlier_summary_path}")
    print(f"Saved top candidate outliers to: {top_outliers_path}")