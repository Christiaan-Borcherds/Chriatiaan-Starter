import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from config import *
from data_overview import parse_exp_user_from_filename, label_to_class


# --------------------------------------------------
# Activity group definitions
# --------------------------------------------------
STATIC_ACTIVITIES = [4, 5, 6]
DYNAMIC_ACTIVITIES = [1, 2, 3]
TRANSITION_ACTIVITIES = [7, 8, 9, 10, 11, 12]

GROUP_TO_ACTIVITIES = {
    "Static": STATIC_ACTIVITIES,
    "Dynamic": DYNAMIC_ACTIVITIES,
    "Transition": TRANSITION_ACTIVITIES,
}

CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# --------------------------------------------------
# Core extraction
# --------------------------------------------------
def extract_activity_segments(dataset_dir, labels_file, acc_pattern, gyro_pattern, target_activity):
    """
    Extract all raw signal segments for a single activity.
    Each returned segment has shape (T, 6):
    [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    """
    acc_files = sorted(glob.glob(os.path.join(dataset_dir, acc_pattern)))
    gyro_files = sorted(glob.glob(os.path.join(dataset_dir, gyro_pattern)))
    labels = np.loadtxt(os.path.join(dataset_dir, labels_file))

    segments = []

    for acc_file, gyro_file in zip(acc_files, gyro_files):
        acc = np.loadtxt(acc_file)
        gyro = np.loadtxt(gyro_file)

        exp_id, user_id = parse_exp_user_from_filename(acc_file)

        for row in labels:
            exp, user, activity, start, end = row

            if int(exp) == exp_id and int(user) == user_id and int(activity) == target_activity:
                start = int(start)
                end = int(end)

                acc_seg = acc[start:end]
                gyro_seg = gyro[start:end]

                segment = np.concatenate((acc_seg, gyro_seg), axis=1)
                segments.append(segment)

    return segments


def get_group_segments(dataset_dir, labels_file, acc_pattern, gyro_pattern, activity_ids):
    """
    Return dictionary:
        {activity_id: [segment1, segment2, ...], ...}
    """
    group_segments = {}

    for activity_id in activity_ids:
        group_segments[activity_id] = extract_activity_segments(
            dataset_dir=dataset_dir,
            labels_file=labels_file,
            acc_pattern=acc_pattern,
            gyro_pattern=gyro_pattern,
            target_activity=activity_id,
        )

    return group_segments


# --------------------------------------------------
# Lightweight summaries for 2.3
# --------------------------------------------------
def summarize_activity_segments_basic(segments):
    """
    Lightweight summary only for behavioural section.
    """
    if len(segments) == 0:
        return {
            "num_segments": 0,
            "avg_duration_seconds": 0.0,
            "min_duration_seconds": 0.0,
            "max_duration_seconds": 0.0,
        }

    lengths = [seg.shape[0] for seg in segments]
    durations = [length / SAMPLING_RATE for length in lengths]

    return {
        "num_segments": len(segments),
        "avg_duration_seconds": round(float(np.mean(durations)), 2),
        "min_duration_seconds": round(float(np.min(durations)), 2),
        "max_duration_seconds": round(float(np.max(durations)), 2),
    }


def print_signal_behaviour_basic_summary(group_name, group_segments):
    """
    Print a lightweight summary for one activity group.
    """
    print("\n" + "-" * 70)
    print(f"{group_name.upper()} ACTIVITIES")
    print("-" * 70)
    print(f"{'ID':>3} | {'Activity':<20} | {'Segments':>8} | {'Avg sec':>8} | {'Min sec':>8} | {'Max sec':>8}")
    print("-" * 70)

    for activity_id, segments in group_segments.items():
        summary = summarize_activity_segments_basic(segments)

        print(
            f"{activity_id:>3} | "
            f"{label_to_class[activity_id]:<20} | "
            f"{summary['num_segments']:>8} | "
            f"{summary['avg_duration_seconds']:>8.2f} | "
            f"{summary['min_duration_seconds']:>8.2f} | "
            f"{summary['max_duration_seconds']:>8.2f}"
        )


# --------------------------------------------------
# Plotting helpers
# --------------------------------------------------
def _plot_single_segment(ax, segment, title, channels=(0, 3)):
    """
    channels default to acc_x and gyro_x
    """
    time = np.arange(segment.shape[0]) / SAMPLING_RATE

    for ch in channels:
        ax.plot(time, segment[:, ch], label=CHANNEL_NAMES[ch])

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.legend()


def plot_activity_examples(group_segments, group_name, num_examples=3, channels=(0, 3)):
    """
    For each activity in a group, show a few example segments.
    Default channels:
        acc_x and gyro_x
    """
    for activity_id, segments in group_segments.items():
        if len(segments) == 0:
            print(f"No segments found for {label_to_class[activity_id]}")
            continue

        n = min(num_examples, len(segments))
        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)
        axes = axes.flatten()

        for i in range(n):
            _plot_single_segment(
                ax=axes[i],
                segment=segments[i],
                title=f"{group_name} - {label_to_class[activity_id]} - Example {i + 1}",
                channels=channels,
            )

        plt.tight_layout()
        plt.show()


def plot_activity_overlay(segments, activity_name, num_examples=5, channel_index=0):
    """
    Overlay several examples of the same activity on one channel.
    Useful to assess within-class consistency visually.
    """
    if len(segments) == 0:
        print(f"No segments found for {activity_name}")
        return

    plt.figure(figsize=(12, 5))

    n = min(num_examples, len(segments))
    for i in range(n):
        seg = segments[i]
        time = np.arange(seg.shape[0]) / SAMPLING_RATE
        plt.plot(time, seg[:, channel_index], alpha=0.8, label=f"Example {i + 1}")

    plt.title(f"{activity_name} overlay ({CHANNEL_NAMES[channel_index]})")
    plt.xlabel("Time (s)")
    plt.ylabel(CHANNEL_NAMES[channel_index])
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_activity_groups_overlay(dataset_dir, groups, num_examples=5, channel_index=0):
    """
    Plot grouped activity overlays in subplot format (3 columns).
    """

    for group_name, activity_ids in groups.items():

        n = len(activity_ids)
        cols = 3
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()  # easier indexing

        for idx, activity_id in enumerate(activity_ids):

            ax = axes[idx]

            activity_segments = get_group_segments(
                dataset_dir=dataset_dir,
                labels_file=LABELS_FILE,
                acc_pattern=ACC_PATTERN,
                gyro_pattern=GYRO_PATTERN,
                activity_ids=[activity_id],
            )[activity_id]

            if len(activity_segments) == 0:
                ax.set_title(f"{label_to_class[activity_id]} (No data)")
                ax.axis("off")
                continue

            n_examples = min(num_examples, len(activity_segments))

            for i in range(n_examples):
                seg = activity_segments[i]
                time = np.arange(seg.shape[0]) / SAMPLING_RATE
                ax.plot(time, seg[:, channel_index], alpha=0.7)

            ax.set_title(label_to_class[activity_id])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(CHANNEL_NAMES[channel_index])

        # Hide unused subplots
        for j in range(len(activity_ids), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"{group_name} Activities Overlay ({CHANNEL_NAMES[channel_index]})",
            fontsize=16
        )

        plt.tight_layout()
        plt.show()


def plot_group_representatives(dataset_dir, labels_file, acc_pattern, gyro_pattern, channels=(0, 3)):
    """
    One representative activity per group:
        Static     -> Standing (5)
        Dynamic    -> Walking (1)
        Transition -> Sit-to-Stand (8)
    """
    representatives = {
        "Static": 5,
        "Dynamic": 1,
        "Transition": 8,
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), squeeze=False)
    axes = axes.flatten()

    for ax, (group_name, activity_id) in zip(axes, representatives.items()):
        segments = extract_activity_segments(
            dataset_dir=dataset_dir,
            labels_file=labels_file,
            acc_pattern=acc_pattern,
            gyro_pattern=gyro_pattern,
            target_activity=activity_id,
        )

        if len(segments) == 0:
            ax.set_title(f"{group_name}: no data found")
            continue

        _plot_single_segment(
            ax=ax,
            segment=segments[0],
            title=f"{group_name} representative - {label_to_class[activity_id]}",
            channels=channels,
        )

    plt.tight_layout()
    plt.show()


def plot_comparison_panel(dataset_dir, labels_file, acc_pattern, gyro_pattern, channel_index=0):
    """
    Compare one representative activity from each group on the same chosen channel.
    """
    representatives = {
        "Static": 5,       # Standing
        "Dynamic": 1,      # Walking
        "Transition": 8,   # Sit-to-Stand
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), squeeze=False)
    axes = axes.flatten()

    for ax, (group_name, activity_id) in zip(axes, representatives.items()):
        segments = extract_activity_segments(
            dataset_dir=dataset_dir,
            labels_file=labels_file,
            acc_pattern=acc_pattern,
            gyro_pattern=gyro_pattern,
            target_activity=activity_id,
        )

        if len(segments) == 0:
            ax.set_title(f"{group_name}: no data found")
            continue

        seg = segments[0]
        time = np.arange(seg.shape[0]) / SAMPLING_RATE

        ax.plot(time, seg[:, channel_index])
        ax.set_title(f"{group_name} comparison - {label_to_class[activity_id]} ({CHANNEL_NAMES[channel_index]})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(CHANNEL_NAMES[channel_index])

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Top-level execution helper for 2.3
# --------------------------------------------------
def run_signal_behaviour_section(dataset_dir, labels_file, acc_pattern, gyro_pattern):
    print("\n" + "=" * 70)
    print("2.3 SIGNAL BEHAVIOUR ACROSS ACTIVITIES")
    print("=" * 70)

    for group_name, activity_ids in GROUP_TO_ACTIVITIES.items():
        group_segments = get_group_segments(
            dataset_dir=dataset_dir,
            labels_file=labels_file,
            acc_pattern=acc_pattern,
            gyro_pattern=gyro_pattern,
            activity_ids=activity_ids,
        )

        print_signal_behaviour_basic_summary(group_name, group_segments)

    print("\nInterpretation focus:")
    print("  - Static activities should appear smoother and less fluctuating.")
    print("  - Dynamic activities should appear more rhythmic and oscillatory.")
    print("  - Transitions should appear short, burst-like, and less regular.")
    print("  - This visual behaviour helps explain why transitions are harder to classify.")