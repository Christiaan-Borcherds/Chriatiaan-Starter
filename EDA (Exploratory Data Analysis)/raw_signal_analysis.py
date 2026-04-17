import numpy as np
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



from config import *
from data_overview import parse_exp_user_from_filename, label_to_class


activity_colors = {
    1: 'lightblue',
    2: 'deepskyblue',
    3: 'dodgerblue',
    4: 'lightgreen',
    5: 'mediumseagreen',
    6: 'plum',
    7: 'gold',
    8: 'orange',
    9: 'salmon',
    10: 'tomato',
    11: 'khaki',
    12: 'orchid',
}


def analyze_signal_composition(dataset_dir, acc_pattern, gyro_pattern):
    acc_files = sorted(glob.glob(os.path.join(dataset_dir, acc_pattern)))
    gyro_files = sorted(glob.glob(os.path.join(dataset_dir, gyro_pattern)))

    all_acc = []
    all_gyro = []

    # Load ALL data (or subset if needed)
    for acc_file, gyro_file in zip(acc_files, gyro_files):
        acc = np.loadtxt(acc_file)
        gyro = np.loadtxt(gyro_file)

        all_acc.append(acc)
        all_gyro.append(gyro)

    all_acc = np.vstack(all_acc)
    all_gyro = np.vstack(all_gyro)

    # Combine into 6D signal
    all_data = np.concatenate((all_acc, all_gyro), axis=1)

    stats = {}

    # Per-channel stats
    channel_names = [
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z"
    ]

    for i, name in enumerate(channel_names):
        channel = all_data[:, i]

        stats[name] = {
            "min": round(np.min(channel),4),
            "max": round(np.max(channel),4),
            "mean": round(np.mean(channel),4),
            "std": round(np.std(channel),4),
        }

    return stats


def print_signal_composition(stats):
    print("\n" + "=" * 70)
    print("2.2.1 SIGNAL COMPOSITION")
    print("=" * 70)

    print(f"{'Channel':<10} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std':>10}")
    print("-" * 65)

    for channel, values in stats.items():
        print(f"{channel:<10} | "
              f"{values['min']:>10.4f} | "
              f"{values['max']:>10.4f} | "
              f"{values['mean']:>10.4f} | "
              f"{values['std']:>10.4f}")




def export_signal_composition(stats, output_dir):
    df = pd.DataFrame(stats).T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "channel"}, inplace=True)

    output_path = os.path.join(output_dir, "signal_composition.csv")
    df.to_csv(output_path, index=False)

    print(f"\nSignal composition exported to: {output_path}")



import numpy as np
import glob
import os


#2.2.2
def analyze_temporal_structure(dataset_dir, acc_pattern, gyro_pattern, sampling_rate=SAMPLING_RATE):
    acc_files = sorted(glob.glob(os.path.join(dataset_dir, acc_pattern)))
    # gyro has the same specs

    recording_info = []

    for acc_file in acc_files:
        acc = np.loadtxt(acc_file)

        num_samples = acc.shape[0]
        duration_seconds = num_samples / sampling_rate

        recording_info.append({
            "file": os.path.basename(acc_file),
            "num_samples": num_samples,
            "duration_seconds": duration_seconds
        })

    return recording_info

# each labeled partition
def analyze_label_intervals(dataset_dir, labels_file, sampling_rate=SAMPLING_RATE):
    labels_path = os.path.join(dataset_dir, labels_file)
    labels = np.loadtxt(labels_path)

    intervals = []

    for row in labels:
        exp_id, user_id, activity, start, end = row
        duration = end - start
        duration_seconds = duration / sampling_rate

        intervals.append({
            "experiment": int(exp_id),
            "user": int(user_id),
            "activity": int(activity),
            "samples": int(duration),
            "seconds": duration_seconds
        })

    return intervals



def check_signal_continuity(dataset_dir, acc_pattern):
    acc_files = sorted(glob.glob(os.path.join(dataset_dir, acc_pattern)))

    continuity_report = []

    for acc_file in acc_files:
        acc = np.loadtxt(acc_file)

        # Check for NaNs
        has_nan = np.isnan(acc).any()

        continuity_report.append({
            "file": os.path.basename(acc_file),
            "num_samples": acc.shape[0],
            "has_nan": has_nan
        })

    return continuity_report





def plot_example_signal(dataset_dir, exp=1, user=1):
    acc_file = os.path.join(dataset_dir, f"acc_exp{str(exp).zfill(2)}_user{str(user).zfill(2)}.txt")
    gyro_file = os.path.join(dataset_dir, f"gyro_exp{str(exp).zfill(2)}_user{str(user).zfill(2)}.txt")

    acc = np.loadtxt(acc_file)
    gyro = np.loadtxt(gyro_file)

    time = np.arange(acc.shape[0]) / 50  # seconds

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, acc[:, 0], label='acc_x')
    plt.plot(time, acc[:, 1], label='acc_y')
    plt.plot(time, acc[:, 2], label='acc_z')
    plt.title("Accelerometer Signal")
    plt.xlabel("Time (s)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, gyro[:, 0], label='gyro_x')
    plt.plot(time, gyro[:, 1], label='gyro_y')
    plt.plot(time, gyro[:, 2], label='gyro_z')
    plt.title("Gyroscope Signal")
    plt.xlabel("Time (s)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_signal_with_labels_single(dataset_dir, labels_file, exp=1, user=1, padding_const = 0.0):
    acc_file = os.path.join(dataset_dir, f"acc_exp{str(exp).zfill(2)}_user{str(user).zfill(2)}.txt")
    gyro_file = os.path.join(dataset_dir, f"gyro_exp{str(exp).zfill(2)}_user{str(user).zfill(2)}.txt")
    acc = np.loadtxt(acc_file)
    gyro = np.loadtxt(gyro_file)

    labels_path = os.path.join(dataset_dir, labels_file)
    labels = np.loadtxt(labels_path)

    time = np.arange(acc.shape[0]) / SAMPLING_RATE

    fig, axes = plt.subplots(
        3, 1,
        sharex=True,
        figsize=(14, 8),
        gridspec_kw={'height_ratios': [1.5, 2, 2]}
    )
    ax_label, ax_acc, ax_gyro = axes

    # ---------- TOP LABEL BAR ----------
    ax_label.set_ylim(0, 1)
    ax_label.set_yticks([])
    ax_label.set_ylabel("Activity")
    ax_label.spines[['left', 'right', 'top']].set_visible(False)

    # ---------- ACC ----------
    ax_acc.plot(time, acc[:, 0], label="acc_x")
    ax_acc.plot(time, acc[:, 1], label="acc_y")
    ax_acc.plot(time, acc[:, 2], label="acc_z")

    acc_ymin, acc_ymax = acc.min(), acc.max()
    acc_padding = padding_const * (acc_ymax - acc_ymin)
    ax_acc.set_ylim(acc_ymin, acc_ymax + acc_padding)

    # ---------- GYRO ----------
    ax_gyro.plot(time, gyro[:, 0], label="gyro_x")
    ax_gyro.plot(time, gyro[:, 1], label="gyro_y")
    ax_gyro.plot(time, gyro[:, 2], label="gyro_z")

    gyro_ymin, gyro_ymax = gyro.min(), gyro.max()
    gyro_padding = padding_const * (gyro_ymax - gyro_ymin)
    ax_gyro.set_ylim(gyro_ymin, gyro_ymax + gyro_padding)

    used_activities = set()

    for row in labels:
        exp_id, user_id, activity, start, end = row

        if int(exp_id) == exp and int(user_id) == user:
            activity = int(activity)
            start_t = start / SAMPLING_RATE
            end_t = end / SAMPLING_RATE
            mid_t = (start_t + end_t) / 2
            color = activity_colors.get(activity, 'gray')

            ax_label.axvspan(start_t, end_t, color=color, alpha=0.30)
            ax_acc.axvspan(start_t, end_t, color=color, alpha=0.20)
            ax_gyro.axvspan(start_t, end_t, color=color, alpha=0.20)

            used_activities.add(activity)

            # label ONLY on top axis
            ax_label.text(
                mid_t,
                0.5,
                label_to_class[activity],
                rotation=90,
                ha='center',
                va='center',
                fontsize=8
            )


    ax_acc.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    ax_gyro.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.0)


    # plt.gca().add_artist(signal_legend)

    activity_patches = [
        Patch(
            facecolor=activity_colors[a],
            edgecolor='none',
            alpha=0.30,
            label=label_to_class[a]
        )
        for a in sorted(used_activities)
    ]

    fig.legend(
        handles=activity_patches,
        loc='center left',
        bbox_to_anchor=(0.845, 0.81),
        fontsize=8
    )

#     plt.legend(
#     handles=activity_patches,
#     loc='upper left',
#     bbox_to_anchor=(1.02, 0.7),
#     borderaxespad=0.0,
#     fontsize=8
# )
    # ---------- LABELS ----------
    ax_acc.set_ylabel("Acceleration")
    ax_gyro.set_ylabel("Angular velocity")
    ax_gyro.set_xlabel("Time (s)")

    fig.suptitle(
        f"Acceleration and Gyroscope Signals with Activity Intervals (Exp {exp}, User {user})",
        fontsize=16,
        fontweight='bold'
    )
    # ---------- LAYOUT ----------
    plt.tight_layout(rect=[0, 0, 0.93, 0.95])

    # plt.title(f"{text} Signal with Activity Intervals (Exp {exp}, User {user})")
    # plt.xlabel("Time (s)")
    # plt.tight_layout(rect=[0, 0, 0.98, 1])
    # plt.legend()
    # signal_type = "acc"  # or "gyro", "filtered", etc.
    filename = f"signal_labels_exp{str(exp).zfill(2)}_user{str(user).zfill(2)}.png"
    plt.savefig(os.path.join(SignalLabelSplitDir_2_2, filename))
    plt.show()


def get_all_exp_user_pairs(dataset_dir):
    pairs = set()

    for file in os.listdir(dataset_dir):
        if file.startswith("acc_exp") and file.endswith(".txt"):
            exp_id, user_id = parse_exp_user_from_filename(file)
            pairs.add((exp_id, user_id))

    return sorted(pairs)

def plot_signal_with_labels_all(dataset_dir, labels_file, padding_constant=0.0):
    pairs = get_all_exp_user_pairs(dataset_dir)

    print(f"Found {len(pairs)} experiment-user combinations\n")

    for exp, user in pairs:
        print(f"Plotting Exp {exp}, User {user}")
        plot_signal_with_labels_single(dataset_dir, labels_file, exp, user, padding_constant)




def print_temporal_structure_report(recording_info, intervals, continuity_report, sampling_rate=50):
    print("\n" + "=" * 70)
    print("2.2.2 TEMPORAL STRUCTURE")
    print("=" * 70)

    # -------------------------
    # Recording-level summary
    # -------------------------
    num_recordings = len(recording_info)
    recording_samples = [item["num_samples"] for item in recording_info]
    recording_durations = [item["duration_seconds"] for item in recording_info]

    total_samples = sum(recording_samples)
    total_duration_seconds = sum(recording_durations)

    print("\nRecording Summary:")
    print(f"  Number of recordings        : {num_recordings}")
    print(f"  Sampling rate              : {sampling_rate} Hz")
    print(f"  Total samples              : {total_samples}")
    print(f"  Total duration             : {total_duration_seconds:.2f} s ({total_duration_seconds / 60:.2f} min)")
    print(f"  Average samples/recording  : {np.mean(recording_samples):.2f}")
    print(f"  Average duration/recording : {np.mean(recording_durations):.2f} s")
    print(f"  Min samples/recording      : {np.min(recording_samples)}")
    print(f"  Max samples/recording      : {np.max(recording_samples)}")
    print(f"  Min duration/recording     : {np.min(recording_durations):.2f} s")
    print(f"  Max duration/recording     : {np.max(recording_durations):.2f} s")

    # -------------------------
    # Interval-level summary
    # -------------------------
    interval_samples = [item["samples"] for item in intervals]
    interval_durations = [item["seconds"] for item in intervals]
    unique_activities = sorted(list(set(item["activity"] for item in intervals)))

    print("\nLabel Interval Summary:")
    print(f"  Number of labelled intervals : {len(intervals)}")
    print(f"  Number of activity classes   : {len(unique_activities)}")
    print(f"  Activity IDs present         : {unique_activities}")
    print(f"  Average interval length      : {np.mean(interval_samples):.2f} samples ({np.mean(interval_durations):.2f} s)")
    print(f"  Min interval length          : {np.min(interval_samples)} samples ({np.min(interval_durations):.2f} s)")
    print(f"  Max interval length          : {np.max(interval_samples)} samples ({np.max(interval_durations):.2f} s)")

    # -------------------------
    # Continuity summary
    # -------------------------
    num_with_nan = sum(1 for item in continuity_report if item["has_nan"])
    num_without_nan = len(continuity_report) - num_with_nan

    print("\nContinuity Check:")
    print(f"  Recordings checked         : {len(continuity_report)}")
    print(f"  Recordings without NaNs    : {num_without_nan}")
    print(f"  Recordings with NaNs       : {num_with_nan}")

    if num_with_nan > 0:
        print("\nFiles containing NaNs:")
        for item in continuity_report:
            if item["has_nan"]:
                print(f"  {item['file']}")




def print_interval_summary_by_activity(intervals):
    from collections import defaultdict

    activity_counts = defaultdict(int)
    activity_samples = defaultdict(int)
    activity_seconds = defaultdict(float)

    for item in intervals:
        activity = item["activity"]
        activity_counts[activity] += 1
        activity_samples[activity] += item["samples"]
        activity_seconds[activity] += item["seconds"]

    total_intervals = len(intervals)

    print("\nInterval summary by activity:")
    print(f"{'Activity':>8} | {'Intervals':>10} | {'% Total':>8} | {'Samples':>12} | {'Seconds':>10} | {'Avg sec/int':>12}")
    print("-" * 78)

    rows = []

    for activity in sorted(activity_counts.keys()):
        count = activity_counts[activity]
        samples = activity_samples[activity]
        seconds = activity_seconds[activity]
        percent_total = round((count / total_intervals) * 100, 2) if total_intervals > 0 else 0.0
        avg_sec = seconds / count if count > 0 else 0.0

        print(f"{activity:>8} | {count:>10} | {percent_total:>7.2f}% | {samples:>12} | {seconds:>10.2f} | {avg_sec:>12.2f}")

        rows.append({
            "activity": activity,
            "intervals": count,
            "percent_total": percent_total,
            "samples": samples,
            "seconds": round(seconds, 2),
            "avg_seconds_per_interval": round(avg_sec, 2),
        })

    df = pd.DataFrame(rows)
    output_path = os.path.join(DatasetOverview_OutputDir_2_1, "interval_summary_by_activity.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved interval summary to: {output_path}")



