from config import *

import os
import glob
import numpy as np
from collections import defaultdict
import pandas as pd


def get_sensor_file_lists(dataset_dir, acc_pattern, gyro_pattern):
    """
    Return sorted lists of accelerometer and gyroscope files.
    """
    acc_files = sorted(glob.glob(os.path.join(dataset_dir, acc_pattern)))
    gyro_files = sorted(glob.glob(os.path.join(dataset_dir, gyro_pattern)))
    return acc_files, gyro_files


def load_example_file(file_path):
    """
    Load a single sensor file and return its numpy array and shape.
    """
    data = np.loadtxt(file_path)
    return data, data.shape


def inspect_data_format(dataset_dir, acc_pattern, gyro_pattern):
    """
    Inspect one accelerometer and one gyroscope file to confirm data format.
    """
    acc_files, gyro_files = get_sensor_file_lists(dataset_dir, acc_pattern, gyro_pattern)

    if not acc_files or not gyro_files:
        raise FileNotFoundError("Could not find accelerometer or gyroscope files.")

    acc_data, acc_shape = load_example_file(acc_files[0])
    gyro_data, gyro_shape = load_example_file(gyro_files[0])

    overview = {
        "example_acc_file": os.path.basename(acc_files[0]),
        "example_gyro_file": os.path.basename(gyro_files[0]),
        "acc_shape": acc_shape,
        "gyro_shape": gyro_shape,
        "acc_num_axes": acc_shape[1] if len(acc_shape) > 1 else 1,
        "gyro_num_axes": gyro_shape[1] if len(gyro_shape) > 1 else 1,
        "acc_first_rows": acc_data[:5],
        "gyro_first_rows": gyro_data[:5],
    }

    return overview


def parse_exp_user_from_filename(file_path):
    """
    Extract experiment ID and user ID from filenames like:
    acc_exp01_user01.txt
    gyro_exp02_user14.txt
    """
    name = os.path.basename(file_path)
    parts = name.replace(".txt", "").split("_")

    exp_id = int(parts[1].replace("exp", ""))
    user_id = int(parts[2].replace("user", ""))

    return exp_id, user_id


def get_dataset_scale(dataset_dir, acc_pattern, gyro_pattern, sampling_rate=50):
    """
    Compute high-level dataset scale information from raw sensor files.
    """
    acc_files, gyro_files = get_sensor_file_lists(dataset_dir, acc_pattern, gyro_pattern)

    if len(acc_files) != len(gyro_files):
        raise ValueError("Mismatch between number of accelerometer and gyroscope files.")

    total_acc_samples = 0
    total_gyro_samples = 0
    total_recordings = len(acc_files)

    subject_sample_counts = defaultdict(int)
    subject_recording_counts = defaultdict(int)
    experiment_sample_counts = defaultdict(int)

    per_file_info = []

    for acc_file, gyro_file in zip(acc_files, gyro_files):
        acc_data = np.loadtxt(acc_file)
        gyro_data = np.loadtxt(gyro_file)

        if acc_data.shape[0] != gyro_data.shape[0]:
            raise ValueError(
                f"Sample mismatch between:\n{acc_file}\nand\n{gyro_file}"
            )

        exp_id, user_id = parse_exp_user_from_filename(acc_file)
        n_samples = acc_data.shape[0]

        total_acc_samples += n_samples
        total_gyro_samples += n_samples

        subject_sample_counts[user_id] += n_samples
        subject_recording_counts[user_id] += 1
        experiment_sample_counts[exp_id] += n_samples

        per_file_info.append({
            "file": os.path.basename(acc_file),
            "experiment_id": exp_id,
            "user_id": user_id,
            "num_samples": n_samples,
            "duration_seconds": n_samples / sampling_rate
        })

    total_duration_seconds = total_acc_samples / sampling_rate
    total_duration_minutes = total_duration_seconds / 60
    total_duration_hours = total_duration_seconds / 3600

    scale_info = {
        "num_acc_files": len(acc_files),
        "num_gyro_files": len(gyro_files),
        "num_recordings": total_recordings,
        "num_subjects": len(subject_sample_counts),
        "num_experiments": len(experiment_sample_counts),
        "total_acc_samples": total_acc_samples,
        "total_gyro_samples": total_gyro_samples,
        "total_duration_seconds": total_duration_seconds,
        "total_duration_minutes": total_duration_minutes,
        "total_duration_hours": total_duration_hours,
        "subject_sample_counts": dict(sorted(subject_sample_counts.items())),
        "subject_recording_counts": dict(sorted(subject_recording_counts.items())),
        "experiment_sample_counts": dict(sorted(experiment_sample_counts.items())),
        "per_file_info": per_file_info,
    }

    return scale_info


def load_labels(dataset_dir, labels_file):
    """
    Load labels.txt and return as numpy array.
    """
    labels_path = os.path.join(dataset_dir, labels_file)
    labels = np.loadtxt(labels_path)
    return labels


def inspect_labels_format(dataset_dir, labels_file):
    """
    Inspect labels.txt structure.
    Expected columns:
    [experiment_id, user_id, activity_id, start, end]
    """
    labels = load_labels(dataset_dir, labels_file)

    overview = {
        "labels_shape": labels.shape,
        "num_label_rows": labels.shape[0],
        "num_label_columns": labels.shape[1] if len(labels.shape) > 1 else 1,
        "first_rows": labels[:5],
        "unique_subjects_in_labels": np.unique(labels[:, 1]).astype(int),
        "unique_experiments_in_labels": np.unique(labels[:, 0]).astype(int),
        "unique_activities_in_labels": np.unique(labels[:, 2]).astype(int),
    }

    return overview


def print_data_format_report(format_info, labels_info):
    print("\n" + "=" * 60)
    print("2.1.3 DATA FORMAT")
    print("=" * 60)

    print(f"Example accelerometer file : {format_info['example_acc_file']}")
    print(f"Example gyroscope file     : {format_info['example_gyro_file']}")
    print(f"Accelerometer shape        : {format_info['acc_shape']}")
    print(f"Gyroscope shape            : {format_info['gyro_shape']}")
    print(f"Accelerometer axes         : {format_info['acc_num_axes']}")
    print(f"Gyroscope axes             : {format_info['gyro_num_axes']}")

    print("\nFirst 5 rows of accelerometer data:")
    print(format_info["acc_first_rows"])

    print("\nFirst 5 rows of gyroscope data:")
    print(format_info["gyro_first_rows"])

    print("\nLabels shape:")
    print(labels_info["labels_shape"])

    print("\nLabel format:")
    print("  Column 1 -> Experiment ID")
    print("  Column 2 -> Subject ID")
    print("  Column 3 -> Activity ID")
    print("  Column 4 -> Start sample index")
    print("  Column 5 -> End sample index")

    print("\nFirst 5 rows of labels:")
    print(labels_info["first_rows"])

    print("\nUnique subjects in labels:")
    print(labels_info["unique_subjects_in_labels"])

    print("\nUnique experiments in labels:")
    print(labels_info["unique_experiments_in_labels"])

    print("\nUnique activity IDs in labels:")
    print(labels_info["unique_activities_in_labels"])



    # print(f"Example accelerometer file : {format_info['example_acc_file']}")
    # print(f"Example gyroscope file     : {format_info['example_gyro_file']}")
    # print(f"Accelerometer shape        : {format_info['acc_shape']}")
    # print(f"Gyroscope shape            : {format_info['gyro_shape']}")
    # print(f"Accelerometer axes         : {format_info['acc_num_axes']}")
    # print(f"Gyroscope axes             : {format_info['gyro_num_axes']}")
    #
    # print("\nFirst 5 rows of accelerometer data:")
    # print(format_info["acc_first_rows"])
    #
    # print("\nFirst 5 rows of gyroscope data:")
    # print(format_info["gyro_first_rows"])
    #
    # print("\nLabels shape:")
    # print(labels_info["labels_shape"])
    #
    # print("\nFirst 5 rows of labels:")
    # print(labels_info["first_rows"])
    #
    # print("\nUnique subjects in labels:")
    # print(labels_info["unique_subjects_in_labels"])
    #
    # print("\nUnique experiments in labels:")
    # print(labels_info["unique_experiments_in_labels"])
    #
    # print("\nUnique activity IDs in labels:")
    # print(labels_info["unique_activities_in_labels"])


def print_dataset_scale_report(scale_info, print_per_subject=True, print_per_experiment=True, print_per_file=False):
    total_samples = scale_info["total_acc_samples"]
    total_recordings = scale_info["num_recordings"]

    num_subjects = scale_info["num_subjects"]

    per_file_samples = [item["num_samples"] for item in scale_info["per_file_info"]]
    per_file_durations = [item["duration_seconds"] for item in scale_info["per_file_info"]]

    avg_experiments_per_subject = total_recordings / num_subjects
    avg_samples_per_recording = sum(per_file_samples) / total_recordings if total_recordings > 0 else 0
    avg_duration_per_recording = sum(per_file_durations) / total_recordings if total_recordings > 0 else 0

    min_samples_per_recording = min(per_file_samples) if per_file_samples else 0
    max_samples_per_recording = max(per_file_samples) if per_file_samples else 0

    min_duration_per_recording = min(per_file_durations) if per_file_durations else 0
    max_duration_per_recording = max(per_file_durations) if per_file_durations else 0

    exp_counts = list(scale_info["subject_recording_counts"].values())
    min_exp = min(exp_counts)
    max_exp = max(exp_counts)

    print("\n" + "=" * 60)
    print("2.1.4 DATASET SCALE")
    print("=" * 60)

    print(f"Number of accelerometer files : {scale_info['num_acc_files']}")
    print(f"Number of gyroscope files     : {scale_info['num_gyro_files']}")
    print(f"Number of recordings          : {scale_info['num_recordings']}")
    print(f"Number of subjects            : {scale_info['num_subjects']}")
    print(f"Number of experiments         : {scale_info['num_experiments']}")
    print(f"Total accelerometer samples   : {scale_info['total_acc_samples']}")
    print(f"Total gyroscope samples       : {scale_info['total_gyro_samples']}")
    print(f"Total duration (seconds)      : {scale_info['total_duration_seconds']:.2f}")
    print(f"Total duration (minutes)      : {scale_info['total_duration_minutes']:.2f}")
    print(f"Total duration (hours)        : {scale_info['total_duration_hours']:.2f}")

    print("\nExperiment summary:")
    print(f"  Average samples/Experiment   : {avg_samples_per_recording:.2f}")
    print(f"  Average duration/Experiment  : {avg_duration_per_recording:.2f} s "
          f"({avg_duration_per_recording / 60:.2f} min)")
    print(f"  Min samples/Experiment       : {min_samples_per_recording}")
    print(f"  Max samples/Experiment       : {max_samples_per_recording}")
    print(f"  Min duration/Experiment      : {min_duration_per_recording:.2f} s")
    print(f"  Max duration/Experiment      : {max_duration_per_recording:.2f} s")
    print(f"  Average experiments per subject: {avg_experiments_per_subject:.2f}")
    print(f"  Min experiments per subject    : {min_exp}")
    print(f"  Max experiments per subject    : {max_exp}")

    # if print_per_subject:
    #     print("\nSamples per subject:")
    #     for subject_id, count in scale_info["subject_sample_counts"].items():
    #         recordings = scale_info["subject_recording_counts"][subject_id]
    #         print(f"  Subject {subject_id:02d}: {count} samples across {recordings} recording(s)")
    if print_per_subject:
        print("\nSamples per subject:")
        print(
            f"{'Subject':>8} | {'Samples':>12} | {'% Total':>8} | {'Recordings':>10} | {'Seconds':>10} | {'Minutes':>10}")
        print("-" * 72)

        for subject_id, count in scale_info["subject_sample_counts"].items():
            recordings = scale_info["subject_recording_counts"][subject_id]
            percent_total = round((count / total_samples) * 100, 2) if total_samples > 0 else 0
            seconds = count / SAMPLING_RATE
            minutes = seconds / 60

            print(
                f"{subject_id:>8} | {count:>12} | {percent_total:>7.2f}% | {recordings:>10} | {seconds:>10.2f} | {minutes:>10.2f}")

    # if print_per_experiment:
    #     print("\nSamples per experiment:")
    #     for exp_id, count in scale_info["experiment_sample_counts"].items():
    #         print(f"  Experiment {exp_id:02d}: {count} samples")
    if print_per_experiment:
        print("\nSamples per experiment:")
        print(f"{'Experiment':>10} | {'Samples':>12} | {'% Total':>8} | {'Seconds':>10} | {'Minutes':>10}")
        print("-" * 62)

        for exp_id, count in scale_info["experiment_sample_counts"].items():
            percent_total = (count / total_samples) * 100 if total_samples > 0 else 0
            seconds = count / SAMPLING_RATE
            minutes = seconds / 60

            print(f"{exp_id:>10} | {count:>12} | {percent_total:>7.2f}% | {seconds:>10.2f} | {minutes:>10.2f}")

    # if print_per_file:
    #     print("\nPer-file information:")
    #     for item in scale_info["per_file_info"]:
    #         print(
    #             f"  {item['file']} | "
    #             f"Exp {item['experiment_id']:02d} | "
    #             f"User {item['user_id']:02d} | "
    #             f"{item['num_samples']} samples | "
    #             f"{item['duration_seconds']:.2f} s"
    #         )
    if print_per_file:
        print("\nPer-file information:")
        print(f"{'File':<28} | {'Exp':>3} | {'User':>4} | {'Samples':>8} | {'% Total':>8} | {'Seconds':>9}")
        print("-" * 80)

        for item in scale_info["per_file_info"]:
            percent_total = (item['num_samples'] / total_samples) * 100 if total_samples > 0 else 0
            print(
                f"{item['file']:<28} | "
                f"{item['experiment_id']:>3} | "
                f"{item['user_id']:>4} | "
                f"{item['num_samples']:>8} | "
                f"{percent_total:>7.2f}% | "
                f"{item['duration_seconds']:>9.2f}"
            )




def export_dataset_scale_to_csv(scale_info, output_dir, sampling_rate=50, export_per_file=False):
    """
    Export dataset scale information to CSV files.

    Files created:
    - dataset_scale_summary.csv
    - dataset_scale_per_subject.csv
    - dataset_scale_per_experiment.csv
    - dataset_scale_per_file.csv (optional)
    """
    os.makedirs(output_dir, exist_ok=True)

    total_samples = scale_info["total_acc_samples"]
    total_recordings = scale_info["num_recordings"]

    per_file_samples = [item["num_samples"] for item in scale_info["per_file_info"]]
    per_file_durations = [item["duration_seconds"] for item in scale_info["per_file_info"]]

    avg_samples_per_recording = sum(per_file_samples) / total_recordings if total_recordings > 0 else 0
    avg_duration_per_recording = sum(per_file_durations) / total_recordings if total_recordings > 0 else 0
    min_samples_per_recording = min(per_file_samples) if per_file_samples else 0
    max_samples_per_recording = max(per_file_samples) if per_file_samples else 0
    min_duration_per_recording = min(per_file_durations) if per_file_durations else 0
    max_duration_per_recording = max(per_file_durations) if per_file_durations else 0

    # -------------------------
    # 1) Overall summary
    # -------------------------
    summary_df = pd.DataFrame([{
        "num_acc_files": scale_info["num_acc_files"],
        "num_gyro_files": scale_info["num_gyro_files"],
        "num_recordings": scale_info["num_recordings"],
        "num_subjects": scale_info["num_subjects"],
        "num_experiments": scale_info["num_experiments"],
        "total_acc_samples": scale_info["total_acc_samples"],
        "total_gyro_samples": scale_info["total_gyro_samples"],
        "total_duration_seconds": scale_info["total_duration_seconds"],
        "total_duration_minutes": scale_info["total_duration_minutes"],
        "total_duration_hours": scale_info["total_duration_hours"],
        "avg_samples_per_recording": avg_samples_per_recording,
        "avg_duration_per_recording_seconds": avg_duration_per_recording,
        "avg_duration_per_recording_minutes": avg_duration_per_recording / 60,
        "min_samples_per_recording": min_samples_per_recording,
        "max_samples_per_recording": max_samples_per_recording,
        "min_duration_per_recording_seconds": min_duration_per_recording,
        "max_duration_per_recording_seconds": max_duration_per_recording,
    }])

    summary_df.to_csv(os.path.join(output_dir, "dataset_scale_summary.csv"), index=False)

    # -------------------------
    # 2) Per subject
    # -------------------------
    subject_rows = []
    for subject_id, count in scale_info["subject_sample_counts"].items():
        recordings = scale_info["subject_recording_counts"][subject_id]
        percent_total = round((count / total_samples) * 100, 2) if total_samples > 0 else 0
        seconds = count / sampling_rate
        minutes = round(seconds / 60,2)

        subject_rows.append({
            "subject": subject_id,
            "samples": count,
            "percent_total": percent_total,
            "recordings": recordings,
            "seconds": seconds,
            "minutes": minutes,
        })

    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(os.path.join(output_dir, "dataset_scale_per_subject.csv"), index=False)

    # -------------------------
    # 3) Per experiment
    # -------------------------
    experiment_rows = []
    for exp_id, count in scale_info["experiment_sample_counts"].items():
        percent_total = round((count / total_samples) * 100, 2) if total_samples > 0 else 0
        seconds = count / sampling_rate
        minutes = round(seconds / 60, 2)

        experiment_rows.append({
            "experiment": exp_id,
            "samples": count,
            "percent_total": percent_total,
            "seconds": seconds,
            "minutes": minutes,
        })

    experiment_df = pd.DataFrame(experiment_rows)
    experiment_df.to_csv(os.path.join(output_dir, "dataset_scale_per_experiment.csv"), index=False)

    # -------------------------
    # 4) Per file (optional)
    # -------------------------
    if export_per_file:
        file_rows = []
        for item in scale_info["per_file_info"]:
            percent_total = (item["num_samples"] / total_samples) * 100 if total_samples > 0 else 0

            file_rows.append({
                "file": item["file"],
                "experiment_id": item["experiment_id"],
                "user_id": item["user_id"],
                "samples": item["num_samples"],
                "percent_total": percent_total,
                "duration_seconds": item["duration_seconds"],
                "duration_minutes": item["duration_seconds"] / 60,
            })

        file_df = pd.DataFrame(file_rows)
        file_df.to_csv(os.path.join(output_dir, "dataset_scale_per_file.csv"), index=False)

    print(f"\nCSV files exported to: {output_dir}")