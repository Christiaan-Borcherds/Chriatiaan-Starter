from config import *
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
import pandas as pd


def get_split(user_id):
    if user_id in TRAIN_SUBJECTS:
        return 0  # train
    elif user_id in VAL_SUBJECTS:
        return 1  # val
    elif user_id in TEST_SUBJECTS:
        return 2  # test
    else:
        return -1



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


def extract_segments(dataset_dir, labels):
    segments = []

    for acc_file, gyro_file in zip(acc_files, gyro_files):
        acc = np.loadtxt(acc_file)
        gyro = np.loadtxt(gyro_file)

        exp_id, user_id = parse_exp_user_from_filename(acc_file)

        for row in labels:
            exp, user, act, start, end = row

            if int(exp) == exp_id and int(user) == user_id:
                segment = {
                    "exp": exp_id,
                    "user": user_id,
                    "activity": int(act),
                    "start": int(start),
                    "end": int(end),
                    "acc": acc[int(start):int(end)],
                    "gyro": gyro[int(start):int(end)]
                }
                segments.append(segment)

    return segments


def get_activity_overlap(activity, overlap):
    """
    overlap can be:
    - int: same overlap for all classes
    - list: class-specific overlap, index activity-1
    """
    if isinstance(overlap, int):
        return overlap

    return overlap[activity - 1]

def create_windows_from_segment(segment, window_size, overlap):
    acc = segment["acc"]
    gyro = segment["gyro"]
    activity = segment["activity"]

    activity_overlap = get_activity_overlap(activity, overlap)
    stride = window_size - activity_overlap

    if stride <= 0:
        raise ValueError(
            f"Invalid overlap {activity_overlap} for window size {window_size}. "
            "Overlap must be smaller than window size."
        )

    windows = []
    start = 0

    while start + window_size <= len(acc):
        acc_win = acc[start:start + window_size]
        gyro_win = gyro[start:start + window_size]

        # windows.append({
        #     "acc": acc_win,
        #     "gyro": gyro_win,
        #     "activity": segment["activity"],
        #     "user": segment["user"]
        # })
        windows.append({
            "acc": acc_win,
            "gyro": gyro_win,
            "activity": activity,
            "user": segment["user"],
            "exp": segment["exp"],
            "segment_start": segment["start"],
            "segment_end": segment["end"],
            "window_start_local": start,
            "window_end_local": start + window_size,
            "window_start_global": segment["start"] + start,
            "window_end_global": segment["start"] + start + window_size,
            "overlap": activity_overlap,
            "stride": stride,
        })

        # start += (window_size - overlap)
        start += stride

    return windows




def fit_scaler(X_train):
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # reshape to (samples*time, features)
    flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(flat)

    return scaler


def apply_scaler(X, scaler):
    shape = X.shape
    flat = X.reshape(-1, shape[-1])
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(shape)


def save_train_val_test_numpy(X_train, y_train, X_val, y_val, X_test, y_test, SaveDir):
    np.save(os.path.join(SaveDir,"X_train.npy"), X_train)
    print(f"X_train saved to: {os.path.join(SaveDir,'X_train.npy')}")
    np.save(os.path.join(SaveDir,"y_train.npy"), y_train)
    print(f"y_train saved to: {os.path.join(SaveDir,'y_train.npy')}")

    np.save(os.path.join(SaveDir,"X_val.npy"), X_val)
    print(f"X_val saved to: {os.path.join(SaveDir,'X_val.npy')}")
    np.save(os.path.join(SaveDir,"y_val.npy"), y_val)
    print(f"y_val saved to: {os.path.join(SaveDir,'y_val.npy')}")

    np.save(os.path.join(SaveDir,"X_test.npy"), X_test)
    print(f"X_test saved to: {os.path.join(SaveDir,'X_test.npy')}")
    np.save(os.path.join(SaveDir,"y_test.npy"), y_test)
    print(f"y_test saved to: {os.path.join(SaveDir,'y_test.npy')}")


    print("Saved numpy arrays containg Train, Val, Test: X & y\n\n")


def save_dataset_scalar_metadata(X_train, dataset, scaler, SaveDir):
    metadata = {
        "window_size": WINDOW_SIZE,
        "overlap": OVERLAP,
        "sampling_rate": SAMPLING_RATE,
        "train_subjects": TRAIN_SUBJECTS,
        "val_subjects": VAL_SUBJECTS,
        "test_subjects": TEST_SUBJECTS,
        "input_shape": X_train.shape[1:],
        "split_encoding": {"train": 0, "val": 1, "test": 2},
        "class_ids": list(range(1, 13)),
        "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    }

    # dataset.to_csv(os.path.join(SaveDir,"dataset.csv"), index=False)
    # print(f"Dataset saved to: {os.path.join(SaveDir,'dataset.csv')}")

    dataset.to_pickle(os.path.join(SaveDir,"dataset.pkl"))
    print(f"Dataset pickle saved to: {os.path.join(SaveDir,'dataset.pkl')}")


    with open(os.path.join(SaveDir,"scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {os.path.join(SaveDir,'scaler.pkl')}")

    with open(os.path.join(SaveDir,"metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {os.path.join(SaveDir,'metadata.json')}")









