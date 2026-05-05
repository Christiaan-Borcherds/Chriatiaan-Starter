import numpy as np
import pandas as pd


# Tensor Dataset
def build_tensor_dataset(windows):
    X = []
    y = []
    users = []

    for w in windows:
        combined = np.stack([w["acc"], w["gyro"]], axis=1)  # (128, 2, 3)
        X.append(combined)
        y.append(w["activity"])
        users.append(w["user"])

    return np.array(X), np.array(y), np.array(users)


# Dataframe
def build_dataframe_dataset(windows, split_map):
    rows = []

    for i, w in enumerate(windows):
        split = split_map(w["user"])

        for t in range(len(w["acc"])): # Each timestep
            row = {
                "acc_x": w["acc"][t][0],
                "acc_y": w["acc"][t][1],
                "acc_z": w["acc"][t][2],
                "gyro_x": w["gyro"][t][0],
                "gyro_y": w["gyro"][t][1],
                "gyro_z": w["gyro"][t][2],
                "instance": w["user"],
                "exp": w["exp"],
                "class": w["activity"],
                "split": split
            }
            rows.append(row)

    return pd.DataFrame(rows)