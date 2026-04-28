import numpy as np
import glob
import os


# =========================
# Paths
# =========================
DatasetDir = "/home/christiaan/Documents/MUST/Starter Project/Datasets/HAPT/RawData"
SaveDir = "/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/Data Preparation/Output/"


# =========================
# Properties
# =========================
SAMPLING_RATE = 50
WINDOW_SIZE = 128
# OVERLAP = 64
OVERLAP = [
    64, 64, 64, 64, 64, 64,
    120, 120, 120, 120, 120, 120
]
subjects = np.arange(1, 31)
np.random.seed(42)  # reproducibility
np.random.shuffle(subjects)

# For 30 subjects and 70:10:20 - we need:
# Train = 21
# Val = 3
# Train = 6
train_size = 21
val_size = 3

TRAIN_SUBJECTS = subjects[:train_size].tolist()
VAL_SUBJECTS   = subjects[train_size:train_size+val_size].tolist()
TEST_SUBJECTS  = subjects[train_size+val_size:].tolist()

# print(f"TRAIN ({len(TRAIN_SUBJECTS)} subjects): {sorted(TRAIN_SUBJECTS)}")
# print(f"VAL   ({len(VAL_SUBJECTS)} subjects): {sorted(VAL_SUBJECTS)}")
# print(f"TEST  ({len(TEST_SUBJECTS)} subjects): {sorted(TEST_SUBJECTS)}")




# =========================
# File naming patterns
# =========================
ACC_PATTERN = 'acc_exp*_user*.txt'
GYRO_PATTERN = 'gyro_exp*_user*.txt'
LABELS_FILE = 'labels.txt'

acc_files = sorted(glob.glob(os.path.join(DatasetDir, ACC_PATTERN)))
gyro_files = sorted(glob.glob(os.path.join(DatasetDir, GYRO_PATTERN)))





