import os


# =========================
# Paths
# =========================
DatasetDir = '/home/christiaan/Documents/MUST/Starter Project/Datasets/HAPT/RawData'

DatasetOverview_OutputDir_2_1 = '/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/EDA (Exploratory Data Analysis)/2.1 Dataset Overview'
SignalLabelSplitDir_2_2 = '/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/EDA (Exploratory Data Analysis)/2.1 Dataset Overview/Label Splits'
OutlierInvestigation_Dir = '/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/EDA (Exploratory Data Analysis)/2.1 Dataset Overview/Outlier Investigation'
DataCacheDir = '/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/EDA (Exploratory Data Analysis)/Data Cache'
# =========================
# Properties
# =========================
SAMPLING_RATE = 50  # Hz


# =========================
# Output control
# =========================
PRINT_PER_FILE = False
PRINT_PER_SUBJECT = True
PRINT_PER_EXPERIMENT = True

EXPORT_PER_FILE_CSV = False


# =========================
# File naming patterns
# =========================
ACC_PATTERN = 'acc_exp*_user*.txt'
GYRO_PATTERN = 'gyro_exp*_user*.txt'
LABELS_FILE = 'labels.txt'


label_to_class = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING',
    7: 'STAND_TO_SIT',
    8: 'SIT_TO_STAND',
    9: 'SIT_TO_LIE',
    10: 'LIE_TO_SIT',
    11: 'STAND_TO_LIE',
    12: 'LIE_TO_STAND',
}

activity_groups = {
    "Static": [4, 5, 6],
    "Dynamic": [1, 2, 3],
    "Transition": [7, 8, 9, 10, 11, 12],
}

channel_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]