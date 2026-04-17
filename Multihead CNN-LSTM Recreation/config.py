import os
from datetime import datetime


DatasetDir = '/home/christiaan/Documents/MUST/Starter Project/Datasets/HAPT/RawData'

ImageDir = '/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/Multihead CNN-LSTM Recreation/Images'
ExampleImageDir = '/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/Multihead CNN-LSTM Recreation/Images/Example'
TestResultsDir = '/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/Multihead CNN-LSTM Recreation/Test_Results'



ReloadFromRawData = False
VisualizeRawDataDistribution = False
TrainModel = True

# Hyperparameters:
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 10
BATCH_SIZE = 100

# Wheights & Biases
run_entity = "christiaanborcherds-north-west-university"
run_notes = "Commit message for run"
run_project = "Starter-HAPT"
run_name = datetime.now().strftime("Run_%Y-%m-%d_%H:%M:%S")
run_config = {
    "architecture": "Multihead CNN & LSTM",
    "dataset": "HAPT",
    "epochs_stage1": EPOCHS_STAGE1,
    "epochs_stage2": EPOCHS_STAGE2,
    "batch_size": BATCH_SIZE,
}

