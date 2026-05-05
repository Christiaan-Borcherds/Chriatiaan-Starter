from pathlib import Path

# -------------------------
# Project paths
# -------------------------
# PROJECT_DIR = Path("/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/Implementation_CrossValidation")
PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = Path("/home/christiaan/Documents/MUST/Starter Project/Christiaan - Starter/Data Preparation/Output")

OUTPUT_DIR = PROJECT_DIR / "Output"
MODEL_DIR = OUTPUT_DIR / "Models"
FIGURE_DIR = OUTPUT_DIR / "Figures"
REPORT_DIR = OUTPUT_DIR / "Reports"
PREDICTION_DIR = OUTPUT_DIR / "Predictions"
KFOLD_OUTPUT_DIR = OUTPUT_DIR / "KFold"

for path in [OUTPUT_DIR, MODEL_DIR, FIGURE_DIR, REPORT_DIR, PREDICTION_DIR, KFOLD_OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# -------------------------
# Dataset files
# -------------------------
X_TRAIN_PATH = DATA_DIR / "X_train.npy"
Y_TRAIN_PATH = DATA_DIR / "y_train.npy"
TRAIN_USERS_PATH = DATA_DIR / "users_train.npy"
X_VAL_PATH = DATA_DIR / "X_val.npy"
Y_VAL_PATH = DATA_DIR / "y_val.npy"
VAL_USERS_PATH = DATA_DIR / "users_val.npy"
X_TEST_PATH = DATA_DIR / "X_test.npy"
Y_TEST_PATH = DATA_DIR / "y_test.npy"
TEST_USERS_PATH = DATA_DIR / "users_test.npy"
METADATA_PATH = DATA_DIR / "metadata.json"

# -------------------------
# Dataset settings
# -------------------------
NUM_CLASSES = 12
WINDOW_SIZE = 128
NUM_SENSORS = 2
NUM_AXES = 3
INPUT_SHAPE = (WINDOW_SIZE, NUM_SENSORS, NUM_AXES)

CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
    "STAND_TO_SIT",
    "SIT_TO_STAND",
    "SIT_TO_LIE",
    "LIE_TO_SIT",
    "STAND_TO_LIE",
    "LIE_TO_STAND",
]

# -------------------------
# Training settings
# -------------------------
SEED = 42
K_FOLDS = 5
DO_DEVELOPMENT = True

# HP Grid

HP_GRID = {
    "cnn_lstm": [
        {
            "batch_size": 100,
            "epochs_stage1": 20,
            "epochs_stage2": 10,
            "lr": 0.001,
            "weight_decay": 0.0,
            "adagrad_lr": 0.001,
        }
    ],
    "cnn": [{"placeholder": True}],
    "lstm": [{"placeholder": True}],
}

# BATCH_SIZE = 100
# EPOCHS = 30
#
# USE_CLASS_WEIGHTS = True
#
#
# # Multihead Stage settings
# EPOCHS_STAGE1 = 20
# EPOCHS_STAGE2 = 10
# LEARNING_RATE = 0.001
# WEIGHT_DECAY = 0.0
#
# ADAGRAD_LEARNING_RATE = 0.001



# -------------------------
# Device
# -------------------------
DEVICE = "cuda"  # use "cuda" if available, otherwise fallback in code

# -------------------------
# Weights & Biases
# -------------------------
WANDB_PROJECT = "Starter-HAPT"
WANDB_ENTITY = "christiaanborcherds-north-west-university"
# WANDB_RUN_PREFIX = "PyTorch_Multihead_CNN_LSTM"
# WANDB_NOTES = "First PyTorch implementation using saved subject-wise split HAPT tensors."
#
# WANDB_CONFIG = {
#     "architecture": "Multihead CNN-LSTM",
#     "dataset": "HAPT",
#     "window_size": WINDOW_SIZE,
#     "input_shape": INPUT_SHAPE,
#     "num_classes": NUM_CLASSES,
#     "batch_size": BATCH_SIZE,
#     "epochs": EPOCHS,
#     "learning_rate": LEARNING_RATE,
#     "use_class_weights": USE_CLASS_WEIGHTS,
# }



# -------------------------
# Model Types
# -------------------------
MulitHeadCNNLSTM_type = "MulitHeadCNNLSTM"


