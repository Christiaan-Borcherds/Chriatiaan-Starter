import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_numpy_data(config):
    X_train = np.load(config.X_TRAIN_PATH)
    y_train = np.load(config.Y_TRAIN_PATH)

    X_val = np.load(config.X_VAL_PATH)
    y_val = np.load(config.Y_VAL_PATH)

    X_test = np.load(config.X_TEST_PATH)
    y_test = np.load(config.Y_TEST_PATH)

    with open(config.METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


def build_loader(dataset_cls, X, y, batch_size, shuffle):
    ds = dataset_cls(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def build_plateau_scheduler(optimizer, config, monitor):
    mode = "min" if monitor == "val_loss" else "max"
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        patience=config.LR_ON_PLATEAU_PATIENCE,
        factor=0.1,
    )


def create_history():
    return {
        "loss": [],
        "accuracy": [],
        "precision_macro": [],
        "recall_macro": [],
        "f1_macro": [],
        "cohen_kappa": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision_macro": [],
        "val_recall_macro": [],
        "val_f1_macro": [],
        "val_cohen_kappa": [],
        "learning_rate": [],
    }
