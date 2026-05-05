import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HAPTDataset(Dataset):
    def __init__(self, X, y):
        """
        X shape: (N, 128, 2, 3)
        y shape: (N,)

        For PyTorch CrossEntropyLoss:
        y must be class indices from 0 to 11, not one-hot.
        """

        self.X = torch.tensor(X, dtype=torch.float32)

        # HAPT labels are usually 1–12, so convert to 0–11
        self.y = torch.tensor(y, dtype=torch.long) - 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        acc = x[:, 0, :]   # shape: (128, 3)
        gyro = x[:, 1, :]  # shape: (128, 3)

        return acc, gyro, y


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


def create_dataloaders(config):
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_numpy_data(config)

    train_dataset = HAPTDataset(X_train, y_train)
    val_dataset = HAPTDataset(X_val, y_val)
    test_dataset = HAPTDataset(X_test, y_test)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, #Batches are then randomised on each epoch, but the temporat structure of the data in each window inside the batches are maintained
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=use_cuda,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=use_cuda,
    )

    return train_loader, val_loader, test_loader, metadata