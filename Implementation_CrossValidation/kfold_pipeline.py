import copy
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

from data_loading import load_numpy_data
from models import build_model
from trainer import calculate_classification_metrics, train_stage, get_predictions
from utils import plot_training_history


class HAPTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) - 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        acc = x[:, 0, :]
        gyro = x[:, 1, :]
        return acc, gyro, self.y[idx]


@dataclass
class ModelSpec:
    name: str
    model_type: str | None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loader(X, y, batch_size, shuffle):
    ds = HAPTDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


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


def load_user_groups(config):
    users_train = np.load(config.TRAIN_USERS_PATH)
    users_val = np.load(config.VAL_USERS_PATH)
    users_test = np.load(config.TEST_USERS_PATH)
    return users_train, users_val, users_test


def run_dev_pipeline(config):
    set_seed(config.SEED)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.KFOLD_OUTPUT_DIR / now
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_numpy_data(config)
    users_train, users_val, users_test = load_user_groups(config)
    X_dev = np.concatenate([X_train, X_val], axis=0)
    y_dev = np.concatenate([y_train, y_val], axis=0)
    users_dev = np.concatenate([users_train, users_val], axis=0)

    if len(users_dev) != len(y_dev):
        raise ValueError(f"users_dev length ({len(users_dev)}) does not match y_dev length ({len(y_dev)})")
    if len(users_test) != len(y_test):
        raise ValueError(f"users_test length ({len(users_test)}) does not match y_test length ({len(y_test)})")

    unique_dev_users = np.unique(users_dev)
    if len(unique_dev_users) < config.K_FOLDS:
        raise ValueError(
            f"GroupKFold requires at least {config.K_FOLDS} unique users, "
            f"but dev data has {len(unique_dev_users)}"
        )

    splits = {
        "seed": config.SEED,
        "k_folds": config.K_FOLDS,
        "splitter": "GroupKFold",
        "dev_size": int(len(y_dev)),
        "test_size": int(len(y_test)),
        "dev_user_count": int(len(unique_dev_users)),
        "test_user_count": int(len(np.unique(users_test))),
        "dev_users": unique_dev_users.tolist(),
        "test_users": np.unique(users_test).tolist(),
    }
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2))

    model_specs = [
        ModelSpec("cnn_lstm", config.MulitHeadCNNLSTM_type),
        ModelSpec("cnn", None),
        ModelSpec("lstm", None),
    ]

    device = torch.device("cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

    summary_rows = []

    for spec in model_specs:
        if spec.model_type is None:
            continue

        hp_grid = config.HP_GRID[spec.name]
        (out_dir / f"config_{spec.name}.json").write_text(json.dumps(hp_grid, indent=2))

        best_hp = None
        best_mean_f1 = -1.0
        best_histories = None

        for hp in hp_grid:
            kf = GroupKFold(n_splits=config.K_FOLDS)
            fold_rows = []
            fold_histories = []

            for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_dev, y_dev, groups=users_dev), start=1):
                set_seed(config.SEED + fold_idx)
                model = build_model(config, spec.model_type).to(device)

                criterion = nn.CrossEntropyLoss()

                train_users = set(users_dev[tr_idx].tolist())
                val_users = set(users_dev[va_idx].tolist())
                overlap = train_users.intersection(val_users)
                if overlap:
                    raise ValueError(f"Fold {fold_idx} has overlapping users: {sorted(overlap)}")

                train_loader = build_loader(X_dev[tr_idx], y_dev[tr_idx], hp["batch_size"], True)
                val_loader = build_loader(X_dev[va_idx], y_dev[va_idx], hp["batch_size"], False)

                history = create_history()
                optimizer1 = optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.99)

                history, best_val_loss, best_epoch, best_state_dict, best_metrics = train_stage(
                    model, train_loader, val_loader, criterion, optimizer1, device,
                    hp["epochs_stage1"], 0, history, scheduler=scheduler, monitor="val_f1_macro",
                )

                optimizer2 = optim.Adagrad(model.parameters(), lr=hp["adagrad_lr"])
                history, best_val_loss, best_epoch, best_state_dict, best_metrics = train_stage(
                    model, train_loader, val_loader, criterion, optimizer2, device,
                    hp["epochs_stage2"], hp["epochs_stage1"], history,
                    best_val_loss=best_val_loss, best_epoch=best_epoch, best_state_dict=best_state_dict,
                    best_metrics=best_metrics
                )

                model.load_state_dict(best_state_dict)
                y_true, y_pred = get_predictions(model, val_loader, device)
                fold_metrics = calculate_classification_metrics(y_true, y_pred)

                fold_rows.append({
                    "fold": fold_idx,
                    "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
                    **fold_metrics,
                    "best_epoch": best_epoch,
                })
                fold_histories.append({"fold": fold_idx, "history": history})

            mean_f1 = float(np.mean([r["f1_macro"] for r in fold_rows]))
            std_f1 = float(np.std([r["f1_macro"] for r in fold_rows]))

            if mean_f1 > best_mean_f1:
                best_mean_f1 = mean_f1
                best_hp = hp
                best_histories = fold_histories

            for row in fold_rows:
                row.update({"mean_f1": mean_f1, "std_f1": std_f1, "hp": json.dumps(hp)})
                summary_rows.append({"model": spec.name, **row})

        with open(out_dir / f"best_hp_{spec.name}.json", "w") as f:
            json.dump({"best_hp": best_hp, "best_mean_f1": best_mean_f1}, f, indent=2)

        run_name = f"kfold_{spec.name}_{now}"
        wb_run = wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY, name=run_name, reinit=True)

        set_seed(config.SEED)
        final_model = build_model(config, spec.model_type).to(device)
        criterion = nn.CrossEntropyLoss()
        dev_loader = build_loader(X_dev, y_dev, best_hp["batch_size"], True)
        holdout_loader = build_loader(X_test, y_test, best_hp["batch_size"], False)

        history = create_history()
        opt1 = optim.Adam(final_model.parameters(), lr=best_hp["lr"], weight_decay=best_hp["weight_decay"])
        sch = optim.lr_scheduler.ExponentialLR(opt1, gamma=0.99)
        history, best_val_loss, best_epoch, best_state_dict, best_metrics = train_stage(final_model, dev_loader, holdout_loader, criterion, opt1, device, best_hp["epochs_stage1"], 0, history, scheduler=sch, wandb_run=wb_run)

        opt2 = optim.Adagrad(final_model.parameters(), lr=best_hp["adagrad_lr"])
        history, best_val_loss, best_epoch, best_state_dict, best_metrics = train_stage(final_model, dev_loader, holdout_loader, criterion, opt2, device, best_hp["epochs_stage2"], best_hp["epochs_stage1"], history, wandb_run=wb_run, best_val_loss=best_val_loss, best_epoch=best_epoch, best_state_dict=best_state_dict, best_metrics=best_metrics)

        final_model.load_state_dict(best_state_dict)
        torch.save(best_state_dict, out_dir / f"best_model_{spec.name}.pt")
        plot_training_history(history, out_dir / f"history_plot_{spec.name}.png")
        with open(out_dir / f"history_{spec.name}.json", "w") as f:
            json.dump(history, f, indent=2)

        y_true, y_pred = get_predictions(final_model, holdout_loader, device)
        cm = confusion_matrix(y_true, y_pred)
        np.save(out_dir / f"confusion_matrix_{spec.name}.npy", cm)
        report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True)
        with open(out_dir / f"classification_report_{spec.name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "precision", "recall", "f1-score", "support"])
            for label, vals in report.items():
                if isinstance(vals, dict):
                    writer.writerow([label, vals.get("precision"), vals.get("recall"), vals.get("f1-score"), vals.get("support")])

        wb_run.finish()

    with open(out_dir / "fold_results_all_models.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "fold",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "cohen_kappa",
                "best_epoch",
                "mean_f1",
                "std_f1",
                "hp",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return out_dir
