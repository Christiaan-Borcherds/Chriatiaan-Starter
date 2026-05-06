import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

# from data_loading import load_numpy_data
from hp_search import generate_hp_candidates
from models import build_model
from run_artifacts import (
    add_timing_row,
    build_model_run_summary,
    update_run_pointers,
    write_best_models_summary,
    write_json,
)
from trainer import calculate_classification_metrics, train_stage, get_predictions
from training_utils import build_loader as build_dataset_loader
from training_utils import build_plateau_scheduler, create_history, set_seed, load_numpy_data
from utils import plot_training_history, save_evaluation_artifacts


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


def build_loader(X, y, batch_size, shuffle):
    return build_dataset_loader(HAPTDataset, X, y, batch_size, shuffle)


def load_user_groups(config):
    users_train = np.load(config.TRAIN_USERS_PATH)
    users_val = np.load(config.VAL_USERS_PATH)
    users_test = np.load(config.TEST_USERS_PATH)
    return users_train, users_val, users_test


def run_dev_pipeline(config):
    pipeline_start = time.perf_counter()
    pipeline_started_at = datetime.now().isoformat(timespec="seconds")
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
        ModelSpec("cnn_lstm", config.CNNLSTM_Type),
        ModelSpec("cnn", config.CNN_Type),
        ModelSpec("lstm", config.LSTM_Type),
        # ModelSpec("multihead_cnn_lstm", config.MulitHeadCNNLSTM_type),

    ]

    device = torch.device("cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    total_training_stages = 2 if config.EPOCHS_STAGE2 > 0 else 1

    summary_rows = []
    timing_rows = []
    model_run_summaries = []

    for spec in model_specs:
        if spec.model_type is None:
            continue

        hp_grid = generate_hp_candidates(config, spec.name)
        if not hp_grid:
            raise ValueError(f"No hyperparameter candidates generated for model {spec.name}")

        (out_dir / f"config_{spec.name}.json").write_text(json.dumps(hp_grid, indent=2))
        (out_dir / f"hp_search_{spec.name}.json").write_text(json.dumps(config.HP_SEARCH, indent=2))

        best_hp = None
        best_mean_f1 = -1.0
        best_histories = None

        for hp_idx, hp in enumerate(hp_grid, start=1):
            hp_start = time.perf_counter()
            hp_started_at = datetime.now().isoformat(timespec="seconds")
            kf = GroupKFold(n_splits=config.K_FOLDS)
            fold_rows = []
            fold_histories = []

            for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_dev, y_dev, groups=users_dev), start=1):
                fold_start = time.perf_counter()
                fold_started_at = datetime.now().isoformat(timespec="seconds")
                set_seed(config.SEED + fold_idx)
                model = build_model(config, spec.model_type, hp).to(device)

                criterion = nn.CrossEntropyLoss()

                train_users = set(users_dev[tr_idx].tolist())
                val_users = set(users_dev[va_idx].tolist())
                overlap = train_users.intersection(val_users)
                if overlap:
                    raise ValueError(f"Fold {fold_idx} has overlapping users: {sorted(overlap)}")

                train_loader = build_loader(X_dev[tr_idx], y_dev[tr_idx], hp["batch_size"], True)
                val_loader = build_loader(X_dev[va_idx], y_dev[va_idx], hp["batch_size"], False)

                history = create_history()
                monitor = "val_f1_macro"
                optimizer1 = optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
                scheduler = build_plateau_scheduler(optimizer1, config, monitor)

                history, best_val_loss, best_epoch, best_state_dict, best_metrics = train_stage(
                    model, train_loader, val_loader, criterion, optimizer1, device,
                    config.EPOCHS_STAGE1, 0, history, scheduler=scheduler, monitor=monitor,
                    early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                    progress_label=(
                        f"Model {spec.name} | HP {hp_idx}/{len(hp_grid)} | "
                        f"Fold {fold_idx}/{config.K_FOLDS} | Stage 1/{total_training_stages}"
                    ),
                )

                if config.EPOCHS_STAGE2 > 0:
                    optimizer2 = optim.Adagrad(model.parameters(), lr=hp["lr"])
                    scheduler2 = build_plateau_scheduler(optimizer2, config, monitor)
                    history, best_val_loss, best_epoch, best_state_dict, best_metrics = train_stage(
                        model, train_loader, val_loader, criterion, optimizer2, device,
                        config.EPOCHS_STAGE2, len(history["loss"]), history, scheduler=scheduler2,
                        best_val_loss=best_val_loss, best_epoch=best_epoch, best_state_dict=best_state_dict,
                        best_metrics=best_metrics,
                        monitor=monitor,
                        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                        progress_label=(
                            f"Model {spec.name} | HP {hp_idx}/{len(hp_grid)} | "
                            f"Fold {fold_idx}/{config.K_FOLDS} | Stage 2/{total_training_stages}"
                        ),
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
                add_timing_row(
                    timing_rows,
                    "fold_training",
                    fold_start,
                    started_at=fold_started_at,
                    model=spec.name,
                    hp_index=hp_idx,
                    hp_total=len(hp_grid),
                    fold=fold_idx,
                    fold_total=config.K_FOLDS,
                    epochs_stage1=config.EPOCHS_STAGE1,
                    epochs_stage2=config.EPOCHS_STAGE2,
                    best_epoch=best_epoch,
                    f1_macro=fold_metrics["f1_macro"],
                    hp_search_strategy=config.HP_SEARCH["strategy"],
                    hp=json.dumps(hp),
                )

            mean_f1 = float(np.mean([r["f1_macro"] for r in fold_rows]))
            std_f1 = float(np.std([r["f1_macro"] for r in fold_rows]))

            if mean_f1 > best_mean_f1:
                best_mean_f1 = mean_f1
                best_hp = hp
                best_histories = fold_histories

            for row in fold_rows:
                row.update({"mean_f1": mean_f1, "std_f1": std_f1, "hp": json.dumps(hp)})
                summary_rows.append({"model": spec.name, **row})

            add_timing_row(
                timing_rows,
                "hp_set_training",
                hp_start,
                started_at=hp_started_at,
                model=spec.name,
                hp_index=hp_idx,
                hp_total=len(hp_grid),
                fold_total=config.K_FOLDS,
                epochs_stage1=config.EPOCHS_STAGE1,
                epochs_stage2=config.EPOCHS_STAGE2,
                f1_macro=mean_f1,
                hp_search_strategy=config.HP_SEARCH["strategy"],
                hp=json.dumps(hp),
            )

        with open(out_dir / f"best_hp_{spec.name}.json", "w") as f:
            json.dump({"best_hp": best_hp, "best_mean_f1": best_mean_f1}, f, indent=2)

        run_name = f"kfold_{spec.name}_{now}"
        wb_run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=run_name,
            reinit=True,
            config={
                "model": spec.name,
                "architecture": spec.model_type,
                "seed": config.SEED,
                "k_folds": config.K_FOLDS,
                "splitter": "GroupKFold",
                "hp_search_strategy": config.HP_SEARCH["strategy"],
                "reference_split": "Val Reference",
                "window_size": config.WINDOW_SIZE,
                "num_classes": config.NUM_CLASSES,
                "epochs_stage1": config.EPOCHS_STAGE1,
                "epochs_stage2": config.EPOCHS_STAGE2,
                "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
                "lr_on_plateau_patience": config.LR_ON_PLATEAU_PATIENCE,
                "lr_on_plateau_factor": 0.1,
                **best_hp,
            },
        )

        final_start = time.perf_counter()
        final_started_at = datetime.now().isoformat(timespec="seconds")
        print(f"\nFinal training: {spec.name}")
        print(f"search_strategy={config.HP_SEARCH['strategy']} | best_mean_f1={best_mean_f1:.4f}")
        print(f"best_hp={json.dumps(best_hp, sort_keys=True)}")

        set_seed(config.SEED)
        final_model = build_model(config, spec.model_type, best_hp).to(device)
        criterion = nn.CrossEntropyLoss()

        train_loader = build_loader(X_train, y_train, best_hp["batch_size"], True)
        val_loader = build_loader(X_val, y_val, best_hp["batch_size"], False)

        history = create_history()
        monitor = "val_f1_macro"
        opt1 = optim.Adam(final_model.parameters(), lr=best_hp["lr"], weight_decay=best_hp["weight_decay"])
        sch = build_plateau_scheduler(opt1, config, monitor)
        history, best_val_loss, best_epoch, best_state_dict, best_metrics = train_stage(
            final_model, train_loader, val_loader, criterion, opt1, device,
            config.EPOCHS_STAGE1, 0, history, scheduler=sch, wandb_run=wb_run,
            monitor=monitor,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            progress_label=f"Final {spec.name} | Val Reference | Stage 1/{total_training_stages}"
        )

        if config.EPOCHS_STAGE2 > 0:
            opt2 = optim.Adagrad(final_model.parameters(), lr=best_hp["lr"])
            sch2 = build_plateau_scheduler(opt2, config, monitor)
            history, best_val_loss, best_epoch, best_state_dict, best_metrics = train_stage(
                final_model, train_loader, val_loader, criterion, opt2, device,
                config.EPOCHS_STAGE2, len(history["loss"]), history, scheduler=sch2, wandb_run=wb_run,
                best_val_loss=best_val_loss, best_epoch=best_epoch, best_state_dict=best_state_dict,
                best_metrics=best_metrics,
                monitor=monitor,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                progress_label=f"Final {spec.name} | Val Reference | Stage 2/{total_training_stages}"
            )

        final_model.load_state_dict(best_state_dict)
        torch.save(best_state_dict, out_dir / f"best_model_{spec.name}.pt")
        plot_training_history(history, out_dir / f"history_plot_{spec.name}.png")
        with open(out_dir / f"history_{spec.name}.json", "w") as f:
            json.dump(history, f, indent=2)

        val_reference_true, val_reference_pred = get_predictions(final_model, val_loader, device)
        val_reference_artifacts = save_evaluation_artifacts(
            y_true=val_reference_true,
            y_pred=val_reference_pred,
            class_names=config.CLASS_NAMES,
            out_dir=out_dir,
            model_name=spec.name,
            prefix="val_reference",
            heading="Val Reference",
        )
        wb_run.log({"Val Reference/confusion_matrix": wandb.Image(str(val_reference_artifacts["confusion_matrix_plot_path"]))})

        with open(out_dir / f"val_reference_metrics_{spec.name}.json", "w") as f:
            json.dump(best_metrics, f, indent=2)

        model_run_summaries.append(
            build_model_run_summary(
                config=config,
                out_dir=out_dir,
                spec=spec,
                best_hp=best_hp,
                best_mean_f1=best_mean_f1,
                best_metrics=best_metrics,
            )
        )

        add_timing_row(
            timing_rows,
            "final_training",
            final_start,
            started_at=final_started_at,
            model=spec.name,
            epochs_stage1=config.EPOCHS_STAGE1,
            epochs_stage2=config.EPOCHS_STAGE2,
            best_epoch=best_epoch,
            f1_macro=best_metrics.get("f1_macro", "") if best_metrics else "",
            hp_search_strategy=config.HP_SEARCH["strategy"],
            hp=json.dumps(best_hp),
        )

        wb_run.finish()

    add_timing_row(
        timing_rows,
        "entire_training_loop",
        pipeline_start,
        started_at=pipeline_started_at,
        epochs_stage1=config.EPOCHS_STAGE1,
        epochs_stage2=config.EPOCHS_STAGE2,
        hp_search_strategy=config.HP_SEARCH["strategy"],
    )

    with open(out_dir / "training_times.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "started_at",
                "ended_at",
                "duration_seconds",
                "model",
                "hp_index",
                "hp_total",
                "fold",
                "fold_total",
                "epochs_stage1",
                "epochs_stage2",
                "best_epoch",
                "f1_macro",
                "hp_search_strategy",
                "hp",
            ],
        )
        writer.writeheader()
        writer.writerows(timing_rows)

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

    best_models_summary_path = out_dir / "best_models_summary.csv"
    write_best_models_summary(best_models_summary_path, model_run_summaries)

    selected_model = max(model_run_summaries, key=lambda row: row["best_mean_f1"])
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(out_dir),
        "run_manifest_path": str(out_dir / "run_manifest.json"),
        "selection_metric": "best_mean_f1",
        "selection_mode": "max",
        "selected_model": selected_model,
        "models": model_run_summaries,
        "fold_results_path": str(out_dir / "fold_results_all_models.csv"),
        "training_times_path": str(out_dir / "training_times.csv"),
        "best_models_summary_path": str(best_models_summary_path),
        "splits_path": str(out_dir / "splits.json"),
        "k_folds": config.K_FOLDS,
        "splitter": "GroupKFold",
        "hp_search": config.HP_SEARCH,
    }
    write_json(out_dir / "run_manifest.json", run_manifest)
    latest_path, best_path, best_by_family_path, best_updated, family_updates = update_run_pointers(config, run_manifest)
    print(f"\nUpdated latest run pointer: {latest_path}")
    if best_updated:
        print(f"Updated best run pointer: {best_path}")
    else:
        print(f"Best run pointer unchanged: {best_path}")
    if family_updates:
        print(f"Updated best model families: {', '.join(family_updates)}")
    else:
        print("Best model families unchanged")
    print(f"Best models by family pointer: {best_by_family_path}")

    return out_dir
