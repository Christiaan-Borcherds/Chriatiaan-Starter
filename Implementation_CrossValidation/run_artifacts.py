import csv
import json
import time
from datetime import datetime
from pathlib import Path


def add_timing_row(timing_rows, scope, start_time, **details):
    end_time = time.perf_counter()
    row = {
        "scope": scope,
        "started_at": details.pop("started_at"),
        "ended_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(end_time - start_time, 3),
        "model": details.pop("model", ""),
        "hp_index": details.pop("hp_index", ""),
        "hp_total": details.pop("hp_total", ""),
        "fold": details.pop("fold", ""),
        "fold_total": details.pop("fold_total", ""),
        "epochs_stage1": details.pop("epochs_stage1", ""),
        "epochs_stage2": details.pop("epochs_stage2", ""),
        "best_epoch": details.pop("best_epoch", ""),
        "f1_macro": details.pop("f1_macro", ""),
        "hp_search_strategy": details.pop("hp_search_strategy", ""),
        "hp": details.pop("hp", ""),
    }
    timing_rows.append(row)
    return end_time


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def ensure_fold_results_path(model_summary):
    if model_summary.get("fold_results_path"):
        return False

    model_path = model_summary.get("model_path")
    if not model_path:
        return False

    model_summary["fold_results_path"] = str(Path(model_path).parent / "fold_results_all_models.csv")
    return True


def build_model_run_summary(config, out_dir, spec, best_hp, best_mean_f1, best_metrics):
    model_path = out_dir / f"best_model_{spec.name}.pt"
    return {
        "model": spec.name,
        "model_type": spec.model_type,
        "selection_metric": "best_mean_f1",
        "selection_mode": "max",
        "best_mean_f1": best_mean_f1,
        "best_hp": best_hp,
        "val_reference_metrics": best_metrics,
        "model_path": str(model_path),
        "best_hp_path": str(out_dir / f"best_hp_{spec.name}.json"),
        "val_reference_metrics_path": str(out_dir / f"val_reference_metrics_{spec.name}.json"),
        "history_path": str(out_dir / f"history_{spec.name}.json"),
        "history_plot_path": str(out_dir / f"history_plot_{spec.name}.png"),
        "val_reference_confusion_matrix_path": str(out_dir / f"val_reference_confusion_matrix_{spec.name}.npy"),
        "val_reference_confusion_matrix_csv_path": str(out_dir / f"val_reference_confusion_matrix_{spec.name}.csv"),
        "val_reference_confusion_matrix_plot_path": str(out_dir / f"val_reference_confusion_matrix_{spec.name}.png"),
        "val_reference_classification_report_path": str(out_dir / f"val_reference_classification_report_{spec.name}.csv"),
        "fold_results_path": str(out_dir / "fold_results_all_models.csv"),
        "epochs_stage1": config.EPOCHS_STAGE1,
        "epochs_stage2": config.EPOCHS_STAGE2,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "lr_on_plateau_patience": config.LR_ON_PLATEAU_PATIENCE,
        "hp_search_strategy": config.HP_SEARCH["strategy"],
    }


def update_run_pointers(config, run_manifest):
    latest_path = config.KFOLD_OUTPUT_DIR / "latest_run.json"
    best_path = config.KFOLD_OUTPUT_DIR / "best_run.json"
    best_by_family_path = config.KFOLD_OUTPUT_DIR / "best_models_by_family.json"

    write_json(latest_path, run_manifest)

    current_best = run_manifest["selected_model"]
    should_update_best = True
    if best_path.exists():
        previous_best = read_json(best_path)["selected_model"]
        should_update_best = current_best["best_mean_f1"] > previous_best["best_mean_f1"]

    if should_update_best:
        write_json(best_path, run_manifest)

    if best_by_family_path.exists():
        best_by_family = read_json(best_by_family_path)
    else:
        best_by_family = {
            "selection_metric": "best_mean_f1",
            "selection_mode": "max",
            "models": {},
        }

    backfilled_existing_entries = any(
        ensure_fold_results_path(model_summary)
        for model_summary in best_by_family["models"].values()
    )

    family_updates = []
    for model_summary in run_manifest["models"]:
        ensure_fold_results_path(model_summary)
        model_name = model_summary["model"]
        previous_summary = best_by_family["models"].get(model_name)
        if previous_summary is None or model_summary["best_mean_f1"] > previous_summary["best_mean_f1"]:
            best_by_family["models"][model_name] = model_summary
            family_updates.append(model_name)

    if family_updates or backfilled_existing_entries:
        best_by_family["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(best_by_family_path, best_by_family)

    return latest_path, best_path, best_by_family_path, should_update_best, family_updates


def write_best_models_summary(path, model_run_summaries):
    fieldnames = [
        "model",
        "model_type",
        "best_mean_f1",
        "best_epoch",
        "val_loss",
        "val_accuracy",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro",
        "val_cohen_kappa",
        "hp_search_strategy",
        "best_hp",
        "model_path",
        "best_hp_path",
        "val_reference_metrics_path",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in model_run_summaries:
            metrics = row.get("val_reference_metrics") or row.get("best_metrics") or {}
            writer.writerow({
                "model": row["model"],
                "model_type": row["model_type"],
                "best_mean_f1": row["best_mean_f1"],
                "best_epoch": metrics.get("epoch"),
                "val_loss": metrics.get("loss"),
                "val_accuracy": metrics.get("accuracy"),
                "val_precision_macro": metrics.get("precision_macro"),
                "val_recall_macro": metrics.get("recall_macro"),
                "val_f1_macro": metrics.get("f1_macro"),
                "val_cohen_kappa": metrics.get("cohen_kappa"),
                "hp_search_strategy": row["hp_search_strategy"],
                "best_hp": json.dumps(row["best_hp"]),
                "model_path": row["model_path"],
                "best_hp_path": row["best_hp_path"],
                "val_reference_metrics_path": row["val_reference_metrics_path"],
            })
