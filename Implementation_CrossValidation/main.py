import json

import torch

import config
from kfold_pipeline import run_dev_pipeline
from models import build_model


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def format_metric(value):
    return "n/a" if value is None else f"{value:.4f}"


def load_best_development_model(config):
    best_run_path = config.KFOLD_OUTPUT_DIR / "best_run.json"
    if not best_run_path.exists():
        raise FileNotFoundError(
            f"No best development run found at {best_run_path}. "
            "Run with DO_DEVELOPMENT=True first."
        )

    with open(best_run_path, "r") as f:
        manifest = json.load(f)

    selected = manifest["selected_model"]
    device = torch.device("cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(config, selected["model_type"], selected["best_hp"]).to(device)
    state_dict = torch.load(selected["model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    total_params, trainable_params = count_parameters(model)

    print("\nBest Development Model Loaded")
    print(f"model={selected['model']}")
    print(f"model_type={selected['model_type']}")
    print(f"device={device}")
    print(f"model_path={selected['model_path']}")
    print(f"run_dir={manifest['run_dir']}")
    print(f"created_at={manifest['created_at']}")
    print(f"selection={manifest['selection_metric']} ({manifest['selection_mode']})")
    print(f"best_mean_f1={selected['best_mean_f1']:.4f}")
    print(f"k_folds={manifest['k_folds']} | splitter={manifest['splitter']}")
    print(f"hp_search_strategy={selected['hp_search_strategy']}")
    print(f"epochs_stage1={selected['epochs_stage1']} | epochs_stage2={selected['epochs_stage2']}")
    print(
        f"early_stopping_patience={selected['early_stopping_patience']} | "
        f"lr_on_plateau_patience={selected['lr_on_plateau_patience']}"
    )
    print(f"parameters={total_params:,} | trainable_parameters={trainable_params:,}")
    print(f"best_hp={json.dumps(selected['best_hp'], sort_keys=True)}")

    val_reference_metrics = selected.get("val_reference_metrics") or selected.get("best_metrics") or {}
    if val_reference_metrics:
        print("\nVal Reference Evaluation Metrics")
        print(f"best_epoch_on_val_reference={val_reference_metrics.get('epoch')}")
        print(f"monitor={val_reference_metrics.get('monitor')}")
        print(f"val_reference_loss={format_metric(val_reference_metrics.get('loss'))}")
        print(f"val_reference_accuracy={format_metric(val_reference_metrics.get('accuracy'))}")
        print(f"val_reference_precision_macro={format_metric(val_reference_metrics.get('precision_macro'))}")
        print(f"val_reference_recall_macro={format_metric(val_reference_metrics.get('recall_macro'))}")
        print(f"val_reference_f1_macro={format_metric(val_reference_metrics.get('f1_macro'))}")
        print(f"val_reference_cohen_kappa={format_metric(val_reference_metrics.get('cohen_kappa'))}")

    print("\nDevelopment Artifacts")
    print(f"best_run_manifest={best_run_path}")
    print(f"run_manifest={manifest['run_manifest_path']}")
    print(f"fold_results={manifest['fold_results_path']}")
    print(f"training_times={manifest['training_times_path']}")
    print(f"best_hp={selected['best_hp_path']}")
    val_reference_metrics_path = selected.get("val_reference_metrics_path")
    print(f"val_reference_metrics={val_reference_metrics_path}")
    print(f"history={selected['history_path']}")
    print(f"history_plot={selected['history_plot_path']}")
    print(f"val_reference_report={selected['val_reference_classification_report_path']}")
    print(f"val_reference_confusion_matrix={selected['val_reference_confusion_matrix_plot_path']}")

    best_by_family_path = config.KFOLD_OUTPUT_DIR / "best_models_by_family.json"
    if best_by_family_path.exists():
        with open(best_by_family_path, "r") as f:
            best_by_family = json.load(f)

        print("\nBest Models By Family")
        print(f"best_models_by_family={best_by_family_path}")
        for model_name, model_summary in best_by_family.get("models", {}).items():
            print(
                f"{model_name}: best_mean_f1={model_summary['best_mean_f1']:.4f} | "
                f"model_path={model_summary['model_path']}"
            )

    return model, manifest

if config.DO_DEVELOPMENT:
    output_dir = run_dev_pipeline(config)
    print(f"Development pipeline completed. Results saved to: {output_dir}")
else:
    model, manifest = load_best_development_model(config)
