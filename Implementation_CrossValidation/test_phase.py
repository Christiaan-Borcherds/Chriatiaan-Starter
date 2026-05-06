# import csv
# import json
# from pathlib import Path
#
# import numpy as np
# import torch
#
# import config
# from data_loading import HAPTDataset
# from models import build_model
# from trainer import calculate_classification_metrics, get_predictions
# from training_utils import build_loader, load_numpy_data, set_seed
# from utils import save_evaluation_artifacts
#
#
# BEST_MODELS_BY_FAMILY_PATH = config.KFOLD_OUTPUT_DIR / "best_models_by_family.json"
# TEST_SUMMARY_CSV_PATH = config.KFOLD_OUTPUT_DIR / "test_performance_summary.csv"
# TEST_SUMMARY_JSON_PATH = config.KFOLD_OUTPUT_DIR / "test_performance_summary.json"
#
# TEST_SUMMARY_FIELDS = [
#     "model",
#     "model_type",
#     "model_path",
#     "test_accuracy",
#     "test_precision_macro",
#     "test_recall_macro",
#     "test_f1_macro",
#     "test_cohen_kappa",
#     "test_metrics_path",
#     "test_confusion_matrix_path",
#     "test_confusion_matrix_csv_path",
#     "test_confusion_matrix_plot_path",
#     "test_classification_report_path",
# ]
#
#
# def read_json(path):
#     with open(path, "r") as f:
#         return json.load(f)
#
#
# def write_json(path, data):
#     with open(path, "w") as f:
#         json.dump(data, f, indent=2)
#
#
# def write_csv(path, rows, fieldnames):
#     with open(path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)
#
#
# def resolve_device(config_module):
#     return torch.device(
#         "cuda"
#         if config_module.DEVICE == "cuda" and torch.cuda.is_available()
#         else "cpu"
#     )
#
#
# def load_test_loader(config_module, batch_size):
#     _, _, _, _, X_test, y_test, _ = load_numpy_data(config_module)
#     return build_loader(HAPTDataset, X_test, y_test, batch_size, shuffle=False)
#
#
# def load_trained_model(config_module, model_summary, device):
#     model_path = Path(model_summary["model_path"])
#     if not model_path.exists():
#         raise FileNotFoundError(model_path)
#
#     model = build_model(
#         config_module,
#         model_summary["model_type"],
#         model_summary["best_hp"],
#     ).to(device)
#
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model
#
#
# def calculate_test_metrics(model_name, model_summary, y_true, y_pred):
#     classification_metrics = calculate_classification_metrics(y_true, y_pred)
#     return {
#         "model": model_name,
#         "model_type": model_summary["model_type"],
#         "dataset": "test",
#         "model_path": model_summary["model_path"],
#         "best_hp": model_summary["best_hp"],
#         "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
#         "precision_macro": float(classification_metrics["precision_macro"]),
#         "recall_macro": float(classification_metrics["recall_macro"]),
#         "f1_macro": float(classification_metrics["f1_macro"]),
#         "cohen_kappa": float(classification_metrics["cohen_kappa"]),
#     }
#
#
# def evaluate_model_on_test_set(config_module, model_name, model_summary, device):
#     batch_size = model_summary["best_hp"]["batch_size"]
#     test_loader = load_test_loader(config_module, batch_size)
#     model = load_trained_model(config_module, model_summary, device)
#
#     y_true, y_pred = get_predictions(model, test_loader, device)
#     metrics = calculate_test_metrics(model_name, model_summary, y_true, y_pred)
#
#     out_dir = Path(model_summary["model_path"]).parent
#     save_evaluation_artifacts(
#         y_true=y_true,
#         y_pred=y_pred,
#         class_names=config_module.CLASS_NAMES,
#         out_dir=out_dir,
#         model_name=model_name,
#         prefix="test",
#         heading="Test",
#     )
#
#     metrics_path = out_dir / f"test_metrics_{model_name}.json"
#     write_json(metrics_path, metrics)
#
#     return {
#         "model": model_name,
#         "model_type": model_summary["model_type"],
#         "model_path": model_summary["model_path"],
#         "test_accuracy": metrics["accuracy"],
#         "test_precision_macro": metrics["precision_macro"],
#         "test_recall_macro": metrics["recall_macro"],
#         "test_f1_macro": metrics["f1_macro"],
#         "test_cohen_kappa": metrics["cohen_kappa"],
#         "test_metrics_path": str(metrics_path),
#         "test_confusion_matrix_path": str(out_dir / f"test_confusion_matrix_{model_name}.npy"),
#         "test_confusion_matrix_csv_path": str(out_dir / f"test_confusion_matrix_{model_name}.csv"),
#         "test_confusion_matrix_plot_path": str(out_dir / f"test_confusion_matrix_{model_name}.png"),
#         "test_classification_report_path": str(out_dir / f"test_classification_report_{model_name}.csv"),
#     }
#
#
# def evaluate_best_models_on_test_set(
#     config_module=config,
#     best_models_by_family_path=BEST_MODELS_BY_FAMILY_PATH,
#     summary_csv_path=TEST_SUMMARY_CSV_PATH,
#     summary_json_path=TEST_SUMMARY_JSON_PATH,
# ):
#     set_seed(config_module.SEED)
#     device = resolve_device(config_module)
#     best_by_family = read_json(best_models_by_family_path)
#
#     summary_rows = []
#     for model_name, model_summary in best_by_family.get("models", {}).items():
#         summary_rows.append(
#             evaluate_model_on_test_set(
#                 config_module=config_module,
#                 model_name=model_name,
#                 model_summary=model_summary,
#                 device=device,
#             )
#         )
#
#     if not summary_rows:
#         raise ValueError(f"No models found in {best_models_by_family_path}")
#
#     write_csv(summary_csv_path, summary_rows, TEST_SUMMARY_FIELDS)
#     write_json(summary_json_path, summary_rows)
#
#     return summary_rows, Path(summary_csv_path), Path(summary_json_path)
#
#
# def main():
#     summary_rows, summary_csv_path, summary_json_path = evaluate_best_models_on_test_set()
#
#     print("Test Performance Summary:")
#     for row in summary_rows:
#         print(
#             f"{row['model']}: "
#             f"accuracy={row['test_accuracy']:.4f} | "
#             f"f1_macro={row['test_f1_macro']:.4f} | "
#             f"cohen_kappa={row['test_cohen_kappa']:.4f}"
#         )
#     print(f"\nSaved test summary CSV to:\n{summary_csv_path}")
#     print(f"Saved test summary JSON to:\n{summary_json_path}")
#
#
# if __name__ == "__main__":
#     main()
