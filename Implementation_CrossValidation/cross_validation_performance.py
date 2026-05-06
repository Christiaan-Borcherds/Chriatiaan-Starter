import csv
import json
from pathlib import Path
from statistics import mean, stdev


PROJECT_DIR = Path(__file__).resolve().parent
KFOLD_OUTPUT_DIR = PROJECT_DIR / "Output" / "KFold"
BEST_MODELS_BY_FAMILY_PATH = KFOLD_OUTPUT_DIR / "best_models_by_family.json"

METRICS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "cohen_kappa",
    "best_epoch",
]

PUBLICATION_METRICS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "cohen_kappa",
]

CROSS_VALIDATION_PERFORMANCE_FIELDS = [
    "model",
    "hp_id",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "cohen_kappa",
    "hp",
]


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def read_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def canonical_hp(hp):
    if isinstance(hp, str):
        hp = json.loads(hp)
    return json.dumps(hp, sort_keys=True)


def metric_mean_std(rows, metric):
    values = [float(row[metric]) for row in rows]
    metric_mean = mean(values)
    metric_std = stdev(values) if len(values) > 1 else 0.0
    return metric_mean, metric_std


def mean_std(mean_value, std_value, decimals=3):
    return f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"


def assign_hp_ids(rows):
    hp_ids_by_model = {}
    for row in rows:
        model = row["model"]
        hp_key = canonical_hp(row["hp"])
        hp_ids = hp_ids_by_model.setdefault(model, {})
        if hp_key not in hp_ids:
            hp_ids[hp_key] = len(hp_ids)
        row["hp_key"] = hp_key
        row["hp_id"] = hp_ids[hp_key]


def load_best_family_fold_rows(best_by_family):
    selected_rows = []

    for model_name, model_summary in best_by_family.get("models", {}).items():
        fold_results_path = model_summary.get("fold_results_path")
        if not fold_results_path:
            raise ValueError(f"Missing fold_results_path for model '{model_name}'")

        fold_results_path = Path(fold_results_path)
        if not fold_results_path.exists():
            raise FileNotFoundError(fold_results_path)

        rows = [
            row
            for row in read_csv(fold_results_path)
            if row["model"] == model_name
        ]
        if not rows:
            raise ValueError(f"No fold rows found for model '{model_name}' in {fold_results_path}")

        assign_hp_ids(rows)
        best_hp_key = canonical_hp(model_summary["best_hp"])
        rows = [row for row in rows if row["hp_key"] == best_hp_key]
        if not rows:
            raise ValueError(
                f"No fold rows found for best_hp of model '{model_name}' in {fold_results_path}"
            )

        for row in rows:
            row["fold_results_path"] = str(fold_results_path)
        selected_rows.extend(rows)

    if not selected_rows:
        raise ValueError(f"No models found in {BEST_MODELS_BY_FAMILY_PATH}")

    return selected_rows


def build_summary(fold_rows):
    grouped_rows = {}
    for row in fold_rows:
        key = (row["model"], row["hp_id"], row["hp"], row["fold_results_path"])
        grouped_rows.setdefault(key, []).append(row)

    summary = []
    for (model, hp_id, hp, fold_results_path), rows in grouped_rows.items():
        summary_row = {
            "model": model,
            "hp_id": hp_id,
            "hp": hp,
            "fold_results_path": fold_results_path,
        }
        for metric in METRICS:
            metric_mean, metric_std = metric_mean_std(rows, metric)
            summary_row[f"{metric}_mean"] = metric_mean
            summary_row[f"{metric}_std"] = metric_std
        summary.append(summary_row)

    return sorted(summary, key=lambda row: row["f1_macro_mean"], reverse=True)


def build_best_models(summary):
    best_by_model = {}
    for row in summary:
        model = row["model"]
        if model not in best_by_model:
            best_by_model[model] = row
    return sorted(
        best_by_model.values(),
        key=lambda row: row["f1_macro_mean"],
        reverse=True,
    )


def build_publication_table(best_models):
    report_rows = []
    for row in best_models:
        report_row = {
            "model": row["model"],
            "hp_id": row["hp_id"],
            "hp": row["hp"],
        }
        for metric in PUBLICATION_METRICS:
            report_row[metric] = mean_std(
                row[f"{metric}_mean"],
                row[f"{metric}_std"],
            )
        report_rows.append(report_row)
    return report_rows


def create_cross_validation_performance_table(
    best_models_by_family_path=BEST_MODELS_BY_FAMILY_PATH,
    output_path=None,
):
    best_models_by_family_path = Path(best_models_by_family_path)
    if output_path is None:
        output_path = best_models_by_family_path.parent / "cross_validation_performance_table.csv"
    output_path = Path(output_path)

    best_by_family = read_json(best_models_by_family_path)
    fold_rows = load_best_family_fold_rows(best_by_family)
    summary = build_summary(fold_rows)
    best_models = build_best_models(summary)
    publication_table = build_publication_table(best_models)

    write_csv(output_path, publication_table, CROSS_VALIDATION_PERFORMANCE_FIELDS)

    return publication_table, output_path


def main():
    publication_table, publication_table_path = create_cross_validation_performance_table()
    print("Cross-Validation Publication Table:")
    for row in publication_table:
        print(row)
    print(f"\nSaved cross-validation performance table to:\n{publication_table_path}")


if __name__ == "__main__":
    main()
