import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def plot_training_history(history, save_path):
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(24, 12))

    plt.subplot(2, 2, 1)
    plt.xlabel("Number of epochs")
    plt.grid(True, linewidth=0.5, linestyle="-.")
    plt.plot(epochs, history["loss"])
    plt.plot(epochs, history["val_loss"])
    plt.legend(["training_loss", "val_loss"])
    plt.title("Training and Validation Loss")

    plt.subplot(2, 2, 2)
    plt.xlabel("Number of epochs")
    plt.grid(True, linewidth=0.5, linestyle="-.")
    plt.plot(epochs, history["accuracy"])
    plt.plot(epochs, history["val_accuracy"])
    plt.legend(["training_accuracy", "val_accuracy"])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 2, 3)
    plt.xlabel("Number of epochs")
    plt.grid(True, linewidth=0.5, linestyle="-.")
    plt.plot(epochs, history["f1_macro"])
    plt.plot(epochs, history["val_f1_macro"])
    plt.legend(["training_f1_macro", "val_f1_macro"])
    plt.title("Training and Validation Macro F1")

    plt.subplot(2, 2, 4)
    plt.xlabel("Number of epochs")
    plt.grid(True, linewidth=0.5, linestyle="-.")
    plt.plot(epochs, history["cohen_kappa"])
    plt.plot(epochs, history["val_cohen_kappa"])
    plt.legend(["training_cohen_kappa", "val_cohen_kappa"])
    plt.title("Training and Validation Cohen Kappa")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()




def save_evaluation_artifacts(y_true, y_pred, class_names, out_dir, model_name, prefix, heading):
    cm = confusion_matrix(y_true, y_pred)

    np.save(out_dir / f"{prefix}_confusion_matrix_{model_name}.npy", cm)

    with open(out_dir / f"{prefix}_confusion_matrix_{model_name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name, *row.tolist()])

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"{heading} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm.max() / 2 if cm.size else 0
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                int(cm[row_idx, col_idx]),
                ha="center",
                va="center",
                color="white" if cm[row_idx, col_idx] > threshold else "black",
            )

    plt.tight_layout()
    plot_path = out_dir / f"{prefix}_confusion_matrix_{model_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    with open(out_dir / f"{prefix}_classification_report_{model_name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label, vals in report.items():
            if isinstance(vals, dict):
                writer.writerow([label, vals.get("precision"), vals.get("recall"), vals.get("f1-score"), vals.get("support")])

    return {
        "confusion_matrix": cm,
        "confusion_matrix_plot_path": plot_path,
        "classification_report": report,
    }
