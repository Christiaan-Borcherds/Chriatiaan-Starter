# utils.py

import matplotlib.pyplot as plt


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
