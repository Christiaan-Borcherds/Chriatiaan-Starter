# main.py

from datetime import datetime

import random
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pandas as pd
import seaborn as sb

import config
from config import MulitHeadCNNLSTM_type
from data_loading import create_dataloaders
from models import build_model
from trainer import train_one_epoch, evaluate, train_stage, get_predictions
from utils import plot_training_history

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt




# -------------------------
# Reproducibility
# -------------------------
# Set the same seed across Python, NumPy, and PyTorch so runs are repeatable.
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

if torch.cuda.is_available():
    # Seed CUDA random number generation for GPU operations as well.
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

torch.backends.cudnn.deterministic = True # Force PyTorch to use deterministic algorithms only
torch.backends.cudnn.benchmark = False # Disable cuDNN’s auto-optimization search, supporting reproducibility


# -------------------------
# DEVICE
# -------------------------
device = torch.device(
    "cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
)

if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")


# -------------------------
# DATA Consistent through all
# -------------------------
train_loader, val_loader, test_loader, metadata = create_dataloaders(config)




# -------------------------
# MultiheadCNNLSTM MODEL
# -------------------------
model = build_model(config, MulitHeadCNNLSTM_type).to(device)
print(model)




# -------------------------
# TRAINING SETUP
# -------------------------
criterion = nn.CrossEntropyLoss()



# -------------------------
# W&B
# -------------------------
run_name = datetime.now().strftime(
    f"{config.WANDB_RUN_PREFIX}_%Y-%m-%d_%Hh%M"
)

run = wandb.init(
    project=config.WANDB_PROJECT,
    entity=config.WANDB_ENTITY,
    name=run_name,
    notes=config.WANDB_NOTES,
    config=config.WANDB_CONFIG,
)

wandb.watch(model, log="all", log_freq=10)

# -------------------------
# -------------------------
# TRAIN LOOP
# -------------------------
# -------------------------
history = {
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "learning_rate": [],
}

best_val_loss = float("inf")
best_model_path = config.MODEL_DIR / f"{run_name}_best.pt"
history_path = config.REPORT_DIR / f"{run_name}_history.json"
config_path = config.REPORT_DIR / f"{run_name}_config.json"
best_state_dict = None

# -------------------------
# Stage 1
# -------------------------
optimizer = optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.99,
)

history, best_val_loss, best_epoch, best_state_dict = train_stage(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=config.EPOCHS_STAGE1,
    start_epoch=0,
    history=history,
    wandb_run=run,
    best_val_loss=best_val_loss,
    best_state_dict=best_state_dict,
)


# -------------------------
# Stage 2
# -------------------------
optimizer = optim.Adagrad(
    model.parameters(),
    lr=config.ADAGRAD_LEARNING_RATE,
)

history, best_val_loss, best_epoch, best_state_dict = train_stage(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=None,
    device=device,
    epochs=config.EPOCHS_STAGE2,
    start_epoch=config.EPOCHS_STAGE1,
    history=history,
    wandb_run=run,
    best_val_loss=best_val_loss,
    best_epoch=best_epoch,
    best_state_dict=best_state_dict,
)

if best_state_dict is None:
    raise RuntimeError("Training completed without capturing a best model state.")

torch.save(best_state_dict, best_model_path)
print(f"Epoch {best_epoch} achieved the lowest validation loss and was saved once at the end.")

# -------------------------
# SAVE HISTORY PLOT
# -------------------------
plot_training_history(
    history,
    config.FIGURE_DIR / f"{run_name}_Train-Val_accuracy_loss.png",
)

with open(history_path, "w") as f:
    json.dump(history, f, indent=4)

with open(config_path, "w") as f:
    json.dump(config.WANDB_CONFIG, f, indent=4)



# -------------------------
# Calculate Metrics EVAL
# -------------------------
model.load_state_dict(torch.load(best_model_path, map_location=device))

val_true, val_pred = get_predictions(model, val_loader, device)

val_report = classification_report(
    val_true,
    val_pred,
    target_names=config.CLASS_NAMES,
    output_dict=True,
)

val_report_df = pd.DataFrame(val_report).transpose()

val_report_df.to_csv(
    config.REPORT_DIR / f"{run_name}_validation_report.csv"
)

val_cm = confusion_matrix(val_true, val_pred)

plt.figure(figsize=(12, 10))
sb.heatmap(
    val_cm,
    annot=True,
    fmt="d",
    xticklabels=config.CLASS_NAMES,
    yticklabels=config.CLASS_NAMES,
)
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Validation Confusion Matrix")
plt.tight_layout()

plt.savefig(
    config.FIGURE_DIR / f"{run_name}_validation_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()



# -------------------------
# TEST
# -------------------------
# model.load_state_dict(torch.load(best_model_path, map_location=device))
#
# test_loss, test_acc = evaluate(
#     model=model,
#     loader=test_loader,
#     criterion=criterion,
#     device=device,
# )
#
# print("\nFinal Test Results")
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_acc:.4f}")
#
# wandb.log(
#     {
#         "test_loss": test_loss,
#         "test_accuracy": test_acc,
#     }
# )
wandb.finish()

