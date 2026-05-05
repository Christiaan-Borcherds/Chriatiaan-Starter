from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
import torch


def calculate_classification_metrics(labels, preds):
    return {
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(labels, preds),
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    total = 0
    all_labels = []
    all_preds = []

    for acc, gyro, labels in loader:
        acc = acc.to(device)
        gyro = gyro.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(acc, gyro)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

    metrics = calculate_classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / total
    metrics["accuracy"] = sum(int(pred == label) for label, pred in zip(all_labels, all_preds)) / total
    return metrics



def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for acc, gyro, labels in loader:
            acc = acc.to(device)
            gyro = gyro.to(device)
            labels = labels.to(device)

            outputs = model(acc, gyro)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

    metrics = calculate_classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / total
    metrics["accuracy"] = sum(int(pred == label) for label, pred in zip(all_labels, all_preds)) / total
    return metrics



def train_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    start_epoch,
    history,
    scheduler=None,
    wandb_run=None,
    best_val_loss=float("inf"),
    best_epoch=None,
    best_state_dict=None,
    best_metrics=None,
    monitor="val_f1_macro",
):
    if best_epoch is None:
        best_epoch = 0

    if monitor == "val_loss":
        best_monitor_value = best_val_loss
    elif best_metrics is not None:
        best_monitor_value = best_metrics.get(monitor.removeprefix("val_"), float("-inf"))
    else:
        best_monitor_value = float("-inf")

    for epoch in range(start_epoch, start_epoch + epochs):

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history.setdefault("loss", []).append(train_metrics["loss"])
        history.setdefault("accuracy", []).append(train_metrics["accuracy"])
        history.setdefault("precision_macro", []).append(train_metrics["precision_macro"])
        history.setdefault("recall_macro", []).append(train_metrics["recall_macro"])
        history.setdefault("f1_macro", []).append(train_metrics["f1_macro"])
        history.setdefault("cohen_kappa", []).append(train_metrics["cohen_kappa"])
        history.setdefault("val_loss", []).append(val_metrics["loss"])
        history.setdefault("val_accuracy", []).append(val_metrics["accuracy"])
        history.setdefault("val_precision_macro", []).append(val_metrics["precision_macro"])
        history.setdefault("val_recall_macro", []).append(val_metrics["recall_macro"])
        history.setdefault("val_f1_macro", []).append(val_metrics["f1_macro"])
        history.setdefault("val_cohen_kappa", []).append(val_metrics["cohen_kappa"])
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch + 1} | "
            f"loss: {train_metrics['loss']:.4f} | "
            f"accuracy: {train_metrics['accuracy']:.4f} | "
            f"f1_macro: {train_metrics['f1_macro']:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_accuracy: {val_metrics['accuracy']:.4f} | "
            f"val_f1_macro: {val_metrics['f1_macro']:.4f} | "
            f"val_kappa: {val_metrics['cohen_kappa']:.4f} | "
            f"lr: {current_lr:.6f}"
        )

        monitor_values = {
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_cohen_kappa": val_metrics["cohen_kappa"],
        }
        if monitor not in monitor_values:
            raise ValueError(f"Unsupported monitor metric: {monitor}")

        current_monitor_value = monitor_values[monitor]
        monitor_improved = (
            current_monitor_value < best_monitor_value
            if monitor == "val_loss"
            else current_monitor_value > best_monitor_value
        )

        # Keep the best weights in memory and save once after training finishes.
        if monitor_improved:
            best_monitor_value = current_monitor_value
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_metrics = {"epoch": best_epoch, "monitor": monitor, **val_metrics}
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            print(f"Updated best model at epoch {epoch + 1} using {monitor}: {current_monitor_value:.4f}")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "loss": train_metrics["loss"],
                    "accuracy": train_metrics["accuracy"],
                    "precision_macro": train_metrics["precision_macro"],
                    "recall_macro": train_metrics["recall_macro"],
                    "f1_macro": train_metrics["f1_macro"],
                    "cohen_kappa": train_metrics["cohen_kappa"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision_macro": val_metrics["precision_macro"],
                    "val_recall_macro": val_metrics["recall_macro"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "val_cohen_kappa": val_metrics["cohen_kappa"],
                    "learning_rate": current_lr,
                }
            )

    return history, best_val_loss, best_epoch, best_state_dict, best_metrics


def get_predictions(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for acc, gyro, labels in loader:
            acc = acc.to(device)
            gyro = gyro.to(device)

            outputs = model(acc, gyro)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds
