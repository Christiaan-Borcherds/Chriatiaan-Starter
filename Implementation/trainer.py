import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

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
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total



def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for acc, gyro, labels in loader:
            acc = acc.to(device)
            gyro = gyro.to(device)
            labels = labels.to(device)

            outputs = model(acc, gyro)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total



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
):
    if best_epoch is None:
        best_epoch = 0

    for epoch in range(start_epoch, start_epoch + epochs):

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rate"].append(current_lr)
        # f1, and other

        print(
            f"Epoch {epoch + 1} | "
            f"loss: {train_loss:.4f} | "
            f"accuracy: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_accuracy: {val_acc:.4f} | "
            f"lr: {current_lr:.6f}"
        )

        # Keep the best weights in memory and save once after training finishes.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            print(f"Updated best model at epoch {epoch + 1}")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "loss": train_loss,
                    "accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": current_lr,
                }
            )

    return history, best_val_loss, best_epoch, best_state_dict


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
