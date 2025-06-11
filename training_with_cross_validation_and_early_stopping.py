import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support


def main():
    # ===== CONFIGURATION =====
    DATA_DIR = r"C:/Users/pc/Downloads/UCMerced/UC_Merced_Train"
    TEST_DIR = r"C:/Users/pc/Downloads/UCMerced/UC_Merced_Test"
    NUM_LABELS = 21
    NUM_EPOCHS = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    K_FOLDS = 3
    PATIENCE = 2
    MIN_DELTA = 0.005

    # ===== SETUP CSV LOGGING & PLOTS DIRECTORY =====
    log_csv = "cv_training_log.csv"
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "epoch", "train_loss", "val_loss", "val_accuracy"])
    plots_dir = "cv_plots"
    os.makedirs(plots_dir, exist_ok=True)

    # ===== DATA PREPARATION =====
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    labels = np.array([label for _, label in full_dataset.samples])

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== TRAIN & VALIDATE FUNCTION FOR ONE FOLD =====
    def train_validate_fold(train_idx, val_idx, fold):
        print(f"\n--- Fold {fold + 1}/{K_FOLDS} ---")
        train_loader = DataLoader(Subset(full_dataset, train_idx),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader   = DataLoader(Subset(full_dataset, val_idx),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        train_losses, val_losses, val_accs = [], [], []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            # --- Training Phase ---
            model.train()
            total_train_loss = 0.0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(images).logits
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # --- Validation Phase ---
            model.eval()
            total_val_loss = correct = total = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images).logits
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            avg_val_loss = total_val_loss / len(val_loader)
            val_acc = 100 * correct / total
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)

            # --- Print & Log ---
            print(f"Fold {fold + 1} Epoch {epoch}/{NUM_EPOCHS} "
                  f"- Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
            with open(log_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([fold + 1, epoch,
                                 avg_train_loss, avg_val_loss, val_acc])

            scheduler.step()

            # --- Early Stopping Check ---
            if best_val_loss - avg_val_loss > MIN_DELTA:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(),
                           f"best_model_fold{fold + 1}.pth")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping triggered for fold {fold + 1}"
                          f" at epoch {epoch}")
                    break

        # --- Plot Loss Curve ---
        epochs = list(range(1, len(train_losses) + 1))
        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses,   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"Fold {fold + 1} Loss Curve")
        plt.legend()
        plt.savefig(f"{plots_dir}/fold{fold + 1}_loss.png")
        plt.close()

        # --- Plot Accuracy Curve ---
        plt.figure()
        plt.plot(epochs, val_accs, marker="o")
        plt.xlabel("Epoch"); plt.ylabel("Val Accuracy (%)")
        plt.title(f"Fold {fold + 1} Accuracy Curve")
        plt.savefig(f"{plots_dir}/fold{fold + 1}_acc.png")
        plt.close()

        return best_val_loss

    # ===== RUN CROSS-VALIDATION =====
    fold_losses = []
    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.arange(len(labels)), labels)):
        best_loss = train_validate_fold(train_idx, val_idx, fold)
        fold_losses.append(best_loss)

    print("\nCross-validation completed.")
    for i, loss in enumerate(fold_losses):
        print(f"Fold {i + 1} best validation loss: {loss:.4f}")

    # ===== HOLD-OUT TEST SET EVALUATION =====
    best_fold = int(np.argmin(fold_losses)) + 1
    print(f"\nBest fold: {best_fold} (using its weights for test)")

    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    best_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    ).to(device)
    best_model.load_state_dict(
        torch.load(f"best_model_fold{best_fold}.pth",
                   map_location=device)
    )
    best_model.eval()

    criterion = nn.CrossEntropyLoss()
    total_test_loss = correct = total = 0.0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            logits = best_model(images).logits
            loss   = criterion(logits, targets)
            total_test_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)

    avg_test_loss = total_test_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f"\nTest Set Results — Loss: {avg_test_loss:.4f}, "
          f"Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    freeze_support()
    main()
