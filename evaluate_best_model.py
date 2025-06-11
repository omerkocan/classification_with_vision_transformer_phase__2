import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # ===== CONFIG =====
    TEST_DIR    = r"C:/Users/pc/Downloads/UCMerced/UC_Merced_Test"
    MODEL_PATH  = "best_model_fold3.pth"
    NUM_LABELS  = 21
    BATCH_SIZE  = 16
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== PREPROCESS =====
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean,
                             std=processor.image_std)
    ])

    # ===== LOAD TEST DATA =====
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
    test_loader  = DataLoader(test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=2)

    # ===== LOAD MODEL =====
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ===== EVALUATION LOOP & TIMING =====
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    start_time = time.time()  # sınıflandırma başlangıç
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images).logits
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds .extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    end_time = time.time()    # sınıflandırma bitiş

    # ===== INFERENCE METRICS =====
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    duration = end_time - start_time

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Inference time: {duration:.2f} seconds")

    # ===== OUTPUT DIRECTORY =====
    out_dir = "eval_plots"
    os.makedirs(out_dir, exist_ok=True)

    # ===== CLASSIFICATION REPORT =====
    class_names = test_dataset.classes
    report_str = classification_report(all_labels, all_preds,
                                       target_names=class_names,
                                       digits=4)
    print("\n=== Classification Report ===")
    print(report_str)
    # Kaydet
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report_str)

    # ===== CONFUSION MATRIX =====
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    # ===== SAVE INFERENCE TIME =====
    time_str = f"Inference time: {duration:.2f} seconds\n"
    with open(os.path.join(out_dir, "inference_time.txt"), "w") as f:
        f.write(time_str)

    print(f"\nClassification report and inference time saved under '{out_dir}'.")


if __name__ == "__main__":
    main()
