# ==========================================================
# ============ MODEL EVALUATION & PERFORMANCE ==============
# ==========================================================
# Evaluate a YOLOv8-CLASSIFICATION model (.pt) on a folder-
# structured test set (one subfolder per class).
# Reports:
#   - Overall Accuracy and Macro F1
#   - Per-class Accuracy, Precision, Recall, and F1
#   - Confusion Matrix visualization
# ==========================================================

# ---------- Imports ----------
import os
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import datasets
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# =================== CONFIGURATION ========================
# ==========================================================
MODEL_PATH = 'office_item_classifier_yolov8cls.pt'       # <-- your new model
TEST_DIR   = 'Classification/dataset/Final_Dataset/test' # folder-per-class
DEVICE     = 0 if torch.cuda.is_available() else 'cpu'   # GPU if available

# ==========================================================
# ================== EVALUATION LOGIC ======================
# ==========================================================
def evaluate_model():
    """
    Evaluates a YOLOv8 classification model on a folder-structured test set.
    """
    print("\n--- Starting YOLOv8-CLS Model Evaluation on Test Set ---\n")

    # ---- Step 1: Load model ----
    model = YOLO(MODEL_PATH)
    yolo_names = model.names  # dict: {idx: class_name}
    print(f"Loaded model with {len(yolo_names)} classes: {yolo_names}\n")

    # ---- Step 2: Load test set (to get file paths + true labels) ----
    # No transforms needed; YOLO does its own preprocessing.
    test_dataset = datasets.ImageFolder(TEST_DIR)
    class_names  = test_dataset.classes              # list of folder names
    num_classes  = len(class_names)
    print(f"Found {num_classes} classes in test set: {', '.join(class_names)}")

    # Build a mapping from class name -> folder index (for consistency)
    name_to_folder_idx = {name: i for i, name in enumerate(class_names)}

    # ---- Step 3: Inference and predictions ----
    print("\n[Step 3] Running predictions...")
    y_true, y_pred = [], []

    for img_path, true_idx in test_dataset.samples:
        # Run YOLO classification on the image path
        results = model(img_path, verbose=False, device=DEVICE)
        res = results[0]

        # Top-1 predicted class index in YOLO's label space
        pred_yolo_idx = int(res.probs.top1)
        pred_name     = yolo_names[pred_yolo_idx]

        # Map YOLO class name -> our folder index
        pred_idx = name_to_folder_idx.get(pred_name, None)
        if pred_idx is None:
            # If names don't match, count as wrong prediction
            pred_idx = -1

        y_true.append(true_idx)
        y_pred.append(pred_idx)

    # If any -1 slipped in (name mismatch), filter them out to avoid metric errors
    valid = [i for i, p in enumerate(y_pred) if p != -1]
    if len(valid) < len(y_pred):
        print(f"Warning: {len(y_pred) - len(valid)} predictions had unmapped class names and were ignored.")
    y_true = [y_true[i] for i in valid]
    y_pred = [y_pred[i] for i in valid]

    print("Predictions complete.\n")

    # ---- Step 4: Overall metrics ----
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    print("=== OVERALL PERFORMANCE ===")
    print(f"{'Accuracy':<15}: {acc:.4f}")
    print(f"{'Macro F1':<15}: {macro_f1:.4f}\n")

    # ---- Step 5: Per-class metrics & confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    per_class_acc  = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    per_class_f1   = f1_score(y_true, y_pred, labels=range(num_classes), average=None)
    per_class_prec = precision_score(y_true, y_pred, labels=range(num_classes),
                                     average=None, zero_division=0)
    per_class_rec  = recall_score(y_true, y_pred, labels=range(num_classes),
                                  average=None, zero_division=0)

    print("=== PER-CLASS METRICS ===")
    print(f"{'Class':<20}{'Acc':>8}{'Prec':>10}{'Rec':>10}{'F1':>10}{'Support':>10}")
    print("-" * 60)
    for i, cls in enumerate(class_names):
        support = cm.sum(axis=1)[i]
        print(f"{cls:<20}{per_class_acc[i]:>8.3f}{per_class_prec[i]:>10.3f}"
              f"{per_class_rec[i]:>10.3f}{per_class_f1[i]:>10.3f}{support:>10}")
    print("-" * 60)

    # ---- Step 6: Confusion matrix plot ----
    print("\n[Step 6] Plotting confusion matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix — YOLOv8 Classification')
    plt.tight_layout()
    plt.show()

    print("\n✅ Evaluation complete. Metrics and visualization generated successfully.")

# ==========================================================
# ======================= MAIN =============================
# ==========================================================
if __name__ == '__main__':
    evaluate_model()
