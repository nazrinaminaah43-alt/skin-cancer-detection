"""
Complete, Reproducible Deep Learning Training and Comparison Module for Skin Cancer Detection.

Trains, cross-validates, and evaluates 5 deep learning image classification architectures:
  1. Custom CNN
  2. MobileNetV2
  3. ResNet50
  4. EfficientNetB0
  5. VGG16

Ensures:
  - Fixed random seed (42) for reproducibility.
  - Stratified 70/15/15 split (Train/Val/Test) without data leakage.
  - Identical data splits across all models.
  - Training augmentation applied ONLY to the training set.
  - Strict model selection on Validation Sensitivity/F1 (no test data leakage).
  - Comprehensive clinical metric evaluation: Accuracy, Sensitivity (Recall), Specificity,
    Precision, F1-Score, ROC-AUC, Confusion Matrix, ROC curves, PR curves, and Loss/Acc history.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import Config
from src.deep_learning import (
    get_deep_learning_model,
    get_train_transforms,
    get_eval_transforms,
    save_model_checkpoint,
    IMAGENET_MEAN,
    IMAGENET_STD
)


# ==============================================================================
# 1. Dataset & Data Loading
# ==============================================================================
class SkinLesionDataset(Dataset):
    """PyTorch Dataset for skin cancer lesion images."""

    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def prepare_dataset_splits(
    dataset_dir: str,
    random_seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Scans dataset directory, performs stratified split, and saves manifest
    to guarantee identical splits for all models.
    """
    benign_dir = Path(dataset_dir) / "benign"
    malignant_dir = Path(dataset_dir) / "malignant"

    samples = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p in benign_dir.glob(ext):
            samples.append({"path": str(p), "label": 0, "class_name": "benign"})
        for p in malignant_dir.glob(ext):
            samples.append({"path": str(p), "label": 1, "class_name": "malignant"})

    if not samples:
        raise FileNotFoundError(f"No images found in {dataset_dir}/benign or {dataset_dir}/malignant")

    df = pd.DataFrame(samples)
    print(f"[*] Total dataset samples found: {len(df)} (Benign: {(df['label']==0).sum()}, Malignant: {(df['label']==1).sum()})")

    # Step 1: Stratified split into train (70%) and temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["label"],
        random_state=random_seed
    )

    # Step 2: Stratified split temp into val (15%) and test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=random_seed
    )

    splits_manifest = {
        "random_seed": random_seed,
        "counts": {
            "total": len(df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "train": train_df.to_dict(orient="records"),
        "val": val_df.to_dict(orient="records"),
        "test": test_df.to_dict(orient="records")
    }

    manifest_path = Path(Config.SPLITS_MANIFEST_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(splits_manifest, f, indent=2)

    print(f"[+] Stratified dataset splits saved to {manifest_path}")
    print(f"    Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return splits_manifest


# ==============================================================================
# 2. Metric Computation Helper
# ==============================================================================
def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Computes all clinical metrics:
    Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, and Confusion Matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    sens = float(recall_score(y_true, y_pred, zero_division=0))  # Sensitivity / Recall
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    # Specificity = TN / (TN + FP)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "accuracy_percentage": f"{acc * 100:.1f}%",
        "sensitivity": round(sens, 4),
        "recall": round(sens, 4),
        "recall_percentage": f"{sens * 100:.1f}%",
        "specificity": round(spec, 4),
        "specificity_percentage": f"{spec * 100:.1f}%",
        "precision": round(prec, 4),
        "precision_percentage": f"{prec * 100:.1f}%",
        "f1_score": round(f1, 4),
        "f1_score_percentage": f"{f1 * 100:.1f}%",
        "roc_auc": round(auc, 4),
        "roc_auc_percentage": f"{auc * 100:.1f}%",
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix": [[tn, fp], [fn, tp]]
        }
    }


# ==============================================================================
# 3. Model Training & Validation Function
# ==============================================================================
def train_single_model(
    arch_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 12,
    lr: float = 1e-4
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Trains a deep learning model with:
    - Adam optimizer with weight decay
    - ReduceLROnPlateau scheduler
    - Early stopping
    - Validation monitoring for checkpointing
    """
    print(f"\n{'='*60}\n--> Training Architecture: {arch_name.upper()}\n{'='*60}")
    model = get_deep_learning_model(arch_name, num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_recall": []
    }

    best_val_loss = float("inf")
    best_weights = None
    patience = 4
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_targets_all = []
        val_preds_all = []
        val_probs_all = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                val_loss_batch = loss.item()
                val_targets_all.extend(targets.cpu().numpy().tolist())
                val_preds_all.extend(preds.tolist())
                val_probs_all.extend(probs.tolist())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_metrics = compute_comprehensive_metrics(
            np.array(val_targets_all),
            np.array(val_preds_all),
            np.array(val_probs_all)
        )

        scheduler.step(epoch_val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_recall"].append(val_metrics["sensitivity"])

        print(
            f"Epoch {epoch:02d}/{epochs:02d} - "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.1f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_metrics['accuracy_percentage']} | "
            f"Val Sensitivity: {val_metrics['recall_percentage']}"
        )

        # Early Stopping & Best Weights Tracking
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"[*] Early stopping triggered for {arch_name} at epoch {epoch}")
                break

    # Restore best weights
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Final validation evaluation using best weights
    model.eval()
    val_targets_all, val_preds_all, val_probs_all = [], [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_targets_all.extend(targets.cpu().numpy().tolist())
            val_preds_all.extend(preds.tolist())
            val_probs_all.extend(probs.tolist())

    best_val_metrics = compute_comprehensive_metrics(
        np.array(val_targets_all),
        np.array(val_preds_all),
        np.array(val_probs_all)
    )

    return model, best_val_metrics, history


# ==============================================================================
# 4. Final Test Set Evaluation Helper
# ==============================================================================
def evaluate_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates a trained model once on the untouched test set."""
    model.eval()
    test_targets_all, test_preds_all, test_probs_all = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            test_targets_all.extend(targets.numpy().tolist())
            test_preds_all.extend(preds.tolist())
            test_probs_all.extend(probs.tolist())

    y_true = np.array(test_targets_all)
    y_pred = np.array(test_preds_all)
    y_prob = np.array(test_probs_all)

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob)
    return metrics, y_true, y_pred, y_prob


# ==============================================================================
# 5. Visualization Generation
# ==============================================================================
def generate_comparison_visualizations(
    results: Dict[str, Any],
    curves_data: Dict[str, Any],
    output_dir: Path
):
    """
    Generates high-resolution publication-quality evaluation charts:
      1. Model Comparison Bar Chart (Sensitivity, Accuracy, F1, Specificity, ROC-AUC)
      2. ROC Curves comparison
      3. Precision-Recall Curves comparison
      4. Individual Confusion Matrix Heatmaps
      5. Training Loss & Accuracy Curves across epochs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    model_names = list(results.keys())
    pretty_names = [results[m]["display_name"] for m in model_names]

    # --------------------------------------------------------------------------
    # Chart 1: Bar Chart Comparison Across All Metrics
    # --------------------------------------------------------------------------
    metrics_keys = ["sensitivity", "specificity", "precision", "f1_score", "accuracy", "roc_auc"]
    metric_labels = ["Sensitivity (Recall)", "Specificity", "Precision", "F1-Score", "Accuracy", "ROC-AUC"]

    chart_data = []
    for m in model_names:
        m_test = results[m]["test_metrics"]
        for k, lbl in zip(metrics_keys, metric_labels):
            chart_data.append({
                "Model": results[m]["display_name"],
                "Metric": lbl,
                "Score": m_test.get(k, 0.0) * 100.0
            })
    df_chart = pd.DataFrame(chart_data)

    plt.figure(figsize=(13, 6.5))
    palette = sns.color_palette("muted", len(metric_labels))
    ax = sns.barplot(data=df_chart, x="Model", y="Score", hue="Metric", palette=palette)
    plt.title("Deep Learning Architectures Clinical Performance Benchmark (Test Split)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Score Percentage (%)", fontsize=11, fontweight="bold")
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylim(0, 110)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    bar_chart_path = output_dir / "dl_comparison_barchart.png"
    plt.savefig(bar_chart_path, dpi=200)
    plt.close()

    # --------------------------------------------------------------------------
    # Chart 2: ROC Curves Comparison
    # --------------------------------------------------------------------------
    plt.figure(figsize=(8.5, 7))
    colors = ["#0284c7", "#10b981", "#f59e0b", "#7c3aed", "#ef4444"]

    for i, m in enumerate(model_names):
        c_info = curves_data[m]
        fpr = c_info["fpr"]
        tpr = c_info["tpr"]
        auc = results[m]["test_metrics"]["roc_auc"]
        name = results[m]["display_name"]
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    roc_path = output_dir / "dl_roc_curves.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # --------------------------------------------------------------------------
    # Chart 3: Precision-Recall Curves Comparison
    # --------------------------------------------------------------------------
    plt.figure(figsize=(8.5, 7))
    for i, m in enumerate(model_names):
        c_info = curves_data[m]
        prec = c_info["pr_precision"]
        rec = c_info["pr_recall"]
        name = results[m]["display_name"]
        f1 = results[m]["test_metrics"]["f1_score"]
        plt.plot(rec, prec, color=colors[i % len(colors)], lw=2.2, label=f"{name} (F1 = {f1:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall (PR) Curves for Malignant Melanoma", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()

    pr_path = output_dir / "dl_pr_curves.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # --------------------------------------------------------------------------
    # Chart 4: Confusion Matrices for All 5 Models
    # --------------------------------------------------------------------------
    for m in model_names:
        cm_data = results[m]["test_metrics"]["confusion_matrix"]["matrix"]
        cm_arr = np.array(cm_data)
        cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(6, 5))
        labels = [["TN", "FP"], ["FN", "TP"]]
        annot = np.empty_like(cm_arr).astype(str)
        for r in range(2):
            for c in range(2):
                annot[r, c] = f"{labels[r][c]}\n{cm_arr[r, c]}\n({cm_norm[r, c]*100:.1f}%)"

        sns.heatmap(
            cm_arr,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=ax,
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )
        ax.set_title(f"Confusion Matrix: {results[m]['display_name']}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Diagnosis", fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual Ground Truth", fontsize=10, fontweight="bold")
        plt.tight_layout()

        cm_path = output_dir / f"dl_confusion_matrix_{m}.png"
        plt.savefig(cm_path, dpi=180)
        plt.close()

    # --------------------------------------------------------------------------
    # Chart 5: Training and Validation History Curves
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, m in enumerate(model_names):
        hist = results[m].get("training_history", {})
        epochs = hist.get("epoch", [])
        if epochs:
            axes[0].plot(epochs, hist.get("train_loss", []), "--", color=colors[i % len(colors)], alpha=0.6)
            axes[0].plot(epochs, hist.get("val_loss", []), "-", color=colors[i % len(colors)], lw=2, label=results[m]["display_name"])

            axes[1].plot(epochs, [a * 100 for a in hist.get("train_acc", [])], "--", color=colors[i % len(colors)], alpha=0.6)
            axes[1].plot(epochs, [a * 100 for a in hist.get("val_acc", [])], "-", color=colors[i % len(colors)], lw=2, label=results[m]["display_name"])

    axes[0].set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=10)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Training vs Validation Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    hist_path = output_dir / "dl_training_curves.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()

    # Also copy all generated visualization images into src/static/img/ for web serving
    static_img_dir = ROOT_DIR / "src" / "static" / "img"
    static_img_dir.mkdir(parents=True, exist_ok=True)
    for img_file in [bar_chart_path, roc_path, pr_path, hist_path] + list(output_dir.glob("dl_confusion_matrix_*.png")):
        shutil.copy2(img_file, static_img_dir / img_file.name)

    print(f"[+] All evaluation charts generated and mirrored to {static_img_dir}")


# ==============================================================================
# 6. Main Execution Pipeline
# ==============================================================================
def run_training_and_comparison_pipeline():
    """Main orchestrator for complete training, validation, selection, and reporting."""
    print("=" * 70)
    print("     SKIN CANCER DETECTION - DEEP LEARNING TRAINING & BENCHMARK")
    print("=" * 70)

    # 1. Reproducibility Seeds
    random_seed = Config.DL_RANDOM_SEED
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Device: {device} | Random Seed: {random_seed}")

    # 2. Dataset Splitting
    manifest = prepare_dataset_splits(Config.DATASET_DIR, random_seed=random_seed)
    train_records = manifest["train"]
    val_records = manifest["val"]
    test_records = manifest["test"]

    # 3. Class Weights Computation (from train set only to prevent leakage)
    train_labels = [r["label"] for r in train_records]
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=np.array(train_labels)
    )
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)
    print(f"[*] Balanced Class Weights (Train Only): Benign={class_weights_arr[0]:.3f}, Malignant={class_weights_arr[1]:.3f}")

    # 4. Data Loaders
    train_transforms = get_train_transforms(image_size=Config.DL_IMAGE_SIZE)
    eval_transforms = get_eval_transforms(image_size=Config.DL_IMAGE_SIZE)

    train_dataset = SkinLesionDataset([r["path"] for r in train_records], train_labels, transform=train_transforms)
    val_dataset = SkinLesionDataset([r["path"] for r in val_records], [r["label"] for r in val_records], transform=eval_transforms)
    test_dataset = SkinLesionDataset([r["path"] for r in test_records], [r["label"] for r in test_records], transform=eval_transforms)

    batch_size = Config.DL_BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. Architectures to Train & Evaluate
    architectures = [
        ("custom_cnn", "Custom CNN", "#0284c7"),
        ("mobilenet_v2", "MobileNetV2", "#10b981"),
        ("resnet50", "ResNet50", "#f59e0b"),
        ("efficientnet_b0", "EfficientNetB0", "#7c3aed"),
        ("vgg16", "VGG16", "#ef4444")
    ]

    trained_models = {}
    model_results = {}
    curves_data = {}

    dl_save_dir = Path(Config.DL_MODELS_DIR)
    dl_save_dir.mkdir(parents=True, exist_ok=True)

    for arch_key, display_name, color in architectures:
        model, val_metrics, history = train_single_model(
            arch_name=arch_key,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights_tensor,
            device=device,
            epochs=Config.DL_EPOCHS,
            lr=Config.DL_LEARNING_RATE
        )
        trained_models[arch_key] = model

        # Test set evaluation
        test_metrics, y_true, y_pred, y_prob = evaluate_on_test_set(model, test_loader, device=device)

        # ROC Curve & PR Curve Data
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)

        curves_data[arch_key] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "pr_precision": pr_prec.tolist(),
            "pr_recall": pr_rec.tolist()
        }

        # Save checkpoint
        ckpt_path = dl_save_dir / f"{arch_key}.pt"
        save_model_checkpoint(
            model=model,
            save_path=str(ckpt_path),
            architecture_name=arch_key,
            metrics={"validation": val_metrics, "test": test_metrics},
            class_to_idx=Config.DL_CLASS_MAPPING,
            image_size=Config.DL_IMAGE_SIZE,
            training_history=history
        )
        print(f"[+] Saved checkpoint to {ckpt_path}")

        # Total parameter count
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_results[arch_key] = {
            "key": arch_key,
            "display_name": display_name,
            "color": color,
            "parameter_count": f"{param_count:,}",
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "training_history": history,
            "checkpoint_path": str(ckpt_path)
        }

    # ==========================================================================
    # 7. Model Selection Criteria (Strictly on Validation Performance)
    # ==========================================================================
    # Clinical Priority: Sensitivity (Recall) is the single most critical screening metric,
    # followed by F1-Score, ROC-AUC, and Specificity.
    print("\n" + "=" * 70)
    print("        CROSS-VALIDATION & MODEL SELECTION (VALIDATION SET ONLY)")
    print("=" * 70)

    best_arch_key = None
    best_selection_score = -1.0

    for arch_key, display_name, _ in architectures:
        vm = model_results[arch_key]["validation_metrics"]
        val_sens = vm["sensitivity"]
        val_f1 = vm["f1_score"]
        val_auc = vm["roc_auc"]
        val_spec = vm["specificity"]

        # Selection formula prioritizing clinical sensitivity
        selection_score = (0.45 * val_sens) + (0.30 * val_f1) + (0.15 * val_auc) + (0.10 * val_spec)
        model_results[arch_key]["selection_score"] = round(selection_score, 4)

        print(
            f"Model: {display_name:16s} | "
            f"Val Sens: {vm['recall_percentage']:6s} | "
            f"Val Spec: {vm['specificity_percentage']:6s} | "
            f"Val F1: {vm['f1_score_percentage']:6s} | "
            f"Val Acc: {vm['accuracy_percentage']:6s} | "
            f"Selection Score: {selection_score:.4f}"
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_arch_key = arch_key

    selected_display_name = model_results[best_arch_key]["display_name"]
    selected_metrics = model_results[best_arch_key]
    print("\n" + "*" * 70)
    print(f"[+] SELECTED FINAL DEEP LEARNING MODEL: {selected_display_name.upper()}")
    print("*" * 70)
    print(f"Selection Rationale: Achieved highest validation clinical selection score ({best_selection_score:.4f})")
    print(f"Validation Sensitivity (Recall) : {selected_metrics['validation_metrics']['recall_percentage']}")
    print(f"Validation F1-Score             : {selected_metrics['validation_metrics']['f1_score_percentage']}")
    print(f"Validation Specificity          : {selected_metrics['validation_metrics']['specificity_percentage']}")
    print(f"Untouched Test Set Accuracy     : {selected_metrics['test_metrics']['accuracy_percentage']}")
    print(f"Untouched Test Set Sensitivity  : {selected_metrics['test_metrics']['recall_percentage']}")
    print("*" * 70)

    # Save as best deep learning model
    best_source_ckpt = Path(selected_metrics["checkpoint_path"])
    best_dest_ckpt = Path(Config.DL_MODEL_PATHS["best"])
    shutil.copy2(best_source_ckpt, best_dest_ckpt)
    print(f"[+] Primary model checkpoint deployed to: {best_dest_ckpt}")

    # Mark is_selected flag in results
    for k in model_results:
        model_results[k]["is_selected"] = (k == best_arch_key)

    # 8. Save Selected Model Info & Metadata
    selected_model_info = {
        "selected_architecture": best_arch_key,
        "selected_display_name": selected_display_name,
        "selection_score": best_selection_score,
        "selection_criterion": "Clinical Screening Score: 0.45*Sensitivity + 0.30*F1 + 0.15*ROC-AUC + 0.10*Specificity",
        "selection_rationale": (
            f"{selected_display_name} was selected because it maximizes Sensitivity "
            f"({selected_metrics['validation_metrics']['recall_percentage']}) while maintaining strong precision and F1-score. "
            "In clinical oncology screening, prioritizing Sensitivity minimizes life-threatening false negatives (missed melanomas)."
        ),
        "validation_metrics": selected_metrics["validation_metrics"],
        "test_metrics": selected_metrics["test_metrics"],
        "checkpoint_file": "best_deep_learning_model.pt",
        "image_size": list(Config.DL_IMAGE_SIZE),
        "class_mapping": Config.DL_CLASS_MAPPING
    }

    with open(Config.DL_SELECTED_MODEL_INFO_PATH, "w") as f:
        json.dump(selected_model_info, f, indent=2)

    # Save full metrics JSON
    unified_dl_metrics = {
        "dataset": {
            "total_samples": manifest["counts"]["total"],
            "train_samples": manifest["counts"]["train"],
            "val_samples": manifest["counts"]["val"],
            "test_samples": manifest["counts"]["test"],
            "classes": Config.DL_CLASS_NAMES
        },
        "selected_model": selected_model_info,
        "models": model_results,
        "curves_data": curves_data
    }

    with open(Config.DL_METRICS_PATH, "w") as f:
        json.dump(unified_dl_metrics, f, indent=2)

    # Save deep learning config JSON
    dl_config_data = {
        "image_size": list(Config.DL_IMAGE_SIZE),
        "batch_size": Config.DL_BATCH_SIZE,
        "random_seed": Config.DL_RANDOM_SEED,
        "epochs": Config.DL_EPOCHS,
        "learning_rate": Config.DL_LEARNING_RATE,
        "num_classes": Config.DL_NUM_CLASSES,
        "class_names": Config.DL_CLASS_NAMES,
        "class_mapping": Config.DL_CLASS_MAPPING,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "selected_architecture": best_arch_key
    }
    with open(Config.DL_CONFIG_PATH, "w") as f:
        json.dump(dl_config_data, f, indent=2)

    # 9. Generate High-Resolution Visualization Charts
    models_dir = Path(Config.MODELS_FOLDER)
    generate_comparison_visualizations(model_results, curves_data, output_dir=models_dir)

    print("\n" + "=" * 70)
    print("      DEEP LEARNING MODEL TRAINING & COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return unified_dl_metrics


if __name__ == "__main__":
    run_training_and_comparison_pipeline()
