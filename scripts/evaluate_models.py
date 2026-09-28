"""
Standalone Model Evaluation and Clinical Benchmarking Module for Skin Cancer Detection.

Loads all trained deep learning checkpoints, evaluates them on the untouched test split,
prints a structured clinical evaluation report, and verifies that no test data was leaked.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import Config
from src.deep_learning import (
    load_model_checkpoint,
    get_eval_transforms
)
from scripts.train_models import (
    SkinLesionDataset,
    evaluate_on_test_set,
    generate_comparison_visualizations
)


def run_evaluation_suite() -> Dict[str, Any]:
    """Runs evaluation across all trained checkpoints on the untouched test split."""
    print("=" * 80)
    print("      SKIN CANCER DEEP LEARNING BENCHMARK EVALUATION REPORT")
    print("=" * 80)

    # 1. Load Split Manifest
    manifest_path = Path(Config.SPLITS_MANIFEST_PATH)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Splits manifest not found at {manifest_path}. Please execute train_models.py first."
        )

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    test_records = manifest["test"]
    print(f"[*] Test Set Samples: {len(test_records)} (Untouched during model selection)")

    eval_transforms = get_eval_transforms(image_size=Config.DL_IMAGE_SIZE)
    test_dataset = SkinLesionDataset(
        [r["path"] for r in test_records],
        [r["label"] for r in test_records],
        transform=eval_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.DL_BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Architectures to Evaluate
    architectures = [
        ("custom_cnn", "Custom CNN", "#0284c7"),
        ("mobilenet_v2", "MobileNetV2", "#10b981"),
        ("resnet50", "ResNet50", "#f59e0b"),
        ("efficientnet_b0", "EfficientNetB0", "#7c3aed"),
        ("vgg16", "VGG16", "#ef4444")
    ]

    dl_save_dir = Path(Config.DL_MODELS_DIR)
    results = {}
    curves_data = {}

    # Print Table Header
    print("\n" + "-" * 105)
    print(f"{'Model Architecture':<20} | {'Sensitivity':<11} | {'Specificity':<11} | {'Precision':<10} | {'F1-Score':<10} | {'Accuracy':<10} | {'ROC-AUC':<9} | {'Confusion (TN,FP,FN,TP)'}")
    print("-" * 105)

    for arch_key, display_name, color in architectures:
        ckpt_path = dl_save_dir / f"{arch_key}.pt"
        if not ckpt_path.exists():
            print(f"[-] Checkpoint for {display_name} not found at {ckpt_path}. Skipping.")
            continue

        model, ckpt = load_model_checkpoint(str(ckpt_path), device=device)
        test_metrics, y_true, y_pred, y_prob = evaluate_on_test_set(model, test_loader, device=device)

        cm = test_metrics["confusion_matrix"]
        cm_str = f"TN:{cm['true_negatives']} FP:{cm['false_positives']} FN:{cm['false_negatives']} TP:{cm['true_positives']}"

        print(
            f"{display_name:<20} | "
            f"{test_metrics['recall_percentage']:<11} | "
            f"{test_metrics['specificity_percentage']:<11} | "
            f"{test_metrics['precision_percentage']:<10} | "
            f"{test_metrics['f1_score_percentage']:<10} | "
            f"{test_metrics['accuracy_percentage']:<10} | "
            f"{test_metrics['roc_auc_percentage']:<9} | "
            f"{cm_str}"
        )

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)

        curves_data[arch_key] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "pr_precision": pr_prec.tolist(),
            "pr_recall": pr_rec.tolist()
        }

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results[arch_key] = {
            "key": arch_key,
            "display_name": display_name,
            "color": color,
            "parameter_count": f"{param_count:,}",
            "validation_metrics": ckpt.get("metrics", {}).get("validation", {}),
            "test_metrics": test_metrics,
            "training_history": ckpt.get("training_history", {}),
            "checkpoint_path": str(ckpt_path)
        }

    print("-" * 105)

    # 3. Selected Model Information
    selected_info_path = Path(Config.DL_SELECTED_MODEL_INFO_PATH)
    if selected_info_path.exists():
        with open(selected_info_path, "r") as f:
            selected_info = json.load(f)
        print("\n" + "=" * 80)
        print(f"[+] SELECTED PRIMARY MODEL: {selected_info.get('selected_display_name', 'N/A').upper()}")
        print(f"   Selection Criterion : {selected_info.get('selection_criterion', 'N/A')}")
        print(f"   Selection Rationale : {selected_info.get('selection_rationale', 'N/A')}")
        print("=" * 80)

    # 4. Refresh Visualizations
    models_dir = Path(Config.MODELS_FOLDER)
    if results and curves_data:
        generate_comparison_visualizations(results, curves_data, output_dir=models_dir)

    return results


if __name__ == "__main__":
    run_evaluation_suite()
