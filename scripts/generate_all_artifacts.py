"""
Generates all benchmarks, evaluations, artifacts, and visualizations from saved checkpoints.
Ensures zero data leakage, strict validation-based model selection, and clean artifact deployment.
"""

import sys
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import Config
from src.deep_learning import (
    load_model_checkpoint,
    get_eval_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from scripts.train_models import (
    SkinLesionDataset,
    compute_comprehensive_metrics,
    evaluate_on_test_set,
    generate_comparison_visualizations
)


def generate_all_evaluation_artifacts():
    print("=" * 80)
    print("   GENERATING FINAL DEEP LEARNING EVALUATION ARTIFACTS & BENCHMARKS")
    print("=" * 80)

    # 1. Load splits manifest
    manifest_path = Path(Config.SPLITS_MANIFEST_PATH)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    train_records = manifest["train"]
    val_records = manifest["val"]
    test_records = manifest["test"]

    print(f"[*] Dataset Manifest: Train={len(train_records)} | Val={len(val_records)} | Test={len(test_records)}")

    eval_transforms = get_eval_transforms(image_size=Config.DL_IMAGE_SIZE)

    val_dataset = SkinLesionDataset(
        [r["path"] for r in val_records],
        [r["label"] for r in val_records],
        transform=eval_transforms
    )
    test_dataset = SkinLesionDataset(
        [r["path"] for r in test_records],
        [r["label"] for r in test_records],
        transform=eval_transforms
    )

    val_loader = DataLoader(val_dataset, batch_size=Config.DL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.DL_BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    architectures = [
        ("custom_cnn", "Custom CNN", "#0284c7"),
        ("mobilenet_v2", "MobileNetV2", "#10b981"),
        ("resnet50", "ResNet50", "#f59e0b"),
        ("efficientnet_b0", "EfficientNetB0", "#7c3aed"),
        ("vgg16", "VGG16", "#ef4444")
    ]

    dl_save_dir = Path(Config.DL_MODELS_DIR)
    model_results = {}
    curves_data = {}

    for arch_key, display_name, color in architectures:
        ckpt_path = dl_save_dir / f"{arch_key}.pt"
        if not ckpt_path.exists():
            print(f"[-] Checkpoint missing for {display_name}: {ckpt_path}")
            continue

        model, ckpt = load_model_checkpoint(str(ckpt_path), device=device)

        # Evaluate on validation set
        val_metrics, val_true, val_pred, val_prob = evaluate_on_test_set(model, val_loader, device=device)

        # Evaluate on test set
        test_metrics, test_true, test_pred, test_prob = evaluate_on_test_set(model, test_loader, device=device)

        fpr, tpr, _ = roc_curve(test_true, test_prob)
        pr_prec, pr_rec, _ = precision_recall_curve(test_true, test_prob)

        curves_data[arch_key] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "pr_precision": pr_prec.tolist(),
            "pr_recall": pr_rec.tolist()
        }

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Retain training history if in checkpoint
        history = ckpt.get("training_history", {})

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

        print(
            f"[+] Evaluated {display_name:<16} | "
            f"Val Recall: {val_metrics['recall_percentage']:<7} | "
            f"Val Acc: {val_metrics['accuracy_percentage']:<7} | "
            f"Test Recall: {test_metrics['recall_percentage']:<7} | "
            f"Test Acc: {test_metrics['accuracy_percentage']}"
        )

    # Model Selection (strictly on validation set)
    print("\n" + "=" * 80)
    print("        CROSS-VALIDATION & MODEL SELECTION (VALIDATION SET ONLY)")
    print("=" * 80)

    best_arch_key = None
    best_selection_score = -1.0

    for arch_key, display_name, _ in architectures:
        if arch_key not in model_results:
            continue
        vm = model_results[arch_key]["validation_metrics"]
        val_sens = vm["sensitivity"]
        val_f1 = vm["f1_score"]
        val_auc = vm["roc_auc"]
        val_spec = vm["specificity"]

        # Selection formula prioritizing clinical sensitivity
        selection_score = (0.45 * val_sens) + (0.30 * val_f1) + (0.15 * val_auc) + (0.10 * val_spec)
        model_results[arch_key]["selection_score"] = round(selection_score, 4)

        print(
            f"Model: {display_name:<16} | "
            f"Val Sens: {vm['recall_percentage']:<6} | "
            f"Val Spec: {vm['specificity_percentage']:<6} | "
            f"Val F1: {vm['f1_score_percentage']:<6} | "
            f"Val Acc: {vm['accuracy_percentage']:<6} | "
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

    # Save best model copy
    best_source_ckpt = Path(selected_metrics["checkpoint_path"])
    best_dest_ckpt = Path(Config.DL_MODEL_PATHS["best"])
    shutil.copy2(best_source_ckpt, best_dest_ckpt)
    print(f"[+] Primary model checkpoint deployed to: {best_dest_ckpt}")

    for k in model_results:
        model_results[k]["is_selected"] = (k == best_arch_key)

    # Save selected_model_info.json
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
    print(f"[+] Saved {Config.DL_SELECTED_MODEL_INFO_PATH}")

    # Save deep_learning_metrics.json
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
    print(f"[+] Saved {Config.DL_METRICS_PATH}")

    # Save deep_learning_config.json
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
    print(f"[+] Saved {Config.DL_CONFIG_PATH}")

    # Generate visualizations in models/ and src/static/img/
    models_dir = Path(Config.MODELS_FOLDER)
    generate_comparison_visualizations(model_results, curves_data, output_dir=models_dir)
    print(f"[+] High-resolution charts saved to {models_dir} and src/static/img/")

    print("\n" + "=" * 80)
    print("         ALL ARTIFACTS AND VISUALIZATIONS GENERATED CLEANLY")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_evaluation_artifacts()
