"""
Train Support Vector Machine (SVM) model for Skin Cancer Detection.
Extracts OpenCV ABCD and Texture features, adheres to ML best practices (no data leakage),
evaluates on held-out test split, and exports model, scaler, and verified metrics.
"""

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import ImagePreprocessor
from src.feature_extraction import FeatureExtractor
from src.config import Config

DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODELS_DIR = Path(Config.MODELS_FOLDER)
STATIC_IMG_DIR = BASE_DIR / "src" / "static" / "img"
STATIC_IMG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def extract_dataset_features():
    """Extract features for all benign and malignant dataset images."""
    meta_path = DATASET_DIR / "dataset_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}. Run download_dataset.py first.")

    df = pd.read_csv(meta_path)
    preprocessor = ImagePreprocessor(target_size=Config.IMAGE_SIZE)
    extractor = FeatureExtractor()

    features_list = []
    labels = []
    valid_records = []

    print(f"Extracting OpenCV ABCD features from {len(df)} images...")
    for idx, row in df.iterrows():
        img_path = BASE_DIR / row["path"]
        if not img_path.exists():
            continue
            
        try:
            preprocessed = preprocessor.process(str(img_path))
            feat_vector = extractor.extract_features(preprocessed)
            features_list.append(feat_vector[0])
            
            # 0 for benign, 1 for malignant
            label = 0 if row["class"].lower() == "benign" else 1
            labels.append(label)
            valid_records.append(row["filename"])
        except Exception as e:
            print(f"Error extracting features for {img_path}: {e}")

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    print(f"Extracted feature matrix shape: {X.shape}, labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)} (0: Benign, 1: Malignant)")
    return X, y, extractor.feature_names


def train_and_evaluate(X: np.ndarray, y: np.ndarray, feature_names: list):
    """
    Train SVM classifier with strict featurization ordering (split BEFORE scaling),
    calculate real performance metrics, and save all artifacts.
    """
    print("\nSplitting dataset into stratified 80% train and 20% test sets...")
    # Stratified split ensures equal class proportions in train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Standardize features: fit ONLY on training set to prevent data leakage!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Base Support Vector Machine Classifier
    print("Training Support Vector Machine (RBF Kernel, balanced weights)...")
    base_svm = SVC(
        C=2.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )
    
    # Calibrated probability model via cross-validation (replaces deprecated probability=True)
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=5)

    # 5-fold cross validation score on train set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(calibrated_svm, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    print(f"5-Fold Cross-Validation Accuracy on Train set: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Fit final calibrated model on training set
    calibrated_svm.fit(X_train_scaled, y_train)

    # Evaluate on held-out test set
    y_pred = calibrated_svm.predict(X_test_scaled)
    y_proba = calibrated_svm.predict_proba(X_test_scaled)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("\n=== Model Performance Evaluation on Test Set ===")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"[[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    # Save model and scaler
    joblib.dump(calibrated_svm, Config.MODEL_PATH)
    joblib.dump(scaler, Config.SCALER_PATH)
    print(f"Model saved to: {Config.MODEL_PATH}")
    print(f"Scaler saved to: {Config.SCALER_PATH}")

    # Generate Confusion Matrix Visualization
    plt.figure(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Benign", "Predicted Malignant"],
        yticklabels=["Actual Benign", "Actual Malignant"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("SVM Confusion Matrix (Test Evaluation)", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    
    # Save confusion matrix plot to both models/ and static/img/
    plt.savefig(Config.CONFUSION_MATRIX_IMG)
    static_cm_path = STATIC_IMG_DIR / "confusion_matrix.png"
    plt.savefig(str(static_cm_path))
    plt.close()
    print(f"Confusion matrix plot saved to: {Config.CONFUSION_MATRIX_IMG} and {static_cm_path}")

    # Save verified metrics to JSON
    metrics_data = {
        "model_name": "Support Vector Machine (SVM)",
        "kernel": "Radial Basis Function (RBF)",
        "hyperparameters": {
            "C": 2.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "calibration": "CalibratedClassifierCV (5-Fold)"
        },
        "classes": ["Benign", "Malignant"],
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "accuracy_percentage": f"{acc * 100:.1f}%",
        "precision_percentage": f"{prec * 100:.1f}%",
        "recall_percentage": f"{rec * 100:.1f}%",
        "f1_score_percentage": f"{f1 * 100:.1f}%",
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": ["Benign", "Malignant"],
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "dataset_split": {
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features_extracted": int(X.shape[1])
        },
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    with open(Config.METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to: {Config.METRICS_PATH}")

    return metrics_data


if __name__ == "__main__":
    X, y, feature_names = extract_dataset_features()
    train_and_evaluate(X, y, feature_names)
