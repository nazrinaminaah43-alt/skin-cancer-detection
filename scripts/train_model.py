"""
Train Multi-Classifier Machine Learning Suite for Skin Cancer Detection.
Trains, evaluates, and exports 5 diverse classifiers:
  1. Support Vector Machine (SVM - RBF Kernel)
  2. Random Forest Classifier
  3. Gradient Boosting Classifier
  4. Logistic Regression
  5. K-Nearest Neighbors (KNN)

Adheres strictly to ML best practices (train/test split BEFORE scaling, no data leakage),
evaluates on held-out test split, and exports models, scaler, confusion matrices, and verified metrics.
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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


def train_and_evaluate_all_classifiers(X: np.ndarray, y: np.ndarray, feature_names: list):
    """
    Train and evaluate 5 separate ML classifiers on identical stratified train/test split.
    Saves individual model artifacts, confusion matrices, and unified comparison metrics.
    """
    print("\nSplitting dataset into stratified 80% train and 20% test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Standardize features: fit ONLY on training set to prevent data leakage!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the fitted scaler
    joblib.dump(scaler, Config.SCALER_PATH)
    print(f"StandardScaler saved to: {Config.SCALER_PATH}")

    # Define the 5 Classifiers
    classifier_definitions = {
        "svm": {
            "name": "Support Vector Machine (SVM)",
            "short_name": "SVM (RBF)",
            "algorithm": "Support Vector Classifier with Radial Basis Function kernel",
            "model": CalibratedClassifierCV(
                estimator=SVC(C=2.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=42),
                cv=5
            ),
            "hyperparameters": {
                "C": 2.0,
                "kernel": "rbf",
                "gamma": "scale",
                "class_weight": "balanced",
                "calibration": "CalibratedClassifierCV (5-Fold)"
            },
            "path": Config.MODEL_PATHS["svm"],
            "color": "#0284c7"
        },
        "random_forest": {
            "name": "Random Forest Classifier",
            "short_name": "Random Forest",
            "algorithm": "Ensemble of 150 bagged decision trees with balanced bootstrapping",
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=4,
                class_weight="balanced",
                random_state=42
            ),
            "hyperparameters": {
                "n_estimators": 150,
                "max_depth": 8,
                "min_samples_split": 4,
                "class_weight": "balanced",
                "criterion": "gini"
            },
            "path": Config.MODEL_PATHS["random_forest"],
            "color": "#10b981"
        },
        "gradient_boosting": {
            "name": "Gradient Boosting Classifier",
            "short_name": "Gradient Boosting",
            "algorithm": "Sequentially boosted decision trees optimizing pseudo-residuals",
            "model": GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=3,
                random_state=42
            ),
            "hyperparameters": {
                "n_estimators": 120,
                "learning_rate": 0.08,
                "max_depth": 3,
                "loss": "log_loss"
            },
            "path": Config.MODEL_PATHS["gradient_boosting"],
            "color": "#f59e0b"
        },
        "logistic_regression": {
            "name": "Logistic Regression",
            "short_name": "Logistic Regression",
            "algorithm": "Regularized linear probabilistic binary classifier with balanced weights",
            "model": LogisticRegression(
                C=1.0,
                penalty="l2",
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),
            "hyperparameters": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 1000
            },
            "path": Config.MODEL_PATHS["logistic_regression"],
            "color": "#6366f1"
        },
        "knn": {
            "name": "K-Nearest Neighbors (KNN)",
            "short_name": "K-Nearest Neighbors",
            "algorithm": "Instance-based non-parametric classifier using distance-weighted voting",
            "model": KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="euclidean"
            ),
            "hyperparameters": {
                "n_neighbors": 5,
                "weights": "distance",
                "metric": "euclidean"
            },
            "path": Config.MODEL_PATHS["knn"],
            "color": "#ec4899"
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trained_metrics = {}
    comparison_summary = []

    print("\n" + "=" * 70)
    print(" TRAINING & EVALUATING ALL 5 CLASSIFIERS")
    print("=" * 70)

    for key, spec in classifier_definitions.items():
        print(f"\n--- Training {spec['name']} ---")
        model = spec["model"]

        # 5-fold cross validation score on train set
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        print(f"5-Fold CV Train Accuracy: {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")

        # Fit model on full training set
        model.fit(X_train_scaled, y_train)

        # Evaluate on held-out test set
        y_pred = model.predict(X_test_scaled)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"Test Accuracy  : {acc * 100:.2f}%")
        print(f"Test Precision : {prec * 100:.2f}%")
        print(f"Test Recall    : {rec * 100:.2f}%")
        print(f"Test F1-Score  : {f1 * 100:.2f}%")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # Save model artifact
        joblib.dump(model, spec["path"])
        print(f"Saved model to: {spec['path']}")

        # Save individual confusion matrix image
        cm_filename = f"confusion_matrix_{key}.png"
        cm_path_models = MODELS_DIR / cm_filename
        cm_path_static = STATIC_IMG_DIR / cm_filename

        plt.figure(figsize=(5.5, 4.5), dpi=150)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if key == "svm" else "Greens" if key == "random_forest" else "Oranges" if key == "gradient_boosting" else "Purples" if key == "logistic_regression" else "PuRd",
            cbar=False,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            annot_kws={"size": 13, "weight": "bold"}
        )
        plt.title(f"{spec['name']}\nConfusion Matrix (Test Evaluation)", fontsize=11, weight="bold", pad=10)
        plt.xlabel("Predicted Label", fontsize=10, weight="bold")
        plt.ylabel("Actual Label", fontsize=10, weight="bold")
        plt.tight_layout()
        plt.savefig(str(cm_path_models))
        plt.savefig(str(cm_path_static))
        plt.close()

        # Build metrics structure
        trained_metrics[key] = {
            "key": key,
            "model_name": spec["name"],
            "short_name": spec["short_name"],
            "algorithm": spec["algorithm"],
            "hyperparameters": spec["hyperparameters"],
            "classes": ["Benign", "Malignant"],
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "accuracy_percentage": f"{acc * 100:.1f}%",
            "precision_percentage": f"{prec * 100:.1f}%",
            "recall_percentage": f"{rec * 100:.1f}%",
            "f1_score_percentage": f"{f1 * 100:.1f}%",
            "cv_mean_accuracy": round(cv_mean, 4),
            "cv_std_accuracy": round(cv_std, 4),
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "labels": ["Benign", "Malignant"],
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            },
            "confusion_matrix_img": f"/static/img/{cm_filename}",
            "color": spec["color"]
        }

        comparison_summary.append({
            "key": key,
            "name": spec["name"],
            "short_name": spec["short_name"],
            "accuracy": round(acc, 4),
            "accuracy_percentage": f"{acc * 100:.1f}%",
            "precision_percentage": f"{prec * 100:.1f}%",
            "recall_percentage": f"{rec * 100:.1f}%",
            "f1_score_percentage": f"{f1 * 100:.1f}%",
            "cv_accuracy": f"{cv_mean * 100:.1f}%",
            "color": spec["color"]
        })

    # Sort summary by accuracy descending
    comparison_summary.sort(key=lambda x: x["accuracy"], reverse=True)
    best_classifier_key = comparison_summary[0]["key"]

    # Also keep default confusion matrix for backwards compatibility
    default_cm = MODELS_DIR / f"confusion_matrix_{best_classifier_key}.png"
    if default_cm.exists():
        import shutil
        shutil.copy(default_cm, Config.CONFUSION_MATRIX_IMG)
        shutil.copy(default_cm, STATIC_IMG_DIR / "confusion_matrix.png")

    # Generate Comparison Bar Chart
    plt.figure(figsize=(10, 5), dpi=150)
    names = [c["short_name"] for c in comparison_summary]
    accuracies = [c["accuracy"] * 100 for c in comparison_summary]
    colors = [c["color"] for c in comparison_summary]

    bars = plt.bar(names, accuracies, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.2)
    plt.ylim(50, 100)
    plt.ylabel("Accuracy (%)", fontsize=11, weight="bold")
    plt.title("Comparative Performance Across 5 ML Classifiers (ISIC Test Split)", fontsize=13, weight="bold", pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    comp_chart_path = MODELS_DIR / "models_comparison_chart.png"
    plt.savefig(str(comp_chart_path))
    plt.savefig(str(STATIC_IMG_DIR / "models_comparison_chart.png"))
    plt.close()

    # Compile Unified Metrics JSON
    best_metrics = trained_metrics[best_classifier_key]
    unified_metrics = {
        # Default top-level properties (backward compatible)
        "model_name": best_metrics["model_name"],
        "kernel": best_metrics.get("hyperparameters", {}).get("kernel", "N/A"),
        "hyperparameters": best_metrics["hyperparameters"],
        "classes": ["Benign", "Malignant"],
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1_score": best_metrics["f1_score"],
        "accuracy_percentage": best_metrics["accuracy_percentage"],
        "precision_percentage": best_metrics["precision_percentage"],
        "recall_percentage": best_metrics["recall_percentage"],
        "f1_score_percentage": best_metrics["f1_score_percentage"],
        "cv_mean_accuracy": best_metrics["cv_mean_accuracy"],
        "cv_std_accuracy": best_metrics["cv_std_accuracy"],
        "confusion_matrix": best_metrics["confusion_matrix"],
        "best_classifier": best_classifier_key,
        "dataset_split": {
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features_extracted": int(X.shape[1])
        },
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        # Multi-classifier properties
        "classifiers": trained_metrics,
        "comparison_summary": comparison_summary
    }

    with open(Config.METRICS_PATH, "w") as f:
        json.dump(unified_metrics, f, indent=2)
    print(f"\nUnified multi-classifier metrics saved to: {Config.METRICS_PATH}")

    print("\n" + "=" * 70)
    print(" CLASSIFIER COMPARISON BENCHMARK (TEST SPLIT)")
    print("=" * 70)
    for c in comparison_summary:
        print(f" • {c['name']:<32} : Acc={c['accuracy_percentage']:<6} | Prec={c['precision_percentage']:<6} | Rec={c['recall_percentage']:<6} | F1={c['f1_score_percentage']}")
    print(f"\nTop Performing Classifier: {trained_metrics[best_classifier_key]['model_name']} ({best_metrics['accuracy_percentage']})")
    print("=" * 70)

    return unified_metrics


if __name__ == "__main__":
    X, y, feature_names = extract_dataset_features()
    train_and_evaluate_all_classifiers(X, y, feature_names)
