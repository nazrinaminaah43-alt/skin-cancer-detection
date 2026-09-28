"""
Configuration settings for Skin Cancer Detection Application.
Supports local SQLite execution and cloud database deployments via environment variables.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"

DL_MODELS_DIR = MODELS_DIR / "deep_learning"

# Ensure directories exist
for folder in [DATA_DIR, UPLOADS_DIR, SAMPLES_DIR, MODELS_DIR, DL_MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

class Config:
    """Base application configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "skin-cancer-detection-secret-key-2026")
    
    # Upload limits & formats
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max image size
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
    
    # Directories
    UPLOADS_FOLDER = str(UPLOADS_DIR)
    SAMPLES_FOLDER = str(SAMPLES_DIR)
    MODELS_FOLDER = str(MODELS_DIR)
    
    # Model Artifacts
    MODEL_PATH = str(MODELS_DIR / "skin_cancer_svm.joblib")
    SCALER_PATH = str(MODELS_DIR / "scaler.joblib")
    METRICS_PATH = str(MODELS_DIR / "metrics.json")
    CONFUSION_MATRIX_IMG = str(MODELS_DIR / "confusion_matrix.png")

    MODEL_PATHS = {
        "svm": str(MODELS_DIR / "skin_cancer_svm.joblib"),
        "random_forest": str(MODELS_DIR / "skin_cancer_rf.joblib"),
        "gradient_boosting": str(MODELS_DIR / "skin_cancer_gb.joblib"),
        "logistic_regression": str(MODELS_DIR / "skin_cancer_lr.joblib"),
        "knn": str(MODELS_DIR / "skin_cancer_knn.joblib"),
    }
    
    # Database Configuration (SQLite local, adaptable to PostgreSQL/MySQL via env var)
    db_url = os.environ.get("DATABASE_URL")
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    SQLALCHEMY_DATABASE_URI = db_url or f"sqlite:///{DATA_DIR / 'database.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Image Preprocessing target resolution
    IMAGE_SIZE = (256, 256)

    # ==============================================================================
    # Deep Learning Configuration & Model Checkpoint Paths
    # ==============================================================================
    DATASET_DIR = str(DATA_DIR / "dataset")
    SPLITS_MANIFEST_PATH = str(DATA_DIR / "dataset" / "splits_manifest.json")
    
    DL_IMAGE_SIZE = (224, 224)
    DL_BATCH_SIZE = 16
    DL_RANDOM_SEED = 42
    DL_EPOCHS = 12
    DL_LEARNING_RATE = 1e-4
    DL_NUM_CLASSES = 2
    DL_CLASS_NAMES = ["benign", "malignant"]
    DL_CLASS_MAPPING = {"benign": 0, "malignant": 1}
    
    DL_MODELS_DIR = str(MODELS_DIR / "deep_learning")
    DL_METRICS_PATH = str(MODELS_DIR / "deep_learning_metrics.json")
    DL_SELECTED_MODEL_INFO_PATH = str(MODELS_DIR / "selected_model_info.json")
    DL_CONFIG_PATH = str(MODELS_DIR / "deep_learning_config.json")
    
    DL_MODEL_PATHS = {
        "custom_cnn": str(MODELS_DIR / "deep_learning" / "custom_cnn.pt"),
        "mobilenet_v2": str(MODELS_DIR / "deep_learning" / "mobilenet_v2.pt"),
        "resnet50": str(MODELS_DIR / "deep_learning" / "resnet50.pt"),
        "efficientnet_b0": str(MODELS_DIR / "deep_learning" / "efficientnet_b0.pt"),
        "vgg16": str(MODELS_DIR / "deep_learning" / "vgg16.pt"),
        "best": str(MODELS_DIR / "deep_learning" / "best_deep_learning_model.pt"),
    }

