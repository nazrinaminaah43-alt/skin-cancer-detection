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

# Ensure directories exist
for folder in [DATA_DIR, UPLOADS_DIR, SAMPLES_DIR, MODELS_DIR]:
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
