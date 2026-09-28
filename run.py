"""
Main application launcher for Skin Cancer Detection Dashboard.
Executes Flask server with local SQLite database and 5 pre-trained ML classifiers.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import Config
from src.app import create_app


def verify_environment():
    """Verifies that all required artifacts and directories are ready."""
    print("=" * 65)
    print("      SKIN CANCER DETECTION - MACHINE LEARNING SYSTEM")
    print("=" * 65)
    print(f"[*] Base Directory   : {BASE_DIR}")
    print(f"[*] Database URI     : {Config.SQLALCHEMY_DATABASE_URI}")
    print(f"[*] Uploads Folder   : {Config.UPLOADS_FOLDER}")
    print(f"[*] Models Folder    : {Config.MODELS_FOLDER}")

    model_paths = getattr(Config, "MODEL_PATHS", {"svm": Config.MODEL_PATH})
    all_models_ready = all(os.path.exists(p) for p in model_paths.values())
    scaler_ready = os.path.exists(Config.SCALER_PATH)
    metrics_ready = os.path.exists(Config.METRICS_PATH)

    if all_models_ready and scaler_ready and metrics_ready:
        print(f"[+] All {len(model_paths)} ML Classifiers, Scaler, and Metrics verified: READY")
    elif os.path.exists(Config.MODEL_PATH) and scaler_ready and metrics_ready:
        print("[+] Primary SVM Model and Scaler artifacts verified: READY")
    else:
        print("[!] Model artifacts missing. Auto-training will be initiated on first run.")

    print("=" * 65)


app = create_app()

if __name__ == "__main__":
    verify_environment()
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1")

    print(f"\n[*] Starting Flask Web Dashboard at: http://{host}:{port}")
    print("[*] Health Check Endpoint        : http://127.0.0.1:5000/api/health")
    print("[*] Model Performance Endpoint   : http://127.0.0.1:5000/api/model-info")
    print("[*] Prediction History Endpoint  : http://127.0.0.1:5000/api/history\n")

    app.run(host=host, port=port, debug=debug)
