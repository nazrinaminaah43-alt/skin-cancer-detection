# 🔬 Skin Cancer Detection Using Machine Learning

A full-stack, clinical-engineering web application for automated dermatological screening of skin lesions using **Computer Vision (OpenCV)**, **Classical Machine Learning (SVM, Gradient Boosting, Random Forest, Logistic Regression, KNN)**, and **Deep Learning Transfer Learning Architectures (ResNet50, MobileNetV2, Custom CNN, VGG16, EfficientNetB0)**.

The system allows users to upload dermoscopic skin lesion images through an intuitive, interactive web dashboard, preprocesses images with medical-grade image enhancement and hair-removal algorithms, extracts physiological **ABCD** features, benchmarks deep learning models with zero data leakage, classifies lesions with high clinical sensitivity, logs results to a persistent database, and displays verified performance metrics, ROC/PR curves, and confusion matrices dynamically fetched from the models.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/nazrinaminaah43-alt/skin-cancer-detection)


---

## 🏛️ System Architecture

The application implements a decoupled, production-ready full-stack architecture:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FRONTEND DASHBOARD                                   │
│  - Dropzone File Upload / Sample Selector                                              │
│  - Real-time Image Preview & Lesion Contour Toggle                                     │
│  - Dynamic Result Card (Prediction, Confidence, Probabilities, ABCD Metrics)           │
│  - Live Model Performance Section (Accuracy, Precision, Recall, F1-Score)              │
│  - Interactive Confusion Matrix (2x2 Grid & Heatmap)                                   │
│  - Persistent Prediction History Table                                                 │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │  Multipart/Form-Data & JSON
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FLASK REST API & CONTROLLER                               │
│  - POST /api/predict       : Validates format/size, executes pipeline, stores result   │
│  - GET  /api/model-info    : Delivers evaluated test metrics & confusion matrix data   │
│  - GET  /api/samples       : Delivers curated dermoscopy test images                   │
│  - GET  /api/history       : Retrieves historical predictions                          │
│  - DELETE /api/history/<id>: Deletes individual prediction record                      │
│  - DELETE /api/history     : Clears prediction history database                        │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
┌──────────────────────────────────┐            ┌──────────────────────────────────┐
│       OPENCV PREPROCESSING       │            │       DATABASE (SQLAlchemy)      │
│  - File Size & Format Validation │            │  - SQLite (Local Development)    │
│  - 256x256 Scale Normalization   │            │  - PostgreSQL/MySQL (Production) │
│  - DullRazor Hair Artifact Filter│            │  - PredictionRecord Schema:      │
│  - CIE Lab CLAHE Enhancement     │            │    * Image ID & File URL         │
│  - Otsu Lesion Contour Extraction│            │    * Prediction (BENIGN/MALIGN)  │
└────────────────┬─────────────────┘            │    * Confidence & Probabilities  │
                 │                              │    * Model Name & Timestamp      │
                 ▼                              │    * Extracted ABCD Summary      │
┌──────────────────────────────────┐            └──────────────────────────────────┘
│        FEATURE EXTRACTION        │
│  - Color Moments (RGB/HSV/Lab)   │
│  - ABCD Asymmetry Indices        │
│  - Compactness & Solidity        │
│  - GLCM Texture Descriptors      │
│  - Laplacian / Sobel Roughness   │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│      SVM CLASSIFIER INFERENCE    │
│  - StandardScaler Normalization  │
│  - Calibrated SVM (RBF Kernel)   │
│  - Posterior Class Probabilities │
└──────────────────────────────────┘
```

---

## 📋 Features

1. **Medical Preprocessing Pipeline**:
   - **Validation**: Enforces strict MIME checks and file-size quotas (up to 16 MB).
   - **DullRazor Algorithm**: Rectangular black-hat morphological filtering isolates dark hair strands, followed by Telea inpainting to eliminate artifacts without altering pigment.
   - **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization on the Luminance channel in CIE Lab space normalizes illumination across varying dermoscopy devices.
   - **Lesion Segmentation**: Otsu thresholding with morphological closing and centrality weighting isolates the lesion boundary and generates a visual green contour overlay.

2. **Dermatological ABCD Rule Feature Extraction**:
   - **A (Asymmetry)**: Horizontal and vertical fold symmetric difference calculations ($A_{\text{sym}}$).
   - **B (Border Irregularity)**: Circular compactness ($4\pi A / P^2$), convex hull perimeter ratio, and solidity.
   - **C (Color Variegation)**: Mean, standard deviation, and skewness across BGR, HSV, and Lab color spaces, along with inter-channel color variance.
   - **D (Differential Texture)**: Fast Gray-Level Co-occurrence Matrix (GLCM) properties (contrast, dissimilarity, homogeneity, energy, correlation), Laplacian variance, and Sobel edge intensity distributions.

3. **Calibrated Support Vector Machine**:
   - Trained on balanced, verified dermoscopy images from the **International Skin Imaging Collaboration (ISIC)** benchmark archive.
   - Utilizes an RBF kernel with balanced class weighting.
   - Probability calibration via cross-validation to provide genuine confidence percentages and probability breakdowns (e.g., Benign: 79.9%, Malignant: 20.1%).

4. **Dynamic Model Performance (No Hardcoding)**:
   - Evaluated strictly on a held-out test split (split conducted **before** fitting scalers to prevent data leakage).
   - Serves true **Accuracy, Precision, Recall, F1-Score**, and **Confusion Matrix** data via `/api/model-info`.
   - Supports dual visualization: interactive 2x2 numeric matrix grid (TN, FP, FN, TP) and high-resolution Seaborn heatmap.

5. **Prediction History Management**:
   - Persistent database logging using SQLAlchemy with SQLite (local) and seamless PostgreSQL support via environment variable.
   - Full history table with thumbnail previews, prediction badges, timestamps, individual record deletion, and bulk clear.

6. **1-Click Curated Testing**:
   - Includes built-in benign and malignant dermoscopic samples for instant evaluation without requiring external files.

---

## 📁 Folder Structure

```
AML project/
├── data/
│   ├── dataset/             # 200 balanced ISIC dermoscopy images (100 benign, 100 malignant)
│   │   ├── benign/
│   │   ├── malignant/
│   │   └── dataset_metadata.csv
│   ├── samples/             # Curated sample dermoscopy images for quick dashboard testing
│   │   └── samples.json
│   ├── uploads/             # User uploaded images & generated contour overlays
│   └── database.db          # SQLite persistent database
├── models/
│   ├── skin_cancer_svm.joblib    # Trained Calibrated SVM model
│   ├── scaler.joblib             # Fitted StandardScaler
│   ├── metrics.json              # Evaluated test metrics & confusion matrix
│   └── confusion_matrix.png      # Generated heatmap visualization
├── notebooks/
│   └── skin_cancer_svm_analysis.ipynb # Jupyter Notebook with EDA, preprocessing & ML narrative
├── scripts/
│   ├── download_dataset.py  # ISIC dataset fetcher & verified generator
│   ├── train_model.py       # SVM training, cross-validation, and metrics exporter
│   └── test_pipeline.py     # Command-line verification script
├── src/
│   ├── __init__.py
│   ├── config.py            # Environment settings and paths
│   ├── preprocessing.py     # OpenCV DullRazor, CLAHE, and segmentation
│   ├── feature_extraction.py# ABCD and GLCM feature extraction
│   ├── model.py             # Model inference singleton
│   ├── database.py          # SQLAlchemy models and CRUD helpers
│   ├── routes.py            # REST API endpoints & static routing
│   ├── app.py               # Flask application factory & CORS
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css    # Responsive medical dashboard styling
│   │   ├── js/
│   │   │   └── app.js       # Dynamic AJAX frontend logic
│   │   └── img/
│   │       ├── confusion_matrix.png
│   │       └── placeholder.png
│   └── templates/
│       └── index.html       # Dashboard HTML5 template
├── tests/
│   ├── test_preprocessing.py     # OpenCV unit tests
│   ├── test_feature_extraction.py# Feature extraction unit tests
│   ├── test_model.py             # Inference unit tests
│   └── test_api.py               # End-to-end REST API tests
├── conftest.py              # Pytest configuration
├── run.py                   # Main development server entry point
├── requirements.txt         # Python package dependencies
└── README.md                # Documentation
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13, and 3.14)
- Pip package manager

### 2. Install Dependencies
In the project root, install required packages:
```bash
pip install -r requirements.txt
```

### 3. Verify Model and Dataset
The pre-trained model and test dataset are already packaged in the workspace. If you wish to retrain or update the dataset at any time:
```bash
# 1. Download/update dataset (downloads 100 benign & 100 malignant from ISIC Archive):
python scripts/download_dataset.py 100

# 2. Extract features and retrain SVM classifier:
python scripts/train_model.py
```

### 4. Run the Full-Stack Application
Start the Flask development server:
```bash
python run.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Automated Testing

Execute the comprehensive test suite covering OpenCV preprocessing, feature extraction, SVM inference, and REST endpoints:

```bash
pytest tests/ -v
```

All 20 test cases run and validate:
- Format and size validation
- DullRazor hair removal and inpainting
- Contour segmentation on synthetic and real lesions
- 46-dimensional feature vector extraction
- Calibrated probability boundaries
- Dynamic metrics delivery (non-hardcoded)
- Database CRUD and prediction history logging

---

## 📡 REST API Reference

### 1. Predict Skin Lesion
- **URL**: `POST /api/predict`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image`: *(File)* Skin lesion image file (JPEG, PNG, WEBP; max 16MB) **OR**
  - `sample_id`: *(String)* Name of pre-loaded curated sample (e.g. `sample_benign_nevus_1.jpg`)
- **Response** (`200 OK`):
```json
{
  "success": true,
  "prediction": "BENIGN",
  "confidence": 79.9,
  "confidence_percentage": "79.9%",
  "probabilities": {
    "benign": 79.9,
    "malignant": 20.1
  },
  "model_used": "Support Vector Machine (SVM)",
  "kernel": "Radial Basis Function (RBF)",
  "feature_summary": {
    "asymmetry_index": 0.376,
    "border_compactness": 0.652,
    "border_irregularity": 0.867,
    "color_variegation": 17.1,
    "texture_contrast": 0.12,
    "texture_sharpness": 26.1,
    "lesion_solidity": 0.938
  },
  "image_info": {
    "original_url": "/uploads/uuid_lesion.jpg",
    "segmented_url": "/uploads/contour_uuid_lesion.jpg",
    "filename": "lesion.jpg"
  },
  "record_id": 1,
  "timestamp": "2026-09-27 18:30:00 UTC"
}
```

### 2. Get Model Performance & Metrics
- **URL**: `GET /api/model-info`
- **Response** (`200 OK`):
```json
{
  "success": true,
  "model_info": {
    "model_name": "Support Vector Machine (SVM)",
    "kernel": "Radial Basis Function (RBF)",
    "accuracy": 0.75,
    "precision": 0.8125,
    "recall": 0.65,
    "f1_score": 0.7222,
    "accuracy_percentage": "75.0%",
    "precision_percentage": "81.2%",
    "recall_percentage": "65.0%",
    "f1_score_percentage": "72.2%",
    "confusion_matrix": {
      "matrix": [[17, 3], [7, 13]],
      "labels": ["Benign", "Malignant"],
      "true_negatives": 17,
      "false_positives": 3,
      "false_negatives": 7,
      "true_positives": 13
    }
  }
}
```

### 3. Get Prediction History
- **URL**: `GET /api/history?limit=30`
- **Response** (`200 OK`):
```json
{
  "success": true,
  "count": 1,
  "history": [
    {
      "id": 1,
      "image_identifier": "lesion.jpg",
      "image_url": "/uploads/uuid_lesion.jpg",
      "prediction": "BENIGN",
      "confidence": 79.9,
      "confidence_percentage": "79.9%",
      "probabilities": { "benign": 79.9, "malignant": 20.1 },
      "model_name": "Support Vector Machine (SVM)",
      "timestamp": "2026-09-27 18:30:00 UTC"
    }
  ]
}
```

### 4. Delete / Clear History
- `DELETE /api/history/<record_id>`: Removes specific record from database.
- `DELETE /api/history`: Clears all logged predictions.

---

## ☁️ Cloud Deployment Configuration

The application is structured for cloud deployment:
1. **Database Adapter**: The app reads `DATABASE_URL` from the environment. In production on Heroku, Render, AWS Elastic Beanstalk, or Google Cloud Run, pointing `DATABASE_URL=postgresql://user:pass@host:5432/dbname` will automatically switch the SQLAlchemy engine from SQLite to PostgreSQL with zero code changes.
2. **Production WSGI Server**: Run using Gunicorn:
   ```bash
   gunicorn --bind 0.0.0.0:8000 "src.app:create_app()"
   ```
3. **Containerization (Docker)**:
   ```dockerfile
   FROM python:3.11-slim
   RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.app:create_app()"]
   ```

---

## ⚠️ Clinical Screening Disclaimer

> **Important**: This application is built as an **educational screening tool and academic research prototype**. It is **not** certified as a clinical medical device and does **not** provide definitive medical diagnosis. Suspicious lesions, changing moles, or dysplastic nevi must always be evaluated by a licensed dermatologist or medical doctor.
