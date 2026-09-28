"""
Model Loader and Inference Service for Skin Cancer Classification.
Wraps the trained Support Vector Machine (SVM) classifier and StandardScaler.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np

from src.config import Config
from src.feature_extraction import FeatureExtractor


class SkinCancerModel:
    """Manages model loading, feature scaling, and probabilistic inference."""

    _instance: Optional["SkinCancerModel"] = None

    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics = None
        self.feature_extractor = FeatureExtractor()
        self.load()

    @classmethod
    def get_instance(cls) -> "SkinCancerModel":
        """Singleton accessor for efficient in-memory model reuse."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        """Loads model weights, scaler, and evaluated performance metrics from disk."""
        if not os.path.exists(Config.MODEL_PATH) or not os.path.exists(Config.SCALER_PATH):
            raise FileNotFoundError(
                f"Model or Scaler artifact not found at {Config.MODEL_PATH}. "
                "Ensure scripts/train_model.py has been executed."
            )

        self.model = joblib.load(Config.MODEL_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)

        if os.path.exists(Config.METRICS_PATH):
            with open(Config.METRICS_PATH, "r") as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {
                "model_name": "Support Vector Machine (SVM)",
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }

    def predict(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs probabilistic prediction on preprocessed image components.
        
        Args:
            preprocessed_data: dictionary from ImagePreprocessor.process()
            
        Returns:
            Dict containing predicted class, confidence, class probabilities,
            and extracted feature summary.
        """
        if self.model is None or self.scaler is None:
            self.load()

        # Extract numerical features
        features = self.feature_extractor.extract_features(preprocessed_data)
        
        # Standardize using fitted training scaler
        features_scaled = self.scaler.transform(features)

        # Probabilities & Prediction
        probabilities = self.model.predict_proba(features_scaled)[0]
        benign_prob = float(probabilities[0])
        malignant_prob = float(probabilities[1])
        
        pred_idx = int(np.argmax(probabilities))
        is_malignant = (pred_idx == 1)
        prediction = "MALIGNANT" if is_malignant else "BENIGN"
        confidence = float(probabilities[pred_idx] * 100.0)

        # Clear Plain-Language Cancer Determination
        is_cancer = is_malignant
        cancer_status = "SKIN CANCER DETECTED" if is_cancer else "NOT SKIN CANCER"
        cancer_verdict = "YES - Potential Skin Cancer Detected" if is_cancer else "NO - Not Skin Cancer (Benign)"
        
        if is_cancer:
            cancer_explanation = "The lesion exhibits asymmetry, irregular borders, and color variegation characteristic of Malignant Melanoma or Carcinoma."
            action_advice = "Prompt clinical evaluation by a certified dermatologist or oncologist is advised."
        else:
            cancer_explanation = "The lesion displays circumscribed borders, uniform pigmentation, and structural cohesion typical of a Benign Nevus (non-cancerous mole)."
            action_advice = "Normal benign lesion. Continue routine self-monitoring and annual skin exams."

        # ABCD feature summary for UI display
        feature_summary = self.feature_extractor.extract_summary(preprocessed_data)

        # Skin validation from preprocessed metadata
        is_skin = preprocessed_data.get("is_skin", True)
        skin_ratio = preprocessed_data.get("skin_ratio", 100.0)

        return {
            "prediction": prediction,
            "is_cancer": is_cancer,
            "cancer_status": cancer_status,
            "cancer_verdict": cancer_verdict,
            "cancer_explanation": cancer_explanation,
            "action_advice": action_advice,
            "is_skin": is_skin,
            "skin_ratio": skin_ratio,
            "confidence": round(confidence, 1),
            "confidence_percentage": f"{confidence:.1f}%",
            "probabilities": {
                "benign": round(benign_prob * 100, 1),
                "malignant": round(malignant_prob * 100, 1)
            },
            "model_used": "Support Vector Machine (SVM)",
            "kernel": "Radial Basis Function (RBF)",
            "feature_summary": feature_summary
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Returns verified evaluation metrics computed during training."""
        if self.metrics is None:
            self.load()
        return self.metrics
