"""
Reusable Prediction and Preprocessing Service for Deep Learning Models.
Loads trained checkpoints and executes deterministic inference on dermoscopy images.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

from src.config import Config
from src.deep_learning import (
    get_eval_transforms,
    load_model_checkpoint,
    IMAGENET_MEAN,
    IMAGENET_STD
)


class DeepLearningPredictor:
    """
    Manages loading of trained PyTorch deep learning models (Custom CNN, MobileNetV2,
    ResNet50, EfficientNetB0, VGG16) and executing clinical inference.
    """

    _instance: Optional["DeepLearningPredictor"] = None

    def __init__(self, default_model: str = "best"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models: Dict[str, Any] = {}
        self.checkpoints: Dict[str, Any] = {}
        self.default_model_key = default_model
        self.eval_transforms = get_eval_transforms(image_size=Config.DL_IMAGE_SIZE)
        self.load_default()

    @classmethod
    def get_instance(cls) -> "DeepLearningPredictor":
        """Singleton accessor for efficient in-memory model reuse."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_key: str = "best") -> bool:
        """Loads a specific deep learning model checkpoint into memory."""
        key = model_key.lower().strip()
        if key in self.loaded_models:
            return True

        checkpoint_path = Config.DL_MODEL_PATHS.get(key)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            # Fallback to best model or any existing model
            best_path = Config.DL_MODEL_PATHS.get("best")
            if best_path and os.path.exists(best_path):
                checkpoint_path = best_path
            else:
                for k, p in Config.DL_MODEL_PATHS.items():
                    if os.path.exists(p):
                        checkpoint_path = p
                        key = k
                        break

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return False

        try:
            model, checkpoint = load_model_checkpoint(checkpoint_path, device=self.device)
            self.loaded_models[key] = model
            self.checkpoints[key] = checkpoint
            return True
        except Exception as e:
            print(f"Warning: Could not load deep learning model '{model_key}': {e}")
            return False

    def load_default(self):
        """Attempts to load the selected best model, falling back to any available."""
        self.load_model(self.default_model_key)

    def preprocess_image(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Converts any valid image input format into the exact normalized tensor
        required by the deep learning models.
        """
        pil_img = None
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if cv_img is None:
                raise ValueError("Could not decode image from provided byte stream.")
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
        elif isinstance(image_input, np.ndarray):
            # If OpenCV BGR (3 channels), convert to RGB
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            elif len(image_input.shape) == 2:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            else:
                rgb_img = image_input
            pil_img = Image.fromarray(rgb_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.eval_transforms(pil_img)
        return tensor.unsqueeze(0).to(self.device)  # Add batch dimension: [1, 3, 224, 224]

    def predict(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes deep learning inference and returns structured clinical probabilities.
        """
        target_key = (model_name or self.default_model_key).lower().strip()
        if target_key not in self.loaded_models:
            loaded = self.load_model(target_key)
            if not loaded:
                # Try default
                if not self.loaded_models:
                    raise FileNotFoundError(
                        "No deep learning model checkpoints found. Please run scripts/train_models.py first."
                    )
                target_key = next(iter(self.loaded_models.keys()))

        model = self.loaded_models[target_key]
        checkpoint = self.checkpoints[target_key]
        arch_title = checkpoint.get("architecture_name", target_key).replace("_", " ").title()

        input_tensor = self.preprocess_image(image_input)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        benign_prob = float(probs[0])
        malignant_prob = float(probs[1])

        pred_idx = int(np.argmax(probs))
        class_to_idx = checkpoint.get("class_to_idx", {"benign": 0, "malignant": 1})
        idx_to_class = checkpoint.get("idx_to_class", {0: "benign", 1: "malignant"})
        pred_label = idx_to_class.get(pred_idx, "malignant" if pred_idx == 1 else "benign").upper()

        is_cancer = (pred_label == "MALIGNANT")
        confidence = (malignant_prob if is_cancer else benign_prob) * 100.0

        if is_cancer:
            cancer_status = "SKIN CANCER DETECTED"
            cancer_verdict = "YES — Potential Skin Cancer Detected (Malignant Melanoma)"
            cancer_explanation = (
                "The deep learning neural network detected high-order structural features "
                "consistent with malignant melanoma. Immediate clinical dermoscopic evaluation is recommended."
            )
        else:
            cancer_status = "NOT SKIN CANCER"
            cancer_verdict = "NO — Not Skin Cancer (Benign Lesion)"
            cancer_explanation = (
                "The deep learning neural network detected benign morphological patterns "
                "characteristic of non-cancerous melanocytic nevi."
            )

        metrics = checkpoint.get("metrics", {})

        return {
            "prediction": pred_label,
            "is_cancer": is_cancer,
            "cancer_status": cancer_status,
            "cancer_verdict": cancer_verdict,
            "cancer_explanation": cancer_explanation,
            "confidence": round(confidence, 1),
            "confidence_percentage": f"{confidence:.1f}%",
            "probabilities": {
                "benign": round(benign_prob * 100.0, 1),
                "malignant": round(malignant_prob * 100.0, 1)
            },
            "model_used": f"{arch_title} (Deep Learning)",
            "architecture": target_key,
            "validation_sensitivity": metrics.get("validation", {}).get("recall_percentage", "N/A"),
            "test_accuracy": metrics.get("test", {}).get("accuracy_percentage", "N/A"),
            "disclaimer": (
                "Educational & Research Screening Tool: This assessment is generated by a deep learning "
                "model for academic and screening demonstration only. It is not a diagnostic test."
            )
        }


def predict_image(image_input: Any, model_name: str = "best") -> Dict[str, Any]:
    """Convenience standalone prediction function."""
    predictor = DeepLearningPredictor.get_instance()
    return predictor.predict(image_input, model_name=model_name)
