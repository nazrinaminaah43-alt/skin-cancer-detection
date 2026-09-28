"""
Unit tests for SVM Model loader and inference.
"""

import unittest
import numpy as np
import cv2

from src.preprocessing import ImagePreprocessor
from src.model import SkinCancerModel


class TestModelInference(unittest.TestCase):
    """Tests model prediction logic, probability boundaries, and metrics loading."""

    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        self.model_service = SkinCancerModel.get_instance()

    def test_model_is_loaded(self):
        """Model and scaler artifacts should be loaded."""
        self.assertIsNotNone(self.model_service.model)
        self.assertIsNotNone(self.model_service.scaler)

    def test_metrics_loaded(self):
        """Metrics should be loaded from metrics.json."""
        metrics = self.model_service.get_performance_metrics()
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_prediction_output_structure(self):
        """Inference should return prediction, confidence, probabilities and feature summary."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 210
        cv2.circle(img, (128, 128), 50, (40, 50, 80), -1)

        preprocessed = self.preprocessor.process(img)
        res = self.model_service.predict(preprocessed)

        self.assertIn(res["prediction"], ["BENIGN", "MALIGNANT"])
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertLessEqual(res["confidence"], 100.0)
        
        benign_p = res["probabilities"]["benign"]
        malignant_p = res["probabilities"]["malignant"]
        self.assertAlmostEqual(benign_p + malignant_p, 100.0, delta=0.5)

    def test_cancer_determination(self):
        """Should explicitly specify whether the lesion is skin cancer or not."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 210
        cv2.circle(img, (128, 128), 50, (40, 50, 80), -1)

        preprocessed = self.preprocessor.process(img)
        res = self.model_service.predict(preprocessed)

        self.assertIn("is_cancer", res)
        self.assertIsInstance(res["is_cancer"], bool)
        self.assertIn("cancer_status", res)
        self.assertIn(res["cancer_status"], ["NOT SKIN CANCER", "SKIN CANCER DETECTED"])
        self.assertIn("cancer_verdict", res)
        self.assertIn("cancer_explanation", res)
        self.assertIn("is_skin", res)


if __name__ == "__main__":
    unittest.main()
