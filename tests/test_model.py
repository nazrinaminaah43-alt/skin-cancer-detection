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
        """All 5 classifier models and scaler artifacts should be loaded."""
        self.assertIsNotNone(self.model_service.model)
        self.assertIsNotNone(self.model_service.scaler)
        expected_keys = ["svm", "random_forest", "gradient_boosting", "logistic_regression", "knn"]
        for key in expected_keys:
            self.assertIn(key, self.model_service.models, f"Classifier '{key}' should be loaded in models dict")

    def test_available_classifiers_list(self):
        """Should return metadata for all 5 classifiers."""
        classifiers = self.model_service.get_available_classifiers()
        self.assertEqual(len(classifiers), 5)
        keys = [c["key"] for c in classifiers]
        for expected in ["svm", "random_forest", "gradient_boosting", "logistic_regression", "knn"]:
            self.assertIn(expected, keys)

    def test_metrics_loaded(self):
        """Metrics should be loaded from metrics.json."""
        metrics = self.model_service.get_performance_metrics()
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_individual_classifier_inference(self):
        """Each of the 5 classifiers should perform separate inference and return valid metrics."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 210
        cv2.circle(img, (128, 128), 50, (40, 50, 80), -1)
        preprocessed = self.preprocessor.process(img)

        classifier_keys = ["svm", "random_forest", "gradient_boosting", "logistic_regression", "knn"]
        for key in classifier_keys:
            res = self.model_service.predict(preprocessed, classifier_key=key)
            self.assertEqual(res["classifier_key"], key)
            self.assertIn(res["prediction"], ["BENIGN", "MALIGNANT"])
            self.assertGreaterEqual(res["confidence"], 50.0)
            self.assertLessEqual(res["confidence"], 100.0)
            self.assertIn("is_cancer", res)
            self.assertIn("model_accuracy", res)
            self.assertIn("classifiers_comparison", res)
            self.assertEqual(len(res["classifiers_comparison"]), 5)

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

    def test_consensus_determination(self):
        """Should calculate multi-classifier consensus agreement metrics."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 210
        cv2.circle(img, (128, 128), 50, (40, 50, 80), -1)

        preprocessed = self.preprocessor.process(img)
        res = self.model_service.predict(preprocessed)

        self.assertIn("consensus", res)
        con = res["consensus"]
        self.assertIn("prediction", con)
        self.assertIn("cancer_votes", con)
        self.assertIn("benign_votes", con)
        self.assertEqual(con["total_classifiers"], 5)
        self.assertEqual(con["cancer_votes"] + con["benign_votes"], 5)
        self.assertIn("agreement_ratio", con)
        self.assertIn("status_label", con)


if __name__ == "__main__":
    unittest.main()

