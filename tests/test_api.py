"""
Integration tests for Flask REST API endpoints and Database operations.
"""

import io
import unittest
import numpy as np
import cv2

from src.app import create_app
from src.config import Config
from src.database import db


class TestApiEndpoints(unittest.TestCase):
    """Tests all REST API endpoints."""

    def setUp(self):
        class TestConfig(Config):
            TESTING = True
            SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"

        self.app = create_app(TestConfig)
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_dashboard_page(self):
        """Should serve dashboard HTML with page title."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"SKIN CANCER DETECTION", resp.data)

    def test_health_check(self):
        """Should return health status."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "healthy")

    def test_model_info_endpoint(self):
        """Should return verified metrics without hardcoded defaults."""
        resp = self.client.get("/api/model-info")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("accuracy", data["model_info"])
        self.assertIn("confusion_matrix", data["model_info"])

    def test_classifiers_endpoint(self):
        """Should return list of all 5 available classifiers with their test accuracies."""
        resp = self.client.get("/api/classifiers")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("classifiers", data)
        self.assertEqual(len(data["classifiers"]), 5)
        keys = [c["key"] for c in data["classifiers"]]
        for expected in ["svm", "random_forest", "gradient_boosting", "logistic_regression", "knn"]:
            self.assertIn(expected, keys)

    def test_model_info_specific_classifier(self):
        """Should return metrics for a specific requested classifier."""
        for c_key in ["svm", "random_forest", "gradient_boosting", "logistic_regression", "knn"]:
            resp = self.client.get(f"/api/model-info?classifier={c_key}")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertEqual(data["model_info"]["key"], c_key)
            self.assertIn("accuracy_percentage", data["model_info"])
            self.assertIn("f1_score_percentage", data["model_info"])

    def test_samples_endpoint(self):
        """Should return sample images."""
        resp = self.client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIsInstance(data["samples"], list)

    def test_predict_no_input(self):
        """Should reject request missing image or sample_id."""
        resp = self.client.post("/api/predict", data={})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])

    def test_predict_with_sample_id(self):
        """Should perform prediction using a curated sample ID."""
        # Use existing sample
        resp = self.client.post("/api/predict", data={"sample_id": "sample_benign_nevus_1.jpg"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn(data["prediction"], ["BENIGN", "MALIGNANT"])
        self.assertIn("confidence", data)
        self.assertIn("feature_summary", data)
        self.assertIn("record_id", data)

    def test_predict_with_different_classifiers(self):
        """Should support selecting any of the 5 classifiers for prediction."""
        for c_key in ["svm", "random_forest", "gradient_boosting", "logistic_regression", "knn"]:
            resp = self.client.post("/api/predict", data={
                "sample_id": "sample_benign_nevus_1.jpg",
                "classifier": c_key
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertEqual(data["classifier_key"], c_key)
            self.assertIn("model_used", data)
            self.assertIn("model_accuracy", data)
            self.assertIn("classifiers_comparison", data)
            self.assertEqual(len(data["classifiers_comparison"]), 5)

    def test_predict_with_file_upload(self):
        """Should accept file upload and return prediction + log to DB."""
        # Create a mock lesion image in memory
        img = np.ones((256, 256, 3), dtype=np.uint8) * 200
        cv2.circle(img, (128, 128), 50, (30, 40, 70), -1)
        _, encoded = cv2.imencode(".jpg", img)

        file_storage = (io.BytesIO(encoded.tobytes()), "test_lesion.jpg")

        resp = self.client.post(
            "/api/predict",
            data={"image": file_storage},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)

    def test_history_endpoints_flow(self):
        """Should log prediction, fetch history, delete record, and clear history."""
        # 1. Run a prediction
        resp = self.client.post("/api/predict", data={"sample_id": "sample_benign_nevus_1.jpg"})
        self.assertEqual(resp.status_code, 200)
        rec_id = resp.get_json()["record_id"]

        # 2. Get history
        hist_resp = self.client.get("/api/history")
        self.assertEqual(hist_resp.status_code, 200)
        hist_data = hist_resp.get_json()
        self.assertEqual(hist_data["count"], 1)
        self.assertEqual(hist_data["history"][0]["id"], rec_id)

        # 3. Delete specific record
        del_resp = self.client.delete(f"/api/history/{rec_id}")
        self.assertEqual(del_resp.status_code, 200)

        # 4. Confirm history is empty
        hist_resp2 = self.client.get("/api/history")
        self.assertEqual(hist_resp2.get_json()["count"], 0)


if __name__ == "__main__":
    unittest.main()
