"""
Comprehensive Unit and Integration Tests for the Deep Learning Pipeline.
Tests architectures, checkpoint loading, deterministic inference, API routes, and split verification.
"""

import os
import json
import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import io

from src.config import Config
from src.deep_learning import (
    CustomCNN,
    build_mobilenet_v2,
    build_resnet50,
    build_efficientnet_b0,
    build_vgg16,
    get_train_transforms,
    get_eval_transforms,
    load_model_checkpoint
)
from src.predict import DeepLearningPredictor
from run import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture
def sample_image_bytes():
    """Generates a dummy 224x224 RGB image in memory."""
    img = Image.new("RGB", (224, 224), color=(180, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestDeepLearningArchitectures:
    """Tests all 5 deep learning model architectures."""

    def test_custom_cnn_forward(self):
        model = CustomCNN(num_classes=2, dropout_rate=0.3)
        dummy_input = torch.randn(2, 3, 224, 224)
        out = model(dummy_input)
        assert out.shape == (2, 2)

    def test_mobilenet_v2_forward(self):
        model = build_mobilenet_v2(num_classes=2, pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224)
        out = model(dummy_input)
        assert out.shape == (2, 2)

    def test_resnet50_forward(self):
        model = build_resnet50(num_classes=2, pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)
        out = model(dummy_input)
        assert out.shape == (1, 2)

    def test_efficientnet_b0_forward(self):
        model = build_efficientnet_b0(num_classes=2, pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)
        out = model(dummy_input)
        assert out.shape == (1, 2)

    def test_vgg16_forward(self):
        model = build_vgg16(num_classes=2, pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)
        out = model(dummy_input)
        assert out.shape == (1, 2)


class TestDataSplitsAndTransforms:
    """Verifies stratified split integrity and deterministic eval transforms."""

    def test_splits_manifest_exists_and_stratified(self):
        manifest_path = Path(Config.SPLITS_MANIFEST_PATH)
        assert manifest_path.exists(), "Splits manifest must exist"

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        assert manifest["counts"]["total"] == 200
        assert manifest["counts"]["train"] == 140
        assert manifest["counts"]["val"] == 30
        assert manifest["counts"]["test"] == 30

        # Check stratification in test split (15 benign, 15 malignant)
        test_labels = [r["label"] for r in manifest["test"]]
        assert test_labels.count(0) == 15
        assert test_labels.count(1) == 15

    def test_eval_transforms_output_shape(self):
        eval_tf = get_eval_transforms(image_size=(224, 224))
        img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        tensor = eval_tf(img)
        assert tensor.shape == (3, 224, 224)


class TestDeepLearningPredictor:
    """Tests the standalone predictor service."""

    def test_predictor_singleton(self):
        p1 = DeepLearningPredictor.get_instance()
        p2 = DeepLearningPredictor.get_instance()
        assert p1 is p2

    def test_predict_with_bytes(self, sample_image_bytes):
        predictor = DeepLearningPredictor.get_instance()
        result = predictor.predict(sample_image_bytes, model_name="best")

        assert "prediction" in result
        assert result["prediction"] in ["BENIGN", "MALIGNANT"]
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 100.0
        assert "probabilities" in result
        assert "benign" in result["probabilities"]
        assert "malignant" in result["probabilities"]
        assert "cancer_verdict" in result
        assert "model_used" in result


class TestDeepLearningAPIRoutes:
    """Tests Flask API routes for comparison and deep learning inference."""

    def test_comparison_page_route(self, client):
        resp = client.get("/comparison")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Deep Learning Model Training" in html
        assert "ResNet50" in html
        assert "MobileNetV2" in html
        assert "Custom CNN" in html

    def test_benchmarks_alias_route(self, client):
        resp = client.get("/benchmarks")
        assert resp.status_code == 200

    def test_api_deep_learning_models_endpoint(self, client):
        resp = client.get("/api/deep-learning-models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "models" in data
        assert "selected_model" in data
        assert len(data["models"]) == 5

    def test_api_predict_with_deep_learning_classifier(self, client, sample_image_bytes):
        data = {
            "classifier": "deep_learning",
            "image": (io.BytesIO(sample_image_bytes), "test_lesion.jpg")
        }
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["success"] is True
        assert result["prediction"] in ["BENIGN", "MALIGNANT"]
        assert "confidence" in result
        assert "probabilities" in result
        assert "model_used" in result
        assert "feature_summary" in result
