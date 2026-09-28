"""
Deep Learning Model Architectures, Preprocessing Transforms, and Utilities
for Skin Cancer (Melanoma vs. Benign Nevus) Classification.

Supported Architectures:
  1. Custom CNN (4 Conv blocks + BatchNorm + Dropout + Dense classification head)
  2. MobileNetV2 (Inverted residual lightweight architecture)
  3. ResNet50 (Residual deep network with skip connections)
  4. EfficientNetB0 (Compound scaling neural network)
  5. VGG16 (16-layer convolutional architecture)
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==============================================================================
# 1. Custom CNN Architecture
# ==============================================================================
class CustomCNN(nn.Module):
    """
    Custom 4-block Convolutional Neural Network designed specifically for
    dermoscopic skin lesion classification.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==============================================================================
# 2. Architecture Factory
# ==============================================================================
def build_custom_cnn(num_classes: int = 2) -> nn.Module:
    """Builds and returns our Custom CNN."""
    return CustomCNN(num_classes=num_classes)


def build_mobilenet_v2(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Builds MobileNetV2 with custom classification head."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Builds ResNet50 with custom classification head."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Builds EfficientNetB0 with custom classification head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_vgg16(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Builds VGG16 with custom classification head."""
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


MODEL_BUILDERS = {
    "custom_cnn": build_custom_cnn,
    "mobilenet_v2": build_mobilenet_v2,
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "vgg16": build_vgg16
}


def get_deep_learning_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Instantiates a deep learning model by architecture key."""
    key = model_name.lower().strip()
    if key not in MODEL_BUILDERS:
        raise ValueError(f"Unknown deep learning architecture '{model_name}'. Available: {list(MODEL_BUILDERS.keys())}")
    builder = MODEL_BUILDERS[key]
    if key == "custom_cnn":
        return builder(num_classes=num_classes)
    return builder(num_classes=num_classes, pretrained=pretrained)


# ==============================================================================
# 3. Data Augmentation & Preprocessing Transforms
# ==============================================================================
def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Applies image resizing, normalization, and appropriate data augmentation
    to the TRAINING set only, preventing data leakage.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_eval_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Consistent evaluation transform for validation and test images.
    Free from any random data augmentation.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


# ==============================================================================
# 4. Checkpoint Persistence & Loading
# ==============================================================================
def save_model_checkpoint(
    model: nn.Module,
    save_path: str,
    architecture_name: str,
    metrics: Dict[str, Any],
    class_to_idx: Optional[Dict[str, int]] = None,
    image_size: Tuple[int, int] = (224, 224),
    training_history: Optional[Dict[str, Any]] = None
):
    """Saves model weights, metadata, class mapping, and preprocessing configuration."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if class_to_idx is None:
        class_to_idx = {"benign": 0, "malignant": 1}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    checkpoint = {
        "architecture_name": architecture_name,
        "state_dict": model.state_dict(),
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "image_size": list(image_size),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "metrics": metrics,
        "training_history": training_history or {}
    }
    torch.save(checkpoint, save_path)


def load_model_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads saved checkpoint and reconstructs model ready for inference.
    
    Returns:
        (model, checkpoint_dict)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint.get("architecture_name", "mobilenet_v2")
    num_classes = checkpoint.get("num_classes", 2)

    model = get_deep_learning_model(arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint
