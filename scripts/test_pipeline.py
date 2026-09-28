"""
Command-line verification script for testing preprocessing, feature extraction,
and SVM prediction pipeline on an arbitrary lesion image file.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import ImagePreprocessor
from src.feature_extraction import FeatureExtractor
from src.model import SkinCancerModel


def run_pipeline(image_path: str):
    """Executes full diagnostic pipeline on the given image path."""
    print(f"\n[*] Testing Skin Cancer Detection Pipeline on: {image_path}")
    
    # 1. Preprocessing
    preprocessor = ImagePreprocessor()
    print("[1] Executing OpenCV preprocessing (Resize, DullRazor hair removal, CLAHE, Otsu segmentation)...")
    preprocessed = preprocessor.process(image_path)
    print(f"    - Lesion mask generated: {preprocessed['lesion_mask'].shape}")
    print(f"    - Contour isolated: {preprocessed['contour'] is not None}")

    # 2. Feature Extraction
    extractor = FeatureExtractor()
    features = extractor.extract_features(preprocessed)
    summary = extractor.extract_summary(preprocessed)
    print(f"[2] Extracted {features.shape[1]} dermatological features:")
    for k, v in summary.items():
        print(f"    - {k}: {v}")

    # 3. Model Inference
    model_service = SkinCancerModel.get_instance()
    res = model_service.predict(preprocessed)
    
    print("\n[3] Model Inference Result:")
    print("=" * 45)
    print(f"  Prediction       : {res['prediction']}")
    print(f"  Model Used       : {res['model_used']}")
    print(f"  Confidence       : {res['confidence_percentage']}")
    print(f"  Probabilities    : Benign: {res['probabilities']['benign']}% | Malignant: {res['probabilities']['malignant']}%")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = str(BASE_DIR / "data" / "samples" / "sample_benign_nevus_1.jpg")
    run_pipeline(test_path)
