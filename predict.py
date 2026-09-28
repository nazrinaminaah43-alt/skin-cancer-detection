"""
Command-line interface and entry point for Skin Cancer Deep Learning Prediction.
Usage:
    python predict.py --image path/to/lesion.jpg [--model mobilenet_v2]
"""

import sys
import argparse
from pathlib import Path
from src.predict import predict_image, DeepLearningPredictor


def main():
    parser = argparse.ArgumentParser(description="Skin Cancer Deep Learning Prediction CLI")
    parser.add_argument("image_pos", nargs="?", default=None, help="Path to lesion image file (positional)")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to lesion image file")
    parser.add_argument("--model", "-m", type=str, default="best", help="Model name: custom_cnn, mobilenet_v2, resnet50, efficientnet_b0, vgg16, or best")
    args = parser.parse_args()

    image_target = args.image or args.image_pos
    if not image_target:
        parser.error("Please provide an image path either as a positional argument or via --image/-i.")

    image_path = Path(image_target)
    if not image_path.exists():
        print(f"Error: File '{image_target}' not found.")
        sys.exit(1)

    result = predict_image(str(image_path), model_name=args.model)
    print("\n" + "=" * 60)
    print("      SKIN CANCER DEEP LEARNING INFERENCE RESULT")
    print("=" * 60)
    print(f"Model Architecture : {result['model_used']}")
    print(f"Prediction         : {result['prediction']}")
    print(f"Confidence         : {result['confidence_percentage']}")
    print(f"Cancer Status      : {result['cancer_status']}")
    print(f"Probabilities      : Benign: {result['probabilities']['benign']}%, Malignant: {result['probabilities']['malignant']}%")
    print(f"Clinical Verdict   : {result['cancer_verdict']}")
    print("=" * 60)
    print(f"Note: {result['disclaimer']}\n")


if __name__ == "__main__":
    main()
