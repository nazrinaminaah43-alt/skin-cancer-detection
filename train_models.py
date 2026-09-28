"""
Root entry point for training all supported Deep Learning models and generating benchmarks.
Usage:
    python train_models.py
"""

from scripts.train_models import run_training_and_comparison_pipeline

if __name__ == "__main__":
    run_training_and_comparison_pipeline()
