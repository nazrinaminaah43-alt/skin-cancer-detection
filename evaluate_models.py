"""
Root entry point for evaluating all trained Deep Learning models and generating clinical reports.
Usage:
    python evaluate_models.py
"""

from scripts.evaluate_models import run_evaluation_suite

if __name__ == "__main__":
    run_evaluation_suite()
