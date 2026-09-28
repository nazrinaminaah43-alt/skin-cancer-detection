"""
Unit tests for Feature Extraction module.
"""

import unittest
import numpy as np
import cv2

from src.preprocessing import ImagePreprocessor
from src.feature_extraction import FeatureExtractor


class TestFeatureExtraction(unittest.TestCase):
    """Tests dermatological visual feature extraction."""

    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        self.extractor = FeatureExtractor()

    def test_feature_names_integrity(self):
        """Should have 46 unique feature names."""
        names = self.extractor.feature_names
        self.assertEqual(len(names), 46)
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("compactness_circularity", names)
        self.assertIn("asymmetry_index", names)
        self.assertIn("glcm_contrast", names)

    def test_feature_vector_shape(self):
        """Extracted feature vector should have shape (1, 46)."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 200
        cv2.circle(img, (128, 128), 50, (50, 60, 90), -1)

        preprocessed = self.preprocessor.process(img)
        feats = self.extractor.extract_features(preprocessed)

        self.assertIsInstance(feats, np.ndarray)
        self.assertEqual(feats.shape, (1, 46))
        # Ensure no NaN or infinite values
        self.assertFalse(np.isnan(feats).any())
        self.assertFalse(np.isinf(feats).any())

    def test_feature_summary(self):
        """Summary dictionary should contain key ABCD metrics."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 200
        cv2.circle(img, (128, 128), 50, (50, 60, 90), -1)

        preprocessed = self.preprocessor.process(img)
        summary = self.extractor.extract_summary(preprocessed)

        self.assertIn("asymmetry_index", summary)
        self.assertIn("border_compactness", summary)
        self.assertIn("border_irregularity", summary)
        self.assertIn("color_variegation", summary)
        self.assertIn("texture_contrast", summary)
        self.assertIn("texture_sharpness", summary)


if __name__ == "__main__":
    unittest.main()
