"""
Unit tests for OpenCV image preprocessing pipeline.
"""

import unittest
import numpy as np
import cv2

from src.preprocessing import ImagePreprocessor


class TestPreprocessing(unittest.TestCase):
    """Tests OpenCV preprocessing, validation, and lesion segmentation."""

    def setUp(self):
        self.preprocessor = ImagePreprocessor(target_size=(256, 256))

    def test_validation_allowed_extensions(self):
        """Should accept valid extensions and reject invalid extensions."""
        allowed = {"png", "jpg", "jpeg", "webp"}
        
        # Valid
        self.preprocessor.validate_image_file("lesion.jpg", 1024, allowed, 10 * 1024 * 1024)
        self.preprocessor.validate_image_file("lesion.png", 2048, allowed, 10 * 1024 * 1024)
        
        # Invalid extension
        with self.assertRaises(ValueError):
            self.preprocessor.validate_image_file("lesion.exe", 1024, allowed, 10 * 1024 * 1024)

    def test_validation_file_size(self):
        """Should reject empty files or files exceeding max size limit."""
        allowed = {"jpg"}
        max_size = 5 * 1024 * 1024
        
        # Empty file
        with self.assertRaises(ValueError):
            self.preprocessor.validate_image_file("lesion.jpg", 0, allowed, max_size)

        # Oversized file
        with self.assertRaises(ValueError):
            self.preprocessor.validate_image_file("lesion.jpg", 6 * 1024 * 1024, allowed, max_size)

    def test_decode_valid_image_bytes(self):
        """Should decode valid image bytes to 3-channel BGR numpy array."""
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 150
        _, encoded = cv2.imencode(".jpg", test_img)
        
        decoded = self.preprocessor.decode_image_bytes(encoded.tobytes())
        self.assertIsInstance(decoded, np.ndarray)
        self.assertEqual(decoded.shape, (100, 100, 3))

    def test_decode_corrupted_bytes(self):
        """Should raise ValueError when given corrupted bytes."""
        with self.assertRaises(ValueError):
            self.preprocessor.decode_image_bytes(b"not_an_image_binary_data")

    def test_hair_removal_and_clahe(self):
        """Should remove dark linear hair lines and return enhanced image."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 200
        # Draw dark line simulating hair
        cv2.line(img, (20, 20), (220, 220), (10, 10, 10), 3)

        clean_img, hair_mask = self.preprocessor.remove_hair(img)
        self.assertEqual(clean_img.shape, (256, 256, 3))
        self.assertGreater(np.count_nonzero(hair_mask), 0)

        enhanced = self.preprocessor.enhance_contrast(clean_img)
        self.assertEqual(enhanced.shape, (256, 256, 3))

    def test_lesion_segmentation(self):
        """Should detect and segment a synthetic circular lesion."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 210
        cv2.circle(img, (128, 128), 45, (40, 50, 80), -1)

        mask, contour, overlay = self.preprocessor.segment_lesion(img)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (256, 256))
        self.assertIsNotNone(contour)
        self.assertGreater(cv2.contourArea(contour), 1000)
        self.assertEqual(overlay.shape, (256, 256, 3))


if __name__ == "__main__":
    unittest.main()
