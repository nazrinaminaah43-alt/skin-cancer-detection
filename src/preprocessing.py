"""
OpenCV Image Preprocessing Pipeline for Skin Cancer Detection.
Implements DullRazor hair artifact removal, CLAHE contrast normalization,
and Otsu-based lesion segmentation with contour extraction.
"""

import os
from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocesses skin lesion dermoscopy images using OpenCV."""

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

    def validate_image_file(self, filename: str, file_size: int, allowed_extensions: set, max_size_bytes: int):
        """Validates filename extension and payload byte size."""
        if not filename or "." not in filename:
            raise ValueError("Invalid file name: missing extension.")
        
        ext = filename.rsplit(".", 1)[1].lower()
        if ext not in allowed_extensions:
            raise ValueError(f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(allowed_extensions))}")
            
        if file_size <= 0:
            raise ValueError("Uploaded file is empty.")
            
        if file_size > max_size_bytes:
            mb_limit = max_size_bytes / (1024 * 1024)
            raise ValueError(f"File size exceeds maximum allowed limit of {mb_limit:.1f} MB.")

    def decode_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Decodes raw bytes into a valid 3-channel BGR numpy array using OpenCV."""
        if not image_bytes:
            raise ValueError("Image byte stream is empty.")
            
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image. File may be corrupted or not a valid image format.")
            
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError("Image must have 3 color channels (RGB/BGR).")
            
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            raise ValueError(f"Image dimensions too small ({w}x{h}). Minimum required is 20x20 pixels.")
            
        return img

    def load_and_validate(self, image_path: str) -> np.ndarray:
        """Loads and validates an image from disk using OpenCV."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")
            
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image from '{image_path}'. File may be corrupt.")
            
        return img

    def resize(self, img: np.ndarray) -> np.ndarray:
        """Standardize image to target dimensions."""
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

    def verify_skin_presence(self, img: np.ndarray, min_skin_ratio: float = 0.10) -> Tuple[bool, float]:
        """
        Validates whether the image contains human skin / dermatological lesion surface
        using multi-color space chromatic modeling (YCrCb and HSV across Fitzpatrick phototypes).
        """
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        skin_mask = ((cr >= 125) & (cr <= 185) & (cb >= 70) & (cb <= 140)) | \
                    (((h <= 55) | (h >= 155)) & (s >= 15) & (v >= 30))

        ratio = float(np.count_nonzero(skin_mask) / skin_mask.size)
        is_skin = ratio >= min_skin_ratio
        return is_skin, ratio

    def remove_hair(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        DullRazor-inspired hair artifact reduction:
        1. Convert to grayscale.
        2. Morphological Black-Hat filter with rectangular kernel isolates dark linear hair hairs.
        3. Threshold to construct hair mask.
        4. Inpaint using Telea algorithm to restore skin pigmentation.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Kernel tuned for typical dermoscopy hair thickness
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Intensity threshold for hair strands
        _, hair_mask = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
        
        # Dilate slightly to ensure complete coverage of hair edges
        hair_mask_dilated = cv2.dilate(hair_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        # Inpaint Telea
        clean_img = cv2.inpaint(img, hair_mask_dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return clean_img, hair_mask_dilated

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        CIE Lab CLAHE Contrast Enhancement:
        Enhances lesion-to-skin contrast without distorting chromatic pigment hues.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        enhanced_lab = cv2.merge((cl, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def segment_lesion(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Segments the skin lesion using Gaussian filtering, Otsu's thresholding,
        and morphological refinement.
        Returns:
            lesion_mask: Binary uint8 mask (255 inside lesion, 0 background)
            contour: Best matching lesion boundary contour
            overlay: Image with annotated green lesion boundary
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Otsu thresholding (lesion is darker than peripheral skin)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological closing to seal internal micro-vacuoles
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        
        # Morphological opening to remove fine peripheral noise
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = img.shape[:2]
        total_pixels = h * w
        center_x, center_y = w // 2, h // 2
        
        best_contour = None
        best_score = -1.0
        
        for c in contours:
            area = cv2.contourArea(c)
            # Filter contours that are too tiny (<1% of image) or take up the entire border (>95%)
            if area < (0.01 * total_pixels) or area > (0.95 * total_pixels):
                continue
                
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist_from_center = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
                # Centrality score favors lesions positioned centrally in the field
                centrality = 1.0 / (1.0 + (dist_from_center / (w / 2)))
                score = area * centrality
                if score > best_score:
                    best_score = score
                    best_contour = c

        # Create final binary mask
        lesion_mask = np.zeros((h, w), dtype=np.uint8)
        if best_contour is not None:
            cv2.drawContours(lesion_mask, [best_contour], -1, 255, -1)
        else:
            # Fallback: central region if segmentation did not isolate a clear single contour
            cv2.circle(lesion_mask, (center_x, center_y), int(min(h, w) * 0.35), 255, -1)
            contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best_contour = contours[0]

        # Create visual overlay for frontend inspection
        overlay = img.copy()
        if best_contour is not None:
            cv2.drawContours(overlay, [best_contour], -1, (0, 230, 115), 2)  # Glowing emerald green boundary
            
        return lesion_mask, best_contour, overlay

    def process(self, img_input: Any) -> Dict[str, Any]:
        """
        Complete end-to-end preprocessing pipeline:
        Takes either a file path or raw image bytes or numpy array.
        """
        if isinstance(img_input, str):
            raw_img = self.load_and_validate(img_input)
        elif isinstance(img_input, bytes):
            raw_img = self.decode_image_bytes(img_input)
        elif isinstance(img_input, np.ndarray):
            raw_img = img_input
        else:
            raise TypeError("Unsupported image input type. Provide path string, bytes, or numpy array.")

        resized_img = self.resize(raw_img)
        is_skin, skin_ratio = self.verify_skin_presence(resized_img)

        clean_img, hair_mask = self.remove_hair(resized_img)
        enhanced_img = self.enhance_contrast(clean_img)
        lesion_mask, contour, overlay_img = self.segment_lesion(enhanced_img)
        
        # Masked lesion isolated against neutral black
        masked_lesion = cv2.bitwise_and(clean_img, clean_img, mask=lesion_mask)
        
        return {
            "original_resized": resized_img,
            "hair_removed": clean_img,
            "hair_mask": hair_mask,
            "enhanced": enhanced_img,
            "lesion_mask": lesion_mask,
            "contour": contour,
            "contour_overlay": overlay_img,
            "masked_lesion": masked_lesion,
            "is_skin": is_skin,
            "skin_ratio": round(skin_ratio * 100, 1)
        }
