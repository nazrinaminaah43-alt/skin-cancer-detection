"""
Dermatological Feature Extraction Pipeline for Skin Lesion Classification.
Extracts clinical ABCD features (Asymmetry, Border irregularity, Color variegation,
Diameter/Shape) along with Texture descriptors (GLCM, Laplacian, Sobel gradients).
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from scipy import stats


class FeatureExtractor:
    """Extracts dermatological visual features from preprocessed lesion images."""

    def __init__(self):
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        """Defines the ordered list of extracted feature names."""
        names = []
        # Color moments: mean, std, skew for BGR, HSV, Lab (9 * 3 = 27)
        for space, channels in [("bgr", ["b", "g", "r"]), ("hsv", ["h", "s", "v"]), ("lab", ["l", "a", "b"])]:
            for ch in channels:
                names.extend([f"{space}_{ch}_mean", f"{space}_{ch}_std", f"{space}_{ch}_skew"])
        
        # Color variance across channels
        names.append("color_channel_var")
        
        # Shape & ABCD Border features
        names.extend([
            "lesion_area",
            "lesion_perimeter",
            "compactness_circularity",
            "border_irregularity",
            "solidity",
            "aspect_ratio",
            "extent",
            "asymmetry_horizontal",
            "asymmetry_vertical",
            "asymmetry_index"
        ])
        
        # Texture features (GLCM + Edges)
        names.extend([
            "glcm_contrast",
            "glcm_dissimilarity",
            "glcm_homogeneity",
            "glcm_energy",
            "glcm_correlation",
            "laplacian_variance",
            "sobel_edge_mean",
            "sobel_edge_std"
        ])
        return names

    def extract_color_moments(self, img_bgr: np.ndarray, mask: np.ndarray) -> List[float]:
        """
        Extracts color moments (mean, std, skewness) in BGR, HSV, and Lab color spaces
        restricted to the lesion mask area.
        """
        features = []
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        
        mask_bool = mask > 0
        if not np.any(mask_bool):
            # Fallback if mask is blank
            mask_bool = np.ones(mask.shape, dtype=bool)

        for space_img in [img_bgr, hsv, lab]:
            for ch in range(3):
                pixels = space_img[:, :, ch][mask_bool].astype(np.float64)
                if len(pixels) > 0:
                    mean_val = float(np.mean(pixels))
                    std_val = float(np.std(pixels))
                    skew_val = float(stats.skew(pixels)) if std_val > 1e-4 else 0.0
                    # Handle possible NaN
                    skew_val = 0.0 if np.isnan(skew_val) else skew_val
                else:
                    mean_val, std_val, skew_val = 0.0, 0.0, 0.0
                features.extend([mean_val, std_val, skew_val])
                
        # Color variance across RGB channels (color variegation metric)
        r_mean = features[6]
        g_mean = features[3]
        b_mean = features[0]
        color_var = float(np.var([r_mean, g_mean, b_mean]))
        features.append(color_var)
        
        return features

    def extract_shape_and_border(self, mask: np.ndarray, contour: np.ndarray) -> List[float]:
        """
        Extracts ABCD shape and border irregularity features:
        - Compactness/circularity: 4*pi*Area / Perimeter^2 (melanoma borders are irregular)
        - Border irregularity: Convex Hull Perimeter / Contour Perimeter
        - Asymmetry indices: Symmetric difference when folded horizontally and vertically
        - Solidity & Extent
        """
        if contour is None or len(contour) < 3:
            return [0.0] * 10

        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        
        # Compactness / Circularity
        compactness = (4.0 * np.pi * area) / (perimeter ** 2 + 1e-6) if perimeter > 0 else 0.0

        # Convex Hull
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        hull_perimeter = float(cv2.arcLength(hull, True))
        
        border_irregularity = (hull_perimeter / (perimeter + 1e-6)) if perimeter > 0 else 0.0
        solidity = (area / (hull_area + 1e-6)) if hull_area > 0 else 0.0

        # Bounding Box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h + 1e-6)
        extent = area / float(w * h + 1e-6)

        # Asymmetry calculation using lesion mask
        # Crop mask to lesion bounding box for normalized orientation
        lesion_crop = mask[y:y+h, x:x+w]
        if lesion_crop.size > 0:
            h_flip = cv2.flip(lesion_crop, 1)  # horizontal flip
            v_flip = cv2.flip(lesion_crop, 0)  # vertical flip
            
            diff_h = np.count_nonzero(cv2.bitwise_xor(lesion_crop, h_flip))
            diff_v = np.count_nonzero(cv2.bitwise_xor(lesion_crop, v_flip))
            total_active = np.count_nonzero(lesion_crop) + 1e-6
            
            asym_h = float(diff_h / total_active)
            asym_v = float(diff_v / total_active)
            asym_index = float((diff_h + diff_v) / (2.0 * total_active))
        else:
            asym_h, asym_v, asym_index = 0.0, 0.0, 0.0

        return [
            area,
            perimeter,
            compactness,
            border_irregularity,
            solidity,
            aspect_ratio,
            extent,
            asym_h,
            asym_v,
            asym_index
        ]

    def extract_texture_features(self, img_bgr: np.ndarray, mask: np.ndarray) -> List[float]:
        """
        Extracts textural descriptors:
        - Fast Gray-Level Co-occurrence Matrix (GLCM): Contrast, Dissimilarity, Homogeneity, Energy, Correlation
        - Laplacian Variance (edge roughness / structural variance)
        - Sobel gradient statistics
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance (sharpness and texture detail)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())
        
        # Sobel edge gradient within lesion
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobel_x, sobel_y)
        
        mask_bool = mask > 0
        if np.any(mask_bool):
            sobel_mean = float(np.mean(sobel_mag[mask_bool]))
            sobel_std = float(np.std(sobel_mag[mask_bool]))
        else:
            sobel_mean = float(np.mean(sobel_mag))
            sobel_std = float(np.std(sobel_mag))

        # Fast GLCM on quantized 16-level grayscale
        levels = 16
        quantized = (gray // (256 // levels)).astype(np.int32)
        
        # Pair co-occurrences horizontally (dx=1, dy=0)
        row_cur = quantized[:, :-1].ravel()
        row_adj = quantized[:, 1:].ravel()
        pairs = row_cur * levels + row_adj
        
        # Histogram of pairs
        hist = np.bincount(pairs, minlength=levels * levels).astype(np.float64)
        p = hist.reshape((levels, levels))
        total_p = np.sum(p)
        if total_p > 0:
            p /= total_p
        else:
            p = np.ones((levels, levels)) / (levels * levels)

        # Coordinate matrices for GLCM properties
        i_coords, j_coords = np.indices((levels, levels))
        diff = np.abs(i_coords - j_coords)
        diff_sq = diff ** 2

        contrast = float(np.sum(p * diff_sq))
        dissimilarity = float(np.sum(p * diff))
        homogeneity = float(np.sum(p / (1.0 + diff_sq)))
        energy = float(np.sum(p ** 2))

        # Correlation
        mu_i = np.sum(i_coords * p)
        mu_j = np.sum(j_coords * p)
        sigma_i = np.sqrt(np.sum(p * (i_coords - mu_i) ** 2))
        sigma_j = np.sqrt(np.sum(p * (j_coords - mu_j) ** 2))
        
        if sigma_i > 1e-6 and sigma_j > 1e-6:
            correlation = float(np.sum(p * (i_coords - mu_i) * (j_coords - mu_j)) / (sigma_i * sigma_j))
        else:
            correlation = 0.0

        return [
            contrast,
            dissimilarity,
            homogeneity,
            energy,
            correlation,
            lap_var,
            sobel_mean,
            sobel_std
        ]

    def extract_features(self, preprocessed: Dict[str, Any]) -> np.ndarray:
        """
        Extracts the full ordered feature vector from preprocessed image components.
        Returns:
            np.ndarray of shape (1, num_features)
        """
        clean_img = preprocessed["hair_removed"]
        mask = preprocessed["lesion_mask"]
        contour = preprocessed["contour"]

        color_feats = self.extract_color_moments(clean_img, mask)
        shape_feats = self.extract_shape_and_border(mask, contour)
        texture_feats = self.extract_texture_features(clean_img, mask)

        all_feats = color_feats + shape_feats + texture_feats
        return np.array(all_feats, dtype=np.float64).reshape(1, -1)

    def extract_summary(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Provides a human-interpretable summary of the ABCD visual features for dashboard display."""
        clean_img = preprocessed["hair_removed"]
        mask = preprocessed["lesion_mask"]
        contour = preprocessed["contour"]

        shape = self.extract_shape_and_border(mask, contour)
        color = self.extract_color_moments(clean_img, mask)
        texture = self.extract_texture_features(clean_img, mask)

        # shape: [area, perimeter, compactness, border_irregularity, solidity, aspect_ratio, extent, asym_h, asym_v, asym_index]
        # color: last element is color_channel_var
        # texture: [contrast, dissimilarity, homogeneity, energy, correlation, lap_var, sobel_mean, sobel_std]
        return {
            "asymmetry_index": round(float(shape[9]), 3),
            "border_compactness": round(float(shape[2]), 3),
            "border_irregularity": round(float(shape[3]), 3),
            "color_variegation": round(float(color[-1]), 1),
            "texture_contrast": round(float(texture[0]), 2),
            "texture_sharpness": round(float(texture[5]), 1),
            "lesion_solidity": round(float(shape[4]), 3)
        }
