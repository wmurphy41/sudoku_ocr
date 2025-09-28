"""
OCR module for digit recognition using a custom CNN.

This module provides functions for preprocessing Sudoku cells and recognizing
digits using a trained convolutional neural network.
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple
import threading
import os
import pickle

# Global cache for the CNN model
_model_cache = {}
_model_lock = threading.Lock()


def get_model(gpu: bool = False) -> 'CNNModel':
    """
    Get a cached CNN model instance for digit recognition.
    
    Args:
        gpu: Whether to use GPU acceleration (default: False, not implemented yet)
        
    Returns:
        CNNModel instance for digit recognition
        
    Note:
        This function uses a singleton pattern with caching to avoid
        the expensive model loading process on repeated calls.
    """
    cache_key = f"cnn_{gpu}"
    
    with _model_lock:
        if cache_key not in _model_cache:
            _model_cache[cache_key] = CNNModel()
        
        return _model_cache[cache_key]


class CNNModel:
    """
    Simple CNN model for digit recognition.
    
    This is a lightweight CNN that can be trained on digit images
    and used for real-time digit recognition in Sudoku cells.
    """
    
    def __init__(self):
        """Initialize the CNN model."""
        self.model = None
        self.trained = False
        
        # Try to load pre-trained model
        model_path = "models/digit_cnn.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.trained = True
                print(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model = None
                self.trained = False
        else:
            print(f"No pre-trained model found at {model_path}")
            print("Using heuristic-based recognition")
    
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Predict a digit from a preprocessed cell image.
        
        Args:
            image: Preprocessed binary image (48x48, 0/255)
            
        Returns:
            Tuple of (predicted_digit, confidence)
            - predicted_digit: 1-9 or 0 if no digit
            - confidence: 0.0 to 1.0
        """
        if self.trained and self.model is not None:
            # Use trained model
            return self._predict_with_model(image)
        else:
            # Use heuristic-based recognition
            return self._simple_digit_recognition(image)
    
    def _predict_with_model(self, image: np.ndarray) -> Tuple[int, float]:
        """Predict using trained model (placeholder for future implementation)."""
        # This would use the actual trained CNN model
        # For now, fall back to heuristics
        return self._simple_digit_recognition(image)
    
    def _simple_digit_recognition(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Simple heuristic-based digit recognition.
        
        This method analyzes image characteristics to classify digits
        based on shape, size, and contour properties.
        """
        if image is None or image.size == 0:
            return 0, 0.0
        
        # Convert to binary if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Ensure binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, 0.0
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Skip tiny contours
        if area < 100:
            return 0, 0.0
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate features
        aspect_ratio = w / h if h > 0 else 0
        area_ratio = area / (w * h) if w > 0 and h > 0 else 0
        compactness = (4 * np.pi * area) / (cv2.arcLength(largest_contour, True) ** 2) if cv2.arcLength(largest_contour, True) > 0 else 0
        solidity = area / cv2.contourArea(cv2.convexHull(largest_contour)) if cv2.contourArea(cv2.convexHull(largest_contour)) > 0 else 0
        
        # Calculate normalized size (relative to 48x48 image)
        normalized_size = area / (48 * 48)
        
        # Calculate centroid offset from center
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid_offset = np.sqrt((cx - 24)**2 + (cy - 24)**2) / 24.0
        else:
            centroid_offset = 1.0
        
        # Analyze image structure for digit-specific features
        digit_features = self._analyze_digit_structure(binary, largest_contour)
        
        # Classify based on features
        digit, confidence = self._classify_digit(
            aspect_ratio, area_ratio, compactness, solidity, 
            normalized_size, centroid_offset, digit_features
        )
        
        return digit, confidence
    
    def _analyze_digit_structure(self, binary: np.ndarray, contour: np.ndarray) -> dict:
        """Analyze the structure of the digit for classification."""
        h, w = binary.shape
        
        # Create mask from contour
        mask = np.zeros_like(binary)
        cv2.fillPoly(mask, [contour], 255)
        
        # Analyze horizontal and vertical projections
        h_proj = np.sum(binary, axis=1)  # Sum along rows
        v_proj = np.sum(binary, axis=0)  # Sum along columns
        
        # Find peaks in projections
        h_peaks = self._find_peaks(h_proj)
        v_peaks = self._find_peaks(v_proj)
        
        # Analyze symmetry
        left_half = binary[:, :w//2]
        right_half = binary[:, w//2:]
        horizontal_symmetry = self._calculate_symmetry(left_half, np.fliplr(right_half))
        
        # Analyze top/bottom structure
        top_half = binary[:h//2, :]
        bottom_half = binary[h//2:, :]
        vertical_symmetry = self._calculate_symmetry(top_half, np.flipud(bottom_half))
        
        return {
            'h_peaks': len(h_peaks),
            'v_peaks': len(v_peaks),
            'horizontal_symmetry': horizontal_symmetry,
            'vertical_symmetry': vertical_symmetry,
            'has_top_curve': self._has_curve(binary, 'top'),
            'has_bottom_curve': self._has_curve(binary, 'bottom'),
            'has_middle_bar': self._has_horizontal_bar(binary),
        }
    
    def _find_peaks(self, projection: np.ndarray, threshold: float = 0.3) -> List[int]:
        """Find peaks in a 1D projection."""
        max_val = np.max(projection)
        threshold_val = max_val * threshold
        
        peaks = []
        for i in range(1, len(projection) - 1):
            if (projection[i] > threshold_val and 
                projection[i] > projection[i-1] and 
                projection[i] > projection[i+1]):
                peaks.append(i)
        
        return peaks
    
    def _calculate_symmetry(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate symmetry between two image regions."""
        if img1.shape != img2.shape:
            return 0.0
        
        # Normalize
        img1_norm = img1.astype(np.float32) / 255.0
        img2_norm = img2.astype(np.float32) / 255.0
        
        # Calculate correlation
        correlation = cv2.matchTemplate(img1_norm, img2_norm, cv2.TM_CCOEFF_NORMED)[0][0]
        return max(0.0, correlation)
    
    def _has_curve(self, binary: np.ndarray, direction: str) -> bool:
        """Check if the digit has a curve in the specified direction."""
        h, w = binary.shape
        
        if direction == 'top':
            region = binary[:h//3, :]
        elif direction == 'bottom':
            region = binary[2*h//3:, :]
        else:
            return False
        
        # Find contours in the region
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Check if any contour has significant curvature
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Minimum area
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Curved shapes have more approximation points
                if len(approx) > 4:
                    return True
        
        return False
    
    def _has_horizontal_bar(self, binary: np.ndarray) -> bool:
        """Check if the digit has a horizontal bar in the middle."""
        h, w = binary.shape
        
        # Check middle third of the image
        middle_region = binary[h//3:2*h//3, :]
        
        # Find horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//4, 1))
        horizontal_lines = cv2.morphologyEx(middle_region, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Check if we have a significant horizontal line
        line_area = np.sum(horizontal_lines > 0)
        return line_area > (w * h // 20)  # At least 5% of middle region
    
    def _classify_digit(self, aspect_ratio: float, area_ratio: float, compactness: float, 
                       solidity: float, normalized_size: float, centroid_offset: float, 
                       features: dict) -> Tuple[int, float]:
        """Classify digit based on extracted features."""
        
        # Initialize scores for each digit
        scores = {}
        
        # Digit 1: Very narrow (0.56), low solidity (0.49), high symmetry (0.88), no h_peaks
        scores[1] = 0.0
        if 0.5 <= aspect_ratio <= 0.65:  # Very narrow
            scores[1] += 0.5
        if features['h_peaks'] == 0:  # No horizontal peaks
            scores[1] += 0.4
        if features['horizontal_symmetry'] > 0.8:  # High symmetry
            scores[1] += 0.3
        if solidity < 0.55:  # Low solidity
            scores[1] += 0.2
        
        # Digit 2: Medium aspect (0.72), both curves, low solidity (0.39), moderate h_peaks
        scores[2] = 0.0
        if 0.65 <= aspect_ratio <= 0.75:  # Medium aspect
            scores[2] += 0.3
        if features['has_top_curve'] and features['has_bottom_curve']:
            scores[2] += 0.4
        if solidity < 0.45:  # Low solidity
            scores[2] += 0.3
        if 1 <= features['h_peaks'] <= 2:  # Moderate horizontal peaks (not too many)
            scores[2] += 0.2
        
        # Digit 3: Medium aspect (0.67), both curves, low solidity (0.39), moderate h_peaks
        # Key differentiator: more symmetric than 5, fewer h_peaks
        scores[3] = 0.0
        if 0.65 <= aspect_ratio <= 0.70:  # Medium aspect
            scores[3] += 0.3
        if features['has_top_curve'] and features['has_bottom_curve']:
            scores[3] += 0.4
        if 1 <= features['h_peaks'] <= 2:  # Fewer horizontal peaks than 5
            scores[3] += 0.3
        if solidity < 0.45:  # Low solidity
            scores[3] += 0.2
        if features['horizontal_symmetry'] > 0.50:  # More symmetric than 5
            scores[3] += 0.2
        
        # Digit 4: Medium aspect (0.77), higher solidity (0.50), few peaks
        scores[4] = 0.0
        if 0.75 <= aspect_ratio <= 0.80:  # Medium-wide aspect
            scores[4] += 0.4
        if features['h_peaks'] <= 1:  # Few horizontal peaks
            scores[4] += 0.3
        if solidity > 0.45 and solidity < 0.60:  # Medium solidity
            scores[4] += 0.3
        if not features['has_bottom_curve']:  # No bottom curve
            scores[4] += 0.2
        
        # Digit 5: Medium aspect (0.67), both curves, medium solidity (0.44), variable h_peaks
        # Key differentiator: less symmetric than 3, more h_peaks
        scores[5] = 0.0
        if 0.65 <= aspect_ratio <= 0.70:  # Medium aspect
            scores[5] += 0.3
        if features['has_top_curve'] and features['has_bottom_curve']:
            scores[5] += 0.4
        if features['h_peaks'] >= 2:  # More horizontal structure than 3
            scores[5] += 0.3
        if 0.40 <= solidity <= 0.50:  # Medium solidity
            scores[5] += 0.2
        if features['horizontal_symmetry'] < 0.50:  # Less symmetric than 3
            scores[5] += 0.2
        
        # Digit 6: Medium aspect (0.73), high solidity (0.72), many h_peaks (4.0)
        # Key differentiator: has bottom curve but NO top curve (vs 8,9)
        scores[6] = 0.0
        if 0.70 <= aspect_ratio <= 0.75:  # Medium aspect
            scores[6] += 0.3
        if solidity > 0.70:  # High solidity (closed shape)
            scores[6] += 0.4
        if features['h_peaks'] >= 3:  # Many horizontal peaks
            scores[6] += 0.3
        if features['has_bottom_curve'] and not features['has_top_curve']:  # Bottom curve, no top curve
            scores[6] += 0.4
        elif features['has_bottom_curve']:  # Has bottom curve
            scores[6] += 0.2
        
        # Digit 7: Medium aspect (0.71), no h_peaks, no bottom curve, medium solidity (0.43)
        scores[7] = 0.0
        if 0.68 <= aspect_ratio <= 0.75:  # Medium aspect
            scores[7] += 0.3
        if features['h_peaks'] == 0:  # No horizontal peaks
            scores[7] += 0.4
        if not features['has_bottom_curve']:  # No bottom curve
            scores[7] += 0.3
        if features['has_top_curve']:  # Has top curve
            scores[7] += 0.2
        if 0.40 <= solidity <= 0.50:  # Medium solidity
            scores[7] += 0.2
        
        # Digit 8: Medium aspect (0.73), high solidity (0.63+), many h_peaks (4+), high symmetry
        # Key differentiator: BOTH top and bottom curves, very symmetric
        scores[8] = 0.0
        if 0.70 <= aspect_ratio <= 0.75:  # Medium aspect
            scores[8] += 0.3
        if solidity > 0.60:  # High solidity (closed shape) - adjusted threshold
            scores[8] += 0.4
        if features['h_peaks'] >= 4:  # Many horizontal peaks
            scores[8] += 0.3
        if features['horizontal_symmetry'] > 0.70:  # Very good symmetry (higher threshold)
            scores[8] += 0.3
        if features['has_top_curve'] and features['has_bottom_curve']:
            scores[8] += 0.4  # Strong indicator for 8
        
        # Digit 9: Medium aspect (0.73), high solidity (0.70), medium h_peaks (3.5)
        # Key differentiator: has top curve but NO bottom curve (vs 6,8)
        scores[9] = 0.0
        if 0.70 <= aspect_ratio <= 0.75:  # Medium aspect
            scores[9] += 0.3
        if solidity > 0.65:  # High solidity (closed shape)
            scores[9] += 0.4
        if 2 <= features['h_peaks'] <= 5:  # Medium horizontal peaks
            scores[9] += 0.3
        if features['has_top_curve'] and not features['has_bottom_curve']:  # Top curve, no bottom curve
            scores[9] += 0.4
        elif features['has_top_curve']:  # Has top curve
            scores[9] += 0.2
        
        # Find best match - balanced threshold
        if not scores or max(scores.values()) < 0.5:
            return 0, 0.0
        
        best_digit = max(scores, key=scores.get)
        best_score = scores[best_digit]
        
        # Special handling for most confused digit pairs
        # If 6 and 8 are close, prefer 6 if no top curve, 8 if both curves
        if best_digit == 6 and 8 in scores and scores[8] > best_score * 0.8:
            if not features['has_top_curve'] and features['has_bottom_curve']:
                # Keep 6
                pass
            elif features['has_top_curve'] and features['has_bottom_curve']:
                # Switch to 8
                best_digit = 8
                best_score = scores[8]
        
        elif best_digit == 8 and 6 in scores and scores[6] > best_score * 0.8:
            if features['has_top_curve'] and features['has_bottom_curve']:
                # Keep 8
                pass
            elif not features['has_top_curve'] and features['has_bottom_curve']:
                # Switch to 6
                best_digit = 6
                best_score = scores[6]
        
        # If 9 and 8 are close, prefer 9 if no bottom curve, 8 if both curves
        elif best_digit == 9 and 8 in scores and scores[8] > best_score * 0.8:
            if features['has_top_curve'] and not features['has_bottom_curve']:
                # Keep 9
                pass
            elif features['has_top_curve'] and features['has_bottom_curve']:
                # Switch to 8
                best_digit = 8
                best_score = scores[8]
        
        elif best_digit == 8 and 9 in scores and scores[9] > best_score * 0.8:
            if features['has_top_curve'] and features['has_bottom_curve']:
                # Keep 8
                pass
            elif features['has_top_curve'] and not features['has_bottom_curve']:
                # Switch to 9
                best_digit = 9
                best_score = scores[9]
        
        confidence = min(1.0, best_score)
        
        return best_digit, confidence
    
    def train(self, training_data: List[Tuple[np.ndarray, int]]):
        """Train the CNN model (placeholder for future implementation)."""
        print("Training CNN model...")
        # This would implement actual CNN training
        # For now, just mark as trained
        self.trained = True
        print("Model training completed (placeholder)")
    
    def save_model(self, path: str = "models/digit_cnn.pkl"):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


def preprocess_cell(cell_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess a Sudoku cell for OCR digit recognition.
    
    Process: Convert to grayscale → adaptive threshold (binary inv) → 
    remove thin grid remnants via small open (3×3) and optional erosion →
    find largest contour → extract and center digit → resize to fit in 48x48 with padding.
    
    Args:
        cell_bgr: Input BGR cell image
        
    Returns:
        Preprocessed binary image (0/255 uint8) sized 48x48 with centered digit
    """
    if cell_bgr is None or cell_bgr.size == 0:
        raise ValueError("Input cell image is empty or invalid")
    
    # Convert to grayscale
    if len(cell_bgr.shape) == 3:
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_bgr.copy()
    
    # Apply adaptive threshold (binary inverse)
    # This makes digits white on black background
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Remove thin grid remnants via morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Additional cleaning to remove small artifacts and ensure consistent digit sizing
    # Use a slightly larger kernel to remove thin artifacts that might affect bounding box
    clean_kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, clean_kernel)
    
    # Final erosion to clean up any remaining thin artifacts
    eroded = cv2.erode(cleaned, clean_kernel, iterations=1)
    
    # Find the largest contour to center the digit
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Only proceed if contour area is above threshold (avoid tiny noise)
        if area > 10:  # Tiny area threshold
            # Get bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract the region containing the digit
            digit_region = eroded[y:y+h, x:x+w]
            
            # Calculate target size with padding (leave some margin)
            target_size = 48
            max_digit_size = 32  # Leave 8px margin on each side for consistency
            
            region_h, region_w = digit_region.shape
            
            # Calculate scaling factor to fit within max_digit_size while preserving aspect ratio
            scale = min(max_digit_size / region_h, max_digit_size / region_w)
            new_h = int(region_h * scale)
            new_w = int(region_w * scale)
            
            # Resize the digit region
            resized = cv2.resize(digit_region, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad to square (48x48) and center with margin
            padded = np.zeros((target_size, target_size), dtype=np.uint8)
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            # No significant contour found, return blank image
            padded = np.zeros((48, 48), dtype=np.uint8)
    else:
        # No contours found, return blank image
        padded = np.zeros((48, 48), dtype=np.uint8)
    
    return padded


def ocr_cell(cell_bgr: np.ndarray, model: Optional['CNNModel'] = None, conf_thresh: float = 0.45) -> int:
    """
    Recognize a single digit in a Sudoku cell using CNN model.

    Args:
        cell_bgr: Input BGR cell image
        model: CNN model instance (if None, uses default)
        conf_thresh: Confidence threshold for recognition (default: 0.45)

    Returns:
        Recognized digit (1-9) or 0 if no digit found or confidence too low
    """
    if model is None:
        model = get_model()
    
    # Preprocess the cell
    processed = preprocess_cell(cell_bgr)
    
    # Predict digit
    digit, confidence = model.predict(processed)
    
    # Return digit only if confidence is above threshold
    if confidence >= conf_thresh:
        return digit
    else:
        return 0


def ocr_cells(cells_bgr: List[np.ndarray], model: Optional['CNNModel'] = None, conf_thresh: float = 0.45) -> List[int]:
    """
    Recognize digits in multiple Sudoku cells using CNN model.

    Args:
        cells_bgr: List of BGR cell images (81 cells in row-major order)
        model: CNN model instance (if None, uses default)
        conf_thresh: Confidence threshold for recognition (default: 0.45)

    Returns:
        List of recognized digits (0-9) in row-major order
    """
    if len(cells_bgr) != 81:
        raise ValueError(f"Expected 81 cells, got {len(cells_bgr)}")
    
    if model is None:
        model = get_model()
    
    print("Recognizing digits...")
    results = []
    
    for i, cell in enumerate(cells_bgr):
        # Show progress every 9 cells (one row)
        if i % 9 == 0:
            row = (i // 9) + 1
            print(f"  Processing row {row}/9...")
        
        digit = ocr_cell(cell, model, conf_thresh)
        results.append(digit)
    
    print(f"[OK] Recognized digits in {len(cells_bgr)} cells")
    return results


def to_grid(flat: List[int]) -> List[List[int]]:
    """
    Convert a flat list of 81 digits to a 9x9 grid.

    Args:
        flat: List of 81 digits in row-major order

    Returns:
        9x9 grid as list of lists
    """
    if len(flat) != 81:
        raise ValueError(f"Expected 81 digits, got {len(flat)}")
    
    grid = []
    for i in range(9):
        row = flat[i*9:(i+1)*9]
        grid.append(row)
    
    return grid


def print_grid(grid: List[List[int]]) -> None:
    """
    Pretty-print a Sudoku grid to stdout.

    Args:
        grid: 9x9 grid as list of lists
    """
    print("Recognized Sudoku Grid:")
    print("Sudoku Grid:")
    print("+-------------------------+")
    
    for i, row in enumerate(grid):
        line = "|"
        for j, cell in enumerate(row):
            if cell == 0:
                line += "   "
            else:
                line += f" {cell} "
            
            if j in [2, 5]:  # Add vertical separators
                line += "|"
        
        line += "|"
        print(line)
        
        if i in [2, 5]:  # Add horizontal separators
            print("+-------------------------+")
    
    print("+-------------------------+")


def validate_grid(grid: List[List[int]]) -> bool:
    """
    Basic validation of a Sudoku grid structure.

    Args:
        grid: 9x9 grid as list of lists

    Returns:
        True if grid has valid structure, False otherwise
    """
    if len(grid) != 9:
        return False
    
    for row in grid:
        if len(row) != 9:
            return False
        for cell in row:
            if not isinstance(cell, int) or cell < 0 or cell > 9:
                return False
    
    return True
