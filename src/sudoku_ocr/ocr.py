"""
OCR module for Sudoku digit recognition using Tesseract.
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Optional, Tuple
import os

# Configure Tesseract path for Chocolatey installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_cell(cell_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess a Sudoku cell for OCR recognition with enhanced preprocessing.
    
    Args:
        cell_bgr: Input BGR cell image
        
    Returns:
        Preprocessed binary image optimized for Tesseract
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return np.zeros((64, 64), dtype=np.uint8)
    
    # Convert to grayscale
    if len(cell_bgr.shape) == 3:
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_bgr.copy()
    
    # Resize to a larger size for better OCR (Tesseract works better with larger text)
    height, width = gray.shape
    if height < 80 or width < 80:
        scale_factor = max(80 / height, 80 / width)
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur to smooth the image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding with different parameters
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2
    )
    
    # Morphological operations to clean up the image
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Remove noise with opening
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    
    # Fill gaps with closing
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large)
    
    # Ensure minimum size for Tesseract
    if binary.shape[0] < 64 or binary.shape[1] < 64:
        binary = cv2.resize(binary, (80, 80), interpolation=cv2.INTER_CUBIC)
    
    return binary

def ocr_cell(cell_bgr: np.ndarray, conf_thresh: float = 0.45) -> int:
    """
    Recognize a single digit in a Sudoku cell using Tesseract with multiple fallback methods.
    
    Args:
        cell_bgr: Input BGR cell image
        conf_thresh: Confidence threshold for recognition (default: 0.45)
        
    Returns:
        Recognized digit (1-9) or 0 if no digit found or confidence too low
    """
    try:
        # Method 1: Enhanced preprocessing
        processed = preprocess_cell(cell_bgr)
        result1 = _tesseract_ocr(processed, conf_thresh)
        if result1 > 0:
            return result1
        
        # Method 2: Alternative preprocessing - larger size
        processed2 = _alternative_preprocessing(cell_bgr, size=120)
        result2 = _tesseract_ocr(processed2, conf_thresh * 0.8)  # Lower threshold for fallback
        if result2 > 0:
            return result2
        
        # Method 3: Simple threshold preprocessing
        processed3 = _simple_preprocessing(cell_bgr)
        result3 = _tesseract_ocr(processed3, conf_thresh * 0.7)  # Even lower threshold
        if result3 > 0:
            return result3
            
        return 0
            
    except Exception as e:
        print(f"OCR error: {e}")
        return 0

def _tesseract_ocr(processed_image: np.ndarray, conf_thresh: float) -> int:
    """Run Tesseract OCR on a preprocessed image."""
    # Configure Tesseract for single digit recognition with multiple PSM modes
    configs = [
        '--psm 10 -c tessedit_char_whitelist=0123456789',  # Single character
        '--psm 8 -c tessedit_char_whitelist=0123456789',   # Single word
        '--psm 7 -c tessedit_char_whitelist=0123456789',   # Single text line
        '--psm 6 -c tessedit_char_whitelist=0123456789',   # Single uniform block
    ]
    
    best_digit = 0
    best_confidence = 0
    
    # Try different PSM modes and pick the best result
    for config in configs:
        try:
            data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and text.isdigit() and conf > best_confidence:
                    digit = int(text)
                    if 1 <= digit <= 9:  # Only accept digits 1-9
                        best_digit = digit
                        best_confidence = conf
        except:
            continue
    
    # Return digit only if confidence is above threshold
    if best_confidence >= conf_thresh * 100:  # Tesseract confidence is 0-100
        return best_digit
    else:
        return 0

def _alternative_preprocessing(cell_bgr: np.ndarray, size: int = 120) -> np.ndarray:
    """Alternative preprocessing with larger size."""
    if cell_bgr is None or cell_bgr.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    
    # Convert to grayscale
    if len(cell_bgr.shape) == 3:
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_bgr.copy()
    
    # Resize to larger size
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_CUBIC)
    
    # Apply Otsu thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def _simple_preprocessing(cell_bgr: np.ndarray) -> np.ndarray:
    """Simple preprocessing method."""
    if cell_bgr is None or cell_bgr.size == 0:
        return np.zeros((80, 80), dtype=np.uint8)
    
    # Convert to grayscale
    if len(cell_bgr.shape) == 3:
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_bgr.copy()
    
    # Resize
    gray = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_CUBIC)
    
    # Simple threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    return binary

def ocr_cells(cells_bgr: List[np.ndarray], conf_thresh: float = 0.45) -> List[int]:
    """
    Recognize digits in multiple Sudoku cells using Tesseract.
    
    Args:
        cells_bgr: List of BGR cell images (81 cells in row-major order)
        conf_thresh: Confidence threshold for recognition (default: 0.45)
        
    Returns:
        List of recognized digits (0-9) in row-major order
    """
    results = []
    
    print("Recognizing digits with Tesseract...")
    
    for i, cell in enumerate(cells_bgr):
        row = i // 9
        col = i % 9
        
        # Progress indicator
        if col == 0:
            print(f"  Processing row {row + 1}/9...")
        
        digit = ocr_cell(cell, conf_thresh)
        results.append(digit)
    
    print(f"[OK] Recognized digits in {len(cells_bgr)} cells")
    return results

def to_grid(flat: List[int]) -> List[List[int]]:
    """
    Convert flat list of 81 digits to 9x9 grid.
    
    Args:
        flat: List of 81 digits in row-major order
        
    Returns:
        9x9 grid as list of lists
        
    Raises:
        ValueError: If input length is not 81
    """
    if len(flat) != 81:
        raise ValueError(f"Expected 81 digits, got {len(flat)}")
    
    grid = []
    for i in range(9):
        row = flat[i * 9:(i + 1) * 9]
        grid.append(row)
    
    return grid

def print_grid(grid: List[List[int]]) -> None:
    """
    Print a nicely formatted Sudoku grid.
    
    Args:
        grid: 9x9 grid as list of lists
    """
    print("\nRecognized Sudoku Grid:")
    print("Sudoku Grid:")
    print("+---------+---------+---------+")
    
    for i, row in enumerate(grid):
        line = "|"
        for j, digit in enumerate(row):
            if digit == 0:
                line += "   "
            else:
                line += f" {digit} "
            
            if j in [2, 5]:
                line += "|"
        
        line += "|"
        print(line)
        
        if i in [2, 5]:
            print("+---------+---------+---------+")
    
    print("+---------+---------+---------+")

def validate_grid(grid: List[List[int]]) -> bool:
    """
    Basic validation of Sudoku grid structure.
    
    Args:
        grid: 9x9 grid as list of lists
        
    Returns:
        True if grid has valid structure
    """
    if len(grid) != 9:
        return False
    
    for row in grid:
        if len(row) != 9:
            return False
        for digit in row:
            if not isinstance(digit, int) or digit < 0 or digit > 9:
                return False
    
    return True

def get_tesseract_version() -> str:
    """
    Get Tesseract version information.
    
    Returns:
        Version string or error message
    """
    try:
        version = pytesseract.get_tesseract_version()
        return f"Tesseract {version}"
    except Exception as e:
        return f"Tesseract not available: {e}"

def test_tesseract_installation() -> bool:
    """
    Test if Tesseract is properly installed and accessible.
    
    Returns:
        True if Tesseract is working, False otherwise
    """
    try:
        # Create a simple test image with a digit
        test_img = np.ones((64, 64), dtype=np.uint8) * 255
        cv2.putText(test_img, '5', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 2)
        
        # Try to recognize the digit
        config = '--psm 10 -c tessedit_char_whitelist=0123456789'
        result = pytesseract.image_to_string(test_img, config=config).strip()
        
        return result == '5'
    except Exception as e:
        print(f"Tesseract test failed: {e}")
        return False