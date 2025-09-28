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
    Preprocess a Sudoku cell for OCR recognition.
    
    Args:
        cell_bgr: Input BGR cell image
        
    Returns:
        Preprocessed binary image optimized for Tesseract
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return np.zeros((48, 48), dtype=np.uint8)
    
    # Convert to grayscale
    if len(cell_bgr.shape) == 3:
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_bgr.copy()
    
    # Resize to a larger size for better OCR (Tesseract works better with larger text)
    height, width = gray.shape
    if height < 64 or width < 64:
        scale_factor = max(64 / height, 64 / width)
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    # Remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Fill gaps
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Ensure minimum size for Tesseract
    if binary.shape[0] < 32 or binary.shape[1] < 32:
        binary = cv2.resize(binary, (64, 64), interpolation=cv2.INTER_CUBIC)
    
    return binary

def ocr_cell(cell_bgr: np.ndarray, conf_thresh: float = 0.45) -> int:
    """
    Recognize a single digit in a Sudoku cell using Tesseract.
    
    Args:
        cell_bgr: Input BGR cell image
        conf_thresh: Confidence threshold for recognition (default: 0.45)
        
    Returns:
        Recognized digit (1-9) or 0 if no digit found or confidence too low
    """
    try:
        # Preprocess the cell
        processed = preprocess_cell(cell_bgr)
        
        # Configure Tesseract for single digit recognition
        config = '--psm 10 -c tessedit_char_whitelist=0123456789'
        
        # Run Tesseract OCR
        data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
        
        # Find the best confidence digit
        best_digit = 0
        best_confidence = 0
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and text.isdigit() and conf > best_confidence:
                digit = int(text)
                if 1 <= digit <= 9:  # Only accept digits 1-9
                    best_digit = digit
                    best_confidence = conf
        
        # Return digit only if confidence is above threshold
        if best_confidence >= conf_thresh * 100:  # Tesseract confidence is 0-100
            return best_digit
        else:
            return 0
            
    except Exception as e:
        print(f"OCR error: {e}")
        return 0

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
    print("+-------------------------+")
    
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
            print("+-------------------------+")
    
    print("+-------------------------+")

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