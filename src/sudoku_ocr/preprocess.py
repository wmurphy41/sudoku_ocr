"""
Image preprocessing functions for Sudoku OCR.

This module handles the initial image processing steps including:
- Converting to grayscale
- Applying blur filters
- Thresholding
- Noise cleanup
"""

import cv2
import numpy as np


def to_binary(img_bgr: np.ndarray, apply_clahe: bool = False) -> np.ndarray:
    """
    Convert BGR image to binary for Sudoku grid detection.
    
    Process: grayscale → (optional CLAHE) → slight Gaussian blur → adaptive threshold (GAUSSIAN_C) 
    → invert so digits/grid are white on black → mild morphological open/close 
    to remove specks.
    
    Args:
        img_bgr: Input BGR image
        apply_clahe: Whether to apply CLAHE for contrast enhancement before thresholding
        
    Returns:
        Binary image (uint8, 0/255) with white digits/grid on black background
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE if requested
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding with GAUSSIAN_C
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Invert so digits/grid are white on black background
    inverted = cv2.bitwise_not(thresh)
    
    # Mild morphological operations to remove specks
    kernel = np.ones((2, 2), np.uint8)
    # Open to remove small noise
    opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    # Close to fill small gaps
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed


def largest_contour(binary: np.ndarray) -> np.ndarray | None:
    """
    Find the largest external contour in a binary image.
    
    Args:
        binary: Binary image (uint8, 0/255)
        
    Returns:
        Largest contour as numpy array, or None if no contours found
    """
    # Find external contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Return the contour with the largest area
    largest = max(contours, key=cv2.contourArea)
    return largest


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Preprocess an image for Sudoku grid detection.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed binary image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance contrast of the image using CLAHE.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Remove small noise from binary image.
    
    Args:
        image: Binary image
        
    Returns:
        Denoised image
    """
    # Remove small connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    
    # Keep only components larger than minimum size
    min_size = 50
    cleaned = np.zeros_like(image)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
            
    return cleaned
