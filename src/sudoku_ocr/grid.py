"""
Grid detection and perspective correction functions.

This module handles:
- Finding the Sudoku grid contour
- Applying perspective transformation
- Correcting for camera angle and distortion
"""

import cv2
import numpy as np


class GridNotFoundError(Exception):
    """Raised when a valid Sudoku grid cannot be detected in the image."""
    pass


def approx_to_quad(cnt: np.ndarray) -> np.ndarray | None:
    """
    Approximate contour to a quadrilateral using cv2.approxPolyDP.
    
    Args:
        cnt: Input contour
        
    Returns:
        4 points if quadrilateral found, None otherwise
        
    Raises:
        GridNotFoundError: If the approximated quadrilateral fails sanity checks
    """
    if cnt is None or len(cnt) < 4:
        return None
        
    # Calculate epsilon as a percentage of the perimeter
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # If we found a quadrilateral
    if len(approx) == 4:
        quad = approx.reshape(4, 2)
        
        # Sanity checks
        if not _is_valid_quad(quad):
            raise GridNotFoundError("Detected quadrilateral failed shape sanity checks")
        
        return quad
    
    return None


def _is_valid_quad(quad: np.ndarray) -> bool:
    """
    Check if a quadrilateral is valid for Sudoku grid detection.
    
    Args:
        quad: 4 corner points of the quadrilateral
        
    Returns:
        True if the quadrilateral passes all sanity checks
    """
    if quad is None or len(quad) != 4:
        return False
    
    # Check if the quadrilateral is convex
    hull = cv2.convexHull(quad)
    if len(hull) != 4:
        return False
    
    # Check aspect ratio (should be roughly square)
    # Calculate bounding rectangle
    x_coords = quad[:, 0]
    y_coords = quad[:, 1]
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    
    if width <= 0 or height <= 0:
        return False
    
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio < 0.8 or aspect_ratio > 1.2:
        return False
    
    # Check minimum area (should be reasonably large)
    area = cv2.contourArea(quad)
    if area < 1000:  # Minimum area threshold
        return False
    
    return True


def create_quad_overlay(img_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Create an overlay showing the quadrilateral with circles at corners and lines for edges.
    
    Args:
        img_bgr: Input BGR image
        quad: 4 corner points of the quadrilateral
        
    Returns:
        BGR image with quad overlay
    """
    overlay = img_bgr.copy()
    
    if quad is None or len(quad) != 4:
        return overlay
    
    # Order the corners
    ordered_quad = order_corners(quad)
    
    # Draw lines connecting the corners
    for i in range(4):
        pt1 = tuple(ordered_quad[i].astype(int))
        pt2 = tuple(ordered_quad[(i + 1) % 4].astype(int))
        cv2.line(overlay, pt1, pt2, (0, 0, 255), 3)  # Red lines
    
    # Draw circles at each corner
    for i, point in enumerate(ordered_quad):
        center = tuple(point.astype(int))
        cv2.circle(overlay, center, 8, (0, 255, 0), -1)  # Green filled circles
        cv2.circle(overlay, center, 8, (0, 0, 0), 2)     # Black outline
    
    return overlay


def order_corners(pts4: np.ndarray) -> np.ndarray:
    """
    Order 4 points in the order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts4: Array of 4 points (4, 2)
        
    Returns:
        Ordered array of points (4, 2) in TL, TR, BR, BL order
    """
    # Initialize ordered points
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Sum and difference of coordinates
    s = pts4.sum(axis=1)
    diff = np.diff(pts4, axis=1)
    
    # Top-left point has smallest sum
    rect[0] = pts4[np.argmin(s)]
    
    # Bottom-right point has largest sum
    rect[2] = pts4[np.argmax(s)]
    
    # Top-right point has smallest difference
    rect[1] = pts4[np.argmin(diff)]
    
    # Bottom-left point has largest difference
    rect[3] = pts4[np.argmax(diff)]
    
    return rect


def warp_to_square(img_bgr: np.ndarray, pts4: np.ndarray, size: int = 450) -> np.ndarray:
    """
    Apply perspective transformation to get a perfect square view of the grid.
    
    Args:
        img_bgr: Input BGR image
        pts4: 4 corner points in TL, TR, BR, BL order
        size: Size of the output square (default: 450)
        
    Returns:
        Warped BGR image as a perfect square
    """
    # Define the target square
    dst_points = np.array([
        [0, 0],           # top-left
        [size, 0],        # top-right
        [size, size],     # bottom-right
        [0, size]         # bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(pts4.astype(np.float32), dst_points)
    
    # Apply the transformation
    warped = cv2.warpPerspective(img_bgr, matrix, (size, size))
    
    return warped


def find_and_warp(img_bgr: np.ndarray, size: int = 450, apply_clahe: bool = False) -> dict:
    """
    Complete pipeline: binary → largest_contour → approx_to_quad → warp.
    
    Args:
        img_bgr: Input BGR image
        size: Size of the output square (default: 450)
        apply_clahe: Whether to apply CLAHE for contrast enhancement
        
    Returns:
        Dictionary with intermediate artifacts: binary, contour_mask, quad, warped
        
    Raises:
        GridNotFoundError: If no valid grid can be detected
    """
    from .preprocess import to_binary, largest_contour
    
    if img_bgr is None or img_bgr.size == 0:
        raise GridNotFoundError("Input image is empty or invalid")
    
    # Step 1: Convert to binary
    binary = to_binary(img_bgr, apply_clahe=apply_clahe)
    
    # Step 2: Find largest contour
    contour = largest_contour(binary)
    if contour is None:
        raise GridNotFoundError("No contours found in the binary image")
    
    # Create contour mask for visualization (white fill)
    contour_mask = np.zeros_like(binary)
    cv2.fillPoly(contour_mask, [contour], 255)
    
    # Step 3: Approximate to quadrilateral
    try:
        quad = approx_to_quad(contour)
    except GridNotFoundError as e:
        raise GridNotFoundError(f"Grid detection failed: {e}")
    
    if quad is None:
        raise GridNotFoundError("Could not approximate contour to a valid quadrilateral")
    
    # Step 4: Warp to square
    try:
        ordered_quad = order_corners(quad)
        warped = warp_to_square(img_bgr, ordered_quad, size=size)
    except Exception as e:
        raise GridNotFoundError(f"Perspective transformation failed: {e}")
    
    return {
        'binary': binary,
        'contour_mask': contour_mask,
        'quad': quad,
        'warped': warped
    }


def find_grid(image: np.ndarray) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Find the largest rectangular contour that represents the Sudoku grid.
    
    Args:
        image: Preprocessed binary image
        
    Returns:
        Array of 4 corner points of the grid
    """
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Look for a rectangular contour
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If we found a quadrilateral
        if len(approx) == 4:
            return approx.reshape(4, 2)
    
    # If no good contour found, return corners of the image
    h, w = image.shape
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


def warp_perspective(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Apply perspective transformation to get a top-down view of the grid.
    
    Args:
        image: Input image
        corners: 4 corner points of the grid
        
    Returns:
        Warped image with corrected perspective
    """
    # Define the target rectangle (square)
    size = 900  # Size of the output square
    dst_points = np.array([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]
    ], dtype=np.float32)
    
    # Calculate perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    
    # Apply the transformation
    warped = cv2.warpPerspective(image, matrix, (size, size))
    
    return warped


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Order points in the order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: Array of 4 points
        
    Returns:
        Ordered array of points
    """
    return order_corners(pts)


def detect_grid_lines(image: np.ndarray) -> tuple:
    """
    Detect horizontal and vertical lines in the grid.
    
    Args:
        image: Binary image of the grid
        
    Returns:
        Tuple of (horizontal_lines, vertical_lines)
    """
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
    
    return horizontal_lines, vertical_lines
