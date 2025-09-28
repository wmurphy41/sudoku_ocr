"""
Cell extraction functions for splitting the grid into individual cells.

This module handles:
- Splitting the 9x9 grid into individual cells
- Adding safe margins around each cell
- Preparing cells for digit recognition
"""

import cv2
import numpy as np


def split_into_cells(warped_bgr: np.ndarray, pad: int = 4) -> list[np.ndarray]:
    """
    Split the warped square into 9x9 crops with inner padding.
    
    For each cell, apply a small inner padding to avoid grid lines.
    Returns a length-81 list of BGR crops in row-major order.
    
    Args:
        warped_bgr: Warped BGR image (square)
        pad: Inner padding to avoid grid lines (default: 4)
        
    Returns:
        List of 81 BGR cell images in row-major order (row 0, col 0-8, row 1, col 0-8, ...)
        
    Raises:
        ValueError: If the input image is not square or too small
    """
    if warped_bgr is None or warped_bgr.size == 0:
        raise ValueError("Input image is empty or invalid")
    
    if len(warped_bgr.shape) != 3:
        raise ValueError("Input image must be a 3-channel BGR image")
    
    h, w = warped_bgr.shape[:2]
    if h != w:
        raise ValueError(f"Input image must be square, got {h}x{w}")
    
    if h < 90:  # Minimum size for 9x9 grid
        raise ValueError(f"Input image too small for 9x9 grid, minimum size is 90x90, got {h}x{w}")
    
    cells = []
    cell_size = h // 9
    
    # Ensure padding doesn't exceed cell size
    max_pad = cell_size // 2 - 1
    actual_pad = min(pad, max_pad) if max_pad > 0 else 0
    
    for row in range(9):
        for col in range(9):
            # Calculate cell boundaries
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size
            
            # Ensure boundaries are within image bounds
            y1 = max(0, y1)
            y2 = min(h, y2)
            x1 = max(0, x1)
            x2 = min(w, x2)
            
            # Extract cell
            cell = warped_bgr[y1:y2, x1:x2].copy()
            
            # Apply inner padding to avoid grid lines
            if actual_pad > 0:
                cell_h, cell_w = cell.shape[:2]
                # Ensure padding doesn't cause negative dimensions
                pad_y = min(actual_pad, cell_h // 3) if cell_h > 6 else 0
                pad_x = min(actual_pad, cell_w // 3) if cell_w > 6 else 0
                
                if pad_y > 0 and pad_x > 0:
                    cell = cell[pad_y:cell_h-pad_y, pad_x:cell_w-pad_x]
            
            # Ensure minimum size of 32x32 only if the cell is too small
            cell_h, cell_w = cell.shape[:2]
            if cell_h < 32 or cell_w < 32:
                # Check if cell is empty or has no content
                if cell_h == 0 or cell_w == 0:
                    # Cell is completely empty, create a 32x32 white cell
                    cell = np.ones((32, 32, 3), dtype=np.uint8) * 255
                else:
                    # Only resize if the cell has some content (not completely white)
                    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
                    if np.var(gray) > 10:  # Check if there's actual content (not just white)
                        # Choose interpolation method based on whether we're scaling up or down
                        if cell_h < 32 and cell_w < 32:
                            # Both dimensions are too small, scale up
                            scale = max(32 / cell_h, 32 / cell_w)
                            new_h = int(cell_h * scale)
                            new_w = int(cell_w * scale)
                            cell = cv2.resize(cell, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        elif cell_h < 32:
                            # Only height is too small
                            scale = 32 / cell_h
                            new_h = 32
                            new_w = int(cell_w * scale)
                            cell = cv2.resize(cell, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        elif cell_w < 32:
                            # Only width is too small
                            scale = 32 / cell_w
                            new_h = int(cell_h * scale)
                            new_w = 32
                            cell = cv2.resize(cell, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    else:
                        # Cell is empty/white, create a 32x32 white cell
                        cell = np.ones((32, 32, 3), dtype=np.uint8) * 255
            
            cells.append(cell)
    
    return cells


def extract_cells(image: np.ndarray, margin: int = 5) -> list:
    """
    Legacy function for backward compatibility.
    Extract individual cells from the Sudoku grid.
    
    Args:
        image: Warped grid image (square)
        margin: Margin to add around each cell
        
    Returns:
        List of 81 cell images (9x9 grid)
    """
    cells = []
    cell_size = image.shape[0] // 9
    
    for row in range(9):
        for col in range(9):
            # Calculate cell boundaries
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size
            
            # Extract cell with margin
            cell = image[y1:y2, x1:x2]
            
            # Add margin if specified
            if margin > 0:
                cell = add_margin(cell, margin)
            
            cells.append(cell)
    
    return cells


def add_margin(cell: np.ndarray, margin: int) -> np.ndarray:
    """
    Add margin around a cell image.
    
    Args:
        cell: Input cell image
        margin: Margin size in pixels
        
    Returns:
        Cell image with added margin
    """
    h, w = cell.shape
    new_h, new_w = h + 2 * margin, w + 2 * margin
    
    # Create new image with white background
    new_cell = np.ones((new_h, new_w), dtype=np.uint8) * 255
    
    # Place original cell in center
    new_cell[margin:margin + h, margin:margin + w] = cell
    
    return new_cell


def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    """
    Preprocess a single cell for digit recognition.
    
    Args:
        cell: Input cell image
        
    Returns:
        Preprocessed cell image
    """
    # Resize to standard size
    cell = cv2.resize(cell, (28, 28))
    
    # Normalize pixel values
    cell = cell.astype(np.float32) / 255.0
    
    # Invert if needed (digits should be dark on light background)
    if np.mean(cell) < 0.5:
        cell = 1.0 - cell
    
    return cell


def is_empty_cell(cell: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Check if a cell is empty (no digit).
    
    Args:
        cell: Cell image
        threshold: Threshold for considering cell empty
        
    Returns:
        True if cell appears to be empty
    """
    # Calculate the ratio of dark pixels
    dark_pixels = np.sum(cell < 128)
    total_pixels = cell.size
    dark_ratio = dark_pixels / total_pixels
    
    return dark_ratio < threshold


def center_digit(cell: np.ndarray) -> np.ndarray:
    """
    Center the digit in the cell by finding the bounding box.
    
    Args:
        cell: Input cell image
        
    Returns:
        Centered cell image
    """
    # Find contours
    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cell
    
    # Find the largest contour (the digit)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Extract the digit region
    digit = cell[y:y+h, x:x+w]
    
    # Calculate padding to center the digit
    cell_h, cell_w = cell.shape
    pad_x = (cell_w - w) // 2
    pad_y = (cell_h - h) // 2
    
    # Create new centered cell
    centered = np.ones_like(cell) * 255
    centered[pad_y:pad_y+h, pad_x:pad_x+w] = digit
    
    return centered
