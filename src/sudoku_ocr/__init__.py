"""
Sudoku OCR - A tool for extracting and solving Sudoku puzzles from images.

This package provides functionality for:
- Image preprocessing (grayscale, blur, threshold, cleanup)
- Grid detection and perspective correction
- Cell extraction and digit recognition
"""

__version__ = "0.1.0"
__author__ = "William Murphy"

from .preprocess import to_binary, largest_contour, preprocess_image
from .grid import find_and_warp, approx_to_quad, order_corners, warp_to_square, create_quad_overlay, GridNotFoundError, find_grid, warp_perspective
from .cells import split_into_cells, extract_cells
from .ocr import preprocess_cell, ocr_cell, ocr_cells, to_grid, print_grid, validate_grid, get_tesseract_version, test_tesseract_installation

__all__ = [
    # New pipeline functions
    "to_binary",
    "largest_contour", 
    "find_and_warp",
    "approx_to_quad",
    "order_corners",
    "warp_to_square",
    "create_quad_overlay",
    "split_into_cells",
    # OCR functions
    "preprocess_cell",
    "ocr_cell",
    "ocr_cells",
    "to_grid",
    "print_grid",
    "validate_grid",
    "get_tesseract_version",
    "test_tesseract_installation",
    # Exceptions
    "GridNotFoundError",
    # Legacy functions
    "preprocess_image",
    "find_grid", 
    "warp_perspective",
    "extract_cells",
]
