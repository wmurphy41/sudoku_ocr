"""
Comprehensive tests for grid detection functionality using synthetic images.

All tests use small synthetic test images built on the fly (NumPy arrays + cv2 drawing)
so tests don't rely on external files.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sudoku_ocr.grid import find_and_warp, approx_to_quad, order_corners, warp_to_square, create_quad_overlay, GridNotFoundError
from sudoku_ocr.cells import split_into_cells
from sudoku_ocr.preprocess import to_binary


def create_sudoku_grid_image(size=400, line_thickness=3, perspective_distort=True):
    """
    Create a synthetic 9x9 Sudoku grid image.
    
    Args:
        size: Size of the square image
        line_thickness: Thickness of grid lines
        perspective_distort: Whether to apply slight perspective distortion
        
    Returns:
        BGR image with Sudoku grid
    """
    # Create white background
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # Draw 9x9 grid lines
    cell_size = size // 9
    
    # Vertical lines
    for i in range(10):
        x = i * cell_size
        if x < size:
            cv2.line(img, (x, 0), (x, size), (0, 0, 0), line_thickness)
    
    # Horizontal lines
    for i in range(10):
        y = i * cell_size
        if y < size:
            cv2.line(img, (0, y), (size, y), (0, 0, 0), line_thickness)
    
    # Add some sample digits (optional)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row in range(9):
        for col in range(9):
            if (row + col) % 3 == 0:  # Add digits in some cells
                digit = str((row * 9 + col) % 9 + 1)
                x = col * cell_size + cell_size // 2 - 10
                y = row * cell_size + cell_size // 2 + 10
                cv2.putText(img, digit, (x, y), font, 0.8, (0, 0, 0), 2)
    
    if perspective_distort:
        # Apply slight perspective distortion
        h, w = img.shape[:2]
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32([
            [10, 20], [w-10, 10], [w-20, h-10], [20, h-20]
        ])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        img = cv2.warpPerspective(img, matrix, (w, h))
    
    return img


def create_simple_square_image(size=300):
    """Create a simple white square on black background for basic tests."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    margin = size // 6
    cv2.rectangle(img, (margin, margin), (size-margin, size-margin), (255, 255, 255), -1)
    return img


class TestGridDetection:
    """Test grid detection functions with synthetic images."""
    
    def test_order_corners(self):
        """Test corner ordering function."""
        # Test with a simple square
        points = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
        ordered = order_corners(points)
        
        # Should be ordered: top-left, top-right, bottom-right, bottom-left
        expected_order = np.array([
            [100, 100],  # top-left
            [200, 100],  # top-right  
            [200, 200],  # bottom-right
            [100, 200]   # bottom-left
        ], dtype=np.float32)
        
        np.testing.assert_array_equal(ordered, expected_order)
    
    def test_warp_to_square(self):
        """Test perspective warping to square with known points."""
        # Create a simple test image
        image = create_simple_square_image(300)
        
        # Define corners (already in correct order)
        corners = np.array([
            [50, 50],   # top-left
            [250, 50],  # top-right
            [250, 250], # bottom-right
            [50, 250]   # bottom-left
        ], dtype=np.float32)
        
        warped = warp_to_square(image, corners, size=450)
        
        # Should be a square image
        assert warped.shape[0] == warped.shape[1]
        assert warped.shape[0] == 450  # Specified size
        assert len(warped.shape) == 3  # BGR image
    
    def test_create_quad_overlay(self):
        """Test quad overlay creation."""
        image = create_simple_square_image(300)
        quad = np.array([
            [50, 50], [250, 50], [250, 250], [50, 250]
        ], dtype=np.float32)
        
        overlay = create_quad_overlay(image, quad)
        
        # Should be same size as input
        assert overlay.shape == image.shape
        # Should be different from original (overlay added)
        assert not np.array_equal(overlay, image)
    
    def test_find_and_warp_with_simple_square(self):
        """Test find_and_warp with a simple square image."""
        image = create_simple_square_image(300)
        
        artifacts = find_and_warp(image, size=400)
        
        # Check that all expected artifacts are present
        assert 'binary' in artifacts
        assert 'contour_mask' in artifacts
        assert 'quad' in artifacts
        assert 'warped' in artifacts
        
        # Check that warped image is square
        assert artifacts['warped'].shape[0] == artifacts['warped'].shape[1]
        assert artifacts['warped'].shape[0] == 400  # Specified size
        assert len(artifacts['warped'].shape) == 3  # BGR image
    
    @pytest.mark.slow
    def test_find_and_warp_with_sudoku_grid(self):
        """Test find_and_warp with a synthetic Sudoku grid."""
        image = create_sudoku_grid_image(size=400, line_thickness=3, perspective_distort=True)
        
        artifacts = find_and_warp(image, size=450)
        
        # Check that all expected artifacts are present
        assert 'binary' in artifacts
        assert 'contour_mask' in artifacts
        assert 'quad' in artifacts
        assert 'warped' in artifacts
        
        # Check that warped image is square
        assert artifacts['warped'].shape[0] == artifacts['warped'].shape[1]
        assert artifacts['warped'].shape[0] == 450  # Specified size
        assert len(artifacts['warped'].shape) == 3  # BGR image
        
        # Check that quad was found
        assert artifacts['quad'] is not None
        assert artifacts['quad'].shape == (4, 2)
    
    def test_find_and_warp_failure_cases(self):
        """Test find_and_warp with images that should fail."""
        # Test with empty image
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(GridNotFoundError):
            find_and_warp(empty_img)
        
        # Test with very small image
        tiny_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        with pytest.raises(GridNotFoundError):
            find_and_warp(tiny_img)
    
    def test_to_binary_with_clahe(self):
        """Test to_binary function with and without CLAHE."""
        image = create_sudoku_grid_image(size=300, line_thickness=2)
        
        # Test without CLAHE
        binary1 = to_binary(image, apply_clahe=False)
        assert binary1.shape == (300, 300)
        assert len(binary1.shape) == 2  # Grayscale
        assert binary1.dtype == np.uint8
        
        # Test with CLAHE
        binary2 = to_binary(image, apply_clahe=True)
        assert binary2.shape == (300, 300)
        assert len(binary2.shape) == 2  # Grayscale
        assert binary2.dtype == np.uint8
        
        # Results should be different (CLAHE affects the image)
        assert not np.array_equal(binary1, binary2)


class TestCellExtraction:
    """Test cell extraction functions with synthetic images."""
    
    def test_split_into_cells_basic(self):
        """Test basic cell splitting functionality."""
        # Create a perfect square image
        size = 450
        image = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # Draw a simple 9x9 grid
        cell_size = size // 9
        for i in range(10):
            x = i * cell_size
            y = i * cell_size
            cv2.line(image, (x, 0), (x, size), (0, 0, 0), 2)
            cv2.line(image, (0, y), (size, y), (0, 0, 0), 2)
        
        cells = split_into_cells(image, pad=4)
        
        # Should return exactly 81 cells
        assert len(cells) == 81
        
        # All cells should be roughly the same size
        cell_sizes = [cell.shape for cell in cells]
        assert all(len(shape) == 3 for shape in cell_sizes)  # BGR images
        
        # Check that cells are roughly equal size (within 2 pixels)
        heights = [shape[0] for shape in cell_sizes]
        widths = [shape[1] for shape in cell_sizes]
        assert max(heights) - min(heights) <= 2
        assert max(widths) - min(widths) <= 2
    
    def test_split_into_cells_with_padding(self):
        """Test cell splitting with different padding values."""
        size = 450
        image = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # Draw a simple 9x9 grid
        cell_size = size // 9
        for i in range(10):
            x = i * cell_size
            y = i * cell_size
            cv2.line(image, (x, 0), (x, size), (0, 0, 0), 2)
            cv2.line(image, (0, y), (size, y), (0, 0, 0), 2)
        
        # Test with different padding values
        for pad in [0, 2, 4, 8]:
            cells = split_into_cells(image, pad=pad)
            assert len(cells) == 81
            
            # With more padding, cells should be smaller
            if pad > 0:
                cell_sizes = [cell.shape for cell in cells]
                heights = [shape[0] for shape in cell_sizes]
                widths = [shape[1] for shape in cell_sizes]
                # Cells should be reasonably sized
                assert all(h > 10 for h in heights)
                assert all(w > 10 for w in widths)
    
    def test_split_into_cells_error_cases(self):
        """Test split_into_cells with invalid inputs."""
        # Test with non-square image
        rect_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        with pytest.raises(ValueError, match="must be square"):
            split_into_cells(rect_image)
        
        # Test with too small image
        tiny_image = np.ones((80, 80, 3), dtype=np.uint8) * 255
        with pytest.raises(ValueError, match="too small"):
            split_into_cells(tiny_image)
        
        # Test with grayscale image
        gray_image = np.ones((300, 300), dtype=np.uint8) * 255
        with pytest.raises(ValueError, match="3-channel BGR"):
            split_into_cells(gray_image)
    
    @pytest.mark.slow
    def test_complete_pipeline_with_sudoku_grid(self):
        """Test complete pipeline: find_and_warp -> split_into_cells."""
        # Create a synthetic Sudoku grid with perspective distortion
        image = create_sudoku_grid_image(size=400, line_thickness=3, perspective_distort=True)
        
        # Run find_and_warp
        artifacts = find_and_warp(image, size=450)
        
        # Extract cells
        cells = split_into_cells(artifacts['warped'], pad=4)
        
        # Should get exactly 81 cells
        assert len(cells) == 81
        
        # All cells should be BGR images
        for cell in cells:
            assert len(cell.shape) == 3
            assert cell.shape[2] == 3
        
        # Cells should be roughly equal size
        cell_sizes = [cell.shape[:2] for cell in cells]
        heights = [size[0] for size in cell_sizes]
        widths = [size[1] for size in cell_sizes]
        
        # All cells should be reasonably sized
        assert all(h > 10 for h in heights)
        assert all(w > 10 for w in widths)
        
        # Size variation should be minimal
        assert max(heights) - min(heights) <= 3
        assert max(widths) - min(widths) <= 3


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
