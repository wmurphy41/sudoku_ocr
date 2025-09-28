import cv2
import numpy as np

# Load the warped image
warped = cv2.imread('out/004_warped.png')
print(f"Warped image shape: {warped.shape}")

# Calculate cell size
cell_size = 450 // 9
print(f"Cell size: {cell_size}x{cell_size}")

# Test manual extraction vs our function
print("\nComparing manual vs function extraction:")

# Manual extraction
def manual_extract_cell(warped, row, col, pad=4):
    cell_size = 450 // 9
    y1 = row * cell_size
    y2 = (row + 1) * cell_size
    x1 = col * cell_size
    x2 = (col + 1) * cell_size
    
    cell = warped[y1:y2, x1:x2].copy()
    
    # Apply padding
    if pad > 0:
        cell_h, cell_w = cell.shape[:2]
        pad_y = min(pad, cell_h // 3) if cell_h > 6 else 0
        pad_x = min(pad, cell_w // 3) if cell_w > 6 else 0
        
        if pad_y > 0 and pad_x > 0:
            cell = cell[pad_y:cell_h-pad_y, pad_x:cell_w-pad_x]
    
    return cell

# Test a few cells
test_cells = [(0, 0), (0, 3), (1, 2), (2, 0), (4, 0)]

for row, col in test_cells:
    # Manual extraction
    manual_cell = manual_extract_cell(warped, row, col, pad=4)
    
    # Our function extraction
    from src.sudoku_ocr.cells import split_into_cells
    function_cells = split_into_cells(warped, pad=4)
    function_cell = function_cells[row * 9 + col]
    
    print(f"\nCell r{row}_c{col}:")
    print(f"  Manual: shape={manual_cell.shape}, range={manual_cell.min()}-{manual_cell.max()}")
    print(f"  Function: shape={function_cell.shape}, range={function_cell.min()}-{function_cell.max()}")
    
    # Check if they're the same
    if manual_cell.shape == function_cell.shape:
        diff = np.sum(np.abs(manual_cell.astype(float) - function_cell.astype(float)))
        print(f"  Difference: {diff}")
    else:
        print(f"  Different shapes!")
    
    # Save both for comparison
    cv2.imwrite(f'out/compare_r{row}_c{col}_manual.png', manual_cell)
    cv2.imwrite(f'out/compare_r{row}_c{col}_function.png', function_cell)

print("\nSaved comparison images to out/compare_*.png")

