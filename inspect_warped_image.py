import cv2
import numpy as np

# Load the warped image
warped = cv2.imread('out/004_warped.png')
print(f"Warped image shape: {warped.shape}")

# Convert to grayscale and apply threshold to see all potential content
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Calculate cell size
cell_size = 450 // 9
print(f"Cell size: {cell_size}x{cell_size}")

# Check a few specific cells manually
print("\nManual inspection of cells:")
for row in range(3):
    for col in range(3):
        # Extract cell manually
        y_start = row * cell_size
        y_end = (row + 1) * cell_size
        x_start = col * cell_size
        x_end = (col + 1) * cell_size
        
        cell = warped[y_start:y_end, x_start:x_end]
        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        cell_thresh = cv2.adaptiveThreshold(cell_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        white_pixels = np.sum(cell_thresh > 0)
        total_pixels = cell_thresh.size
        area_ratio = white_pixels / total_pixels
        
        print(f"Cell r{row}_c{col}: white_pixels={white_pixels}, area_ratio={area_ratio:.3f}")
        
        # Save the cell for inspection
        cv2.imwrite(f'out/manual_inspect_r{row}_c{col}.png', cell)
        cv2.imwrite(f'out/manual_inspect_r{row}_c{col}_thresh.png', cell_thresh)

# Also check the overall thresholded image
cv2.imwrite('out/warped_thresh.png', thresh)
print(f"\nSaved thresholded warped image to out/warped_thresh.png")

# Count total white pixels in the thresholded image
total_white = np.sum(thresh > 0)
total_pixels = thresh.size
print(f"Overall white pixels: {total_white}/{total_pixels} ({total_white/total_pixels:.3f})")

