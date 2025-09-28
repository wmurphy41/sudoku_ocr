import cv2
import numpy as np

def analyze_digit_patterns():
    """Analyze the patterns in the manual cell extractions to understand digit characteristics."""
    
    # Load and analyze several cells that likely contain digits
    cells_to_analyze = [
        ('r0_c0', 'First cell - likely digit 1'),
        ('r1_c2', 'Second row, third column - likely a digit'),
        ('r2_c0', 'Third row, first column - likely a digit'),
        ('r2_c2', 'Third row, third column - likely a digit'),
        ('r3_c2', 'Fourth row, third column - likely a digit'),
        ('r4_c0', 'Fifth row, first column - likely a digit'),
    ]
    
    print("Analyzing digit patterns:")
    print("=" * 50)
    
    for cell_name, description in cells_to_analyze:
        # Load the thresholded image
        thresh_path = f'out/manual_cell_{cell_name}_thresh.png'
        thresh = cv2.imread(thresh_path, cv2.IMREAD_GRAYSCALE)
        
        if thresh is None:
            print(f"{cell_name}: Could not load image")
            continue
            
        # Basic stats
        white_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        area_ratio = white_pixels / total_pixels
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"{cell_name}: No contours found")
            continue
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate some additional features
        # 1. Centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
            
        # 2. Compactness (perimeter^2 / area)
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = (perimeter * perimeter) / area if area > 0 else 0
        
        # 3. Solidity (area / convex_hull_area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 4. Position in cell (normalized)
        cell_size = 50
        norm_x = x / cell_size
        norm_y = y / cell_size
        norm_w = w / cell_size
        norm_h = h / cell_size
        
        print(f"\n{cell_name} ({description}):")
        print(f"  Area ratio: {area_ratio:.3f}")
        print(f"  Contour area: {area:.1f}")
        print(f"  Aspect ratio: {aspect_ratio:.2f}")
        print(f"  Bounding box: ({x},{y},{w},{h})")
        print(f"  Centroid: ({cx},{cy})")
        print(f"  Compactness: {compactness:.2f}")
        print(f"  Solidity: {solidity:.3f}")
        print(f"  Normalized pos: ({norm_x:.2f}, {norm_y:.2f})")
        print(f"  Normalized size: ({norm_w:.2f}, {norm_h:.2f})")
        
        # Try to identify the digit based on these features
        digit_guess = guess_digit(area_ratio, aspect_ratio, compactness, solidity, norm_w, norm_h)
        print(f"  Digit guess: {digit_guess}")

def guess_digit(area_ratio, aspect_ratio, compactness, solidity, norm_w, norm_h):
    """Guess the digit based on feature analysis."""
    
    # Very basic heuristics based on the patterns I'm seeing
    if area_ratio < 0.1:
        return 0  # Empty
    
    # Check for very wide shapes (likely 1, 4, 7)
    if aspect_ratio > 2.0:
        return 1
    
    # Check for very tall shapes (likely 1, 6, 8, 9)
    if aspect_ratio < 0.5:
        return 1
    
    # Check for compact, solid shapes (likely 0, 6, 8, 9)
    if compactness < 20 and solidity > 0.8:
        if norm_w > 0.6 and norm_h > 0.6:  # Large and square-ish
            return 8
        else:
            return 6
    
    # Check for medium-sized shapes
    if 0.3 < area_ratio < 0.5:
        if aspect_ratio > 1.2:
            return 4
        elif aspect_ratio < 0.8:
            return 6
        else:
            return 5
    
    # Check for small, compact shapes
    if area_ratio < 0.3:
        if aspect_ratio > 1.5:
            return 1
        else:
            return 7
    
    # Default guess
    return 5

if __name__ == "__main__":
    analyze_digit_patterns()

