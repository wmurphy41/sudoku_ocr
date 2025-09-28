import cv2
import numpy as np

# Load the warped image
warped = cv2.imread('out/004_warped.png')
print(f"Warped image shape: {warped.shape}")

# Convert to grayscale and apply threshold to see all potential digits
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} contours")

# Filter contours by area (potential digits)
digit_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  # Reasonable size for a digit
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio for digits
            digit_contours.append((contour, area, x, y, w, h))

print(f"Found {len(digit_contours)} potential digit contours")

# Sort by area (largest first)
digit_contours.sort(key=lambda x: x[1], reverse=True)

for i, (contour, area, x, y, w, h) in enumerate(digit_contours[:10]):  # Show top 10
    print(f"Contour {i}: area={area:.1f}, bbox=({x},{y},{w},{h}), aspect_ratio={w/h:.2f}")

# Create a visualization
vis = warped.copy()
for i, (contour, area, x, y, w, h) in enumerate(digit_contours[:5]):  # Draw top 5
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
    cv2.putText(vis, f"{i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imwrite('out/debug_digit_contours.png', vis)
print("Saved debug visualization to out/debug_digit_contours.png")

