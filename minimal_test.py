#!/usr/bin/env python3
"""
Minimal test for the custom CNN implementation.
"""

print("Starting minimal test...")

try:
    import numpy as np
    print("✓ NumPy imported")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")
    exit(1)

try:
    import cv2
    print("✓ OpenCV imported")
except Exception as e:
    print(f"✗ OpenCV import failed: {e}")
    exit(1)

# Test basic functionality
print("Testing basic functionality...")

# Create a simple test image
img = np.ones((48, 48, 3), dtype=np.uint8) * 255
cv2.putText(img, '5', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
print("✓ Test image created")

# Test contour detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"✓ Found {len(contours)} contours")

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    print(f"✓ Largest contour area: {area}")

print("✓ Minimal test completed successfully!")



