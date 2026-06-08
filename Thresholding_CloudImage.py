import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

stratus = cv.imread('dataset/stratus1.jpg')

# Convert to grayscale
gray_stratus = cv.cvtColor(stratus, cv.COLOR_BGR2GRAY)

# Otsu's thresholding
Threshold_value, otsu_thresholded = cv.threshold(gray_stratus, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print(f"Otsu threshold value: {Threshold_value}")

# ─── MORPHOLOGICAL CLEANUP ──────────────────────────────────────
kernel = np.ones((5, 5), np.uint8)
cleaned = cv.morphologyEx(otsu_thresholded, cv.MORPH_OPEN, kernel)   # remove specks
cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)           # fill holes

# ─── HISTOGRAM OF PIXEL INTENSITIES ─────────────────────────────
plt.hist(gray_stratus.ravel(), 256)
plt.title('Histogram of pixel intensities')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')
plt.show()

# Visualize the result
plt.subplot(1, 3, 1)
plt.imshow(gray_stratus, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(otsu_thresholded, cmap='gray')
plt.title("After Otsu's Method Thresholding")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cleaned, cmap='gray')
plt.title("Morphologically Cleaned")
plt.axis('off')

plt.show()