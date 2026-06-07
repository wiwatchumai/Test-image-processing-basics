import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

stratus = cv.imread('stratus1.jpg')

# Convert to grayscale
gray_stratus = cv.cvtColor(stratus, cv.COLOR_BGR2GRAY)

# Otsu's thresholding
Threshold_value, otsu_thresholded = cv.threshold(gray_stratus, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print(f"Otsu threshold value: {Threshold_value}")

# ─── MORPHOLOGICAL CLEANUP ──────────────────────────────────────
kernel = np.ones((5, 5), np.uint8)
cleaned = cv.morphologyEx(otsu_thresholded, cv.MORPH_OPEN, kernel)   # remove specks
cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)           # fill holes

# ─── IoU SCORING ────────────────────────────────────────────────
def compute_iou(pred, true):
    pred = pred > 127
    true = true > 127
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return intersection / union

# Load the ground truth mask (adjust filename to match your SWIMSEG file)
true_mask = cv.imread('stratus1.jpg', cv.IMREAD_GRAYSCALE)

if true_mask is None:
    print("⚠ Ground truth file not found — skipping IoU. Check the filename/path.")
else:
    iou = compute_iou(cleaned, true_mask)
    print(f"IoU score: {iou:.3f}")
# ────────────────────────────────────────────────────────────────

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