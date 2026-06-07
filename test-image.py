import cv2 as cv 
import numpy as np 
from matplotlib import pyplot as plt

image = cv.imread('dataset/Stratus2.png')
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

otsu_threshold, otsu_thresholded = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print(f"Otsu threshold value: {otsu_threshold}")

kernel = np.ones((5, 5), np.uint8)
cleaned = cv.morphologyEx(otsu_thresholded, cv.MORPH_OPEN, kernel)
cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)

plt.hist(gray_image.ravel(), 256, (0, 256))
plt.title('Histogram of pixel intensities')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')
plt.show()

plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(otsu_thresholded, cmap='gray')
plt.title("After Otsu's Method Thresholding")

plt.subplot(1, 3, 3)
plt.imshow(cleaned, cmap='gray')
plt.title("Morphologically Cleaned")
plt.show()