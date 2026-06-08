import cv2 as cv 
import numpy as np 
from matplotlib import pyplot as plt
import os

folder = 'dataset'
files = os.listdir(folder)

for filenames in files:
    path = os.path.join(folder, filenames)
    image = cv.imread(path)
    if image is None:
        continue
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    otsu_threshold, otsu_thresholded = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    print(f"Otsu threshold value: {otsu_threshold}")

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv.morphologyEx(otsu_thresholded, cv.MORPH_OPEN, kernel)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)

    cloud_cover = np.sum(cleaned == 255) / cleaned.size

    if cloud_cover > 0.8 and cloud_cover < 0.9:
        status = "Cloudy"
    elif cloud_cover >= 0.9 and cloud_cover <= 0.5:
        status = "FAIL"
    else:
        status = "VALID"    

    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(cleaned, cmap='gray')
    plt.title("Morphologically Cleaned")

    plt.show()

    print(f"{filenames} — Otsu T: {otsu_threshold}  Cloud cover: {cloud_cover:.1%} {status}")

# To find sum of all pixels in the intersection between images, you can use intersection.sum() which gives you the total remaining pixels you are looking for
# intersection.sum() does not count matrix element with 0 value, so it will give you only 255 as we are using the gray-scale image converted to black-white 

# If we have ground truth, we can compute IoU to see if the image is accurately segmented.
# If no ground truth, just look at it by eyes