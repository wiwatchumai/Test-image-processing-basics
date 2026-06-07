import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


image = cv.imread('Cumulonimbus.jpg')
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Set threshold value
threshold_value = 127

height = gray_image.shape[0]
width = gray_image.shape[1]


# 0-255 is the level of intensity from black to white
# Convert to grayscale only has one channel, therefore white = RGB(255), black = RGB(0)
# If the image is in RGB which has three channels, 
        # the thresholding will be applied to each channal separately as white = RGB(255, 255, 255), black = RGB(0, 0, 0)

for row in range(height):
    for col in range(width):
        if gray_image[row,col]>threshold_value:
            gray_image[row,col] = 255
        else:
            gray_image[row,col] = 0

plt.xlabel('Width')
plt.ylabel('Height')
plt.title('Thresholded Image')
plt.imshow(gray_image, cmap='gray')
print(gray_image.shape)
plt.show()