from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

image = mpimg.imread('Cumulonimbus.jpg')

sharpen_edge = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# To use kernel (filtering image), use cv.filter2D function
sharpened_image = cv.filter2D(image, -1, sharpen_edge)

plt.xlabel('Width')
plt.ylabel('Height')
plt.title('Cloud image')
plt.imshow(image)
print(image.shape)
plt.show()