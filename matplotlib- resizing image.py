from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2 as cv
import numpy as np

# Read the image
image = mpimg.imread('Cumulonimbus.jpg')
reimage = cv.resize(image, (50, 50), fx=2.5, fy=2.5, interpolation=cv.INTER_LINEAR)

# Convert image to numpy array
array = np.array(reimage)
# print (array)

# Try merging the image pixels for segmentation
# The manual way to do it is to loop through the image and merge the pixels, but we can use numpy to do it more efficiently
k=1
for i in range(0, reimage.shape[0], k): # Loop through height 
    for j in range(0, reimage.shape[1], k): # Loop through width

        # Identify a block of pixel
        block = array[i:i+k, j:j+k]

        # Calculate the average color of the block
        avg_color = np.mean(block, axis=(0, 1))

        # Set the color of the block to the average color
        array[i:i+k, j:j+k] = avg_color




# Plot the resized image
plt.xlabel('Width')
plt.ylabel('Height')
plt.title('Resized Image')
plt.imshow(reimage)
print(reimage.shape)
plt.show()

