import numpy as np
import cv2
from matplotlib import pyplot as plt

# load image
image = cv2.imread('image.png')
# convert to RGB if necesaary
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Flatten into 2D array
pixels = image.reshape(-1, 3).astype(np.float32)
#Define K-Means parameters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3
# Apply K-Means
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# Reshape labels to size of original image
labels = labels.reshape(image.shape[0], image.shape[1])
# Create k segments
segmented_image = np.zeros_like(image)
for i in range(k):
    segmented_image[labels == i] = centers[i]

# Display segments
plt.figure()
plt.imshow(image)
plt.title('Original Image')
plt.show()

plt.figure()
plt.imshow(segmented_image)
plt.title('Segmented Image')
plt.show()