import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read image and convert from BGR to RGB
image_path = './imgs/car.webp' 
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the RGB image
h, w = image.shape[:2]
plt.subplot(1, 1, 1)
plt.imshow(image_rgb)
plt.title('RGB Image')
plt.axis('off')
plt.show()

# Step 2: Translation, rotation, and scaling
fig, axes = plt.subplots(1, 3, figsize=(12, 6))  # 1 row, 2 columns

# Translation
tx, ty = 50, 50  # Translate by 50 pixels in x and y
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(image_rgb, translation_matrix, (w, h))

# Rotation
angle = 45
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (w, h))

# Scaling
scale_factor = 2
scaled_image = cv2.resize(image_rgb, (1050, 1050))


# Display the transformations
axes[0].set_title('Translated')
axes[0].imshow(translated_image)
axes[0].axis('off')

axes[1].imshow(rotated_image)
axes[1].set_title('Rotated')
axes[1].axis('off')

# Scaled image display with adjusted layout
axes[2].imshow(scaled_image)
axes[2].set_title('Scaled')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Step 3: Edge detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobel_edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
laplacian_edges = cv2.Laplacian(gray_image, cv2.CV_64F)
canny_edges = cv2.Canny(gray_image, 100, 200)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Sobel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian_edges, cmap='gray')
plt.title('Laplacian')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny')
plt.axis('off')
plt.show()

# Step 4: Global and adaptive thresholding
_, global_thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
adaptive_thresh = cv2.adaptiveThreshold(
    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(global_thresh, cmap='gray')
plt.title('Global Thresholding')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title('Adaptive Thresholding')
plt.axis('off')
plt.show()


# Step 5: K-means clustering for segmentation
image_flatten = image.reshape((-1, 3))
image_flatten = np.float32(image_flatten)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4  # Number of clusters
_, labels, centers = cv2.kmeans(image_flatten, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_image = centers[labels.flatten()].reshape(image.shape)

plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Segmented Image (K-means)')
plt.axis('off')
plt.show()

# Step 6: SIFT keypoints detection
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray_image, None)
keypoint_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints')
plt.axis('off')
plt.show()
