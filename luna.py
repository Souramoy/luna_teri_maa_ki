# Lunar DEM Generation using Shape-from-Shading (Beginner Python Version)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Step 1: Load the lunar image (grayscale)
image_path = "img.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found. Make sure the path is correct.")

# Normalize image between 0 and 1
img = img.astype(np.float32) / 255.0

# Step 2: Estimate surface normals using brightness
# Simplified Lambertian reflectance model

# Assume sun coming from top-left (example only)
light_vector = np.array([1, 1, 1])
light_vector = light_vector / np.linalg.norm(light_vector)

# Compute gradients (brightness slope approximation)
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# Smooth gradients
gx = gaussian_filter(gx, sigma=1)
gy = gaussian_filter(gy, sigma=1)

# Estimate surface normals (Z = height)
normals = np.dstack((-gx, -gy, np.ones_like(img)))
norm_mags = np.linalg.norm(normals, axis=2)
normals /= norm_mags[..., np.newaxis] + 1e-8  # Normalize

# Step 3: Estimate surface depth from gradients
# Integrate gradients to get height map (simple integration)
z = np.zeros_like(img)
for y in range(1, z.shape[0]):
    z[y, 0] = z[y-1, 0] + gy[y, 0]
for x in range(1, z.shape[1]):
    z[:, x] = z[:, x-1] + gx[:, x]

# Normalize depth map
z = z - np.min(z)
z = z / np.max(z)

# Step 4: Visualize results
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Input Lunar Image')

plt.subplot(1, 3, 2)
plt.quiver(normals[::10, ::10, 0], -normals[::10, ::10, 1])
plt.title('Estimated Normals')

plt.subplot(1, 3, 3)
plt.imshow(z, cmap='terrain')
plt.colorbar(label='Normalized Height')
plt.title('Estimated DEM')

plt.tight_layout()
plt.show()
