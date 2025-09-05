import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

img_path = "images/einstein.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

# Sobel kernels
sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]], dtype=np.float32)

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], dtype=np.float32)

# (a) Sobel using filter2D
gx_a = cv2.filter2D(img, cv2.CV_32F, sobel_x)
gy_a = cv2.filter2D(img, cv2.CV_32F, sobel_y)
mag_a = cv2.magnitude(gx_a, gy_a)
cv2.imwrite("results/q6_sobel_a.png", np.uint8(mag_a))

# (b) Manual convolution
def conv2d(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    out = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)
    return out

gx_b = conv2d(img, sobel_x)
gy_b = conv2d(img, sobel_y)
mag_b = np.sqrt(gx_b**2 + gy_b**2)
cv2.imwrite("results/q6_sobel_b.png", np.uint8(mag_b))

# (c) Separable filtering
# Separable filters: [1,2,1]^T and [1,0,-1]
kcol = np.array([1, 2, 1], dtype=np.float32).reshape(3,1)
krow = np.array([1, 0, -1], dtype=np.float32).reshape(1,3)

gx_c = cv2.sepFilter2D(img, cv2.CV_32F, krow, kcol)  # horizontal
gy_c = cv2.sepFilter2D(img, cv2.CV_32F, kcol, krow)  # vertical
mag_c = cv2.magnitude(gx_c, gy_c)
cv2.imwrite("results/q6_sobel_c.png", np.uint8(mag_c))

# Save side-by-side comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 6))
axes[0].imshow(img, cmap='gray'); axes[0].set_title("Original")
axes[1].imshow(mag_a, cmap='gray'); axes[1].set_title("Sobel filter2D (a)")
axes[2].imshow(mag_b, cmap='gray'); axes[2].set_title("Manual conv (b)")
axes[3].imshow(mag_c, cmap='gray'); axes[3].set_title("Separable (c)")
for ax in axes: ax.axis('off')
plt.savefig("results/q6_comparison.png")

print("✅ Q6 done. Check 'results' folder for sobel outputs (a, b, c).")
