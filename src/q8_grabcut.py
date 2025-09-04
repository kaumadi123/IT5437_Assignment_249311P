import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ensure results folder exists
os.makedirs("results", exist_ok=True)

# Load image
img = cv2.imread("images/daisy.jpg")
if img is None:
    raise FileNotFoundError("Image not found: images/daisy.jpg")

# Initialize mask
mask = np.zeros(img.shape[:2], np.uint8)

# GrabCut models
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Define rectangle around the flower (adjust if needed)
h, w = img.shape[:2]
rect = (int(w*0.2), int(h*0.1), int(w*0.6), int(h*0.8))  # (x,y,w,h)

# Apply GrabCut
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Final mask: probable/definite foreground -> 1, background -> 0
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

# Foreground and background
foreground = img * mask2[:, :, np.newaxis]
background = img * (1 - mask2[:, :, np.newaxis])

# Save mask, fg, bg
cv2.imwrite("results/q8_mask.png", mask2 * 255)
cv2.imwrite("results/q8_foreground.png", foreground)
cv2.imwrite("results/q8_background.png", background)

# (b) Enhance with blurred background
blurred_bg = cv2.GaussianBlur(img, (25, 25), 0)
enhanced = blurred_bg * (1 - mask2[:, :, np.newaxis]) + foreground

cv2.imwrite("results/q8_enhanced.png", enhanced)

# Show side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original")
axes[1].imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)); axes[1].set_title("Foreground")
axes[2].imshow(cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_BGR2RGB)); axes[2].set_title("Enhanced (blurred background)")
for ax in axes: ax.axis("off")
plt.savefig("results/q8_comparison.png")
plt.close()

print("✅ Q8 done. Check results folder.")
