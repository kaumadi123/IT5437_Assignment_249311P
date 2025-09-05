# Q9 — Rice grain counting
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ensure results folder exists
os.makedirs("results", exist_ok=True)

# Load rice image (place rice.png in images/)
img = cv2.imread("images/rice.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found: images/rice.png")

# (a) light Gaussian denoise
g_d = cv2.GaussianBlur(img, (3, 3), 0.8)

# (b) background flattening with large morphological opening
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
bg = cv2.morphologyEx(g_d, cv2.MORPH_OPEN, k)
flat = cv2.subtract(g_d, bg)
flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)

# (c) Otsu threshold (ensure grains are white)
_, th_raw = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if np.mean(flat[th_raw > 0]) < np.mean(flat[th_raw == 0]):
    th_raw = cv2.bitwise_not(th_raw)

# (d) morphology clean-up
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(th_raw, cv2.MORPH_OPEN, se, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=2)

# remove tiny junk by connected components
num, lab = cv2.connectedComponents(mask)
areas = np.bincount(lab.ravel())
keep = np.ones(num, np.uint8)
keep[0] = 0
minA, maxA = 30, 10000
for i, a in enumerate(areas):
    if i == 0: 
        continue
    if a < minA or a > maxA:
        keep[i] = 0
mask = (keep[lab] * 255).astype(np.uint8)

# fill holes function
def fill_holes(bin255):
    inv = cv2.bitwise_not(bin255)
    h, w = inv.shape
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    flood = inv.copy()
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(bin255, holes)

mask_filled = fill_holes(mask)

# (optional) watershed split of touching grains
bin8 = (mask_filled > 0).astype(np.uint8)
dist = cv2.distanceTransform(bin8, cv2.DIST_L2, 5)
markers = (dist > 0.45 * dist.max()).astype(np.uint8)
_, markers = cv2.connectedComponents(markers)
markers = markers + 1
markers[bin8 == 0] = 0
rgb_for_ws = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
ws = cv2.watershed(rgb_for_ws.copy(), markers.astype(np.int32))
seg = (ws > 1).astype(np.uint8) * 255

# (e) count rice grains
n_final, _ = cv2.connectedComponents((seg > 0).astype(np.uint8))
count = n_final - 1

# Save outputs
cv2.imwrite("results/q9_denoised.png", g_d)
cv2.imwrite("results/q9_background.png", bg)
cv2.imwrite("results/q9_flattened.png", flat)
cv2.imwrite("results/q9_mask_filled.png", mask_filled)
cv2.imwrite("results/q9_final.png", seg)

# comparison figure
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0,0].imshow(img, cmap="gray"); axes[0,0].set_title("Original")
axes[0,1].imshow(g_d, cmap="gray"); axes[0,1].set_title("Denoised")
axes[0,2].imshow(bg, cmap="gray"); axes[0,2].set_title("Background")
axes[1,0].imshow(flat, cmap="gray"); axes[1,0].set_title("Flattened")
axes[1,1].imshow(mask_filled, cmap="gray"); axes[1,1].set_title("Mask Filled")
axes[1,2].imshow(seg, cmap="gray"); axes[1,2].set_title(f"Final (count={count})")
for ax in axes.ravel(): ax.axis("off")
plt.tight_layout()
plt.savefig("results/q9_comparison.png")
plt.close()

print(f"✅ Q9 done. Estimated number of rice grains: {count}. Check results folder.")
