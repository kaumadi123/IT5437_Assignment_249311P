# File: src/q10_sapphire_area.py
# Q10 — Two sapphires on a table: segment → fill → pixel areas → mm² areas

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ensure results folder exists
os.makedirs("results", exist_ok=True)

# -----------------------------
# (a) Load and segment sapphires (color-based in HSV)
# -----------------------------
img = cv2.imread("images/sapphire.jpg", cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("Image not found: images/sapphire.jpg")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV range for blue sapphires
lower = np.array([100, 60, 40], dtype=np.uint8)
upper = np.array([140, 255, 255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower, upper)

# Morphological cleanup
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=2)

# -----------------------------
# (b) Fill holes
# -----------------------------
def fill_holes(bin255):
    inv = cv2.bitwise_not(bin255)
    h, w = inv.shape
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    flood = inv.copy()
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(bin255, holes)

filled = fill_holes(mask)

# Keep only the two largest components
num, lab, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
areas_sorted = sorted(areas, key=lambda t: t[1], reverse=True)[:2]
keep = {i for i, _ in areas_sorted}
mask_two = np.where(np.isin(lab, list(keep)), 255, 0).astype(np.uint8)
mask_two = fill_holes(mask_two)

cv2.imwrite("results/q10_mask.png", mask_two)

# -----------------------------
# (c) Areas in pixels
# -----------------------------
num2, lab2, stats2, cents2 = cv2.connectedComponentsWithStats(mask_two, connectivity=8)
areas_px = [int(stats2[i, cv2.CC_STAT_AREA]) for i in range(1, num2)]

# -----------------------------
# (d) Convert to mm²
# -----------------------------
f_mm = 8.0
Z_mm = 480.0
pixel_pitch_mm = 0.0048  # 4.8 µm default
mm_per_px = (Z_mm / f_mm) * pixel_pitch_mm
mm2_per_px = mm_per_px ** 2
areas_mm2 = [round(a * mm2_per_px, 3) for a in areas_px]

print(f"Scale: {mm_per_px:.6f} mm/px  →  {mm2_per_px:.6f} mm²/px  (pixel_pitch={pixel_pitch_mm} mm)")
print("Areas (pixels):", areas_px)
print("Areas (mm²):   ", areas_mm2)

# -----------------------------
# Save annotated visualization
# -----------------------------
color_img = rgb.copy()
for i in range(1, num2):
    x, y, w, h, area = stats2[i]
    label_text = f"Sapphire {i}: {areas_mm2[i-1]:.1f} mm²"
    cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(color_img, label_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("results/q10_annotated.png", cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

# Comparison figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rgb); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(mask_two, cmap="gray"); axes[1].set_title("Mask (2 sapphires)"); axes[1].axis("off")
axes[2].imshow(color_img); axes[2].set_title("Annotated"); axes[2].axis("off")
plt.tight_layout()
plt.savefig("results/q10_comparison.png")
plt.close()

print("✅ Q10 done. Check results folder for masks and annotated outputs.")
