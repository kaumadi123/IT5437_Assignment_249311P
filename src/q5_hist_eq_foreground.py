import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ensure results folder exists
os.makedirs("results", exist_ok=True)

# load image
img_path = "images/taylor.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# -------------------------------
# Step (a) Save H, S, V planes
# -------------------------------
cv2.imwrite("results/q5_H.png", h)
cv2.imwrite("results/q5_S.png", s)
cv2.imwrite("results/q5_V.png", v)

# -------------------------------
# Step (b) Threshold (Otsu on V)
# -------------------------------
_, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("results/q5_mask.png", mask)

# -------------------------------
# Step (c) Extract foreground
# -------------------------------
fg = cv2.bitwise_and(v, v, mask=mask)
cv2.imwrite("results/q5_foreground.png", fg)

# compute histogram of foreground
hist = cv2.calcHist([v], [0], mask, [256], [0, 256]).flatten()

# -------------------------------
# Step (d) Cumulative sum
# -------------------------------
cdf = hist.cumsum()
cdf_min = cdf[cdf > 0][0]
N = mask.sum() / 255  # number of foreground pixels

# -------------------------------
# Step (e) Histogram equalization mapping
# -------------------------------
lut = np.floor((cdf - cdf_min) / (N - cdf_min) * 255).clip(0, 255).astype(np.uint8)

# apply LUT only on foreground
v_eq = v.copy()
v_eq[mask == 255] = lut[v[mask == 255]]

# -------------------------------
# Step (f) Recombine with background
# -------------------------------
hsv_eq = cv2.merge([h, s, v_eq])
result = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

# save results
cv2.imwrite("results/q5_result.png", result)

# -------------------------------
# Extra: save histogram plots
# -------------------------------
plt.figure()
plt.hist(v[mask == 255].ravel(), bins=256, color='gray')
plt.title("Foreground Histogram (Original)")
plt.savefig("results/q5_hist_foreground_original.png")

plt.figure()
plt.hist(v_eq[mask == 255].ravel(), bins=256, color='blue')
plt.title("Foreground Histogram (Equalized)")
plt.savefig("results/q5_hist_foreground_equalized.png")

print("✅ Q5 done. Check 'results' folder for H,S,V planes, mask, and equalized foreground result.")
