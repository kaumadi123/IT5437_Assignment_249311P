import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ensure results folder exists
os.makedirs("results", exist_ok=True)

# load image (grayscale)
img_path = "images/brain_proton_density_slice.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

# ----------------------------
# Step 1: plot histogram
# ----------------------------
plt.figure()
plt.hist(img.ravel(), bins=256, color='black')
plt.title("Brain image histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.savefig("results/q2_hist.png")

# ----------------------------
# Step 2: create LUT functions
# ----------------------------
def make_piecewise(points):
    """points = [(x_in, x_out), ...]"""
    xs, ys = zip(*points)
    lut = np.interp(np.arange(256), xs, ys).astype('uint8')
    return lut

# Accentuate white matter
points_white = [(0,0), (100,60), (140,180), (200,255), (255,255)]
lut_white = make_piecewise(points_white)
out_white = cv2.LUT(img, lut_white)

# Accentuate gray matter
points_gray = [(0,0), (60,60), (120,200), (160,220), (255,255)]
lut_gray = make_piecewise(points_gray)
out_gray = cv2.LUT(img, lut_gray)

# ----------------------------
# Step 3: save outputs
# ----------------------------
cv2.imwrite("results/q2_white.png", out_white)
cv2.imwrite("results/q2_gray.png", out_gray)

# plot transformation curves
plt.figure()
plt.plot(np.arange(256), lut_white, label="White matter LUT")
plt.plot(np.arange(256), lut_gray, label="Gray matter LUT")
plt.legend()
plt.title("Q2 Intensity Transformation Curves")
plt.xlabel("Input intensity")
plt.ylabel("Output intensity")
plt.savefig("results/q2_mappings.png")

# plot histograms of outputs
plt.figure()
plt.hist(out_white.ravel(), bins=256, color='blue')
plt.title("Output histogram (White matter emphasized)")
plt.savefig("results/q2_hist_white.png")

plt.figure()
plt.hist(out_gray.ravel(), bins=256, color='green')
plt.title("Output histogram (Gray matter emphasized)")
plt.savefig("results/q2_hist_gray.png")

print("✅ Q2 done. Check 'results' folder for white/gray matter outputs & plots.")
