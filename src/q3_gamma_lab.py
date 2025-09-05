import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

img_path = "images/highlights_and_shadows.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, a, b = cv2.split(lab)

L_norm = L / 255.0

gamma = 0.8
L_gamma = np.power(L_norm, gamma)

L_corrected = np.clip(L_gamma * 255.0, 0, 255).astype("uint8")

lab_corrected = cv2.merge([L_corrected, a, b])
img_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

cv2.imwrite("results/q3_gamma_corrected.png", img_corrected)

# Histograms
plt.figure()
plt.hist(L.ravel(), bins=256, color="gray")
plt.title("Histogram of Original L channel")
plt.savefig("results/q3_hist_L_original.png")

plt.figure()
plt.hist(L_corrected.ravel(), bins=256, color="blue")
plt.title("Histogram of Corrected L channel (gamma=0.8)")
plt.savefig("results/q3_hist_L_corrected.png")

print("✅ Q3 done. Gamma correction applied (gamma=0.8). Check 'results' folder.")
