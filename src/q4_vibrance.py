import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

img_path = "images/spider.png"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

sigma = 70.0
a = 0.6

def vibrance_curve_gaussian(a=0.6, sigma=70.0):
    x = np.arange(256, dtype=np.float32)
    bump = a * 128.0 * np.exp(-((x - 128.0) ** 2) / (2.0 * sigma ** 2))
    y = np.minimum(x + bump, 255.0)
    return y.astype(np.uint8)

# Build LUT and apply it to the S channel
lut = vibrance_curve_gaussian(a=a, sigma=sigma)
s_enhanced = cv2.LUT(s, lut)

# Recombine and convert back to BGR
hsv_enhanced = cv2.merge([h, s_enhanced, v])
img_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

cv2.imwrite("results/q4_original.png", img)
cv2.imwrite("results/q4_vibrance.png", img_enhanced)

# Plot transformation curve (the LUT)
plt.figure()
plt.plot(np.arange(256), lut, label=f"a={a}, sigma={sigma}")
plt.title("Vibrance Transformation Curve")
plt.xlabel("Input Saturation (x)")
plt.ylabel("Output Saturation f(x)")
plt.ylim(0, 255)
plt.legend()
plt.grid(True)
plt.savefig("results/q4_transform_curve.png")

print(f"✅ Q4 done. Vibrance enhanced with a={a}. Check 'results' folder.")
