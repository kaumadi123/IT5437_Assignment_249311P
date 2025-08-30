import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# load the image
img_path = "images/spider.png"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# parameters
sigma = 70.0
a = 0.6 

# define vibrance function
def vibrance_transform(s_channel, a, sigma=70.0):
    x = s_channel.astype(np.float32)
    bump = a * 128.0 * np.exp(-((x - 128.0)**2) / (2 * sigma**2))
    out = np.clip(x + bump, 0, 255)
    return out.astype(np.uint8)

# apply transform to saturation channel
s_enhanced = vibrance_transform(s, a, sigma)

# recombine and convert back to BGR
hsv_enhanced = cv2.merge([h, s_enhanced, v])
img_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

# save results
cv2.imwrite("results/q4_original.png", img)
cv2.imwrite("results/q4_vibrance.png", img_enhanced)

# plot transformation curve
x_vals = np.arange(256)
bump_curve = a * 128.0 * np.exp(-((x_vals - 128.0)**2) / (2 * sigma**2))
y_vals = np.clip(x_vals + bump_curve, 0, 255)

plt.figure()
plt.plot(x_vals, y_vals, label=f"a={a}, sigma={sigma}")
plt.title("Vibrance Transformation Curve")
plt.xlabel("Input Saturation (x)")
plt.ylabel("Output Saturation f(x)")
plt.legend()
plt.savefig("results/q4_transform_curve.png")

print(f"✅ Q4 done. Vibrance enhanced with a={a}. Check 'results' folder.")
