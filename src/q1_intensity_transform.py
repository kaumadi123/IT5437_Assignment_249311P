import cv2, numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread("images/emma.jpg", cv2.IMREAD_GRAYSCALE)

# Control points (matching the first LUT logic)
points = [(0,0), (49,49), (50,100), (150,150), (255,255)]

xs, ys = zip(*points)
lut = np.interp(np.arange(256), xs, ys).astype('uint8')

# Apply LUT
out = cv2.LUT(img, lut)

# Save transformed image
cv2.imwrite("results/q1_out.png", out)

# Plot mapping
plt.figure()
plt.plot(np.arange(256), lut)
plt.title("Intensity mapping")
plt.xlabel("Input")
plt.ylabel("Output")
plt.savefig("results/q1_mapping.png")

# Histograms
plt.figure()
plt.hist(img.ravel(), bins=256)
plt.title("input hist")
plt.savefig("results/q1_hist_in.png")

plt.figure()
plt.hist(out.ravel(), bins=256)
plt.title("output hist")
plt.savefig("results/q1_hist_out.png")
