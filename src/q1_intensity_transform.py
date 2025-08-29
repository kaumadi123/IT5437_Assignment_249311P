import cv2, numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/emma.jpg", cv2.IMREAD_GRAYSCALE)

# define mapping points as (x_in, x_out). Replace with points from Fig1a.
points = [(0,0), (60,30), (140,220), (255,255)]

# build LUT via linear interpolation
xs, ys = zip(*points)
lut = np.interp(np.arange(256), xs, ys).astype('uint8')

out = cv2.LUT(img, lut)

# save
cv2.imwrite("results/q1_out.png", out)

# plot mapping
plt.figure()
plt.plot(np.arange(256), lut); plt.title("Intensity mapping"); plt.xlabel("Input"); plt.ylabel("Output")
plt.savefig("results/q1_mapping.png")

# histograms
plt.figure(); plt.hist(img.ravel(), bins=256); plt.title("input hist")
plt.savefig("results/q1_hist_in.png")
plt.figure(); plt.hist(out.ravel(), bins=256); plt.title("output hist")
plt.savefig("results/q1_hist_out.png")
