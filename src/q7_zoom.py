import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# Zoom function
def zoom(img, s, method="nearest"):
    h, w = img.shape[:2]
    new_h, new_w = int(h * s), int(w * s)
    out = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype) if img.ndim == 3 else np.zeros((new_h, new_w), dtype=img.dtype)

    for y in range(new_h):
        for x in range(new_w):
            src_x = x / s
            src_y = y / s

            if method == "nearest":
                nx = int(round(src_x))
                ny = int(round(src_y))
                nx = min(nx, w - 1)
                ny = min(ny, h - 1)
                out[y, x] = img[ny, nx]

            elif method == "bilinear":
                x0 = int(np.floor(src_x))
                x1 = min(x0 + 1, w - 1)
                y0 = int(np.floor(src_y))
                y1 = min(y0 + 1, h - 1)
                dx = src_x - x0
                dy = src_y - y0

                if img.ndim == 2:  
                    val = (img[y0, x0] * (1 - dx) * (1 - dy) +
                           img[y0, x1] * dx * (1 - dy) +
                           img[y1, x0] * (1 - dx) * dy +
                           img[y1, x1] * dx * dy)
                else: 
                    val = (img[y0, x0, :] * (1 - dx) * (1 - dy) +
                           img[y0, x1, :] * dx * (1 - dy) +
                           img[y1, x0, :] * (1 - dx) * dy +
                           img[y1, x1, :] * dx * dy)
                out[y, x] = np.clip(val, 0, 255)

    return out.astype(img.dtype)

# Normalized SSD
def normalized_ssd(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    ssd = np.sum((img1 - img2) ** 2)
    norm_ssd = ssd / img1.size
    return norm_ssd

# Process image pairs
pairs = [
    ("images/im01.png", "images/im01small.png"),
    ("images/im02.png", "images/im02small.png")
]

scale = 4
for i, (large_path, small_path) in enumerate(pairs, start=1):
    large = cv2.imread(large_path)
    small = cv2.imread(small_path)

    if small is None or large is None:
        raise FileNotFoundError(f"Check paths for pair {i}: {large_path}, {small_path}")

    # zoom small
    zoom_nearest = zoom(small, scale, method="nearest")
    zoom_bilinear = zoom(small, scale, method="bilinear")

    cv2.imwrite(f"results/q7_pair{i}_nearest.png", zoom_nearest)
    cv2.imwrite(f"results/q7_pair{i}_bilinear.png", zoom_bilinear)

    ssd_nearest = normalized_ssd(zoom_nearest, large)
    ssd_bilinear = normalized_ssd(zoom_bilinear, large)

    print(f"=== Pair {i} ===")
    print(f"Normalized SSD (nearest): {ssd_nearest:.4f}")
    print(f"Normalized SSD (bilinear): {ssd_bilinear:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(large, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original Large")
    axes[1].imshow(cv2.cvtColor(zoom_nearest, cv2.COLOR_BGR2RGB)); axes[1].set_title("Nearest x4")
    axes[2].imshow(cv2.cvtColor(zoom_bilinear, cv2.COLOR_BGR2RGB)); axes[2].set_title("Bilinear x4")
    for ax in axes: ax.axis('off')
    plt.savefig(f"results/q7_pair{i}_comparison.png")
    plt.close()

print("✅ Q7 completed. Check 'results' folder for outputs.")
