import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img_path = r"C:\Users\Anurag\Downloads\ch2_ohr_ncp_20211228T2209123959_d_img_d18\data\calibrated\20211228\ch2_ohr_ncp_20211228T2209123959_d_img_d18.img"

n_lines = 79796
n_samples = 12000

col_min, col_max = 11500, 12000
row_min, row_max = 22500, 24700

n_rows_crop = row_max - row_min
n_cols_crop = col_max - col_min

with open(img_path, "rb") as f:
    data = np.zeros((n_rows_crop, n_cols_crop), dtype=np.uint8)
    for i, row in enumerate(range(row_min, row_max)):
        f.seek(row * n_samples + col_min)
        data[i, :] = np.frombuffer(f.read(n_cols_crop), dtype=np.uint8)

print("Cropped OHRC shape:", data.shape)
print("Min/Max pixel value:", data.min(), data.max())

plt.figure(figsize=(8, 10))
plt.imshow(data, cmap='gray')
plt.title("OHRC crop - matches ice-cluster region")
plt.savefig("ohrc_ice_region_crop.png", dpi=150)
print("Saved ohrc_ice_region_crop.png")