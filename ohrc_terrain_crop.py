import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img_path = r"C:\Users\Anurag\Downloads\ch2_ohr_ncp_20211228T2209123959_d_img_d18\data\calibrated\20211228\ch2_ohr_ncp_20211228T2209123959_d_img_d18.img"

n_lines = 79796
n_samples = 12000

row_center, col_center = 79560, 4440
half_h, half_w = 1000, 1500

row_start = max(0, row_center - half_h)
row_end = min(n_lines, row_center + half_h)
col_start = max(0, col_center - half_w)
col_end = min(n_samples, col_center + half_w)

print(f"Cropping rows {row_start}-{row_end}, cols {col_start}-{col_end}")

mm = np.memmap(img_path, dtype=np.uint8, mode='r', shape=(n_lines, n_samples))
terrain_crop = np.array(mm[row_start:row_end, col_start:col_end]).copy()
del mm

print("Terrain crop shape:", terrain_crop.shape)
print("Min/Max:", terrain_crop.min(), terrain_crop.max())

p2, p98 = np.percentile(terrain_crop, (2, 98))
stretched = np.clip((terrain_crop.astype(float) - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

plt.figure(figsize=(10, 8))
plt.imshow(stretched, cmap='gray')
plt.title("OHRC rim/terrain crop (for rover path cost map)")
plt.savefig("ohrc_terrain_crop.png", dpi=150)
print("Saved ohrc_terrain_crop.png")

np.save("ohrc_terrain_array.npy", terrain_crop)
print("Saved ohrc_terrain_array.npy for rover path use")