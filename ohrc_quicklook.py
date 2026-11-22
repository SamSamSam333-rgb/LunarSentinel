import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img_path = r"C:\Users\Anurag\Downloads\ch2_ohr_ncp_20211228T2209123959_d_img_d18\data\calibrated\20211228\ch2_ohr_ncp_20211228T2209123959_d_img_d18.img"

n_lines = 79796
n_samples = 12000

print("Memory-mapping file...")
mm = np.memmap(img_path, dtype=np.uint8, mode='r', shape=(n_lines, n_samples))

print("Downsampling (this should be quick)...")
quicklook = np.array(mm[::40, ::20]).copy()  # every 40th row, every 20th col

print("Quicklook shape:", quicklook.shape)
print("Min/Max:", quicklook.min(), quicklook.max())

p2, p98 = np.percentile(quicklook, (2, 98))
stretched = np.clip((quicklook.astype(float) - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

plt.figure(figsize=(8, 14))
plt.imshow(stretched, cmap='gray')
plt.title("OHRC full-strip quicklook (contrast stretched)")
plt.savefig("ohrc_full_quicklook.png", dpi=150)
print("Saved ohrc_full_quicklook.png")

from scipy.ndimage import uniform_filter
smoothed = uniform_filter(quicklook.astype(float), size=15)
brightest_row, brightest_col = np.unravel_index(np.argmax(smoothed), smoothed.shape)
print(f"Brightest region approx at full-res row={brightest_row*40}, col={brightest_col*20}")

del mm  # release the memmap