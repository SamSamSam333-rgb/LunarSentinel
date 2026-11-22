# Lunar Ice Sentinel

Detecting water-ice candidate zones near the lunar south pole (Shackleton crater) from ISRO Chandrayaan-2 data.
Capstone project for HIMSHIKHAR 2026 (IIT Mandi, AI & ML track). Team Leader: Anurag Kasi.

## What it does
- **DFSAR radar:** reads Chandrayaan-2 DFSAR polarimetric products (CPR, SRD, T-Ratio) and builds physics-based weak labels for ice-likely pixels.
- **Model:** trains a Random Forest on those labels to produce an ice-probability map and ice clusters.
- **OHRC terrain:** reads raw OHRC PDS4 `.img` data with NumPy memmap (no GDAL) to build a high-resolution terrain crop.
- **Rover path:** plans an A* rover route over the OHRC terrain to the ice candidates.
- **Dashboard:** Flask + Folium web app showing the maps and route.

## Files
| File | Purpose |
|---|---|
| `dfsat.py` | DFSAR processing, weak labels, Random Forest, ice-probability map |
| `ohrc.py`, `ohrc_quicklook.py`, `ohrc_terrain_crop.py` | OHRC image reading and terrain crop |
| `rover_path.py` | A* rover path planning |
| `app.py` | Dashboard |
| `copy_images.py` | Copies output images into `static/` |
| `*.pkl`, `*.npy` | Cached model and intermediate results |
| `Lunar_Ice_Sentinel.pdf` / `.pptx` | Presentation |

## Data
Raw Chandrayaan-2 data is not included (download from ISRO PRADAN):
- DFSAR: `ch2_sar_ndxl_20250630mpcpspwest_d_fp_xxx`
- OHRC: `ch2_ohr_ncp_20211228T2209123959_d_img_d18`

Update the input paths at the top of `dfsat.py` and the `ohrc*.py` scripts to where you saved them.
`cached_arrays.pkl` is not committed (117 MB); `dfsat.py` recreates it on first run.

## Run
```bash
pip install numpy pandas scipy scikit-learn rasterio pyproj matplotlib folium flask flask-cors requests
python dfsat.py
python ohrc_terrain_crop.py
python rover_path.py
python app.py
```
