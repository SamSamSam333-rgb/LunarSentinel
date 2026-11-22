import shutil
import os

os.makedirs("static", exist_ok=True)

files_to_copy = {
    "ohrc_ice_region_crop.png": "static/ohrc_psr_floor.png",
    "ohrc_terrain_crop.png": "static/ohrc_terrain.png",
    "rover_path.png": "static/rover_path.png",
}

for src, dst in files_to_copy.items():
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"MISSING: {src} not found in current folder")