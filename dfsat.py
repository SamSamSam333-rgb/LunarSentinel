import rasterio
import numpy as np
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import pyproj
import math

# ---------------- FILE PATHS ----------------
cpr_path = r"C:\Users\Anurag\Downloads\ch2_sar_ndxl_20250630mpcpspwest_d_fp_xxx\data\derived\20250630\ch2_sar_ndxl_20250630mpcpspwest_d_cpr_xx_fp_xx_xxx.tif"
srd_path = r"C:\Users\Anurag\Downloads\ch2_sar_ndxl_20250630mpcpspwest_d_fp_xxx\data\derived\20250630\ch2_sar_ndxl_20250630mpcpspwest_d_srd_xx_fp_xx_xxx.tif"
trt_path = r"C:\Users\Anurag\Downloads\ch2_sar_ndxl_20250630mpcpspwest_d_fp_xxx\data\derived\20250630\ch2_sar_ndxl_20250630mpcpspwest_d_trt_xx_fp_xx_xxx.tif"

CACHE_FILE = "cached_arrays.pkl"
MODEL_CACHE = "cached_model.pkl"
HALF_SIZE_M = 40000

def crop_around_pole(filepath, half_size_m=HALF_SIZE_M):
    with rasterio.open(filepath) as src:
        transform = src.transform
        row, col = rasterio.transform.rowcol(transform, 0, 0)
        px_size = abs(transform[0])
        win_px = int(half_size_m / px_size)
        window = Window(col - win_px, row - win_px, win_px * 2, win_px * 2)
        data = src.read(1, window=window)
    return data

if os.path.exists(CACHE_FILE):
    print("Loading cached crop data...")
    with open(CACHE_FILE, "rb") as f:
        cpr, srd, trt = pickle.load(f)
else:
    print("Cropping rasters...")
    cpr = crop_around_pole(cpr_path)
    srd = crop_around_pole(srd_path)
    trt = crop_around_pole(trt_path)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump((cpr, srd, trt), f)

valid_mask_full = ~np.isnan(cpr) & ~np.isnan(srd) & ~np.isnan(trt)
ice_label = ((cpr > 1.0) & (srd < 0.3)).astype(int)

X = np.stack([cpr.flatten(), srd.flatten(), trt.flatten()], axis=1)
y = ice_label.flatten()
mask_flat = valid_mask_full.flatten()
X_clean, y_clean = X[mask_flat], y[mask_flat]

if os.path.exists(MODEL_CACHE):
    print("Loading cached model...")
    with open(MODEL_CACHE, "rb") as f:
        clf = pickle.load(f)
else:
    print("Training model...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
    clf.fit(X_clean, y_clean)
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(clf, f)

proba_clean = clf.predict_proba(X_clean)[:, 1]
prob_map = np.full(y.shape, np.nan, dtype=np.float32)
prob_map[mask_flat] = proba_clean
prob_map = prob_map.reshape(cpr.shape)

labeled, num_clusters = ndimage.label(prob_map > 0.5)
cluster_sizes = []
for i in range(1, num_clusters + 1):
    cr, cc = np.where(labeled == i)
    cluster_sizes.append((len(cr), cr.mean(), cc.mean(), i))
cluster_sizes.sort(reverse=True)

# ---------------- COORDINATE CONVERSION ----------------
with rasterio.open(cpr_path) as src:
    full_transform = src.transform
    row0, col0 = rasterio.transform.rowcol(full_transform, 0, 0)
    px_size = abs(full_transform[0])
    win_px = int(HALF_SIZE_M / px_size)
    row_off, col_off = row0 - win_px, col0 - win_px

moon_proj = pyproj.CRS.from_wkt(
    'PROJCS["Moon_2000_South_Pole_Stereographic",GEOGCS["GCS_Moon_2000",'
    'DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],'
    'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]'
)
moon_geographic = pyproj.CRS.from_wkt(
    'GEOGCS["GCS_Moon_2000",DATUM["D_Moon_2000",'
    'SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)

transformer = pyproj.Transformer.from_crs(moon_proj, moon_geographic, always_xy=True)

def pixel_to_latlon(local_row, local_col):
    global_row = row_off + local_row
    global_col = col_off + local_col
    x, y_ = rasterio.transform.xy(full_transform, global_row, global_col)
    lon, lat = transformer.transform(x, y_)
    return lat, lon

# ---------------- SAVE RESULTS (WITH PIXEL COORDS FOR DASHBOARD) ----------------
print("\n=== TOP CLUSTER COORDINATES ===")
results = []
for size, row_c, col_c, cluster_id in cluster_sizes[:10]:
    lat, lon = pixel_to_latlon(row_c, col_c)
    print(f"Cluster {cluster_id}: {size}px, lat={lat:.4f}, lon={lon:.4f}, row={row_c:.0f}, col={col_c:.0f}")
    results.append({
        "cluster_id": cluster_id,
        "pixels": size,
        "lat": lat,
        "lon": lon,
        "row": float(row_c),
        "col": float(col_c)
    })

with open("ice_clusters.pkl", "wb") as f:
    pickle.dump(results, f)
print("Saved ice_clusters.pkl (now includes pixel row/col).")

with open("prob_map_shape.pkl", "wb") as f:
    pickle.dump(prob_map.shape, f)
print("Saved prob_map_shape.pkl:", prob_map.shape)

# ---------------- CLEAN IMAGE FOR DASHBOARD (no axes/title/colorbar) ----------------
os.makedirs("static", exist_ok=True)

cmap_clean = matplotlib.colormaps['inferno'].copy()
cmap_clean.set_bad(color='#1a1a1a')

fig, ax = plt.subplots(figsize=(prob_map.shape[1]/100, prob_map.shape[0]/100), dpi=100)
ax.imshow(prob_map, cmap=cmap_clean, vmin=0, vmax=1)
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("static/ice_map_clean.png", dpi=100, pad_inches=0)
plt.close(fig)
print("Saved static/ice_map_clean.png")

# ---------------- FEATURE IMPORTANCE + SCATTER DATA FOR DASHBOARD ----------------
importances = clf.feature_importances_
feature_names = ["CPR", "SRD", "T-Ratio"]
feature_importance_data = list(zip(feature_names, importances.tolist()))
print("\n=== FEATURE IMPORTANCE ===")
for name, val in feature_importance_data:
    print(f"{name}: {val:.4f}")

with open("feature_importance.pkl", "wb") as f:
    pickle.dump(feature_importance_data, f)

# Sample ~2500 valid pixels for a CPR vs SRD scatter plot (downsample for speed/size)
np.random.seed(42)
valid_idx = np.where(mask_flat)[0]
sample_idx = np.random.choice(valid_idx, size=min(2500, len(valid_idx)), replace=False)

scatter_data = []
for idx in sample_idx:
    scatter_data.append({
        "cpr": float(X[idx, 0]),
        "srd": float(X[idx, 1]),
        "proba": float(clf.predict_proba(X[idx:idx+1])[0, 1])
    })

with open("scatter_sample.pkl", "wb") as f:
    pickle.dump(scatter_data, f)
print(f"Saved scatter_sample.pkl with {len(scatter_data)} points")

# ============================================================
# NEW — ICE MASS & VOLUME ESTIMATE
# ============================================================
PIXEL_SIZE_M = 24.0
pixel_area_m2 = PIXEL_SIZE_M ** 2

total_ice_pixels = sum(c['pixels'] for c in results)
total_area_m2 = total_ice_pixels * pixel_area_m2
total_area_km2 = total_area_m2 / 1e6

REGOLITH_DENSITY_KG_M3 = 1500
ASSUMED_DEPTH_M = 1.0
ICE_CONCENTRATION_LOW = 0.01
ICE_CONCENTRATION_MID = 0.056   # LCROSS Cabeus measured value
ICE_CONCENTRATION_HIGH = 0.10

def estimate_mass(concentration):
    volume_m3 = total_area_m2 * ASSUMED_DEPTH_M
    mass_kg = volume_m3 * REGOLITH_DENSITY_KG_M3 * concentration
    return mass_kg / 1000  # tonnes

print("\n=== ICE MASS & VOLUME ESTIMATE (order-of-magnitude) ===")
print(f"Total ice-candidate area: {total_area_km2:.5f} km^2 ({total_area_m2:.0f} m^2)")
print(f"Assumed depth: {ASSUMED_DEPTH_M} m | Regolith density: {REGOLITH_DENSITY_KG_M3} kg/m^3")
print(f"Low estimate  (1% conc.):   {estimate_mass(ICE_CONCENTRATION_LOW):,.1f} tonnes")
print(f"Mid estimate  (5.6% conc. - LCROSS Cabeus value): {estimate_mass(ICE_CONCENTRATION_MID):,.1f} tonnes")
print(f"High estimate (10% conc.):  {estimate_mass(ICE_CONCENTRATION_HIGH):,.1f} tonnes")

mass_estimate_data = {
    "total_area_km2": total_area_km2,
    "low_tonnes": estimate_mass(ICE_CONCENTRATION_LOW),
    "mid_tonnes": estimate_mass(ICE_CONCENTRATION_MID),
    "high_tonnes": estimate_mass(ICE_CONCENTRATION_HIGH),
    "assumptions": f"{ASSUMED_DEPTH_M}m depth, {REGOLITH_DENSITY_KG_M3} kg/m3 density"
}
with open("mass_estimate.pkl", "wb") as f:
    pickle.dump(mass_estimate_data, f)
print("Saved mass_estimate.pkl")

# ============================================================
# NEW — MULTI-CRATER ICE ATLAS (crater attribution)
# ============================================================
MOON_RADIUS_KM = 1737.4

KNOWN_CRATERS = {
    "Shackleton":  (-89.9, 0.0),
    "Shoemaker":   (-88.1, 44.9),
    "Faustini":    (-87.3, 77.0),
    "de Gerlache": (-88.5, -87.1),
    "Sverdrup":    (-88.5, -152.0),
    "Cabeus":      (-84.9, -35.5),
}

def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * MOON_RADIUS_KM * math.asin(math.sqrt(a))

print("\n=== MULTI-CRATER ICE ATLAS ===")
atlas = []
for c in results:
    distances = {name: haversine_km(c['lat'], c['lon'], clat, clon)
                 for name, (clat, clon) in KNOWN_CRATERS.items()}
    nearest = min(distances, key=distances.get)
    atlas.append({**c, "nearest_crater": nearest, "distance_km": distances[nearest]})
    print(f"Cluster {c['cluster_id']}: {c['pixels']}px \u2192 nearest crater: {nearest} ({distances[nearest]:.1f} km away)")

with open("crater_atlas.pkl", "wb") as f:
    pickle.dump(atlas, f)
print("Saved crater_atlas.pkl")