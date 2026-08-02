"""
compute_winter_peak_and_sweep.py
שלב 1: חישוב winter_peak.tif לכל עונה×טייל (ממוצע 3 תאריכי השיא, דצ-מרץ)
שלב 2: threshold sweep עם התנאי הכפול: delta_fill >= X  AND  winter_peak > 0
"""

import csv
import time
import contextlib
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
import rasterio.mask
import geopandas as gpd
import shapely.geometry
import pandas as pd
from rasterio.enums import Resampling
from tqdm import tqdm

MNDWI_DIR     = Path("/Volumes/ElementsData/menasim_volonteer/mndwi")
DETECTION_DIR = Path("/Volumes/ElementsData/menasim_volonteer/vernal_pool_detection")
POOLS_PATH    = Path("/Users/gilikurtser/Desktop/gili - personal/menasim_volonteer/data/reference/Vernal.pools.update.gpkg")
CSV_PATH      = Path(__file__).parent / "used_dates.csv"
OUT_CSV       = Path(__file__).parent / "threshold_sweep_peak.csv"

TILE_NAMES = {
    "36SXA": "36SXA_מרכז - שרון ומישור החוף",
    "36SYB": "36SYB_גליל וגולן",
    "36SXB": "36SXB_חיפה והכרמל",
    "36SYA": "36SYA_שומרון וירושלים",
    "37SBS": "37SBS_צפון-מזרח - גולן וחרמון",
    "36RXV": "36RXV_דרום-מערב - נגב מערבי",
}
WINTER_PEAK_MONTHS = {12, 1, 2, 3}
SPATIAL_CHUNK = 500
FILL_THRESHOLDS = [0.00, 0.03, 0.05, 0.07, 0.10, 0.15]


def read_chunk(src_list, row0, row1, w):
    ch = row1 - row0
    out = np.full((len(src_list), ch, w), np.nan, dtype=np.float32)
    win = rasterio.windows.Window(0, row0, w, ch)
    for i, src in enumerate(src_list):
        arr = src.read(1, window=win, out_shape=(ch, w),
                       resampling=Resampling.nearest).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        out[i] = arr
    return out


def save_tif(path, arr, profile):
    tmp = path.with_suffix(".tmp.tif")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr, 1)
    tmp.rename(path)


def compute_winter_peak(groups):
    print("=== שלב 1: חישוב winter_peak.tif ===")
    for (tile, season), date_paths in sorted(groups.items()):
        tile_folder = TILE_NAMES.get(tile, tile)
        out_path = DETECTION_DIR / season / tile_folder / "winter_peak.tif"
        if out_path.exists():
            continue

        peak_paths = [p for d, p in sorted(date_paths.items())
                      if int(d[4:6]) in WINTER_PEAK_MONTHS]
        if not peak_paths:
            continue

        with rasterio.open(peak_paths[0]) as ref:
            h, w = ref.height, ref.width
            profile = ref.profile.copy()
        profile.update(driver="GTiff", dtype="float32", count=1,
                       nodata=np.nan, compress="deflate", predictor=3,
                       tiled=True, blockxsize=512, blockysize=512)

        winter_peak = np.full((h, w), np.nan, dtype=np.float32)

        with contextlib.ExitStack() as stack:
            srcs = [stack.enter_context(rasterio.open(p)) for p in peak_paths]
            for row0 in range(0, h, SPATIAL_CHUNK):
                row1 = min(row0 + SPATIAL_CHUNK, h)
                chunk = read_chunk(srcs, row0, row1, w)
                sorted_desc = np.sort(chunk, axis=0)[::-1]
                top3 = sorted_desc[:min(3, chunk.shape[0])]
                with np.errstate(all="ignore"):
                    winter_peak[row0:row1] = np.nanmean(top3, axis=0).astype(np.float32)

        save_tif(out_path, winter_peak, profile)
        print(f"  {tile}/{season}  (>0: {int((winter_peak > 0).sum()):,} פיקסלים)")


def sweep_with_peak(groups):
    print("\n=== שלב 2: threshold sweep עם winter_peak > 0 ===")
    pools = gpd.read_file(POOLS_PATH)
    n_pools = len(pools)

    # Collect per tile: list of (delta_fill_path, winter_peak_path)
    by_tile = defaultdict(list)
    for (tile, season) in groups:
        tile_folder = TILE_NAMES.get(tile, tile)
        df_path = DETECTION_DIR / season / tile_folder / "delta_fill.tif"
        wp_path = DETECTION_DIR / season / tile_folder / "winter_peak.tif"
        if df_path.exists() and wp_path.exists():
            by_tile[tile_folder].append((df_path, wp_path))

    # Per pool: collect (delta_fill_median, winter_peak_median) per season
    pool_data = defaultdict(lambda: defaultdict(lambda: {"fill": [], "peak": []}))

    for tile_folder, season_paths in sorted(by_tile.items()):
        print(f"\n{tile_folder}")
        with rasterio.open(season_paths[0][0]) as ref:
            tif_crs = ref.crs
            tb = ref.bounds
            tile_box = shapely.geometry.box(tb.left, tb.bottom, tb.right, tb.top)

        pools_reproj = pools.to_crs(tif_crs)
        pools_in_tile = pools_reproj[pools_reproj.intersects(tile_box)].copy()
        print(f"  {len(pools_in_tile)} בריכות")

        for df_path, wp_path in tqdm(season_paths, desc="  עונות", leave=False):
            with rasterio.open(df_path) as src_fill, rasterio.open(wp_path) as src_peak:
                for idx, pool in pools_in_tile.iterrows():
                    try:
                        fill_arr, _ = rasterio.mask.mask(src_fill, [pool.geometry],
                                                          crop=True, nodata=np.nan)
                        peak_arr, _ = rasterio.mask.mask(src_peak, [pool.geometry],
                                                          crop=True, nodata=np.nan)
                        fill_vals = fill_arr[0].astype(np.float32).ravel()
                        peak_vals = peak_arr[0].astype(np.float32).ravel()
                        valid = ~np.isnan(fill_vals) & ~np.isnan(peak_vals)
                        if valid.sum() == 0:
                            pool_data[idx][tile_folder]["fill"].append(np.nan)
                            pool_data[idx][tile_folder]["peak"].append(np.nan)
                        else:
                            pool_data[idx][tile_folder]["fill"].append(
                                float(np.nanmedian(fill_vals[valid])))
                            pool_data[idx][tile_folder]["peak"].append(
                                float(np.nanmedian(peak_vals[valid])))
                    except Exception:
                        pool_data[idx][tile_folder]["fill"].append(np.nan)
                        pool_data[idx][tile_folder]["peak"].append(np.nan)

    print("\n--- תוצאות ---")
    print(f"{'Δ_fill ≥':>10} | {'peak>0':>6} | {'≥1':>7} | {'≥2':>7} | {'≥3':>7}")
    print("-" * 50)
    rows = []
    for fill_thr, use_peak in product(FILL_THRESHOLDS, [False, True]):
        counts = [0, 0, 0]
        for pool_idx in range(n_pools):
            best = 0
            for tile_folder, arrays in pool_data[pool_idx].items():
                fill_arr = np.array(arrays["fill"])
                peak_arr = np.array(arrays["peak"])
                ok = fill_arr >= fill_thr
                if use_peak:
                    ok &= peak_arr > 0
                best = max(best, int(np.sum(ok)))
            for i, thr in enumerate([1, 2, 3]):
                if best >= thr:
                    counts[i] += 1

        pct = [f"{c/n_pools*100:.1f}%" for c in counts]
        peak_str = "כן" if use_peak else "לא"
        print(f"  ≥{fill_thr:.2f}     |  {peak_str:>4}  | {pct[0]:>7} | {pct[1]:>7} | {pct[2]:>7}")
        rows.append({"delta_fill_thr": fill_thr, "peak_gt0": use_peak,
                     "pct_ge1": round(counts[0]/n_pools*100,1),
                     "pct_ge2": round(counts[1]/n_pools*100,1),
                     "pct_ge3": round(counts[2]/n_pools*100,1)})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nנשמר: {OUT_CSV}")


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(dict)
    for row in rows:
        tile_folder = TILE_NAMES.get(row["tile"], row["tile"])
        path = MNDWI_DIR / row["season"] / tile_folder / row["date"] / "MNDWI.tif"
        if path.exists():
            groups[(row["tile"], row["season"])][row["date"]] = path

    compute_winter_peak(groups)
    sweep_with_peak(groups)


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except (OSError, rasterio.errors.RasterioIOError) as e:
            print(f"\n⚠️ שגיאת כונן: {e}")
            time.sleep(60)
