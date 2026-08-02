"""
sweep_seasonal.py
Δ_seasonal = winter_peak − august_MNDWI  per pixel per season.
Sweeps thresholds against known vernal pool polygons.
No files written — everything in memory.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
import shapely.geometry
import pandas as pd
from tqdm import tqdm

DETECTION_DIR = Path("/Volumes/ElementsData/menasim_volonteer/vernal_pool_detection")
MNDWI_DIR     = Path("/Volumes/ElementsData/menasim_volonteer/mndwi")
POOLS_PATH    = Path("/Users/gilikurtser/Desktop/gili - personal/menasim_volonteer/data/reference/Vernal.pools.update.gpkg")
OUT_CSV       = Path(__file__).parent / "sweep_seasonal_results.csv"

TILE_NAMES = {
    "36SXA": "36SXA_מרכז - שרון ומישור החוף",
    "36SYB": "36SYB_גליל וגולן",
    "36SXB": "36SXB_חיפה והכרמל",
    "36SYA": "36SYA_שומרון וירושלים",
    "37SBS": "37SBS_צפון-מזרח - גולן וחרמון",
    "36RXV": "36RXV_דרום-מערב - נגב מערבי",
}
WINTERS = ["winter_2021_2022","winter_2022_2023","winter_2023_2024",
           "winter_2024_2025","winter_2025_2026"]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def get_summer_path(tile_folder, season):
    year = int(season.split("_")[1])
    paths = list((MNDWI_DIR / f"summer_{year}" / tile_folder).glob("*/MNDWI.tif"))
    return paths[0] if paths else None


def main():
    pools = gpd.read_file(POOLS_PATH)
    n_pools = len(pools)
    print(f"Loaded {n_pools} pools")

    # pool_data[pool_idx][tile_folder] → list of delta_seasonal medians (one per season)
    pool_data = defaultdict(lambda: defaultdict(list))

    for tile_code, tile_folder in TILE_NAMES.items():
        print(f"\n{tile_folder}")

        # get tile CRS and bounds from first available winter_peak
        ref_path = None
        for season in WINTERS:
            p = DETECTION_DIR / season / tile_folder / "winter_peak.tif"
            if p.exists():
                ref_path = p
                break
        if not ref_path:
            print("  no winter_peak found, skipping")
            continue

        with rasterio.open(ref_path) as src:
            tif_crs = src.crs
            tb = src.bounds
            tile_box = shapely.geometry.box(tb.left, tb.bottom, tb.right, tb.top)

        pools_reproj = pools.to_crs(tif_crs)
        pools_in_tile = pools_reproj[pools_reproj.intersects(tile_box)].copy()
        print(f"  {len(pools_in_tile)} pools in tile")

        for season in tqdm(WINTERS, desc="  seasons", leave=False):
            wp_path  = DETECTION_DIR / season / tile_folder / "winter_peak.tif"
            aug_path = get_summer_path(tile_folder, season)

            if not wp_path.exists() or aug_path is None:
                continue

            with rasterio.open(wp_path) as wp_src, rasterio.open(aug_path) as aug_src:
                for idx, pool in pools_in_tile.iterrows():
                    try:
                        wp_arr,  _ = rasterio.mask.mask(wp_src,  [pool.geometry], crop=True, nodata=np.nan)
                        aug_arr, _ = rasterio.mask.mask(aug_src, [pool.geometry], crop=True, nodata=np.nan)
                        wp_vals  = wp_arr[0].astype(np.float32).ravel()
                        aug_vals = aug_arr[0].astype(np.float32).ravel()
                        # grids should be identical (same tile, same 10m res); if not, align by min size
                        n = min(len(wp_vals), len(aug_vals))
                        wp_vals, aug_vals = wp_vals[:n], aug_vals[:n]
                        valid = ~np.isnan(wp_vals) & ~np.isnan(aug_vals)
                        if valid.sum() == 0:
                            pool_data[idx][tile_folder].append(np.nan)
                        else:
                            delta = float(np.nanmedian(wp_vals[valid] - aug_vals[valid]))
                            pool_data[idx][tile_folder].append(delta)
                    except Exception:
                        pool_data[idx][tile_folder].append(np.nan)

    print("\n=== Threshold sweep — Δ_seasonal only ===")
    print(f"{'threshold':>10} | {'≥1':>7} | {'≥2':>7} | {'≥3':>7} | {'≥4':>7}")
    print("-" * 50)

    rows = []
    for thr in THRESHOLDS:
        counts = [0, 0, 0, 0]
        for pool_idx in range(n_pools):
            best = 0
            for tile_folder, deltas in pool_data[pool_idx].items():
                seasons_ok = sum(1 for d in deltas if not np.isnan(d) and d >= thr)
                best = max(best, seasons_ok)
            for i, min_seasons in enumerate([1, 2, 3, 4]):
                if best >= min_seasons:
                    counts[i] += 1

        pcts = [f"{c/n_pools*100:.1f}%" for c in counts]
        print(f"  ≥{thr:.2f}     | {pcts[0]:>7} | {pcts[1]:>7} | {pcts[2]:>7} | {pcts[3]:>7}")
        rows.append({"threshold": thr,
                     "pct_ge1": round(counts[0]/n_pools*100,1),
                     "pct_ge2": round(counts[1]/n_pools*100,1),
                     "pct_ge3": round(counts[2]/n_pools*100,1),
                     "pct_ge4": round(counts[3]/n_pools*100,1)})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
