"""
sweep_seasonal_v2.py
Reads pre-computed delta_seasonal.tif (from detect_vernal_pools.py) per tile×season.
Masks each known pool polygon, computes median Δ_seasonal.
Sweeps thresholds and reports recall against 433 known pools.
"""

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
POOLS_PATH    = Path("/Users/gilikurtser/Desktop/gili - personal/menasim_volonteer/data/reference/Vernal.pools.update.gpkg")
OUT_CSV       = Path(__file__).parent / "sweep_seasonal_v2_results.csv"

TILE_NAMES = {
    "36SXA": "36SXA_מרכז - שרון ומישור החוף",
    "36SYB": "36SYB_גליל וגולן",
    "36SXB": "36SXB_חיפה והכרמל",
    "36SYA": "36SYA_שומרון וירושלים",
    "37SBS": "37SBS_צפון-מזרח - גולן וחרמון",
    "36RXV": "36RXV_דרום-מערב - נגב מערבי",
}
WINTERS    = ["winter_2021_2022","winter_2022_2023","winter_2023_2024",
              "winter_2024_2025","winter_2025_2026"]
THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def main():
    pools = gpd.read_file(POOLS_PATH)
    n_pools = len(pools)
    print(f"Loaded {n_pools} pools")

    # pool_deltas[pool_idx] = list of (delta_seasonal median) across all tile×season combos
    pool_best_deltas = defaultdict(list)   # per pool: best median delta per season (across tiles)
    pool_coverage = defaultdict(int)       # how many seasons have valid data for this pool

    for tile_code, tile_folder in TILE_NAMES.items():
        print(f"\n{tile_folder}")

        # find first valid delta_seasonal to get CRS/bounds
        ref_ds_path = None
        for season in WINTERS:
            p = DETECTION_DIR / season / tile_folder / "delta_seasonal.tif"
            if p.exists():
                ref_ds_path = p
                break
        if not ref_ds_path:
            print("  no delta_seasonal found, skipping")
            continue

        with rasterio.open(ref_ds_path) as src:
            tif_crs = src.crs
            tb = src.bounds
            tile_box = shapely.geometry.box(tb.left, tb.bottom, tb.right, tb.top)

        pools_reproj = pools.to_crs(tif_crs)
        pools_in_tile = pools_reproj[pools_reproj.intersects(tile_box)].copy()
        print(f"  {len(pools_in_tile)} pools within delta_seasonal extent")

        for season in tqdm(WINTERS, desc="  seasons", leave=False):
            ds_path = DETECTION_DIR / season / tile_folder / "delta_seasonal.tif"
            if not ds_path.exists():
                continue

            with rasterio.open(ds_path) as ds_src:
                for idx, pool in pools_in_tile.iterrows():
                    try:
                        arr, _ = rasterio.mask.mask(ds_src, [pool.geometry],
                                                     crop=True, nodata=np.nan)
                        vals = arr[0].ravel().astype(np.float32)
                        valid = ~np.isnan(vals)
                        if valid.sum() == 0:
                            continue
                        delta = float(np.nanmedian(vals[valid]))
                        pool_best_deltas[idx].append((season, delta))
                    except Exception:
                        pass

    print(f"\nPools with any seasonal data: {len(pool_best_deltas)}/{n_pools}")

    # group per-pool deltas by season (take best tile if pool spans multiple tiles)
    # pool_best_deltas[idx] = [(season, delta), ...]
    # we need: per pool, per season → best delta across tiles (already done above since tiles are separate)
    # reshape: pool_seasonal[pool_idx][season] = max delta across tiles
    pool_seasonal = defaultdict(dict)
    for pool_idx, season_deltas in pool_best_deltas.items():
        for season, delta in season_deltas:
            if season not in pool_seasonal[pool_idx] or delta > pool_seasonal[pool_idx][season]:
                pool_seasonal[pool_idx][season] = delta

    print("\n=== Threshold sweep — Δ_seasonal only (using delta_seasonal.tif) ===")
    print(f"{'thr':>6} | {'≥1 seas':>8} | {'≥2 seas':>8} | {'≥3 seas':>8}")
    print("-" * 45)

    rows = []
    for thr in THRESHOLDS:
        c1 = c2 = c3 = 0
        for pool_idx in range(n_pools):
            seasons_above = sum(1 for d in pool_seasonal[pool_idx].values() if d >= thr)
            if seasons_above >= 1: c1 += 1
            if seasons_above >= 2: c2 += 1
            if seasons_above >= 3: c3 += 1

        print(f"  ≥{thr:.2f} | {c1:5} {c1/n_pools*100:5.1f}% | {c2:5} {c2/n_pools*100:5.1f}% | {c3:5} {c3/n_pools*100:5.1f}%")
        rows.append({"threshold": thr,
                     "n_ge1": c1, "pct_ge1": round(c1/n_pools*100,1),
                     "n_ge2": c2, "pct_ge2": round(c2/n_pools*100,1),
                     "n_ge3": c3, "pct_ge3": round(c3/n_pools*100,1)})

    # also print pools that NEVER pass threshold 0.10
    never_detected = [i for i in range(n_pools) if i not in pool_seasonal]
    print(f"\nPools with NO valid delta_seasonal at all: {len(never_detected)}")
    print("(these are outside all tile extents or fully NaN)")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
