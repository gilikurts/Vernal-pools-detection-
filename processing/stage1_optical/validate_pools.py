"""
validate_pools.py – Sensitivity analysis of vernal pool detection.

For each known pool polygon:
  - Sample persistence.tif (0–5, number of seasons detected)
  - Compute fraction of pixels with persistence >= P (pixel threshold)
  - Pool is "detected" if that fraction >= C (coverage threshold)

Reports Recall = detected / total for all (P, C) combinations.
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
import shapely.geometry
import pandas as pd
from tqdm import tqdm

DETECTION_DIR = Path("/Volumes/ElementsData/menasim_volonteer/vernal_pool_detection")
POOLS_PATH    = Path("/Users/gilikurtser/Desktop/gili - personal/menasim_volonteer/data/reference/Vernal.pools.update.gpkg")
OUT_CSV       = Path(__file__).parent / "validation_results.csv"

TILE_NAMES = {
    "36SXA": "36SXA_מרכז - שרון ומישור החוף",
    "36SYB": "36SYB_גליל וגולן",
    "36SXB": "36SXB_חיפה והכרמל",
    "36SYA": "36SYA_שומרון וירושלים",
    "37SBS": "37SBS_צפון-מזרח - גולן וחרמון",
    "36RXV": "36RXV_דרום-מערב - נגב מערבי",
}

PIXEL_THRESHOLDS    = [1, 2, 3, 4, 5]   # persistence >= P
COVERAGE_THRESHOLDS = [0.20, 0.40, 0.60, 0.80]


def main():
    pools = gpd.read_file(POOLS_PATH)
    n_pools = len(pools)
    print(f"Loaded {n_pools} pools")

    # pool_coverage[pool_idx] = fraction of pixels with persistence >= P, per P
    # store raw persistence values per pool (list of pixel values)
    pool_persistence: dict[int, np.ndarray] = {}

    for tile_folder in TILE_NAMES.values():
        pers_path = DETECTION_DIR / "persistence" / tile_folder / "persistence.tif"
        if not pers_path.exists():
            print(f"  missing: {pers_path}")
            continue

        with rasterio.open(pers_path) as src:
            tif_crs = src.crs
            tb      = src.bounds
            tile_box = shapely.geometry.box(tb.left, tb.bottom, tb.right, tb.top)

        pools_r   = pools.to_crs(tif_crs)
        pools_in  = pools_r[pools_r.intersects(tile_box)].copy()
        print(f"  {tile_folder}: {len(pools_in)} pools")

        with rasterio.open(pers_path) as src:
            for idx, pool in tqdm(pools_in.iterrows(), total=len(pools_in),
                                   desc="    masking", leave=False):
                try:
                    arr, _ = rasterio.mask.mask(src, [pool.geometry],
                                                 crop=True, nodata=255)
                    vals = arr[0].ravel()
                    vals = vals[vals != 255].astype(np.uint8)
                    if len(vals) == 0:
                        continue
                    # merge with any values already from another tile
                    if idx in pool_persistence:
                        pool_persistence[idx] = np.concatenate(
                            [pool_persistence[idx], vals])
                    else:
                        pool_persistence[idx] = vals
                except Exception:
                    pass

    print(f"\nPools with pixel data: {len(pool_persistence)}/{n_pools}")

    # sensitivity analysis
    rows = []
    print("\n=== Sensitivity Analysis: Recall ===")
    print(f"{'':12}", end="")
    for C in COVERAGE_THRESHOLDS:
        print(f"  cov≥{int(C*100)}%", end="")
    print()
    print("-" * (12 + 10 * len(COVERAGE_THRESHOLDS)))

    for P in PIXEL_THRESHOLDS:
        label = f"  pers={'=5' if P==5 else f'≥{P}'}  "
        print(label, end="")
        for C in COVERAGE_THRESHOLDS:
            detected = 0
            for pool_idx in range(n_pools):
                if pool_idx not in pool_persistence:
                    continue
                vals = pool_persistence[pool_idx]
                if P == 5:
                    frac = (vals == 5).sum() / len(vals)
                else:
                    frac = (vals >= P).sum() / len(vals)
                if frac >= C:
                    detected += 1
            recall = detected / n_pools * 100
            print(f"  {recall:6.1f}%  ", end="")
            rows.append({"pixel_thresh": P, "coverage_thresh": C,
                         "detected": detected, "recall_pct": round(recall, 1)})
        print()

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
