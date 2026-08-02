"""
overlap_analysis.py – Compute overlap between known vernal pools and persistence map.

For each known pool polygon, extract persistence values underneath and compute
the median — this is the "detection score" for that pool.

Output: overlap_results.csv  (one row per pool)

Usage:
    python overlap_analysis.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from tqdm import tqdm

POOLS_PATH     = Path("/Users/gilikurtser/Desktop/gili - personal/menasim_volonteer/data/reference/Vernal.pools.update.gpkg")
PERSISTENCE_DIR = Path("/Volumes/ElementsData/menasim_volonteer/vernal_pool_detection/persistence")
OUT_CSV        = Path(__file__).parent / "overlap_results.csv"


def main():
    pools = gpd.read_file(POOLS_PATH)
    print(f"נטענו {len(pools)} בריכות (CRS: {pools.crs})")

    persistence_files = sorted(PERSISTENCE_DIR.rglob("persistence.tif"))
    print(f"נמצאו {len(persistence_files)} קבצי persistence")

    results = []

    for tif_path in persistence_files:
        tile_folder = tif_path.parent.name
        print(f"\n{tile_folder}")

        with rasterio.open(tif_path) as src:
            tif_crs = src.crs

        # Reproject pools to match this tile's CRS
        pools_reproj = pools.to_crs(tif_crs)

        with rasterio.open(tif_path) as src:
            tile_bounds = src.bounds
            tile_box = gpd.GeoDataFrame(
                geometry=[gpd.GeoSeries.from_wkt([
                    f"POLYGON(({tile_bounds.left} {tile_bounds.bottom}, "
                    f"{tile_bounds.right} {tile_bounds.bottom}, "
                    f"{tile_bounds.right} {tile_bounds.top}, "
                    f"{tile_bounds.left} {tile_bounds.top}, "
                    f"{tile_bounds.left} {tile_bounds.bottom}))"
                ])[0]],
                crs=tif_crs,
            )

            # Only process pools that intersect this tile
            pools_in_tile = pools_reproj[pools_reproj.intersects(tile_box.geometry[0])].copy()
            print(f"  {len(pools_in_tile)} בריכות בטייל זה")

            for idx, pool in tqdm(pools_in_tile.iterrows(), total=len(pools_in_tile), leave=False):
                try:
                    masked, _ = rasterio.mask.mask(
                        src, [pool.geometry], crop=True, nodata=255
                    )
                    values = masked[0]
                    valid = values[values != 255]
                    if valid.size == 0:
                        continue

                    results.append({
                        "pool_name":       pool.get("Name", ""),
                        "pool_type":       pool.get("Type", ""),
                        "area_dunam":      pool.get("Area_dunam", np.nan),
                        "tile":            tile_folder,
                        "n_pixels":        int(valid.size),
                        "median_persistence": float(np.median(valid)),
                        "max_persistence": int(valid.max()),
                        "pct_ge3":         float((valid >= 3).sum() / valid.size * 100),
                    })
                except Exception as e:
                    tqdm.write(f"  שגיאה בבריכה {pool.get('Name', idx)}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nנשמר: {OUT_CSV}  ({len(df)} שורות)")

    # Summary
    print("\n=== סיכום ===")
    print(f"בריכות שנמצאו בלפחות טייל אחד: {df['pool_name'].nunique()}")
    print(f"\nחציון persistence לפי סף:")
    for threshold in [1, 2, 3, 4, 5]:
        n = (df["median_persistence"] >= threshold).sum()
        print(f"  חציון ≥ {threshold}: {n} בריכות ({n/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
