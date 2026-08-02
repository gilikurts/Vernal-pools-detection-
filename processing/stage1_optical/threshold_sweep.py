"""
threshold_sweep.py – בדיקת ספים שונים של delta_fill ו-delta_recession
על גבי delta tifs קיימים, ללא כתיבה מחדש.

לכל שילוב ספים: ספירת עונות שהתנאי מתקיים לכל פיקסל ←
מיצוי תחת כל בריכה ידועה ← חישוב חציון ← סיכום.
"""

from itertools import product
from pathlib import Path
from collections import defaultdict

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
import pandas as pd
from tqdm import tqdm

DETECTION_DIR = Path("/Volumes/ElementsData/menasim_volonteer/vernal_pool_detection")
POOLS_PATH    = Path("/Users/gilikurtser/Desktop/gili - personal/menasim_volonteer/data/reference/Vernal.pools.update.gpkg")
OUT_CSV       = Path(__file__).parent / "threshold_sweep_results.csv"

TILE_NAMES = {
    "36SXA": "36SXA_מרכז - שרון ומישור החוף",
    "36SYB": "36SYB_גליל וגולן",
    "36SXB": "36SXB_חיפה והכרמל",
    "36SYA": "36SYA_שומרון וירושלים",
    "37SBS": "37SBS_צפון-מזרח - גולן וחרמון",
    "36RXV": "36RXV_דרום-מערב - נגב מערבי",
}

FILL_THRESHOLDS      = [0.00, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
RECESSION_THRESHOLDS = [0.00, 0.03, 0.05, 0.10]


def main():
    pools = gpd.read_file(POOLS_PATH)

    # Collect all (season, tile_folder) delta paths
    # Structure: by_tile[tile_folder] → list of (delta_fill_path, delta_recession_path)
    by_tile: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for season_dir in sorted(DETECTION_DIR.iterdir()):
        if not season_dir.is_dir() or season_dir.name == "persistence":
            continue
        for tile_dir in sorted(season_dir.iterdir()):
            if not tile_dir.is_dir():
                continue
            df_path = tile_dir / "delta_fill.tif"
            dr_path = tile_dir / "delta_recession.tif"
            if df_path.exists() and dr_path.exists():
                by_tile[tile_dir.name].append((df_path, dr_path))

    print(f"נמצאו {sum(len(v) for v in by_tile.values())} קבצי delta בסה\"כ ({len(by_tile)} טיילים)")

    # For each tile: per pool, collect (fill_values, recession_values) across seasons
    # fill_values shape: (n_seasons,) – median delta_fill under pool for each season
    # same for recession

    # Results: pool_id → tile → {'fill': array(n_seasons), 'recession': array(n_seasons)}
    pool_data: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: {"fill": [], "recession": []}))

    for tile_folder, season_paths in sorted(by_tile.items()):
        print(f"\n{tile_folder}  ({len(season_paths)} עונות)")

        with rasterio.open(season_paths[0][0]) as ref:
            tif_crs = ref.crs

        pools_reproj = pools.to_crs(tif_crs)

        with rasterio.open(season_paths[0][0]) as ref:
            tb = ref.bounds
            import shapely.geometry
            tile_box = shapely.geometry.box(tb.left, tb.bottom, tb.right, tb.top)

        pools_in_tile = pools_reproj[pools_reproj.intersects(tile_box)].copy()
        print(f"  {len(pools_in_tile)} בריכות בטייל")

        for df_path, dr_path in tqdm(season_paths, desc="  עונות", leave=False):
            with rasterio.open(df_path) as src_fill, rasterio.open(dr_path) as src_rec:
                for idx, pool in pools_in_tile.iterrows():
                    try:
                        fill_arr, _ = rasterio.mask.mask(src_fill, [pool.geometry], crop=True, nodata=np.nan)
                        rec_arr, _  = rasterio.mask.mask(src_rec,  [pool.geometry], crop=True, nodata=np.nan)

                        fill_vals = fill_arr[0].astype(np.float32).ravel()
                        rec_vals  = rec_arr[0].astype(np.float32).ravel()

                        valid = ~np.isnan(fill_vals) & ~np.isnan(rec_vals)
                        if valid.sum() == 0:
                            fill_med = np.nan
                            rec_med  = np.nan
                        else:
                            fill_med = float(np.nanmedian(fill_vals[valid]))
                            rec_med  = float(np.nanmedian(rec_vals[valid]))

                        pool_data[idx][tile_folder]["fill"].append(fill_med)
                        pool_data[idx][tile_folder]["recession"].append(rec_med)

                    except Exception:
                        pool_data[idx][tile_folder]["fill"].append(np.nan)
                        pool_data[idx][tile_folder]["recession"].append(np.nan)

    print("\n=== סריקת ספים ===")

    n_pools = len(pools)
    rows = []

    for fill_thr, rec_thr in product(FILL_THRESHOLDS, RECESSION_THRESHOLDS):
        # For each pool: count seasons (across all tiles, take max tile) where both conditions met
        detected_any  = 0   # pools with persistence ≥ 1 in at least one tile
        detected_ge2  = 0
        detected_ge3  = 0

        for pool_idx in range(len(pools)):
            best_persistence = 0
            for tile_folder, arrays in pool_data[pool_idx].items():
                fill_arr = np.array(arrays["fill"])
                rec_arr  = np.array(arrays["recession"])
                # seasons where both thresholds met
                seasons_ok = np.sum(
                    (fill_arr >= fill_thr) & (rec_arr >= rec_thr)
                )
                best_persistence = max(best_persistence, int(seasons_ok))

            if best_persistence >= 1: detected_any += 1
            if best_persistence >= 2: detected_ge2 += 1
            if best_persistence >= 3: detected_ge3 += 1

        rows.append({
            "delta_fill_thr":      fill_thr,
            "delta_recession_thr": rec_thr,
            "pct_ge1": round(detected_any / n_pools * 100, 1),
            "pct_ge2": round(detected_ge2 / n_pools * 100, 1),
            "pct_ge3": round(detected_ge3 / n_pools * 100, 1),
            "n_ge1":   detected_any,
            "n_ge2":   detected_ge2,
            "n_ge3":   detected_ge3,
        })
        print(f"  fill≥{fill_thr:.2f}  rec≥{rec_thr:.2f}  →  "
              f"≥1: {detected_any}/{n_pools} ({detected_any/n_pools*100:.1f}%)  "
              f"≥2: {detected_ge2} ({detected_ge2/n_pools*100:.1f}%)  "
              f"≥3: {detected_ge3} ({detected_ge3/n_pools*100:.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nנשמר: {OUT_CSV}")


if __name__ == "__main__":
    main()
