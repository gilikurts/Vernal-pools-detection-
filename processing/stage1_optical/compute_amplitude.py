"""
compute_amplitude.py – Compute per-pixel MNDWI amplitude across a season.

For each tile × season:
  1. Stack all MNDWI.tif files (one per date) → (n_dates, height, width)
  2. Compute amplitude = nanmax - nanmin along the time axis
  3. Save amplitude.tif

Output: amplitude/{season}/{tile_folder}/amplitude.tif

Usage:
    python compute_amplitude.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

MNDWI_DIR = Path("/Volumes/ElementsData/menasim_volonteer/mndwi")
OUT_DIR   = Path("/Volumes/ElementsData/menasim_volonteer/amplitude")
CSV_PATH  = Path(__file__).parent / "used_dates.csv"

TILE_NAMES = {
    "36SXA": "36SXA_מרכז - שרון ומישור החוף",
    "36SYB": "36SYB_גליל וגולן",
    "36SXB": "36SXB_חיפה והכרמל",
    "36SYA": "36SYA_שומרון וירושלים",
    "37SBS": "37SBS_צפון-מזרח - גולן וחרמון",
    "36RXV": "36RXV_דרום-מערב - נגב מערבי",
}


def compute_amplitude(tile: str, season: str, dates: list[str]):
    tile_folder = TILE_NAMES.get(tile, tile)
    out_path    = OUT_DIR / season / tile_folder / "amplitude.tif"

    if out_path.exists():
        print(f"  דילוג (קיים): {tile} / {season}")
        return

    mndwi_paths = []
    for date in sorted(dates):
        p = MNDWI_DIR / season / tile_folder / date / "MNDWI.tif"
        if p.exists():
            mndwi_paths.append(p)

    if not mndwi_paths:
        print(f"  אין קבצים: {tile} / {season}")
        return

    # Read reference profile from first file
    with rasterio.open(mndwi_paths[0]) as ref:
        profile = ref.profile.copy()
        h, w    = ref.height, ref.width

    # Stack all dates into (n_dates, h, w) — resample to reference shape if needed
    stack = np.full((len(mndwi_paths), h, w), np.nan, dtype=np.float32)
    for i, path in enumerate(mndwi_paths):
        with rasterio.open(path) as src:
            stack[i] = src.read(
                1, out_shape=(h, w), resampling=Resampling.bilinear
            ).astype(np.float32)

    amplitude = (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)).astype(np.float32)

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(amplitude, 1)
    tmp_path.rename(out_path)

    print(f"  נשמר: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group dates by tile × season
    groups = defaultdict(list)
    for row in rows:
        groups[(row["tile"], row["season"])].append(row["date"])

    print(f"{len(groups)} שילובי טייל × עונה")

    for (tile, season), dates in tqdm(sorted(groups.items()), desc="אמפליטודה"):
        compute_amplitude(tile, season, dates)

    print(f"\nהושלם. נתונים ב: {OUT_DIR}")


if __name__ == "__main__":
    import time
    while True:
        try:
            main()
            break
        except (PermissionError, OSError, rasterio.errors.RasterioIOError) as e:
            print(f"\n⚠️ שגיאת כונן: {e}")
            print("ממתין 60 שניות ומנסה מחדש...")
            time.sleep(60)
