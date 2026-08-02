"""
compute_mndwi.py – Compute MNDWI per date from masked_scenes/.

Input:  used_dates.csv  (produced by list_used_dates.py)
        masked_scenes/  (produced by mask_scenes.py)

Output: mndwi/{season}/{tile_folder}/{date}/MNDWI.tif

MNDWI = (B03 - B11) / (B03 + B11)
B11 (20m) is resampled to 10m to match B03.
NaN pixels (cloud/shadow/nodata) propagate through the formula.

Usage:
    python compute_mndwi.py
"""

import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

MASKED_DIR = Path("/Volumes/ElementsData/menasim_volonteer/masked_scenes")
OUT_DIR    = Path("/Volumes/ElementsData/menasim_volonteer/mndwi")
CSV_PATH   = Path(__file__).parent / "used_dates.csv"

TILE_NAMES = {
    "36SXA": "36SXA_מרכז - שרון ומישור החוף",
    "36SYB": "36SYB_גליל וגולן",
    "36SXB": "36SXB_חיפה והכרמל",
    "36SYA": "36SYA_שומרון וירושלים",
    "37SBS": "37SBS_צפון-מזרח - גולן וחרמון",
    "36RXV": "36RXV_דרום-מערב - נגב מערבי",
}


def compute_mndwi_for_date(date_dir: Path, out_path: Path):
    b03_path = date_dir / "B03.tif"
    b11_path = date_dir / "B11.tif"

    if not b03_path.exists() or not b11_path.exists():
        tqdm.write(f"  חסר B03 או B11 ב-{date_dir.name}, מדלג")
        return

    with rasterio.open(b03_path) as b03_src:
        b03     = b03_src.read(1).astype(np.float32)
        profile = b03_src.profile.copy()
        h, w    = b03_src.shape

    with rasterio.open(b11_path) as b11_src:
        b11 = b11_src.read(
            1,
            out_shape=(h, w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = ((b03 - b11) / (b03 + b11)).astype(np.float32)

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
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mndwi, 1)


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"נטענו {len(rows)} תאריכים מ-{CSV_PATH.name}")

    for row in tqdm(rows, desc="מחשב MNDWI"):
        tile_folder = TILE_NAMES.get(row["tile"], row["tile"])
        date_dir    = MASKED_DIR / row["season"] / tile_folder / row["date"]
        out_path    = OUT_DIR    / row["season"] / tile_folder / row["date"] / "MNDWI.tif"

        if out_path.exists():
            continue

        if not date_dir.exists():
            tqdm.write(f"  לא נמצא: {date_dir}")
            continue

        compute_mndwi_for_date(date_dir, out_path)

    print(f"\nהושלם. נתונים ב: {OUT_DIR}")


if __name__ == "__main__":
    main()
