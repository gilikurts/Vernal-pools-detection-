"""
mask_scenes.py – Apply SCL cloud/shadow mask to all approved scenes and save
as ready-to-use GeoTIFFs.

Input:  used_dates.csv  (produced by list_used_dates.py)
        sentinel2_data/ on the external drive

Output: masked_scenes/ on the external drive, same structure:
          season / tile_folder / date / B02.tif … B12.tif

Masked pixels (SCL classes 1,3,8,9,10,11) are set to nodata=nan.
SCL band itself is not copied to the output.

Usage:
    python mask_scenes.py
    python mask_scenes.py --dry-run   # preview without writing
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

DATA_DIR    = Path("/Volumes/ElementsData/menasim_volonteer/cdse_pipeline/sentinel2_data")
OUT_DIR     = Path("/Volumes/ElementsData/menasim_volonteer/masked_scenes")
CSV_PATH    = Path(__file__).parent / "used_dates.csv"

BAD_SCL_CLASSES = {1, 3, 8, 9, 10, 11}

TILE_NAMES = {
    "36SXA": "36SXA_מרכז - שרון ומישור החוף",
    "36SYB": "36SYB_גליל וגולן",
    "36SXB": "36SXB_חיפה והכרמל",
    "36SYA": "36SYA_שומרון וירושלים",
    "37SBS": "37SBS_צפון-מזרח - גולן וחרמון",
    "36RXV": "36RXV_דרום-מערב - נגב מערבי",
}

BANDS_10M = {"B02", "B03", "B04", "B08"}
BANDS_20M = {"B11", "B12"}


def find_band_files(date_dir: Path) -> dict[str, Path]:
    """Return {band_name: path} for all spectral bands in a date folder."""
    files = list(date_dir.glob("*.jp2")) + list(date_dir.glob("*.tif"))
    result = {}
    for f in files:
        parts = f.stem.split("_")
        # band name is the second-to-last part, e.g. T36SYB_20241110T082151_B02_10m
        if len(parts) >= 2:
            band = parts[-2]
            if band in BANDS_10M | BANDS_20M | {"SCL"}:
                result[band] = f
    return result


def process_date(date_dir: Path, out_date_dir: Path):
    """Mask cloud/shadow pixels for all spectral bands and save as GeoTIFF."""
    band_files = find_band_files(date_dir)

    scl_path = band_files.get("SCL")
    if scl_path is None:
        print(f"  אין SCL ב-{date_dir.name}, מדלג")
        return

    with rasterio.open(scl_path) as scl_src:
        scl      = scl_src.read(1)
        scl_h, scl_w = scl_src.shape
        scl_transform = scl_src.transform
        scl_crs       = scl_src.crs

    bad_mask = np.isin(scl, list(BAD_SCL_CLASSES))  # True = mask out

    for band_name in sorted(BANDS_10M | BANDS_20M):
        band_path = band_files.get(band_name)
        if band_path is None:
            continue

        out_path = out_date_dir / f"{band_name}.tif"
        if out_path.exists():
            continue

        with rasterio.open(band_path) as src:
            arr     = src.read(1).astype(np.float32)
            profile = src.profile.copy()

            if band_name in BANDS_20M:
                # Upsample bad_mask to 10m to match these 20m bands
                # (bad_mask is at 20m resolution, same as SCL)
                band_mask = bad_mask
            else:
                # bad_mask is 20m, band is 10m — upsample mask with nearest
                from rasterio.warp import reproject
                band_mask_10m = np.zeros(arr.shape, dtype=np.uint8)
                reproject(
                    source=bad_mask.astype(np.uint8),
                    destination=band_mask_10m,
                    src_transform=scl_transform,
                    src_crs=scl_crs,
                    dst_transform=src.transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest,
                )
                band_mask = band_mask_10m.astype(bool)

        arr[arr == 0]    = np.nan   # original nodata
        arr[band_mask]   = np.nan   # cloud/shadow pixels

        profile.update(
            driver="GTiff",
            dtype="float32",
            nodata=np.nan,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        out_date_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)


def load_used_dates(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be processed without writing files")
    args = parser.parse_args()

    rows = load_used_dates(CSV_PATH)
    print(f"נטענו {len(rows)} תאריכים מ-{CSV_PATH.name}")

    if args.dry_run:
        print("=== DRY RUN ===")
        for r in rows:
            tile_folder = TILE_NAMES.get(r["tile"], r["tile"])
            src = DATA_DIR / r["season"] / tile_folder / r["date"]
            dst = OUT_DIR  / r["season"] / tile_folder / r["date"]
            print(f"  {src} → {dst}")
        return

    for row in tqdm(rows, desc="מסכה סצנות"):
        tile_folder = TILE_NAMES.get(row["tile"], row["tile"])
        date_dir    = DATA_DIR / row["season"] / tile_folder / row["date"]
        out_dir     = OUT_DIR  / row["season"] / tile_folder / row["date"]

        if not date_dir.exists():
            tqdm.write(f"  לא נמצא: {date_dir}")
            continue

        process_date(date_dir, out_dir)

    print(f"\nהושלם. נתונים מוכנים ב: {OUT_DIR}")


if __name__ == "__main__":
    main()
