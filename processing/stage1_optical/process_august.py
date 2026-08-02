"""
process_august.py – מסכה + MNDWI לסצנות אוגוסט.

קורא מ-manifest_august.csv, מעבד כל סצנה:
  1. מסכה SCL → masked_scenes/summer_YYYY/tile/date/
  2. MNDWI    → mndwi/summer_YYYY/tile/date/MNDWI.tif
"""

import csv
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from tqdm import tqdm

RAW_DIR     = Path("/Volumes/ElementsData/menasim_volonteer/sentinel2_data")
MASKED_DIR  = Path("/Volumes/ElementsData/menasim_volonteer/masked_scenes")
MNDWI_DIR   = Path("/Volumes/ElementsData/menasim_volonteer/mndwi")
MANIFEST    = Path(__file__).parent.parent.parent / "cdse_pipeline" / "manifest_august.csv"

TILE_NAMES = {
    "36SXA": "36SXA_מרכז - שרון ומישור החוף",
    "36SYB": "36SYB_גליל וגולן",
    "36SXB": "36SXB_חיפה והכרמל",
    "36SYA": "36SYA_שומרון וירושלים",
    "37SBS": "37SBS_צפון-מזרח - גולן וחרמון",
    "36RXV": "36RXV_דרום-מערב - נגב מערבי",
}

BAD_SCL     = {1, 3, 8, 9, 10, 11}
BANDS_10M   = {"B02", "B03", "B04", "B08"}
BANDS_20M   = {"B11", "B12"}


def find_band_files(date_dir: Path) -> dict[str, Path]:
    result = {}
    for f in list(date_dir.glob("*.jp2")) + list(date_dir.glob("*.tif")):
        parts = f.stem.split("_")
        if len(parts) >= 2:
            band = parts[-2]
            if band in BANDS_10M | BANDS_20M | {"SCL"}:
                result[band] = f
    return result


def mask_scene(date_dir: Path, out_dir: Path):
    band_files = find_band_files(date_dir)
    scl_path = band_files.get("SCL")
    if not scl_path:
        tqdm.write(f"  אין SCL: {date_dir}")
        return False

    with rasterio.open(scl_path) as scl_src:
        scl = scl_src.read(1)
        scl_transform = scl_src.transform
        scl_crs = scl_src.crs

    bad_mask = np.isin(scl, list(BAD_SCL))

    for band_name in sorted(BANDS_10M | BANDS_20M):
        band_path = band_files.get(band_name)
        if not band_path:
            continue
        out_path = out_dir / f"{band_name}.tif"
        if out_path.exists():
            continue

        with rasterio.open(band_path) as src:
            arr = src.read(1).astype(np.float32)
            profile = src.profile.copy()

            if band_name in BANDS_10M:
                mask_10m = np.zeros(arr.shape, dtype=np.uint8)
                reproject(
                    source=bad_mask.astype(np.uint8),
                    destination=mask_10m,
                    src_transform=scl_transform,
                    src_crs=scl_crs,
                    dst_transform=src.transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest,
                )
                band_mask = mask_10m.astype(bool)
            else:
                band_mask = bad_mask

        arr[arr == 0] = np.nan
        arr[band_mask] = np.nan

        profile.update(driver="GTiff", dtype="float32", nodata=np.nan,
                       compress="deflate", predictor=3,
                       tiled=True, blockxsize=512, blockysize=512)

        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.tif")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(arr, 1)
        tmp.rename(out_path)

    return True


def compute_mndwi(masked_date_dir: Path, out_path: Path):
    b03_path = masked_date_dir / "B03.tif"
    b11_path = masked_date_dir / "B11.tif"
    if not b03_path.exists() or not b11_path.exists():
        return

    with rasterio.open(b03_path) as src:
        b03 = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        h, w = src.shape

    with rasterio.open(b11_path) as src:
        b11 = src.read(1, out_shape=(h, w),
                       resampling=Resampling.bilinear).astype(np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = ((b03 - b11) / (b03 + b11)).astype(np.float32)

    profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan,
                   compress="deflate", predictor=3,
                   tiled=True, blockxsize=512, blockysize=512)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(mndwi, 1)
    tmp.rename(out_path)


def main():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]

    print(f"נמצאו {len(rows)} סצנות אוגוסט להעיבוד")

    for row in tqdm(rows, desc="מעבד אוגוסט"):
        tile        = row["tile"]
        tile_folder = TILE_NAMES.get(tile, tile)
        season      = row["season"]   # e.g. summer_2021
        date        = row["date"]

        raw_dir    = RAW_DIR   / season / tile_folder / date
        masked_dir = MASKED_DIR / season / tile_folder / date
        mndwi_path = MNDWI_DIR  / season / tile_folder / date / "MNDWI.tif"

        if not raw_dir.exists():
            tqdm.write(f"  לא נמצא: {raw_dir}")
            continue

        # שלב 1: מסכה
        needs_mask = not all((masked_dir / f"{b}.tif").exists()
                             for b in ["B03", "B11"])
        if needs_mask:
            mask_scene(raw_dir, masked_dir)

        # שלב 2: MNDWI
        if not mndwi_path.exists():
            compute_mndwi(masked_dir, mndwi_path)

        tqdm.write(f"  ✓ {tile}/{season}/{date}")

    print(f"\nהושלם!")
    print(f"  מסכות: {MASKED_DIR}/summer_*/")
    print(f"  MNDWI:  {MNDWI_DIR}/summer_*/")


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except (OSError, rasterio.errors.RasterioIOError) as e:
            print(f"\n⚠️ שגיאת כונן: {e}")
            time.sleep(60)
