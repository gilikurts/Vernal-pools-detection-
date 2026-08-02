"""
detect_vernal_pools.py – Vernal pool detection via seasonal MNDWI curve shape.

Algorithm (per pixel, per season):
  1. nov_mndwi    = median(MNDWI, November)
  2. winter_mndwi = median(MNDWI, January–February)
  3. aug_mndwi    = August MNDWI (single scene, pre-downloaded)

  delta_fill     = winter_mndwi − nov_mndwi      (rise into winter)
  delta_seasonal = winter_mndwi − aug_mndwi      (fall back to dry)

  candidate = (delta_fill > 0) AND (delta_seasonal > 0)
  i.e. pixel must be wetter in Jan–Feb than both November and August.

Stage 2: persistence.tif = sum of candidate years per pixel (0–5).

Usage:
    python detect_vernal_pools.py
    python detect_vernal_pools.py --force   # recompute even if output exists
"""

import contextlib
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from rasterio.enums import Resampling
from tqdm import tqdm

MNDWI_DIR = Path("/Volumes/ElementsData/menasim_volonteer/mndwi")
OUT_DIR   = Path("/Volumes/ElementsData/menasim_volonteer/vernal_pool_detection")
CSV_PATH  = Path(__file__).parent / "used_dates.csv"

SPATIAL_CHUNK = 500
FORCE = "--force" in sys.argv

TILE_NAMES = {
    "36SXA": "36SXA_מרכז - שרון ומישור החוף",
    "36SYB": "36SYB_גליל וגולן",
    "36SXB": "36SXB_חיפה והכרמל",
    "36SYA": "36SYA_שומרון וירושלים",
    "37SBS": "37SBS_צפון-מזרח - גולן וחרמון",
    "36RXV": "36RXV_דרום-מערב - נגב מערבי",
}

PERIOD_MONTHS = {
    "november": {11},
    "winter":   {1, 2},
}

OUTPUT_FILES = ["nov_mndwi.tif", "winter_mndwi.tif",
                "delta_fill.tif", "delta_seasonal.tif", "candidates.tif"]


def month_of(date_str: str) -> int:
    return int(date_str[4:6])


def read_chunk(src_list: list, row0: int, row1: int, w: int) -> np.ndarray:
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


def period_median(stack: np.ndarray) -> np.ndarray:
    if stack.shape[0] == 0:
        return np.full(stack.shape[1:], np.nan, dtype=np.float32)
    with np.errstate(all="ignore"):
        return np.nanmedian(stack, axis=0).astype(np.float32)


def save_tif(path: Path, arr: np.ndarray, profile: dict):
    tmp = path.with_suffix(".tmp.tif")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr, 1)
    tmp.rename(path)


def get_aug_path(tile_folder: str, season: str) -> Path | None:
    start_year = int(season.split("_")[1])  # winter_2021_2022 → 2021
    summer_dir = MNDWI_DIR / f"summer_{start_year}" / tile_folder
    paths = list(summer_dir.glob("*/MNDWI.tif"))
    return paths[0] if paths else None


def process_tile_season(tile: str, season: str, date_paths: dict[str, Path]):
    tile_folder = TILE_NAMES.get(tile, tile)
    out_dir     = OUT_DIR / season / tile_folder

    if not FORCE and all((out_dir / f).exists() for f in OUTPUT_FILES):
        return

    period_paths: dict[str, list[Path]] = {p: [] for p in PERIOD_MONTHS}
    for date_str, path in sorted(date_paths.items()):
        m = month_of(date_str)
        for period, months in PERIOD_MONTHS.items():
            if m in months:
                period_paths[period].append(path)

    nov_paths    = period_paths["november"]
    winter_paths = period_paths["winter"]
    aug_path     = get_aug_path(tile_folder, season)

    if not nov_paths or not winter_paths:
        print(f"  ⚠️  {tile}/{season}: חסרים תאריכים ({len(nov_paths)} נוב, {len(winter_paths)} ינ-פב)")
        return
    if aug_path is None:
        print(f"  ⚠️  {tile}/{season}: אין MNDWI אוגוסט")
        return

    all_paths = nov_paths + winter_paths
    with rasterio.open(all_paths[0]) as ref:
        h, w    = ref.height, ref.width
        profile = ref.profile.copy()

    float_profile = profile.copy()
    float_profile.update(driver="GTiff", dtype="float32", count=1,
                         nodata=np.nan, compress="deflate", predictor=3,
                         tiled=True, blockxsize=512, blockysize=512)
    uint8_profile = profile.copy()
    uint8_profile.update(driver="GTiff", dtype="uint8", count=1,
                         nodata=None, compress="deflate", predictor=2,
                         tiled=True, blockxsize=512, blockysize=512)

    nov_arr    = np.full((h, w), np.nan, dtype=np.float32)
    winter_arr = np.full((h, w), np.nan, dtype=np.float32)
    delta_fill     = np.full((h, w), np.nan, dtype=np.float32)
    delta_seasonal = np.full((h, w), np.nan, dtype=np.float32)

    with contextlib.ExitStack() as stack:
        nov_srcs    = [stack.enter_context(rasterio.open(p)) for p in nov_paths]
        winter_srcs = [stack.enter_context(rasterio.open(p)) for p in winter_paths]
        aug_src     = stack.enter_context(rasterio.open(aug_path))

        for row0 in tqdm(range(0, h, SPATIAL_CHUNK),
                         desc=f"  {tile}/{season[:11]}", leave=False):
            row1 = min(row0 + SPATIAL_CHUNK, h)

            nov_chunk    = period_median(read_chunk(nov_srcs,    row0, row1, w))
            winter_chunk = period_median(read_chunk(winter_srcs, row0, row1, w))
            aug_chunk    = read_chunk([aug_src], row0, row1, w)[0]

            nov_arr[row0:row1]    = nov_chunk
            winter_arr[row0:row1] = winter_chunk
            delta_fill[row0:row1]     = winter_chunk - nov_chunk
            delta_seasonal[row0:row1] = winter_chunk - aug_chunk

    # candidate: winter > november AND winter > august (curve shape)
    candidates = (
        (delta_fill     > 0) &
        (delta_seasonal > 0)
    ).astype(np.uint8)

    save_tif(out_dir / "nov_mndwi.tif",      nov_arr,        float_profile)
    save_tif(out_dir / "winter_mndwi.tif",   winter_arr,     float_profile)
    save_tif(out_dir / "delta_fill.tif",     delta_fill,     float_profile)
    save_tif(out_dir / "delta_seasonal.tif", delta_seasonal, float_profile)
    save_tif(out_dir / "candidates.tif",     candidates,     uint8_profile)

    n_cand = int(candidates.sum())
    pct    = n_cand / (candidates.size) * 100
    print(f"  {tile}/{season} — נוב:{len(nov_paths)} ינ-פב:{len(winter_paths)} | מועמדים: {n_cand:,} ({pct:.1f}%)")


def sum_across_seasons(groups: dict):
    print("\n=== שלב 2: persistence ===")

    by_tile: dict[str, list[Path]] = defaultdict(list)
    for (tile, season) in groups:
        tile_folder = TILE_NAMES.get(tile, tile)
        p = OUT_DIR / season / tile_folder / "candidates.tif"
        if p.exists():
            by_tile[tile_folder].append(p)

    for tile_folder, paths in sorted(by_tile.items()):
        out_path = OUT_DIR / "persistence" / tile_folder / "persistence.tif"
        if not FORCE and out_path.exists():
            print(f"  דילוג (קיים): {tile_folder}")
            continue

        with rasterio.open(paths[0]) as ref:
            h, w    = ref.height, ref.width
            profile = ref.profile.copy()

        profile.update(driver="GTiff", dtype="uint8", count=1,
                       nodata=None, compress="deflate", predictor=2,
                       tiled=True, blockxsize=512, blockysize=512)

        total = np.zeros((h, w), dtype=np.uint8)
        for p in paths:
            with rasterio.open(p) as src:
                total += src.read(1, out_shape=(h, w),
                                  resampling=Resampling.nearest).astype(np.uint8)

        save_tif(out_path, total, profile)

        print(f"  {tile_folder} ({len(paths)} עונות)")
        for v in range(1, len(paths) + 1):
            n = int((total == v).sum())
            print(f"    persistence={v}: {n:,} פיקסלים ({n/total.size*100:.2f}%)")


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        tile_folder = TILE_NAMES.get(row["tile"], row["tile"])
        path = MNDWI_DIR / row["season"] / tile_folder / row["date"] / "MNDWI.tif"
        if path.exists():
            groups[(row["tile"], row["season"])][row["date"]] = path

    print(f"=== שלב 1: זיהוי מועמדים ===")
    print(f"{len(groups)} שילובי טייל × עונה\n")

    for (tile, season), date_paths in sorted(groups.items()):
        process_tile_season(tile, season, date_paths)

    sum_across_seasons(groups)
    print(f"\nהושלם. נתונים ב: {OUT_DIR}")


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except (PermissionError, OSError, rasterio.errors.RasterioIOError) as e:
            print(f"\n⚠️ שגיאת כונן: {e}")
            time.sleep(60)
