"""
list_used_dates.py – Scan sentinel2_data and output a CSV of all date folders
that pass the scene filters (month + cloud cover), without computing any indices.

Filters:
  - Month: November–April only (11, 12, 1, 2, 3, 4)
  - Cloud cover: ≤ 5% cloud pixels (SCL classes 8/9/10)

Output CSV columns: tile, season, date, cloud_pct

Usage:
    python list_used_dates.py
    python list_used_dates.py --output /path/to/used_dates.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio

DATA_DIR = Path("/Volumes/ElementsData/menasim_volonteer/cdse_pipeline/sentinel2_data")
DEFAULT_OUT = Path(__file__).parent / "used_dates.csv"

KEEP_MONTHS       = {11, 12, 1, 2, 3, 4}
MAX_CLOUD_FRAC    = 0.05
CLOUD_SCL_CLASSES = {8, 9, 10}


def is_valid_month(date_dir: Path) -> bool:
    return int(date_dir.name[4:6]) in KEEP_MONTHS


def scene_cloud_fraction(date_dir: Path) -> float:
    scl_files = list(date_dir.glob("*_SCL_*.jp2")) + list(date_dir.glob("*_SCL_*.tif"))
    if not scl_files:
        return 0.0
    with rasterio.open(scl_files[0]) as src:
        scl = src.read(1)
    valid = scl != 0
    if not valid.any():
        return 0.0
    cloud = np.isin(scl, list(CLOUD_SCL_CLASSES))
    return float(cloud[valid].sum()) / float(valid.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUT),
                        help=f"Output CSV path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    rows = []

    for season_dir in sorted(DATA_DIR.iterdir()):
        if not season_dir.is_dir():
            continue
        season = season_dir.name
        for tile_dir in sorted(season_dir.iterdir()):
            if not tile_dir.is_dir():
                continue
            tile = tile_dir.name.split("_")[0]
            for date_dir in sorted(d for d in tile_dir.iterdir() if d.is_dir()):
                if not is_valid_month(date_dir):
                    continue
                frac = scene_cloud_fraction(date_dir)
                if frac > MAX_CLOUD_FRAC:
                    continue
                rows.append({
                    "tile": tile,
                    "season": season,
                    "date": date_dir.name,
                    "cloud_pct": f"{frac * 100:.1f}",
                })

    out_path = Path(args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tile", "season", "date", "cloud_pct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"נשמר: {out_path}  ({len(rows)} תאריכים)")


if __name__ == "__main__":
    main()
