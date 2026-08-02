"""
compute_persistence.py – Sum candidate masks across seasons per tile.

For each tile: sum candidates.tif across all 5 winters → values 0-5
(how many seasons the pixel was a candidate).

Output: persistence/{tile_folder}/persistence.tif  (uint8, values 0-5)

Usage:
    python compute_persistence.py
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

CANDIDATES_DIR = Path("/Volumes/ElementsData/menasim_volonteer/candidates")
OUT_DIR        = Path("/Volumes/ElementsData/menasim_volonteer/persistence")


def main():
    # Group candidate files by tile
    groups = defaultdict(list)
    for path in sorted(CANDIDATES_DIR.rglob("candidates.tif")):
        tile_folder = path.parent.name
        groups[tile_folder].append(path)

    print(f"נמצאו {len(groups)} טיילים")

    for tile_folder, paths in sorted(groups.items()):
        out_path = OUT_DIR / tile_folder / "persistence.tif"
        if out_path.exists():
            print(f"  דילוג (קיים): {tile_folder}")
            continue

        print(f"  {tile_folder} — {len(paths)} עונות")

        # Read reference shape from first file
        with rasterio.open(paths[0]) as ref:
            profile = ref.profile.copy()
            h, w    = ref.height, ref.width

        total = np.zeros((h, w), dtype=np.uint8)
        for p in paths:
            with rasterio.open(p) as src:
                arr = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
            total += arr.astype(np.uint8)

        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=None,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp.tif")
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(total, 1)
        tmp_path.rename(out_path)

        for v in range(1, len(paths) + 1):
            n = int((total == v).sum())
            print(f"    {v} עונות: {n:,} פיקסלים")

    print(f"\nהושלם. נתונים ב: {OUT_DIR}")


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except (PermissionError, OSError, rasterio.errors.RasterioIOError) as e:
            print(f"\n⚠️ שגיאת כונן: {e}")
            print("ממתין 60 שניות ומנסה מחדש...")
            time.sleep(60)
