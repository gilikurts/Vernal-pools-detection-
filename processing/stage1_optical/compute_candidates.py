"""
compute_candidates.py – Binary candidate mask from amplitude.tif.

For each tile × season:
  - Threshold = 80th percentile of valid (non-NaN) amplitude pixels
  - 1 = candidate (top 20% amplitude)
  - 0 = not a candidate

Output: candidates/{season}/{tile_folder}/candidates.tif  (uint8)

Usage:
    python compute_candidates.py
"""

import time
from pathlib import Path

import numpy as np
import rasterio

AMPLITUDE_DIR = Path("/Volumes/ElementsData/menasim_volonteer/amplitude")
OUT_DIR       = Path("/Volumes/ElementsData/menasim_volonteer/candidates")

PERCENTILE = 80


def process(amp_path: Path, out_path: Path):
    with rasterio.open(amp_path) as src:
        amplitude = src.read(1).astype(np.float32)
        profile   = src.profile.copy()

    valid = amplitude[~np.isnan(amplitude)]
    if valid.size == 0:
        print(f"  אין פיקסלים תקינים: {amp_path.parent.name}")
        return

    threshold = np.percentile(valid, PERCENTILE)
    candidates = (amplitude >= threshold).astype(np.uint8)

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
        dst.write(candidates, 1)
    tmp_path.rename(out_path)

    n_candidates = int(candidates.sum())
    print(f"  נשמר: {out_path.parent.parent.name}/{out_path.parent.name} "
          f"| סף={threshold:.3f} | מועמדים={n_candidates:,}")


def main():
    amp_files = sorted(AMPLITUDE_DIR.rglob("amplitude.tif"))
    print(f"נמצאו {len(amp_files)} קבצי amplitude")

    for amp_path in amp_files:
        # mirror the directory structure under OUT_DIR
        relative = amp_path.parent.relative_to(AMPLITUDE_DIR)
        out_path = OUT_DIR / relative / "candidates.tif"

        if out_path.exists():
            print(f"  דילוג (קיים): {relative}")
            continue

        process(amp_path, out_path)

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
