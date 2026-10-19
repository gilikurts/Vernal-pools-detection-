"""
run_depression_extraction.py
============================
Step 1 of the Michigan Tech vernal pool pipeline: "Isolating the Depressions".

Uses a contour tree approach (Wu et al. 2015) to identify topographic
depressions with no outflow, then filters them to vernal pool candidates
based on size, depth, and shape.

Usage:
    python run_depression_extraction.py --dem path/to/dem.tif --out_dir output/

Output:
    output/
        all_depressions.gpkg        - all depressions found
        vernal_pool_candidates.gpkg - filtered candidates only
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd

# make sure the local lidar package is importable
sys.path.insert(0, str(Path(__file__).parent / "lidar_pkg"))

from lidar import MedianFilter, ExtractSinks

# ── Vernal pool filter parameters ──────────────────────────────────────────
MIN_PIXELS    = 5        # minimum depression size in pixels (avoids noise)
MIN_AREA_M2   = 500      # minimum area in m²  (~0.05 dunam)
MAX_AREA_M2   = 500_000  # maximum area in m²  (~50 dunam)
MIN_DEPTH_M   = 0.5      # minimum max depth in meters (DEM vertical resolution ~5m)
MAX_DEPTH_M   = 10.0     # maximum max depth in meters
MAX_ELONGATED = 6.0      # elongatedness ratio — filters out stream/wadi-like shapes
# ───────────────────────────────────────────────────────────────────────────


def run(dem_path: str, out_dir: str = "output", kernel_size: int = 3):
    dem_path = Path(dem_path).resolve()
    out_dir  = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"DEM:        {dem_path}")
    print(f"Output dir: {out_dir}")
    print()

    # Step 1: smooth DEM with median filter
    print("=== Smoothing DEM ===")
    smoothed_dem = str(tmp_dir / "dem_smoothed.tif")
    MedianFilter(str(dem_path), kernel_size=kernel_size, out_file=smoothed_dem)

    # Step 2: extract all depressions (contour tree)
    print("\n=== Extracting depressions (contour tree) ===")
    ExtractSinks(
        in_dem     = smoothed_dem,
        min_size   = MIN_PIXELS,
        out_dir    = str(tmp_dir),
        keep_files = True,
    )

    gpkg_files = list(tmp_dir.glob("*.gpkg"))
    if not gpkg_files:
        print("ERROR: no GeoPackage output found.")
        sys.exit(1)
    gpkg_path = gpkg_files[0]

    # Step 3: load and save all depressions
    print("\n=== Loading results ===")
    gdf = gpd.read_file(gpkg_path)
    print(f"Total depressions found: {len(gdf)}")

    all_out = out_dir / "all_depressions.gpkg"
    gdf.to_file(all_out, driver="GPKG")
    print(f"Saved: {all_out}")

    # Step 4: filter to vernal pool candidates
    print("\n=== Filtering to vernal pool candidates ===")
    candidates = gdf[
        (gdf["area"]          >= MIN_AREA_M2) &
        (gdf["area"]          <= MAX_AREA_M2) &
        (gdf["max_depth"]     >= MIN_DEPTH_M) &
        (gdf["max_depth"]     <= MAX_DEPTH_M) &
        (gdf["elongatedness"] <= MAX_ELONGATED)
    ].copy()

    print(f"After filtering: {len(candidates)} candidates")
    print(f"  area:          {MIN_AREA_M2}-{MAX_AREA_M2} m²")
    print(f"  max_depth:     {MIN_DEPTH_M}-{MAX_DEPTH_M} m")
    print(f"  elongatedness: <= {MAX_ELONGATED}")

    cand_out = out_dir / "vernal_pool_candidates.gpkg"
    candidates.to_file(cand_out, driver="GPKG")
    print(f"\nSaved: {cand_out}")
    print(f"\nDone. Open {cand_out} in QGIS to review.")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Extract depression candidates from DEM")
    parser.add_argument("--dem",     required=True, help="Path to input DEM (.tif)")
    parser.add_argument("--out_dir", default="output", help="Output directory (default: output/)")
    parser.add_argument("--kernel",  type=int, default=3, help="Median filter kernel size (default: 3)")
    args = parser.parse_args()

    run(args.dem, args.out_dir, args.kernel)


if __name__ == "__main__":
    main()
