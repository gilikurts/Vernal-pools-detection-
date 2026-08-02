"""Retroactively crop downloaded Sentinel-2 band files to the land area
within each tile, masking out sea pixels (nodata=0) and re-saving as
compressed GeoTIFF instead of jp2. Tiles that are (almost) entirely land
are left untouched - there is nothing to gain there.

Land boundary comes from Natural Earth 10m land polygons
(coastline/ne_10m_land/ne_10m_land.shp).
"""

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping, shape
from tqdm import tqdm

from config import OUTPUT_DIR

LAND_SHP = "coastline/ne_10m_land/ne_10m_land.shp"

# If the land area's bounding box still covers this fraction (or more) of
# the tile, cropping wouldn't shrink the raster meaningfully - skip the tile.
CROP_BBOX_THRESHOLD = 0.95


def get_tile_code(tile_folder_name):
    return tile_folder_name.split("_")[0]


def land_geoms_for_bounds(land_gdf, crs, bounds):
    tile_box_wgs = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=crs).to_crs(land_gdf.crs)
    land_clip = gpd.clip(land_gdf, tile_box_wgs)
    land_clip = land_clip[~land_clip.geometry.is_empty]
    land_utm = land_clip.to_crs(crs)
    return [mapping(g) for g in land_utm.geometry if g is not None and not g.is_empty]


def crop_bbox_fraction(geoms, tile_bounds):
    bounds = [shape(g).bounds for g in geoms]
    minx = min(b[0] for b in bounds)
    miny = min(b[1] for b in bounds)
    maxx = max(b[2] for b in bounds)
    maxy = max(b[3] for b in bounds)

    tile_area = (tile_bounds[2] - tile_bounds[0]) * (tile_bounds[3] - tile_bounds[1])
    crop_area = (maxx - minx) * (maxy - miny)
    return crop_area / tile_area


def process_file(jp2_path, geoms):
    with rasterio.open(jp2_path) as src:
        out_image, out_transform = mask(src, geoms, crop=True, nodata=0)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": 0,
                "driver": "GTiff",
                "compress": "deflate",
                "predictor": 2,
            }
        )

    out_path = jp2_path.with_suffix(".tif")
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_image)

    with rasterio.open(out_path) as check:
        if check.shape != out_image.shape[1:]:
            raise RuntimeError("shape mismatch after crop")
        check.read(1)  # make sure the written file is actually readable

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Directory containing season/tile/date folders to crop (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    land = gpd.read_file(LAND_SHP)
    tile_geoms_cache = {}

    total_before = 0
    total_after = 0
    n_converted = 0
    skipped_tiles = set()
    processed_tiles = set()

    for season_dir in sorted(Path(args.output_dir).glob("*")):
        if not season_dir.is_dir():
            continue
        for tile_dir in sorted(season_dir.glob("*")):
            if not tile_dir.is_dir():
                continue
            tile_code = get_tile_code(tile_dir.name)

            jp2_files = sorted(tile_dir.glob("*/*.jp2"))
            if not jp2_files:
                continue

            if tile_code not in tile_geoms_cache:
                with rasterio.open(jp2_files[0]) as src:
                    crs = src.crs
                    bounds = src.bounds
                geoms = land_geoms_for_bounds(land, crs, bounds)
                frac = crop_bbox_fraction(geoms, bounds)
                tile_geoms_cache[tile_code] = (geoms, frac)
                print(f"{tile_code}: land crop bbox = {frac * 100:.0f}% of tile area")

            geoms, frac = tile_geoms_cache[tile_code]
            if frac >= CROP_BBOX_THRESHOLD:
                skipped_tiles.add(tile_code)
                continue

            processed_tiles.add(tile_code)
            for jp2_path in tqdm(jp2_files, desc=tile_dir.name):
                before = jp2_path.stat().st_size
                try:
                    out_path = process_file(jp2_path, geoms)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR on {jp2_path}: {exc}")
                    continue

                after = out_path.stat().st_size
                total_before += before
                total_after += after
                n_converted += 1
                jp2_path.unlink()

    print(f"\nProcessed tiles: {sorted(processed_tiles)}")
    print(f"Skipped tiles (mostly/all land, left as jp2): {sorted(skipped_tiles)}")
    print(f"Converted {n_converted} files")
    if total_before:
        print(
            f"Before: {total_before / 1e9:.2f} GB, After: {total_after / 1e9:.2f} GB "
            f"({total_after / total_before * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
