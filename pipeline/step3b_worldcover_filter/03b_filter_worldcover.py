"""
Step 3b: drop candidates that sit mostly on cropland, built-up land, or a
plastic-covered greenhouse, per two satellite-derived rasters (not OSM):
  - ESA WorldCover (10m, Sentinel-1+2, see download_worldcover.py)
  - Global-PCG-10 (10m, Sentinel-2, plastic-covered greenhouses only - see
    download_pcg_greenhouses.py). WorldCover has no greenhouse class of its
    own (greenhouses mostly read as "Built-up", only partial coverage), so
    this adds a dedicated, purpose-built source for the ones that slip past.

Why rasters at all, not just OSM: OSM's agricultural-field and greenhouse
tagging is very incomplete in Israel (visually confirmed - plowed fields and
real greenhouses full of false-positive candidates with no OSM tag at all).
Both rasters are satellite-derived global classifications, not dependent on
manual mapping, so they catch what OSM misses.

WorldCover classes relevant here: 40 = Cropland, 50 = Built-up.
Global-PCG-10: binary, 1 = plastic-covered greenhouse.
A candidate is dropped if RASTER_OVERLAP_FRAC or more of its area falls on
any of these.

Input:
  pipeline/step3_osm_filter/output/<period>/pools_<period>.gpkg         from step 3
  pipeline/step3b_worldcover_filter/data/worldcover_clipped.tif         from download_worldcover.py
  pipeline/step3b_worldcover_filter/data/pcg_greenhouse_clipped.tif     from download_pcg_greenhouses.py
Output (pipeline/step3b_worldcover_filter/output/<period>/):
  pools_<period>.gpkg     candidates that survived
  removed_<period>.gpkg   candidates that were removed, with `reason`

Usage: python 03b_filter_worldcover.py [periods...]  (default: every folder in step3 output)
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
STEP3_OUT = PIPELINE / "step3_osm_filter" / "output"
OUT_ROOT = HERE / "output"

RASTER_OVERLAP_FRAC = 0.5
# A candidate sitting in a narrow gap between greenhouse/warehouse rows (not
# on a roof itself) can slip past the direct check above: its own footprint
# reads as bare ground, but it's still an optical artifact from reflective
# plastic/metal roofs nearby, not a real pool. Catch those by also checking
# the candidate's surroundings (CONTEXT_BUFFER_M) - if that's dominated by
# built-up/greenhouse, drop it too.
CONTEXT_BUFFER_M = 20.0
CONTEXT_OVERLAP_FRAC = 0.6
# raster path -> {pixel value: reason name}
RASTERS = {
    HERE / "data" / "worldcover_clipped.tif": {40: "cropland", 50: "built_up"},
    HERE / "data" / "pcg_greenhouse_clipped.tif": {1: "greenhouse_pcg"},
}


def classify_candidate(src, geom, codes):
    """Fraction of geom's area on each of `codes`, via a windowed read."""
    try:
        window = from_bounds(*geom.bounds, transform=src.transform)
    except Exception:
        return {}
    window = window.round_offsets().round_lengths()
    if window.width <= 0 or window.height <= 0:
        return {}
    try:
        out_image, out_transform = mask(src, [geom.__geo_interface__], crop=True,
                                         all_touched=True, filled=True, nodata=0)
    except Exception:
        return {}
    band = out_image[0]
    total = band.size
    if total == 0:
        return {}
    return {code: float(np.sum(band == code)) / total for code in codes}


def filter_period(period):
    stage3 = STEP3_OUT / period / f"pools_{period}.gpkg"
    if not stage3.exists():
        print(f"{period}: no step3 output found, skipping", flush=True)
        return
    cands = gpd.read_file(stage3)
    print(f"{period}: stage 3 candidates {len(cands)}", flush=True)
    out = OUT_ROOT / period
    out.mkdir(parents=True, exist_ok=True)

    reasons = [""] * len(cands)
    for raster_path, codes in RASTERS.items():
        if not raster_path.exists():
            print(f"  WARNING: {raster_path} not found, skipping that raster", flush=True)
            continue
        with rasterio.open(raster_path) as src:
            cands_raster_crs = cands.to_crs(src.crs)
            for i, geom in enumerate(cands_raster_crs.geometry):
                if reasons[i]:
                    continue
                fracs = classify_candidate(src, geom, codes)
                for code, name in codes.items():
                    if fracs.get(code, 0.0) >= RASTER_OVERLAP_FRAC:
                        reasons[i] = name
                        break

    # second pass: candidates that survived the direct check, but whose
    # surroundings are mostly built-up/greenhouse (see CONTEXT_BUFFER_M above).
    # Buffer in `cands.crs` (projected, meters) before reprojecting to the
    # raster's CRS (geographic, degrees) - buffering directly in degrees
    # would make CONTEXT_BUFFER_M meaningless.
    context_geoms = cands.geometry.buffer(CONTEXT_BUFFER_M)
    for raster_path, codes in RASTERS.items():
        if not raster_path.exists():
            continue
        with rasterio.open(raster_path) as src:
            context_crs = gpd.GeoSeries(context_geoms, crs=cands.crs).to_crs(src.crs)
            for i, geom in enumerate(context_crs):
                if reasons[i]:
                    continue
                fracs = classify_candidate(src, geom, codes)
                for code, name in codes.items():
                    if fracs.get(code, 0.0) >= CONTEXT_OVERLAP_FRAC:
                        reasons[i] = f"{name}_context"
                        break
    cands["reason"] = reasons

    all_names = sorted({name for codes in RASTERS.values() for name in codes.values()})
    all_names += [f"{name}_context" for name in all_names]
    for name in all_names:
        print(f"{name}: {(cands['reason'] == name).sum()}", flush=True)

    removed = cands[cands["reason"] != ""]
    final = cands[cands["reason"] == ""].drop(columns="reason")
    removed.to_file(out / f"removed_{period}.gpkg", driver="GPKG")
    final.to_file(out / f"pools_{period}.gpkg", driver="GPKG")
    print(f"{period}: final {len(final)}  removed {len(removed)}", flush=True)


def main():
    periods = sys.argv[1:] or sorted(p.name for p in STEP3_OUT.iterdir() if p.is_dir())
    failed = []
    for period in periods:
        try:
            filter_period(period)
        except Exception as e:
            print(f"{period}: FAILED ({e}), continuing to next period", flush=True)
            failed.append(period)
    if failed:
        print(f"\nFailed periods (rerun these): {failed}", flush=True)


if __name__ == "__main__":
    main()
