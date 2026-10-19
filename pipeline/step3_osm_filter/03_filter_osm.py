"""
Step 3: one pass over the stage 2 candidates, dropping water, greenhouses,
industrial/power-plant land, construction sites, buildings, roads and
waterways (streams/rivers). Each removed candidate gets a `reason` (first
match wins). Water uses an area-overlap-fraction threshold (not plain
intersects) so a candidate that only clips a thin river-channel water
polygon at the edge survives; everything else still uses intersects.

Inputs:
  pipeline/step2_planet/output/<period>/pools_<period>.gpkg        from step 2
  pipeline/step3_osm_filter/data/osm_unified_filter_layer.gpkg     from step1_depressions/clean_depressions.py
    (water, building, greenhouse, industrial, construction, quarry, pool, and
    already-buffered road/waterway polygons, one `category` column)
Outputs (pipeline/step3_osm_filter/output/<period>/):
  pools_<period>.gpkg     candidates that survived all four checks
  removed_<period>.gpkg   candidates that were removed, with `reason`

Usage: python 03_filter_osm.py [periods...]  (default: every folder in step2 output)
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

HERE = Path(__file__).resolve().parent
PLANET = HERE.parent / "step2_planet"
DATA = HERE / "data"
OUT_ROOT = HERE / "output"
WATER_OVERLAP_FRAC = 0.5  # candidate must have most of its area inside OSM water to be dropped

_unified = None


def load_unified_layer(crs):
    global _unified
    if _unified is None:
        _unified = gpd.read_file(DATA / "osm_unified_filter_layer.gpkg").to_crs(crs)
    return _unified


def filter_period(period):
    stage2 = PLANET / "output" / period / f"pools_{period}.gpkg"
    cands = gpd.read_file(stage2)
    print(f"{period}: stage 2 candidates {len(cands)}", flush=True)
    out = OUT_ROOT / period
    out.mkdir(parents=True, exist_ok=True)
    cands["reason"] = ""

    unified = load_unified_layer(cands.crs)

    water_union = unified[unified["category"] == "water"].geometry.union_all()
    frac_in_water = cands.geometry.intersection(water_union).area / cands.geometry.area
    hit = (frac_in_water >= WATER_OVERLAP_FRAC).values
    cands.loc[hit, "reason"] = "water"
    print(f"water: {hit.sum()}", flush=True)

    for kind in ["greenhouse", "industrial", "construction", "quarry", "pool", "building", "road", "waterway",
                 "military", "wastewater"]:
        layer = unified[unified["category"] == kind]
        if layer.empty:
            continue
        todo = (cands["reason"] == "").values
        hit = np.zeros(len(cands), dtype=bool)
        hit[todo] = cands.geometry[todo].intersects(layer.geometry.union_all()).values
        cands.loc[hit, "reason"] = kind
        print(f"{kind}: {hit.sum()}", flush=True)

    # broad landuse zones (a whole neighbourhood/industrial park/port) use the
    # same overlap-fraction rule as water: a candidate merely touching the
    # edge of a city shouldn't be dropped, only one that's mostly inside it
    for kind in ["residential", "commercial", "retail", "port"]:
        layer = unified[unified["category"] == kind]
        if layer.empty:
            continue
        todo = (cands["reason"] == "").values
        layer_union = layer.geometry.union_all()
        frac = cands.geometry[todo].intersection(layer_union).area / cands.geometry[todo].area
        hit = np.zeros(len(cands), dtype=bool)
        hit[todo] = (frac >= WATER_OVERLAP_FRAC).values
        cands.loc[hit, "reason"] = kind
        print(f"{kind}: {hit.sum()}", flush=True)

    removed = cands[cands["reason"] != ""]
    final = cands[cands["reason"] == ""].drop(columns="reason")
    removed.to_file(out / f"removed_{period}.gpkg", driver="GPKG")
    final.to_file(out / f"pools_{period}.gpkg", driver="GPKG")
    print(f"{period}: final {len(final)}  removed {len(removed)}", flush=True)


def main():
    periods = sys.argv[1:] or sorted(p.name for p in (PLANET / "output").iterdir() if p.is_dir())
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
