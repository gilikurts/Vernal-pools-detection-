"""
Step 2: persistence scoring on each period's NDWI cache, then vectorize the wet core.

Scores every period folder under cache/ (or the ones given on the command line),
so one run covers all years. Each output row carries its period, so a pool seen
in two different winters appears once per winter.

Inputs (pipeline/step2_planet/cache/<period>/):
  candidates_in_planet.gpkg  from step 1 (has dep_id, the stable depression id)
  ndwi_delta.pkl             from step 1, an append-only log of {idx: {date: array}}
                              batches - merged into one dict here, in memory, once

Output:
  pipeline/step2_planet/output/<period>/pools_<period>.gpkg
  one row per wet component; geometry = wet core + 6 m buffer (EPSG:2039)

Usage: python 02_score_candidates.py [2024-2025 2025-2026 ...]
"""
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "cache"
OUTPUT_ROOT = HERE.parent / "output"

RISE_THRESHOLD = 0.10
ABS_WET_FLOOR = -0.20
SUMMER_POSITIVE_FRAC_CAP = 0.10
MIN_REPEAT = 3
DILATE_PIXELS = 1  # spatial tolerance for "same spot" across dates, in pixels (1px = 3m)
MIN_CORE_PIXELS = 10
WET_ABS_MARGIN = 0.10
PIXEL_SIZE = 3.0
PIXEL_AREA_M2 = PIXEL_SIZE * PIXEL_SIZE
BUFFER_M = 6.0
SUMMER_MONTHS = ("07", "08")
WINTER_MONTHS = ("10", "11", "12", "01", "02", "03", "04")


def is_summer(date_str):
    return date_str[4:6] in SUMMER_MONTHS


def is_winter(date_str):
    return date_str[4:6] in WINTER_MONTHS


def score_candidate(per_date):
    summer = [d for d in per_date if is_summer(d)]
    winter = [d for d in per_date if is_winter(d)]
    if not summer or len(winter) < 3:
        return None
    shape_arr = per_date[summer[0]].shape
    if any(per_date[d].shape != shape_arr for d in summer + winter):
        return None

    with np.errstate(invalid="ignore"):
        baseline = np.nanmean(np.stack([per_date[d] for d in summer]), axis=0)
    base_valid = baseline[~np.isnan(baseline)]
    if base_valid.size < 3 or np.mean(base_valid > 0) > SUMMER_POSITIVE_FRAC_CAP:
        return None

    count = np.zeros(shape_arr, dtype=int)
    for d in winter:
        img = per_date[d]
        diff = img - baseline
        wet_today = (~np.isnan(diff) & (diff > RISE_THRESHOLD) & (img > ABS_WET_FLOOR))
        # dilate by one pixel so a wet spot that shifts slightly day to day
        # (a real pool's edge expanding/contracting) still counts as the same
        # recurring spot, instead of needing the exact same pixel every time
        count += ndimage.binary_dilation(wet_today, iterations=DILATE_PIXELS).astype(int)

    core = count >= MIN_REPEAT
    labeled, n = ndimage.label(core)
    if n == 0:
        return None

    if core.any() and (~core).any():
        surround = baseline[~core & ~np.isnan(baseline)]
    else:
        surround = np.array([])

    components = []
    for comp_id in range(1, n + 1):
        mask = labeled == comp_id
        core_pixels = int(mask.sum())
        if core_pixels < MIN_CORE_PIXELS:
            continue
        if surround.size and np.nanmean(baseline[mask]) > np.nanmedian(surround) + WET_ABS_MARGIN:
            continue
        components.append({"comp_id": comp_id, "core_pixels": core_pixels, "mask": mask})

    return {"n_dates": len(winter), "components": components}


def core_polygon(mask, geom_utm):
    minx, _, _, maxy = geom_utm.bounds
    origin_x = np.floor(minx / PIXEL_SIZE) * PIXEL_SIZE
    origin_y = np.ceil(maxy / PIXEL_SIZE) * PIXEL_SIZE
    transform = rasterio.transform.from_origin(origin_x, origin_y, PIXEL_SIZE, PIXEL_SIZE)
    polys = [shape(g) for g, v in rio_shapes(mask.astype(np.uint8), mask=mask, transform=transform) if v == 1]
    return unary_union(polys).buffer(BUFFER_M)


N_BUCKETS = 24  # 16GB-RAM machine, periods can hold 40+GB of NDWI arrays total;
                 # bucketing by idx keeps each pass to roughly total/N_BUCKETS


def bucket_ndwi_delta(cache, bucket_dir):
    """Partition ndwi_delta.pkl (an append-only log of {idx: {date: array}}
    batches, see 01_extract_ndwi.py) into N_BUCKETS files by idx % N_BUCKETS,
    so every one candidate's data ends up in exactly one bucket. Streams
    through the log one batch at a time and flushes each bucket's own small
    buffer periodically - never holds more than a fraction of the period's
    data in memory, unlike reading it all into one dict (which doesn't fit
    in RAM for a big period and is what caused repeated OOM kills before)."""
    bucket_dir.mkdir(exist_ok=True)
    buffers = [{} for _ in range(N_BUCKETS)]
    buffer_sizes = [0] * N_BUCKETS
    FLUSH_EVERY = 2000  # entries per bucket buffer before writing out

    def flush(b):
        if not buffers[b]:
            return
        with open(bucket_dir / f"bucket_{b}.pkl", "ab") as f:
            pickle.dump(buffers[b], f)
        buffers[b].clear()
        buffer_sizes[b] = 0

    with open(cache / "ndwi_delta.pkl", "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
            except EOFError:
                break
            for idx, by_date in batch.items():
                b = idx % N_BUCKETS
                buffers[b].setdefault(idx, {}).update(by_date)
                buffer_sizes[b] += len(by_date)
                if buffer_sizes[b] >= FLUSH_EVERY:
                    flush(b)
    for b in range(N_BUCKETS):
        flush(b)


def score_period(period):
    cache = CACHE_ROOT / period
    candidates = gpd.read_file(cache / "candidates_in_planet.gpkg")
    geom_utm = dict(zip(candidates["idx"], candidates.to_crs("EPSG:32636").geometry))
    dep_id = dict(zip(candidates["idx"], candidates["dep_id"]))

    bucket_dir = cache / "score_buckets"
    print(f"{period}: bucketing ndwi_delta.pkl into {N_BUCKETS} pieces...", flush=True)
    bucket_ndwi_delta(cache, bucket_dir)

    rows = []
    n_scored = 0
    for b in range(N_BUCKETS):
        bucket_path = bucket_dir / f"bucket_{b}.pkl"
        if not bucket_path.exists():
            continue
        ndwi = {}
        with open(bucket_path, "rb") as f:
            while True:
                try:
                    batch = pickle.load(f)
                except EOFError:
                    break
                for idx, by_date in batch.items():
                    ndwi.setdefault(idx, {}).update(by_date)
        for idx, per_date in ndwi.items():
            res = score_candidate(per_date)
            if res is None:
                continue
            n_scored += 1
            for comp in res["components"]:
                rows.append({
                    "period": period,
                    "dep_id": int(dep_id[idx]),
                    "idx": int(idx),
                    "sub_id": comp["comp_id"],
                    "core_area_m2": comp["core_pixels"] * PIXEL_AREA_M2,
                    "n_dates": res["n_dates"],
                    "geometry": core_polygon(comp["mask"], geom_utm[idx]),
                })
        ndwi = None  # drop this bucket's data before moving to the next

    import shutil
    shutil.rmtree(bucket_dir, ignore_errors=True)

    print(f"{period}: scored {n_scored}, pools passing {len(rows)}", flush=True)
    return rows


def main():
    periods = sys.argv[1:] or sorted(p.name for p in CACHE_ROOT.iterdir()
                                     if p.is_dir() and (p / "ndwi_delta.pkl").exists())
    print(f"Periods: {periods}", flush=True)

    rows = []
    for period in periods:
        rows.extend(score_period(period))

    for period in periods:
        part = [r for r in rows if r["period"] == period]
        out_dir = OUTPUT_ROOT / period
        out_dir.mkdir(parents=True, exist_ok=True)
        out = gpd.GeoDataFrame(part, geometry="geometry", crs="EPSG:32636").to_crs("EPSG:2039")
        out_path = out_dir / f"pools_{period}.gpkg"
        out.to_file(out_path, driver="GPKG")
        print(f"Saved {len(out)} rows: {out_path}", flush=True)


if __name__ == "__main__":
    main()
