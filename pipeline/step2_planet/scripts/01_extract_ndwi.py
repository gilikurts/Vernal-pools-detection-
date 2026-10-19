"""
Step 1: DEM depression candidates x Planet scenes -> NDWI cache.

Two cases, one cache:
- regular candidates (area <= GIANT_AREA_M2): cropped from the single scene whose
  footprint contains the candidate's centroid (one scene is enough)
- giant candidates (area > GIANT_AREA_M2): for each date, every scene that overlaps
  the polygon is merged onto one canvas covering the polygon's full extent, then
  masked to the polygon, so the whole polygon is seen

Inputs:
  pipeline/step1_depressions/output/depressions_clean.gpkg
  <PLANET_RAW>/<period>/ (monthly ZIPs and the extracted january_2025/ folder)

Usage: python 01_extract_ndwi.py [2024-2025 | 2025-2026] [chunk_size]
  chunk_size: if given, process at most this many NEW scenes+giant-polygons
  then checkpoint and exit (prints CHUNK_DONE if more work remains, ALL_DONE
  if nothing was left). Run the same command again to continue, or use
  run_chunked.sh to loop automatically. Keeps peak memory low by letting the
  OS reclaim everything between short-lived processes, instead of one long
  run accumulating memory until it gets killed.

Outputs (in pipeline/step2_planet/cache/<period>/):
  candidates_in_planet.gpkg  candidates kept inside Planet coverage; `idx` is the key below
  progress.pkl               done_scenes/done_giant sets only (tiny) - the sole thing read
                              back into memory on resume, so restarting never re-holds
                              everything ever produced (that was the old design, and the
                              cause of repeated OOM kills on long/many-giant periods: each
                              restart started already holding the full history, then grew
                              further before the next kill)
  ndwi_delta.pkl              the actual data: an ever-growing append-only log of
                              {idx: {date: 2D NDWI array}} batches, one per checkpoint (every
                              20 scenes / 25 giant polygons). Never read back into memory
                              during extraction - only once, after everything is done, by
                              02_score_candidates.py.
"""
import os
import pickle
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.ops import transform as shapely_transform

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]
CANDIDATES = PIPELINE / "step1_depressions/output/depressions_clean.gpkg"
PERIOD = sys.argv[1] if len(sys.argv) > 1 else "2024-2025"
CHUNK_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else None
PLANET_RAW_ROOT = Path(os.environ.get("PLANET_RAW_ROOT", "/Volumes/ElementsData/menasim_volonteer/data/planet_raw"))
PLANET_RAW = PLANET_RAW_ROOT / PERIOD
CACHE = HERE.parent / "cache" / PERIOD
CACHE.mkdir(parents=True, exist_ok=True)


class Budget:
    def __init__(self, n):
        self.n = n

    def tick(self):
        if self.n is not None:
            self.n -= 1

    def done(self):
        return self.n is not None and self.n <= 0

def record(delta, idx, date, arr):
    """Write into the since-last-checkpoint delta. checkpoint() persists and
    clears it every 20-25 items, so this never accumulates beyond that - the
    full historical dataset lives on disk in ndwi_delta.pkl, never in RAM
    during extraction (see main() for why that matters)."""
    delta.setdefault(idx, {})[date] = arr


SCL_CLEAR_UDM2 = 1
READ_RETRIES = 4
READ_RETRY_SLEEP = 10
GIANT_AREA_M2 = 500_000
PIXEL_SIZE = 3.0
PLANET_CRS = "EPSG:32636"
MAX_CANVAS_PIXELS = 4_000_000


def gather_scene_sources(raw_dir):
    sources = []
    jan_dir = raw_dir / "january_2025" / "PSScene"
    for f in sorted(jan_dir.glob("*_3B_AnalyticMS_SR*.tif")):
        clip = f.name.endswith("_clip.tif")
        sr_suffix = "_3B_AnalyticMS_SR_clip.tif" if clip else "_3B_AnalyticMS_SR.tif"
        udm_suffix = "_3B_udm2_clip.tif" if clip else "_3B_udm2.tif"
        sources.append({"date": f.name[:8], "sr": str(f), "udm2": str(f.with_name(f.name.replace(sr_suffix, udm_suffix)))})

    for z in sorted(raw_dir.glob("Israeli_Coastal_Plain_*.zip")):
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
        for sr_name in [n for n in names if n.endswith("_3B_AnalyticMS_SR.tif") or n.endswith("_3B_AnalyticMS_SR_clip.tif")]:
            clip = sr_name.endswith("_clip.tif")
            udm_name = sr_name.replace("_3B_AnalyticMS_SR_clip.tif", "_3B_udm2_clip.tif") if clip \
                else sr_name.replace("_3B_AnalyticMS_SR.tif", "_3B_udm2.tif")
            sources.append({"date": Path(sr_name).name[:8],
                            "sr": f"/vsizip/{z}/{sr_name}", "udm2": f"/vsizip/{z}/{udm_name}"})
    return sources


def scene_wgs84_bounds(sr_path):
    with rasterio.open(sr_path) as ds:
        b = ds.bounds
        tr = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
    lo1, la1 = tr.transform(b.left, b.bottom)
    lo2, la2 = tr.transform(b.right, b.top)
    return min(lo1, lo2), min(la1, la2), max(lo1, lo2), max(la1, la2)


def ndwi_from_scene(src):
    for attempt in range(1, READ_RETRIES + 1):
        try:
            with rasterio.open(src["sr"]) as sr_ds, rasterio.open(src["udm2"]) as udm_ds:
                green = sr_ds.read(2).astype(np.float32)
                nir = sr_ds.read(4).astype(np.float32)
                clear = udm_ds.read(1) == SCL_CLEAR_UDM2
                valid = (green > 0) & (nir > 0) & clear
                ndwi = np.full(green.shape, np.nan, dtype=np.float32)
                ndwi[valid] = (green[valid] - nir[valid]) / (green[valid] + nir[valid])
                return ndwi, sr_ds.transform, sr_ds.crs
        except Exception:
            if attempt == READ_RETRIES:
                raise
            time.sleep(READ_RETRY_SLEEP)


def extract_regular(candidates_wgs, lon, lat, sources, delta, done_scenes, checkpoint, budget):
    processed = 0
    for n, src in enumerate(sources, 1):
        if src["sr"] in done_scenes:
            continue
        if budget.done():
            checkpoint()
            return True
        try:
            b = scene_wgs84_bounds(src["sr"])
            inside = (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])
            if not inside.any():
                done_scenes.add(src["sr"])
                processed += 1
                if processed % 20 == 0:
                    print(f"  regular: scene {n}/{len(sources)}", flush=True)
                    checkpoint()
                continue
            ndwi, transform, crs = ndwi_from_scene(src)

            to_scene = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            for idx, geom_wgs in zip(candidates_wgs["idx"].values[inside], candidates_wgs.geometry.values[inside]):
                geom = shapely_transform(to_scene.transform, geom_wgs)
                try:
                    win = from_bounds(*geom.bounds, transform=transform)
                except Exception:
                    continue
                r0, c0 = max(0, int(np.floor(win.row_off))), max(0, int(np.floor(win.col_off)))
                r1 = min(ndwi.shape[0], int(np.ceil(win.row_off + win.height)))
                c1 = min(ndwi.shape[1], int(np.ceil(win.col_off + win.width)))
                if r1 <= r0 or c1 <= c0:
                    continue
                sub = ndwi[r0:r1, c0:c1]
                sub_transform = rasterio.transform.from_origin(
                    transform.c + c0 * transform.a, transform.f + r0 * transform.e,
                    transform.a, -transform.e)
                mask = geometry_mask([geom], transform=sub_transform, out_shape=sub.shape, invert=True)
                cropped = np.where(mask, sub, np.nan)
                if np.sum(~np.isnan(cropped)) < 3:
                    continue
                record(delta, int(idx), src["date"], cropped)
            done_scenes.add(src["sr"])
            budget.tick()
            processed += 1
            if processed % 20 == 0:
                print(f"  regular: scene {n}/{len(sources)}", flush=True)
                checkpoint()
        except Exception as e:
            print(f"  ERROR {src['sr']}: {e}", flush=True)
    return False


def extract_giant(giant_utm, giant_wgs, sources, bboxes, delta, done_giant, checkpoint, budget):
    processed = 0
    for n, (idx, poly_utm, poly_wgs) in enumerate(zip(giant_utm["idx"], giant_utm.geometry, giant_wgs.geometry), 1):
        if int(idx) in done_giant:
            continue
        if budget.done():
            checkpoint()
            return True
        minx, miny, maxx, maxy = poly_utm.bounds
        origin_x = np.floor(minx / PIXEL_SIZE) * PIXEL_SIZE
        origin_y = np.ceil(maxy / PIXEL_SIZE) * PIXEL_SIZE
        width = int(np.ceil((maxx - origin_x) / PIXEL_SIZE))
        height = int(np.ceil((origin_y - miny) / PIXEL_SIZE))
        if width * height > MAX_CANVAS_PIXELS:
            print(f"  skip giant {idx}: canvas too large ({width}x{height})", flush=True)
            done_giant.add(int(idx))
            processed += 1
            if processed % 25 == 0:
                print(f"  giant: polygon {n}/{len(giant_utm)}", flush=True)
                checkpoint()
            continue
        canvas_transform = rasterio.transform.from_origin(origin_x, origin_y, PIXEL_SIZE, PIXEL_SIZE)
        poly_mask = geometry_mask([poly_utm], transform=canvas_transform, out_shape=(height, width), invert=True)

        wx0, wy0, wx1, wy1 = poly_wgs.bounds
        overlapping = [s for s, b in zip(sources, bboxes)
                       if not (b[2] < wx0 or b[0] > wx1 or b[3] < wy0 or b[1] > wy1)]

        per_date = {}
        for src in overlapping:
            read_ok = False
            for attempt in range(1, READ_RETRIES + 1):
                try:
                    with rasterio.open(src["sr"]) as sr_ds, rasterio.open(src["udm2"]) as udm_ds:
                        win = from_bounds(minx, miny, maxx, maxy, transform=sr_ds.transform)
                        win = win.round_offsets().round_lengths()
                        win = win.intersection(Window(0, 0, sr_ds.width, sr_ds.height))
                        if win.width <= 0 or win.height <= 0:
                            read_ok = None
                            break
                        green = sr_ds.read(2, window=win).astype(np.float32)
                        nir = sr_ds.read(4, window=win).astype(np.float32)
                        clear = udm_ds.read(1, window=win) == SCL_CLEAR_UDM2
                        win_transform = sr_ds.transform * rasterio.Affine.translation(win.col_off, win.row_off)
                        read_ok = True
                        break
                except Exception:
                    if attempt == READ_RETRIES:
                        read_ok = False
                    else:
                        time.sleep(READ_RETRY_SLEEP)
            if not read_ok:
                continue
            valid = (green > 0) & (nir > 0) & clear
            ndwi = np.full(green.shape, np.nan, dtype=np.float32)
            ndwi[valid] = (green[valid] - nir[valid]) / (green[valid] + nir[valid])

            dst_col = int(round((win_transform.c - origin_x) / PIXEL_SIZE))
            dst_row = int(round((origin_y - win_transform.f) / PIXEL_SIZE))
            canvas = per_date.setdefault(src["date"], np.full((height, width), np.nan, dtype=np.float32))
            sh, sw = ndwi.shape
            r0, c0 = max(0, dst_row), max(0, dst_col)
            r1, c1 = min(height, dst_row + sh), min(width, dst_col + sw)
            if r1 <= r0 or c1 <= c0:
                continue
            block = ndwi[r0 - dst_row:r1 - dst_row, c0 - dst_col:c1 - dst_col]
            region = canvas[r0:r1, c0:c1]
            fill = np.isnan(region) & ~np.isnan(block)
            region[fill] = block[fill]
            canvas[r0:r1, c0:c1] = region

        for date, canvas in per_date.items():
            masked = np.where(poly_mask, canvas, np.nan)
            if np.sum(~np.isnan(masked)) >= 3:
                record(delta, int(idx), date, masked)
        done_giant.add(int(idx))
        budget.tick()
        processed += 1
        if processed % 25 == 0:
            print(f"  giant: polygon {n}/{len(giant_utm)}", flush=True)
            checkpoint()
    return False


def main():
    candidates = gpd.read_file(CANDIDATES)
    candidates["dep_id"] = candidates.index
    candidates = candidates.reset_index(drop=True)
    sources = gather_scene_sources(PLANET_RAW)
    print(f"Candidates: {len(candidates)}  Scenes: {len(sources)}", flush=True)

    bboxes = [scene_wgs84_bounds(s["sr"]) for s in sources]
    union = (min(b[0] for b in bboxes), min(b[1] for b in bboxes),
             max(b[2] for b in bboxes), max(b[3] for b in bboxes))

    c_wgs = candidates.to_crs("EPSG:4326")
    lon = c_wgs.geometry.centroid.x.values
    lat = c_wgs.geometry.centroid.y.values
    in_union = (lon >= union[0]) & (lon <= union[2]) & (lat >= union[1]) & (lat <= union[3])
    candidates = candidates[in_union].reset_index(drop=True)
    candidates["idx"] = candidates.index
    print(f"Candidates inside Planet coverage: {len(candidates)}", flush=True)
    candidates.to_file(CACHE / "candidates_in_planet.gpkg", driver="GPKG")

    candidates_utm = candidates.to_crs(PLANET_CRS)
    is_giant = (candidates_utm.geometry.area > GIANT_AREA_M2).values
    print(f"Giant: {is_giant.sum()}  regular: {(~is_giant).sum()}", flush=True)

    delta_path = CACHE / "ndwi_delta.pkl"
    progress_path = CACHE / "progress.pkl"
    done_scenes, done_giant = set(), set()
    if progress_path.exists():
        try:
            with open(progress_path, "rb") as f:
                progress = pickle.load(f)
            done_scenes, done_giant = progress["done_scenes"], progress["done_giant"]
            print(f"Resuming: {len(done_scenes)} scenes and {len(done_giant)} giant polygons already done "
                  f"(their data already lives in {delta_path.name}, not reloaded into memory here)", flush=True)
        except Exception as e:
            print(f"Checkpoint file unreadable ({e}), starting fresh", flush=True)
            done_scenes, done_giant = set(), set()

    # `delta` only ever holds what THIS run has produced since its last
    # checkpoint (cleared after every checkpoint() call) - never the full
    # historical dataset. That dataset lives permanently in ndwi_delta.pkl
    # (an ever-growing append-only log of small batches) and is never loaded
    # back into memory during extraction, only once, later, by
    # 02_score_candidates.py. Reloading full history into RAM on every
    # resume (the previous design) is what caused repeated OOM kills on
    # long/many-giant periods: each restart started already holding
    # everything ever produced, then grew further before the next kill.
    delta = {}

    def checkpoint():
        if delta:
            with open(delta_path, "ab") as f:
                pickle.dump(delta, f)
            delta.clear()

        tmp_progress = progress_path.with_suffix(".tmp")
        with open(tmp_progress, "wb") as f:
            pickle.dump({"done_scenes": done_scenes, "done_giant": done_giant}, f)
        os.replace(tmp_progress, progress_path)

    budget = Budget(CHUNK_SIZE)

    regular = candidates[~is_giant]
    c_reg = regular.to_crs("EPSG:4326")
    truncated = extract_regular(c_reg, c_reg.geometry.centroid.x.values, c_reg.geometry.centroid.y.values,
                                 sources, delta, done_scenes, checkpoint, budget)
    checkpoint()

    if not truncated:
        giant_utm = candidates_utm[is_giant]
        giant_wgs = candidates[is_giant].to_crs("EPSG:4326")
        truncated = extract_giant(giant_utm, giant_wgs, sources, bboxes, delta, done_giant, checkpoint, budget)
        checkpoint()

    if not truncated:
        print(f"All scenes and giant polygons done. Data lives in {delta_path} "
              f"(read by 02_score_candidates.py).", flush=True)
    if CHUNK_SIZE is not None:
        print("CHUNK_DONE" if truncated else "ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
