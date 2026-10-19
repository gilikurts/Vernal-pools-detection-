"""
Bulk download of broad OSM landuse zones (residential, commercial, retail,
port/harbour) plus military land and wastewater treatment plants, tiled the
same way as osm_obstacles_bulk.py. These are zone-level polygons (a whole
neighbourhood, industrial park, port, or base), not individual building
footprints - added after visual review showed a lot of false-positive pools
sitting in the gaps between buildings across entire built-up areas (city
blocks, hospital/school campuses, ports), not just on top of one building.

Appends its results (with a `category` column) onto the existing
pipeline/step3_osm_filter/data/osm_unified_filter_layer.gpkg instead of
creating a separate file.

Usage: python osm_landuse_zones_bulk.py [--workers N]
"""
import argparse
import json
import os
import pickle
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import geopandas as gpd
from shapely.geometry import Polygon

DATA = Path(__file__).resolve().parent / "data"
CKPT = DATA / "osm_landuse_zones_ckpt.pkl"
UNIFIED = DATA / "osm_unified_filter_layer.gpkg"
BBOX = (34.190641, 31.389999, 35.439727, 32.922476)
TILE_DEG = 0.1
RETRIES = 3

STMTS_TEMPLATE = [
    'way["landuse"="residential"]({box});',
    'way["landuse"="commercial"]({box});',
    'way["landuse"="retail"]({box});',
    'way["landuse"="port"]({box});',
    'way["landuse"="harbour"]({box});',
    'way["landuse"="military"]({box});',
    'way["military"]({box});',
    'way["man_made"="wastewater_plant"]({box});',
]
TAG_TO_CATEGORY = {
    ("landuse", "residential"): "residential",
    ("landuse", "commercial"): "commercial",
    ("landuse", "retail"): "retail",
    ("landuse", "port"): "port",
    ("landuse", "harbour"): "port",
    ("landuse", "military"): "military",
    ("man_made", "wastewater_plant"): "wastewater",
}


def tiles_for(bbox):
    minx, miny, maxx, maxy = bbox
    out = []
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            out.append((x, y, min(x + TILE_DEG, maxx), min(y + TILE_DEG, maxy)))
            x += TILE_DEG
        y += TILE_DEG
    return out


def classify(tags):
    for (k, v), category in TAG_TO_CATEGORY.items():
        if tags.get(k) == v:
            return category
    if "military" in tags:
        return "military"
    return None


def fetch_tile(tminx, tminy, tmaxx, tmaxy):
    box = f"{tminy},{tminx},{tmaxy},{tmaxx}"
    stmts = [s.format(box=box) for s in STMTS_TEMPLATE]
    query = "[out:json][timeout:180];(" + "".join(stmts) + ");out geom;"
    body = ("data=" + urllib.parse.quote(query)).encode()
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(
                "https://overpass-api.de/api/interpreter", data=body,
                headers={"User-Agent": "research-script/1.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=200) as resp:
                return json.loads(resp.read())["elements"]
        except Exception as e:
            print(f"  tile ({tminx:.1f},{tminy:.1f}) attempt {attempt} failed: {e}", flush=True)
            time.sleep(20 * attempt)
    print(f"  tile ({tminx:.1f},{tminy:.1f}) gave up after {RETRIES} attempts, skipping", flush=True)
    return None


def elements_to_rows(elements):
    polys, cats = [], []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        category = classify(tags)
        if category is None:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
        if len(coords) < 4:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        if poly.is_valid and poly.area > 0:
            polys.append(poly)
            cats.append(category)
    return polys, cats


def save_checkpoint(done, results):
    tmp = CKPT.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"done": done, "results": results}, f)
    os.replace(tmp, CKPT)


def load_checkpoint():
    if not CKPT.exists():
        return set(), {}
    try:
        with open(CKPT, "rb") as f:
            state = pickle.load(f)
        return state["done"], state["results"]
    except Exception as e:
        print(f"checkpoint unreadable ({e}), starting fresh", flush=True)
        return set(), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    tiles = tiles_for(BBOX)
    print(f"{len(tiles)} tiles, {args.workers} workers", flush=True)

    done, results = load_checkpoint()
    todo = [t for t in tiles if t not in done]
    print(f"{len(done)} tiles already done, {len(todo)} remaining", flush=True)

    lock = Lock()
    completed_count = len(done)

    def worker(tile):
        elements = fetch_tile(*tile)
        return tile, elements

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, t): t for t in todo}
        for fut in as_completed(futures):
            tile, elements = fut.result()
            if elements is None:
                continue
            with lock:
                done.add(tile)
                results[tile] = elements
                completed_count += 1
                if completed_count % 5 == 0 or completed_count == len(tiles):
                    print(f"  {completed_count}/{len(tiles)} tiles done", flush=True)
                    save_checkpoint(done, results)

    save_checkpoint(done, results)
    if len(done) < len(tiles):
        print(f"{len(tiles) - len(done)} tiles never succeeded; rerun this script to retry just those", flush=True)
        return

    all_elements = [el for tile_elements in results.values() for el in tile_elements]
    print(f"total raw elements: {len(all_elements)}", flush=True)
    polys, cats = elements_to_rows(all_elements)
    print(f"zones: {len(polys)} polygons  ({dict((c, cats.count(c)) for c in set(cats))})", flush=True)

    new_layer = gpd.GeoDataFrame({"category": cats}, geometry=polys, crs="EPSG:4326")
    if UNIFIED.exists():
        existing = gpd.read_file(UNIFIED)
        new_layer = new_layer.to_crs(existing.crs)
        combined = gpd.GeoDataFrame(
            gpd.pd.concat([existing[["category", "geometry"]], new_layer], ignore_index=True), crs=existing.crs)
    else:
        combined = new_layer
    combined.to_file(UNIFIED, driver="GPKG")
    print(f"Appended to {UNIFIED}: now {len(combined)} features total", flush=True)


if __name__ == "__main__":
    main()
