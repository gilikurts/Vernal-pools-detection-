"""
Bulk, one-time download of all OSM obstacle layers (buildings, greenhouses,
industrial/power-plant land, construction sites, quarries, pools, roads,
waterways) over the whole Planet-coverage bbox, tiled on a 0.1-degree grid
and fetched with a few workers in parallel. Replaces the old per-run,
per-candidate Overpass queries in osm_obstacles_for_candidates.py: that
script re-queried Overpass from scratch on every single 03_filter_osm.py
run (and per period), which is what was timing out against the public
Overpass API. This downloads the same categories once, as two layers:

  data/osm_obstacles_bulk.gpkg   polygons, column `kind` = building |
                                  greenhouse | industrial | construction |
                                  quarry | pool
  data/osm_roads_bulk.gpkg       lines, column `kind` = road | waterway

Resumable: progress is checkpointed per tile (atomic write) so a crash or
Ctrl-C only loses the in-flight tile, not the whole run.

Usage: python osm_obstacles_bulk.py [--workers N]
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
from shapely.geometry import LineString, Polygon

DATA = Path(__file__).resolve().parent / "data"
CKPT = DATA / "osm_obstacles_bulk_ckpt.pkl"
BBOX = (34.190641, 31.389999, 35.439727, 32.922476)  # same as PLANET_BBOX_WGS84
TILE_DEG = 0.1
RETRIES = 3  # fail fast and move to the next tile; rerun the script later to retry stragglers

STMTS_TEMPLATE = [
    'way["building"]({box});',
    'way["building"="greenhouse"]({box});',
    'way["landuse"="greenhouse_horticulture"]({box});',
    'way["man_made"="greenhouse"]({box});',
    'way["landuse"="industrial"]({box});',
    'way["power"="plant"]({box});',
    'way["power"="substation"]({box});',
    'way["man_made"="works"]({box});',
    'way["landuse"="construction"]({box});',
    'way["landuse"="quarry"]({box});',
    'way["leisure"="swimming_pool"]({box});',
    'way["leisure"="water_park"]({box});',
    'way["highway"]({box});',
    'way["waterway"]({box});',
]


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
    polys, polys_kind, lines, lines_kind = [], [], [], []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
        if len(coords) < 2:
            continue
        closed = coords[0] == coords[-1] and len(coords) >= 4
        is_greenhouse = (tags.get("building") == "greenhouse" or tags.get("landuse") == "greenhouse_horticulture"
                          or tags.get("man_made") == "greenhouse")
        is_industrial = (tags.get("landuse") == "industrial" or tags.get("power") in ("plant", "substation")
                          or tags.get("man_made") == "works")
        is_construction = tags.get("landuse") == "construction"
        is_quarry = tags.get("landuse") == "quarry"
        is_pool = tags.get("leisure") in ("swimming_pool", "water_park")
        is_area = closed and ("building" in tags or is_greenhouse or is_industrial or is_construction or is_quarry or is_pool)
        if is_area:
            kind = ("greenhouse" if is_greenhouse else "industrial" if is_industrial
                    else "construction" if is_construction else "quarry" if is_quarry
                    else "pool" if is_pool else "building")
            poly = Polygon(coords)
            if poly.is_valid and poly.area > 0:
                polys.append(poly)
                polys_kind.append(kind)
        elif "highway" in tags:
            lines.append(LineString(coords))
            lines_kind.append("road")
        elif "waterway" in tags:
            lines.append(LineString(coords))
            lines_kind.append("waterway")
    return polys, polys_kind, lines, lines_kind


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
    ap.add_argument("--workers", type=int, default=3)
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
                continue  # gave up after retries - leave it out of `done` so a resume retries it
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

    all_elements = [el for tile_elements in results.values() for el in tile_elements]
    print(f"total raw elements: {len(all_elements)}", flush=True)
    polys, polys_kind, lines, lines_kind = elements_to_rows(all_elements)

    gpd.GeoDataFrame({"kind": polys_kind}, geometry=polys, crs="EPSG:4326").to_file(
        DATA / "osm_obstacles_bulk.gpkg", driver="GPKG")
    gpd.GeoDataFrame({"kind": lines_kind}, geometry=lines, crs="EPSG:4326").to_file(
        DATA / "osm_roads_bulk.gpkg", driver="GPKG")
    print(f"obstacles: {len(polys)} polygons, roads/waterways: {len(lines)} lines", flush=True)


if __name__ == "__main__":
    main()
