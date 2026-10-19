"""
Fetch OSM relations (multipolygons) for every category in the unified filter
layer. Every other fetch script in this folder (osm_obstacles_bulk.py,
osm_landuse_zones_bulk.py) only queried `way[...]`, never `relation[...]` -
but large/complex real-world areas (a whole port, a refinery, an industrial
zone, a residential neighbourhood) are very often mapped as a relation
(multipolygon) in OSM, not a single way. That silently dropped them from
every category - confirmed for Ashdod port ("נמל אשדוד"), the refineries
("בתי הזיקוק אשדוד"), and the northern industrial zone, all missing from the
pools filter despite the polygons clearly sitting inside them.

Not tiled (relations are far rarer than ways/buildings, one query per
category over the whole bbox is small enough), but still resumable: each
category is fetched and checkpointed independently.

Appends its results (with the same `category` values as the other fetch
scripts) onto pipeline/step3_osm_filter/data/osm_unified_filter_layer.gpkg.

Usage: python osm_relations_bulk.py
"""
import json
import pickle
import time
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

DATA = Path(__file__).resolve().parent / "data"
CKPT = DATA / "osm_relations_ckpt.pkl"
UNIFIED = DATA / "osm_unified_filter_layer.gpkg"
BBOX = "31.389999,34.190641,32.922476,35.439727"  # south,west,north,east
RETRIES = 4

# category -> list of Overpass relation selectors
CATEGORY_QUERIES = {
    "water": ['relation["natural"="water"]', 'relation["landuse"="reservoir"]'],
    "building": ['relation["building"]'],
    "greenhouse": ['relation["landuse"="greenhouse_horticulture"]'],
    "industrial": ['relation["landuse"="industrial"]', 'relation["power"="plant"]'],
    "construction": ['relation["landuse"="construction"]'],
    "quarry": ['relation["landuse"="quarry"]'],
    "military": ['relation["landuse"="military"]', 'relation["military"]'],
    "wastewater": ['relation["man_made"="wastewater_plant"]'],
    "residential": ['relation["landuse"="residential"]'],
    "commercial": ['relation["landuse"="commercial"]'],
    "retail": ['relation["landuse"="retail"]'],
    "port": ['relation["landuse"="port"]', 'relation["landuse"="harbour"]'],
}


def fetch_category(selectors):
    stmts = "".join(f"{sel}({BBOX});" for sel in selectors)
    query = f"[out:json][timeout:180];({stmts});out geom;"
    body = ("data=" + urllib.parse.quote(query)).encode()
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(
                "https://overpass-api.de/api/interpreter", data=body,
                headers={"User-Agent": "research-script/1.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=200) as resp:
                return json.loads(resp.read())["elements"]
        except Exception as e:
            print(f"  attempt {attempt} failed: {e}", flush=True)
            time.sleep(25 * attempt)
    print("  gave up after all retries", flush=True)
    return None


def relation_to_polygons(el):
    """Outer rings only - good enough for a 'does a pool sit inside this area' filter."""
    polys = []
    for member in el.get("members", []):
        if member.get("role") != "outer" or "geometry" not in member:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in member["geometry"]]
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        if poly.is_valid and poly.area > 0:
            polys.append(poly)
    return polys


def load_checkpoint():
    if not CKPT.exists():
        return {}
    try:
        with open(CKPT, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"checkpoint unreadable ({e}), starting fresh", flush=True)
        return {}


def save_checkpoint(done):
    tmp = CKPT.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(done, f)
    import os
    os.replace(tmp, CKPT)


def main():
    done = load_checkpoint()  # category -> list of elements

    for category, selectors in CATEGORY_QUERIES.items():
        if category in done:
            print(f"{category}: already fetched ({len(done[category])} elements), skipping", flush=True)
            continue
        print(f"fetching {category}...", flush=True)
        elements = fetch_category(selectors)
        if elements is None:
            print(f"{category}: failed, will retry on next run", flush=True)
            continue
        done[category] = elements
        print(f"{category}: {len(elements)} relations", flush=True)
        save_checkpoint(done)

    if len(done) < len(CATEGORY_QUERIES):
        missing = set(CATEGORY_QUERIES) - set(done)
        print(f"missing categories: {missing}; rerun this script to retry", flush=True)
        return

    rows = []
    for category, elements in done.items():
        for el in elements:
            if el.get("type") != "relation":
                continue
            for poly in relation_to_polygons(el):
                rows.append({"category": category, "geometry": poly})
    print(f"total relation polygons: {len(rows)}", flush=True)

    new_layer = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
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
