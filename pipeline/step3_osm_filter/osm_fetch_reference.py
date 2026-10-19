"""
Download the reference geometry used by the OSM cross-check step:
- Israel and West Bank boundaries (simplified, public GeoJSON)
- Natural Earth 10m land and lakes (Sea of Galilee lives in lakes)
- OSM water bodies over the study area (Overpass)
Writes everything into step3_osm_filter/data/.
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)

DOWNLOADS = {
    "israel_boundary.geojson": "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/ISR.geo.json",
    "palestine_boundary.geojson": "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/PSE.geo.json",
    "ne_10m_land.geojson": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson",
    "ne_10m_lakes.geojson": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_lakes.geojson",
}

for name, url in DOWNLOADS.items():
    dest = DATA / name
    if dest.exists():
        continue
    urllib.request.urlretrieve(url, dest)
    print(f"downloaded {name}", flush=True)

OVERPASS_QUERY = """[out:json][timeout:180];
(
way["natural"="water"](29.4,34.0,33.5,36.0);
way["landuse"="reservoir"](29.4,34.0,33.5,36.0);
way["landuse"="basin"](29.4,34.0,33.5,36.0);
way["water"](29.4,34.0,33.5,36.0);
relation["natural"="water"](29.4,34.0,33.5,36.0);
relation["landuse"="reservoir"](29.4,34.0,33.5,36.0);
);
out geom;"""

req = urllib.request.Request(
    "https://overpass-api.de/api/interpreter",
    data=("data=" + urllib.parse.quote(OVERPASS_QUERY)).encode(),
    headers={"User-Agent": "research-script/1.0", "Accept": "application/json"},
)
with urllib.request.urlopen(req, timeout=180) as resp:
    osm = json.loads(resp.read())

polys = []
for el in osm["elements"]:
    if el.get("type") == "way" and "geometry" in el:
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        if poly.is_valid and poly.area > 0:
            polys.append(poly)
    elif el.get("type") == "relation":
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

gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326").to_file(DATA / "osm_water.gpkg", driver="GPKG")
print(f"osm_water.gpkg: {len(polys)} polygons", flush=True)
