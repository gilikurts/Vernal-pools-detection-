"""
Cleanup of the raw DEM depression layer: geography, OSM, Planet coverage.

Input:  output/all_depressions.gpkg               (raw, from run_depression_extraction.py)
Output: output/depressions_clean.gpkg

Steps, in order:
1. Remove the Negev: centroid south of latitude 31.3.
2. Keep only what lies on land inside Israel or the West Bank.
   A polygon is kept if more than 50% of its area is inside that region;
   polygons fully inside are kept without a fraction check.
3. Remove polygons intersecting the Sea of Galilee (Kinneret).
4. Remove the 10 largest polygons by area (DEM-processing artifacts).
5. Drop depressions that sit almost entirely (>=90% of their area) inside an
   OSM water body, building, greenhouse, industrial/quarry/construction site,
   swimming pool, or a buffered road/waterway - these can never score as a
   vernal pool later (step2 needs a dry summer baseline) so there is no
   point carrying them through the slow Planet/NDWI stages. Deliberately
   conservative: only rejects near-total overlap, not a depression that
   merely touches or partially clips an OSM feature.
6. Drop depressions outside every period's actual Planet imagery footprint
   (union of every scene's bounding box, across all 5 periods). Without
   this, step2's NDWI scoring silently drops them one period at a time
   instead; doing it once here makes the cut explicit and final. Cached to
   output/planet_coverage.gpkg since listing ~1800 scene headers takes a
   couple of minutes.

Boundary/lake files (israel_boundary.geojson, palestine_boundary.geojson,
ne_10m_land.geojson, ne_10m_lakes.geojson) live in pipeline/step3_osm_filter/data/
- regenerate them with step3_osm_filter's osm_fetch_reference.py if missing.
The OSM filter layer is pipeline/step3_osm_filter/data/osm_unified_filter_layer.gpkg
- water, building, greenhouse, industrial/construction/quarry/pool, broad landuse
zones (residential/commercial/retail/port/military/wastewater), and
already-buffered road/waterway polygons, one `category` column, clipped to
Planet coverage. If it's missing, regenerate it with step3_osm_filter's
osm_obstacles_bulk.py, osm_landuse_zones_bulk.py, and osm_relations_bulk.py.

Needs PLANET_RAW_ROOT set (env var) if your Planet data isn't at the default
path below - see the repo README.
"""
import os
import zipfile
from pathlib import Path

import geopandas as gpd
import rasterio
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
DATA = PIPELINE / "step3_osm_filter" / "data"
OUT = HERE / "output"
INPUT = OUT / "all_depressions.gpkg"
OUTPUT = OUT / "depressions_clean.gpkg"
COVERAGE_CACHE = OUT / "planet_coverage.gpkg"
UNIFIED_OSM_LAYER = DATA / "osm_unified_filter_layer.gpkg"
PLANET_RAW_ROOT = Path(os.environ.get("PLANET_RAW_ROOT", "/Volumes/ElementsData/menasim_volonteer/data/planet_raw"))
PLANET_PERIODS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]

NEGEV_LAT_CUTOFF = 31.3
LAND_BBOX = (34.0, 29.4, 36.0, 33.5)
N_LARGEST_TO_DROP = 10
OSM_CONTAINMENT_FRAC = 0.90


def geographic_cleanup(gdf):
    g = gdf.to_crs("EPSG:4326")

    lat = g.geometry.centroid.y.values
    north_of_negev = lat >= NEGEV_LAT_CUTOFF
    print(f"After Negev cut: {north_of_negev.sum()}", flush=True)

    israel = gpd.read_file(DATA / "israel_boundary.geojson").to_crs("EPSG:4326")
    west_bank = gpd.read_file(DATA / "palestine_boundary.geojson").to_crs("EPSG:4326")
    land = gpd.read_file(DATA / "ne_10m_land.geojson")
    minx, miny, maxx, maxy = LAND_BBOX
    land_union = land.cx[minx:maxx, miny:maxy].geometry.union_all()
    country_union = gpd.GeoSeries([israel.geometry.iloc[0], west_bank.geometry.iloc[0]], crs="EPSG:4326").union_all()
    valid_region = country_union.intersection(land_union)

    sub = g[north_of_negev]
    fully_within = sub.geometry.within(valid_region).values
    remainder = sub[~fully_within]
    straddle = remainder[remainder.geometry.intersects(valid_region).values]
    frac = straddle.geometry.intersection(valid_region).area.values / straddle.geometry.area.values
    keep_idx = set(sub.index[fully_within]) | set(straddle.index[frac > 0.5])
    print(f"After land/country check: {len(keep_idx)}", flush=True)

    lakes = gpd.read_file(DATA / "ne_10m_lakes.geojson")
    kinneret = lakes[lakes["name"] == "Sea of Galilee"].geometry.union_all()
    in_kinneret = set(g.loc[list(keep_idx)][g.loc[list(keep_idx)].geometry.intersects(kinneret)].index)
    keep_idx -= in_kinneret
    print(f"After Kinneret cut: {len(keep_idx)} (removed {len(in_kinneret)})", flush=True)

    out = gdf.loc[sorted(keep_idx)].copy()
    largest = out.assign(_a=out.geometry.area).nlargest(N_LARGEST_TO_DROP, "_a").index
    out = out.drop(index=largest)
    print(f"After dropping {N_LARGEST_TO_DROP} largest: {len(out)}", flush=True)
    return out


def _intersection_area_per_row(gdf, obstacles):
    """Intersection area of each gdf row with whichever obstacles it hits, summed per row."""
    obstacles = obstacles[["geometry"]]
    hits = gpd.sjoin(gdf[["geometry"]], obstacles, predicate="intersects", how="inner")
    if hits.empty:
        return gpd.pd.Series(0.0, index=gdf.index)
    dep_geom = gdf.geometry.loc[hits.index]
    obs_geom = gpd.GeoSeries(obstacles.geometry.loc[hits["index_right"]].values, index=hits.index, crs=gdf.crs)
    inter_area = dep_geom.intersection(obs_geom).area
    return inter_area.groupby(hits.index).sum().reindex(gdf.index, fill_value=0.0)


def drop_osm_contained(gdf):
    """Drop depressions whose area is >=OSM_CONTAINMENT_FRAC inside OSM water, any
    non-pool obstacle (building, greenhouse, industrial, construction, quarry, pool),
    or a buffered road/waterway - all read from the single unified OSM layer (roads
    and waterways are already buffered when that layer is built)."""
    if not UNIFIED_OSM_LAYER.exists():
        raise FileNotFoundError(
            f"{UNIFIED_OSM_LAYER} not found. Regenerate it with osm_fetch_reference.py and "
            f"osm_obstacles_bulk.py (both in step3_osm_filter/) - the raw per-source OSM files "
            f"were deleted after the unified layer was built to avoid keeping duplicate data."
        )
    unified = gpd.read_file(UNIFIED_OSM_LAYER).to_crs(gdf.crs)

    covered = _intersection_area_per_row(gdf, unified)
    frac = (covered / gdf.geometry.area).clip(upper=1.0)
    keep = gdf[frac < OSM_CONTAINMENT_FRAC]
    print(f"After OSM containment filter (water/building/greenhouse/industrial/"
          f"construction/quarry/pool/road/waterway): {len(keep)} (removed {len(gdf) - len(keep)})", flush=True)
    return keep


def gather_scene_paths(raw_dir):
    paths = []
    jan_dir = raw_dir / "january_2025" / "PSScene"
    if jan_dir.exists():
        paths.extend(str(f) for f in jan_dir.glob("*_3B_AnalyticMS_SR*.tif"))
    for z in sorted(raw_dir.glob("Israeli_Coastal_Plain_*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
        except Exception as e:
            print(f"  skip corrupt zip {z}: {e}", flush=True)
            continue
        for n in names:
            if n.endswith("_3B_AnalyticMS_SR.tif") or n.endswith("_3B_AnalyticMS_SR_clip.tif"):
                paths.append(f"/vsizip/{z}/{n}")
    return paths


def scene_wgs84_box(path):
    with rasterio.open(path) as ds:
        b = ds.bounds
        tr = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
    lo1, la1 = tr.transform(b.left, b.bottom)
    lo2, la2 = tr.transform(b.right, b.top)
    return box(min(lo1, lo2), min(la1, la2), max(lo1, lo2), max(la1, la2))


def planet_coverage():
    if COVERAGE_CACHE.exists():
        print(f"Using cached Planet coverage: {COVERAGE_CACHE}", flush=True)
        return gpd.read_file(COVERAGE_CACHE).geometry.iloc[0]

    boxes = []
    for period in PLANET_PERIODS:
        raw_dir = PLANET_RAW_ROOT / period
        if not raw_dir.exists():
            print(f"{period}: no raw data dir, skipping", flush=True)
            continue
        paths = gather_scene_paths(raw_dir)
        print(f"{period}: {len(paths)} scenes", flush=True)
        for p in paths:
            try:
                boxes.append(scene_wgs84_box(p))
            except Exception as e:
                print(f"  could not read {p}: {e}", flush=True)

    coverage = unary_union(boxes)
    gpd.GeoDataFrame(geometry=[coverage], crs="EPSG:4326").to_file(COVERAGE_CACHE, driver="GPKG")
    print(f"Saved Planet coverage: {COVERAGE_CACHE}", flush=True)
    return coverage


def clip_to_planet_coverage(gdf):
    coverage = planet_coverage()
    gdf_wgs = gdf.to_crs("EPSG:4326")
    inside = gdf_wgs.geometry.centroid.intersects(coverage).values
    keep = gdf[inside]
    print(f"After Planet-coverage clip: {len(keep)} (removed {(~inside).sum()})", flush=True)
    return keep


def main():
    gdf = gpd.read_file(INPUT).reset_index(drop=True)
    print(f"Input: {len(gdf)}", flush=True)

    out = geographic_cleanup(gdf)
    out = drop_osm_contained(out)
    out = clip_to_planet_coverage(out)

    out.to_file(OUTPUT, driver="GPKG")
    print(f"Saved: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
