import time
from collections import defaultdict
from datetime import date, timedelta

import requests

from config import (
    BBOX,
    CATALOGUE_URL,
    MAX_CLOUD_COVER,
    N_YEARS,
    REQUIRE_ALL_TILES,
    TILE_WHITELIST,
    WANTED_MONTHS,
)
from download import parse_tile_and_date


def build_date_ranges(n_years=N_YEARS, wanted_months=WANTED_MONTHS, start=None, end=None):
    """Return a list of (start_date, end_date) month-long windows
    covering the wanted months between `start` and `end` (end exclusive).
    Defaults to the last n_years up to today."""
    today = date.today()
    if start is None:
        start = today.replace(year=today.year - n_years)
    if end is None:
        end = today + timedelta(days=1)

    ranges = []
    cur = date(start.year, start.month, 1)
    while cur < end:
        if cur.month in wanted_months:
            if cur.month == 12:
                nxt = date(cur.year + 1, 1, 1)
            else:
                nxt = date(cur.year, cur.month + 1, 1)
            range_end = min(nxt, end)
            ranges.append((cur, range_end))

        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

    return ranges


def _aoi_wkt(bbox=BBOX):
    min_lon, min_lat, max_lon, max_lat = bbox
    coords = (
        f"{min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}"
    )
    return f"POLYGON (({coords}))"


def get_attribute(product, name):
    for attr in product.get("Attributes", []):
        if attr.get("Name") == name:
            return attr.get("Value")
    return None


def search_products(
    start_date,
    end_date,
    bbox=BBOX,
    max_cloud_cover=MAX_CLOUD_COVER,
    require_all_tiles=REQUIRE_ALL_TILES,
):
    """Query the CDSE OData catalogue for Sentinel-2 L2A products
    intersecting `bbox` within [start_date, end_date), filtered by
    scene-level cloud cover. Returns a list of product dicts."""
    aoi = _aoi_wkt(bbox)
    odata_filter = (
        "Collection/Name eq 'SENTINEL-2' and "
        "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' "
        "and a/Value eq 'S2MSI2A') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi}') and "
        f"ContentDate/Start ge {start_date.isoformat()}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date.isoformat()}T00:00:00.000Z"
    )

    params = {
        "$filter": odata_filter,
        "$expand": "Attributes",
        "$top": 1000,
        "$orderby": "ContentDate/Start asc",
    }

    products = []
    url = CATALOGUE_URL
    while url:
        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt * 5)
        payload = resp.json()
        products.extend(payload.get("value", []))
        url = payload.get("@odata.nextLink")
        params = None  # nextLink already contains the full query string

    whitelisted = []
    for product in products:
        tile, _ = parse_tile_and_date(product["Name"])
        if tile in TILE_WHITELIST:
            whitelisted.append(product)

    if require_all_tiles:
        tiles_per_date = defaultdict(set)
        for product in whitelisted:
            _, sensing_date = parse_tile_and_date(product["Name"])
            tile, _ = parse_tile_and_date(product["Name"])
            tiles_per_date[sensing_date].add(tile)

        synced_dates = {
            d for d, tiles in tiles_per_date.items() if tiles == TILE_WHITELIST
        }
        whitelisted = [
            p for p in whitelisted
            if parse_tile_and_date(p["Name"])[1] in synced_dates
        ]

    kept = []
    for product in whitelisted:
        cloud_cover = get_attribute(product, "cloudCover")
        if cloud_cover is not None and float(cloud_cover) > max_cloud_cover:
            continue
        kept.append(product)

    return kept
