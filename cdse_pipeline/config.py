import os
from dotenv import load_dotenv

load_dotenv()

CDSE_USERNAME = os.environ.get("CDSE_USERNAME")
CDSE_PASSWORD = os.environ.get("CDSE_PASSWORD")

# Area of interest - bounding box around the vernal pools extent
# (min_lon, min_lat, max_lon, max_lat)
BBOX = (34.4, 31.2, 35.9, 33.4)

# Sentinel-2 MGRS tiles that actually cover the vernal pools (chosen by
# counting pool centroids per tile footprint - these 6 cover ~98.5% of
# the 390 pools; the other tiles intersecting BBOX only touch empty corners)
TILE_WHITELIST = {"36SXA", "36SYB", "36SXB", "36SYA", "37SBS", "36RXV"}

# Hebrew region label for each tile, used in output folder names
TILE_NAMES = {
    "36SXA": "מרכז - שרון ומישור החוף",
    "36SYB": "גליל וגולן",
    "36SXB": "חיפה והכרמל",
    "36SYA": "שומרון וירושלים",
    "37SBS": "צפון-מזרח - גולן וחרמון",
    "36RXV": "דרום-מערב - נגב מערבי",
}


def tile_folder_name(tile):
    label = TILE_NAMES.get(tile, "")
    return f"{tile}_{label}" if label else tile

# Only keep acquisition dates where ALL 6 tiles above were captured
# (the ~5-day synchronized pass), so every date gives a consistent
# wall-to-wall mosaic for comparison. Drops the extra overlap-only
# passes that some tiles get from neighboring orbits.
REQUIRE_ALL_TILES = True

# Bands to download, grouped by native resolution / IMG_DATA subfolder
BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B11", "B12", "SCL"]

# Scene-level cloud cover above this is skipped entirely
MAX_CLOUD_COVER = 50.0

# How many years back from today to search
N_YEARS = 5

# Months of interest: wet season (Oct-Apr) + dry season reference (Aug)
WANTED_MONTHS = {1, 2, 3, 4, 8, 10, 11, 12}

OUTPUT_DIR = "sentinel2_data"
MANIFEST_PATH = "manifest.csv"

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
