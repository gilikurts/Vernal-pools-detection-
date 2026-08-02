import re
import time
from pathlib import Path

import requests

from config import BANDS_10M, BANDS_20M, DOWNLOAD_URL, tile_folder_name

TILE_RE = re.compile(r"_T(\d{2}[A-Z]{3})_")
DATE_RE = re.compile(r"_MSIL2A_(\d{8})T\d{6}_")


def parse_tile_and_date(product_name):
    """product_name is the .SAFE name, e.g.
    S2A_MSIL2A_20230615T083601_N0509_R136_T36RXU_20230615T125629.SAFE"""
    tile_match = TILE_RE.search(product_name)
    date_match = DATE_RE.search(product_name)
    tile = tile_match.group(1) if tile_match else "UNKNOWN_TILE"
    sensing_date = date_match.group(1) if date_match else "UNKNOWN_DATE"
    return tile, sensing_date


def get_season(sensing_date):
    """Map a YYYYMMDD date string to a season folder name:
    months 10-12 and 1-4 -> winter_<startyear>_<endyear>
    month 8             -> summer_<year>"""
    if sensing_date == "UNKNOWN_DATE":
        return "unknown_season"

    year = int(sensing_date[:4])
    month = int(sensing_date[4:6])

    if month == 8:
        return f"summer_{year}"
    if month in (10, 11, 12):
        return f"winter_{year}_{year + 1}"
    if month in (1, 2, 3, 4):
        return f"winter_{year - 1}_{year}"
    return "unknown_season"


def _odata_node(name):
    # CDSE Nodes() segments use the raw name, unquoted
    return f"Nodes({name})"


def _request_with_retry(method, url, auth, max_retries=4, **kwargs):
    last_exc = None
    for attempt in range(max_retries):
        try:
            headers = kwargs.pop("headers", {})
            headers["Authorization"] = f"Bearer {auth.get_token()}"
            resp = requests.request(method, url, headers=headers, timeout=120, **kwargs)
            if resp.status_code == 401:
                # token might have just expired - force refresh and retry once
                auth._access_token = None
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
    raise last_exc


def list_child_nodes(auth, product_id, path_segments):
    """path_segments: list of node names from the product root down to
    the folder whose children we want to list."""
    url = f"{DOWNLOAD_URL}({product_id})"
    for seg in path_segments:
        url += f"/{_odata_node(seg)}"
    url += "/Nodes"
    resp = _request_with_retry("GET", url, auth)
    return resp.json().get("result", [])


def find_band_files(auth, product_id, safe_name):
    """Locate the band files of interest inside the SAFE product and
    return a dict {band_name: path_segments_to_file}."""
    band_files = {}

    granule_root = [safe_name, "GRANULE"]
    granules = list_child_nodes(auth, product_id, granule_root)
    if not granules:
        return band_files
    granule_name = granules[0]["Name"]

    for resolution, bands in (("R10m", BANDS_10M), ("R20m", BANDS_20M)):
        img_data_path = [safe_name, "GRANULE", granule_name, "IMG_DATA", resolution]
        files = list_child_nodes(auth, product_id, img_data_path)
        for f in files:
            fname = f["Name"]
            for band in bands:
                if f"_{band}_" in fname and fname.lower().endswith(".jp2"):
                    band_files[band] = img_data_path + [fname]

    return band_files


def download_node(auth, product_id, path_segments, out_path):
    url = f"{DOWNLOAD_URL}({product_id})"
    for seg in path_segments:
        url += f"/{_odata_node(seg)}"
    url += "/$value"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    with requests.Session() as session:
        headers = {"Authorization": f"Bearer {auth.get_token()}"}
        with session.get(url, headers=headers, stream=True, timeout=300) as resp:
            if resp.status_code == 401:
                headers["Authorization"] = f"Bearer {auth.get_token()}"
                resp = session.get(url, headers=headers, stream=True, timeout=300)
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    tmp_path.rename(out_path)


def download_product_bands(auth, product, output_dir):
    """Download all configured bands for a single product.
    Returns the list of local file paths that were downloaded (or
    already existed)."""
    product_id = product["Id"]
    safe_name = product["Name"]
    tile, sensing_date = parse_tile_and_date(safe_name)
    season = get_season(sensing_date)

    band_files = find_band_files(auth, product_id, safe_name)
    if not band_files:
        return []

    out_dir = Path(output_dir) / season / tile_folder_name(tile) / sensing_date
    downloaded = []

    for band, path_segments in band_files.items():
        fname = path_segments[-1]
        out_path = out_dir / fname
        if out_path.exists():
            downloaded.append(out_path)
            continue
        download_node(auth, product_id, path_segments, out_path)
        downloaded.append(out_path)

    return downloaded
