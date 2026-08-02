"""
download_august.py – הורדת תמונת אוגוסט אחת לכל טייל לכל שנה.

הבדלים מ-main.py:
  - חודש 8 בלבד (WANTED_MONTHS = {8})
  - עננות ≤ 5% (במקום 50%)
  - REQUIRE_ALL_TILES = False  (כל טייל עצמאי, לא חייבים אותו יום)
  - מניפסט נפרד: manifest_august.csv
  - פלט: /Volumes/ElementsData/menasim_volonteer/sentinel2_data/
"""

import csv
import time
from datetime import date
from pathlib import Path

from tqdm import tqdm

from auth import CDSEAuth
from config import N_YEARS, TILE_WHITELIST
from download import download_product_bands, get_season, parse_tile_and_date
from search import build_date_ranges, get_attribute, search_products

OUTPUT_DIR   = Path("/Volumes/ElementsData/menasim_volonteer/sentinel2_data")
MANIFEST     = Path(__file__).parent / "manifest_august.csv"
MAX_CLOUD    = 5.0
WANTED_MONTHS_AUG = {8}


def load_done(manifest):
    if not manifest.exists():
        return set()
    with open(manifest, newline="", encoding="utf-8") as f:
        return {row["product_id"] for row in csv.DictReader(f)}


def append_row(manifest, row, write_header):
    with open(manifest, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    auth = CDSEAuth()
    date_ranges = build_date_ranges(n_years=N_YEARS, wanted_months=WANTED_MONTHS_AUG)
    print(f"מחפש {len(date_ranges)} חלונות אוגוסט ({N_YEARS} שנים אחרונות)...")
    print(f"עננות מקסימלית: {MAX_CLOUD}%\n")

    done_ids   = load_done(MANIFEST)
    write_header = not MANIFEST.exists()

    # Track best scene per (tile, year) — lowest cloud cover
    best: dict[tuple, dict] = {}

    for start, end in date_ranges:
        print(f"=== {start} -> {end} ===")
        time.sleep(2)
        products = search_products(
            start, end,
            max_cloud_cover=MAX_CLOUD,
            require_all_tiles=False,
        )
        print(f"  נמצאו {len(products)} מוצרים עם עננות ≤ {MAX_CLOUD}%")

        for product in products:
            tile, sensing_date = parse_tile_and_date(product["Name"])
            if tile not in TILE_WHITELIST:
                continue
            year = int(sensing_date[:4])
            cloud = float(get_attribute(product, "cloudCover") or 100)
            key = (tile, year)
            if key not in best or cloud < float(
                get_attribute(best[key], "cloudCover") or 100
            ):
                best[key] = product

    print(f"\nנבחרו {len(best)} סצנות (הכי נקיות לכל טייל×שנה):")
    for (tile, year), product in sorted(best.items()):
        _, sensing_date = parse_tile_and_date(product["Name"])
        cloud = get_attribute(product, "cloudCover")
        print(f"  {tile} / {year}  →  {sensing_date}  ({cloud}%)")

    print("\nמתחיל הורדה...")
    for (tile, year), product in tqdm(sorted(best.items()), desc="סצנות"):
        product_id = product["Id"]
        if product_id in done_ids:
            tqdm.write(f"  דילוג (קיים): {tile}/{year}")
            continue

        _, sensing_date = parse_tile_and_date(product["Name"])
        season = get_season(sensing_date)
        cloud  = get_attribute(product, "cloudCover")

        try:
            files  = download_product_bands(auth, product, str(OUTPUT_DIR))
            status = "ok" if files else "no_bands_found"
        except Exception as e:
            files  = []
            status = f"error: {e}"
            tqdm.write(f"  ⚠️ שגיאה {tile}/{year}: {e}")

        append_row(MANIFEST, {
            "product_id":  product_id,
            "name":        product["Name"],
            "season":      season,
            "tile":        tile,
            "date":        sensing_date,
            "cloud_cover": cloud,
            "n_files":     len(files),
            "status":      status,
        }, write_header)
        write_header = False
        done_ids.add(product_id)

    print(f"\nהושלם. מניפסט: {MANIFEST}")


if __name__ == "__main__":
    main()
