import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path

from tqdm import tqdm

from auth import CDSEAuth
from config import MANIFEST_PATH, MAX_CLOUD_COVER, N_YEARS, OUTPUT_DIR
from download import download_product_bands, get_season, parse_tile_and_date
from search import build_date_ranges, get_attribute, search_products


def load_manifest_ids(manifest_path):
    if not Path(manifest_path).exists():
        return set()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return {row["product_id"] for row in csv.DictReader(f)}


def append_manifest(manifest_path, row, write_header):
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years", type=int, default=N_YEARS,
        help=f"How many years back to search (default: {N_YEARS})",
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Download only the 'pool year' Y (summer_Y + winter_Y_Y+1), instead of the last N years",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Directory to save downloaded files into (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--products-file", default=None,
        help="JSON file from prefetch_products.py — skip catalogue search entirely",
    )
    args = parser.parse_args()

    auth = CDSEAuth()

    if args.products_file:
        with open(args.products_file) as f:
            cached = json.load(f)
        done_ids = load_manifest_ids(MANIFEST_PATH)
        write_header = not Path(MANIFEST_PATH).exists()
        products_to_download = [p for p in cached if p["id"] not in done_ids]
        print(f"Loaded {len(products_to_download)} products from {args.products_file} "
              f"(skipping {len(cached) - len(products_to_download)} already done)")
        for entry in tqdm(products_to_download, desc="Downloading"):
            try:
                files = download_product_bands(auth, entry["product"], args.output_dir)
                status = "ok" if files else "no_bands_found"
            except Exception as exc:  # noqa: BLE001
                files = []
                status = f"error: {exc}"
            append_manifest(MANIFEST_PATH, {
                "product_id": entry["id"],
                "name": entry["name"],
                "season": entry["season"],
                "tile": entry["tile"],
                "date": entry["date"],
                "cloud_cover": entry["cloud_cover"],
                "n_files": len(files),
                "status": status,
            }, write_header)
            write_header = False
        return

    if args.year is not None:
        date_ranges = build_date_ranges(start=date(args.year, 8, 1), end=date(args.year + 1, 5, 1))
        print(
            f"Searching {len(date_ranges)} monthly windows for pool-year {args.year} "
            f"(summer_{args.year} + winter_{args.year}_{args.year + 1})..."
        )
    else:
        date_ranges = build_date_ranges(n_years=args.years)
        print(f"Searching {len(date_ranges)} monthly windows over the last {args.years} year(s)...")

    done_ids = load_manifest_ids(MANIFEST_PATH)
    write_header = not Path(MANIFEST_PATH).exists()

    for start, end in date_ranges:
        print(f"\n=== {start} -> {end} ===")
        time.sleep(2)  # avoid hammering the catalogue search API
        products = search_products(start, end)
        print(f"  Found {len(products)} products with cloud cover <= {MAX_CLOUD_COVER}%")

        for product in tqdm(products, desc="Downloading"):
            product_id = product["Id"]
            if product_id in done_ids:
                continue

            tile, sensing_date = parse_tile_and_date(product["Name"])
            season = get_season(sensing_date)
            cloud_cover = get_attribute(product, "cloudCover")

            try:
                files = download_product_bands(auth, product, args.output_dir)
                status = "ok" if files else "no_bands_found"
            except Exception as exc:  # noqa: BLE001 - log and continue
                files = []
                status = f"error: {exc}"

            append_manifest(
                MANIFEST_PATH,
                {
                    "product_id": product_id,
                    "name": product["Name"],
                    "season": season,
                    "tile": tile,
                    "date": sensing_date,
                    "cloud_cover": cloud_cover,
                    "n_files": len(files),
                    "status": status,
                },
                write_header,
            )
            write_header = False
            done_ids.add(product_id)


if __name__ == "__main__":
    main()
