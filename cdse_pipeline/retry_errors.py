"""Re-download all manifest entries that have status starting with 'error'."""

import csv
from pathlib import Path

from auth import CDSEAuth
from config import MANIFEST_PATH, OUTPUT_DIR
from download import download_product_bands
from search import get_attribute

MANIFEST = Path(MANIFEST_PATH)


def load_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_manifest(rows):
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = load_manifest()
    errors = [r for r in rows if r["status"].startswith("error")]
    print(f"נמצאו {len(errors)} שגיאות להורדה מחדש")

    auth = CDSEAuth()
    status_by_id = {r["product_id"]: i for i, r in enumerate(rows)}

    for row in errors:
        product_id = row["product_id"]
        print(f"  מוריד: {row['name'][:60]}...")

        product = {"Id": product_id, "Name": row["name"]}
        try:
            files = download_product_bands(auth, product, OUTPUT_DIR)
            status = "ok" if files else "no_bands_found"
        except Exception as exc:
            files = []
            status = f"error: {exc}"

        print(f"    → {status} ({len(files)} קבצים)")
        rows[status_by_id[product_id]]["status"] = status
        rows[status_by_id[product_id]]["n_files"] = len(files)
        save_manifest(rows)


if __name__ == "__main__":
    main()
