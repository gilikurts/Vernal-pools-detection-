"""
How many of the merged pools (step4's output) were detected in exactly how
many of the 5 years. Output: recurrence_table.csv in step4_cross_year/output/,
printed too.
"""
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent
POOLS = sorted((HERE / "output").glob("pools_*.gpkg"))[-1]

pools = gpd.read_file(POOLS)
counts = pools["n_periods"].value_counts().sort_index()
total = len(pools)

rows = [{"years_seen": y, "n_pools": int(n), "pct_of_total": round(100 * n / total, 1)}
        for y, n in counts.items()]
rows.append({"years_seen": "TOTAL", "n_pools": total, "pct_of_total": 100.0})

import pandas as pd
table = pd.DataFrame(rows)
print(table.to_string(index=False))

out = HERE / "output" / "recurrence_table.csv"
table.to_csv(out, index=False)
print(f"\nSaved: {out}")
