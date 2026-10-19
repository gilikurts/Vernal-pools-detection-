"""
Pairwise overlap matrix between years: for each pair of periods, what
percentage of the row period's pools also overlap a pool in the column
period - same rule as cross_year.py: buffer each polygon by BUFFER_M and
check intersects (not an area-overlap-fraction rule).

Output: pairwise_overlap.csv in step4_cross_year/output/, printed too.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
STEP3_OUT = HERE.parent / "step3b_worldcover_filter" / "output"
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)

BUFFER_M = 6.0

periods = sorted(p.name for p in STEP3_OUT.iterdir() if p.is_dir())
layers = {p: gpd.read_file(STEP3_OUT / p / f"pools_{p}.gpkg") for p in periods}

rows = []
for a in periods:
    row = {"period": a, "n_pools": len(layers[a])}
    for b in periods:
        if a == b:
            row[b] = 100.0
            continue
        geoms_a = layers[a].geometry.buffer(BUFFER_M).values
        union_b = layers[b].geometry.buffer(BUFFER_M).union_all()
        if len(geoms_a) == 0:
            row[b] = 0.0
            continue
        hits = sum(1 for g in geoms_a if g.intersects(union_b))
        row[b] = round(hits / len(geoms_a) * 100, 1)
    rows.append(row)

table = pd.DataFrame(rows)
print("Row = source year, column = % of the row year's pools also found in that year\n")
print(table.to_string(index=False))

out = OUT_DIR / "pairwise_overlap.csv"
table.to_csv(out, index=False)
print(f"\nSaved: {out}")
