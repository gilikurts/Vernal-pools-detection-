"""
Step 4: merge the same pool across years, then check against the ground truth.

Two candidates (from any period, including the same one) are the same pool
when their geometries, each buffered by BUFFER_M, intersect. Pools are
grouped transitively (union-find). `n_periods` counts UNIQUE years, not raw
member count - a pool can have two members from the same year (e.g. the DEM
depression happened to split there) merge together; such pools are flagged
with a `*` suffix on `pool_label` (`same_year_duplicate=True`) since their
member count and year count disagree. `core_area_m2` is the MEDIAN across
members, not the max, so one oversized duplicate member doesn't dominate.

Input:
  pipeline/step3b_worldcover_filter/output/<period>/pools_<period>.gpkg   one per year, from step 3b
Outputs:
  pipeline/step4_cross_year/output/pools_<first-year>-<last-year>.gpkg   one row per pool
      pool_id, pool_label, n_periods, periods, same_year_duplicate, core_area_m2, geometry
  printed ground-truth check (GT layer, Planet-covered polygons only)

Usage: python cross_year.py
"""
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
FINAL_FILES = sorted((PIPELINE / "step3b_worldcover_filter" / "output").glob("*/pools_*.gpkg"))
GT_DIR = Path(os.environ.get("GT_DIR", "/Volumes/ElementsData/menasim_volonteer/data/reference"))
GT = next((f for f in GT_DIR.glob("*.gpkg") if "בריכות" in f.name or "Vernal" in f.name), None)
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)

BUFFER_M = 6.0
GT_CHECK_BUFFER_M = 4.0
PLANET_BBOX_WGS84 = (34.190641, 31.389999, 35.439727, 32.922476)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def group_pools(cands):
    buffered = cands.geometry.buffer(BUFFER_M).values
    uf = UnionFind(len(cands))
    tree = STRtree(buffered)
    pairs_i, pairs_j = tree.query(buffered, predicate="intersects")
    for i, j in zip(pairs_i, pairs_j):
        if i >= j:
            continue
        uf.union(i, j)
    return np.array([uf.find(i) for i in range(len(cands))])


def build_pool_layer(cands):
    cands = cands.reset_index(drop=True)
    labels = group_pools(cands)
    cands["_group"] = labels
    rows = []
    for pool_id, (_, grp) in enumerate(cands.groupby("_group")):
        period_list = list(grp["period"])
        unique_periods = sorted(set(period_list))
        same_year_dup = len(period_list) > len(unique_periods)
        rows.append({
            "pool_id": pool_id,
            "pool_label": f"{pool_id}*" if same_year_dup else str(pool_id),
            "n_periods": len(unique_periods),
            "periods": ",".join(unique_periods),
            "same_year_duplicate": same_year_dup,
            "core_area_m2": float(grp["core_area_m2"].median()),
            "geometry": grp.geometry.union_all(),
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=cands.crs)


def ground_truth_check(pools):
    gt = gpd.read_file(GT).to_crs(pools.crs)
    gt_wgs = gt.to_crs("EPSG:4326")
    cx, cy = gt_wgs.geometry.centroid.x.values, gt_wgs.geometry.centroid.y.values
    x0, y0, x1, y1 = PLANET_BBOX_WGS84
    covered = gt[(cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)]
    print(f"\nGround truth polygons: {len(gt)}  inside Planet coverage: {len(covered)}", flush=True)

    covered_buf = covered.geometry.buffer(GT_CHECK_BUFFER_M)

    pool_union = pools.geometry.buffer(GT_CHECK_BUFFER_M).union_all()
    hit = covered_buf.intersects(pool_union).values
    print(f"Overlapped by any pool: {hit.sum()} / {len(covered)} ({hit.sum() / len(covered):.1%})", flush=True)

    multi = pools[pools["n_periods"] > 1].geometry.buffer(GT_CHECK_BUFFER_M).union_all()
    hit_multi = covered_buf.intersects(multi).values
    print(f"Overlapped by a pool seen in 2+ years: {hit_multi.sum()}", flush=True)


def main():
    cands = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in FINAL_FILES], ignore_index=True), crs="EPSG:2039")
    print(f"Final candidates in: {len(cands)}  periods: {sorted(set(cands['period']))}", flush=True)

    periods = sorted(set(cands["period"]))
    years = sorted({y for period in periods for y in period.split("-")})
    span = f"{years[0]}-{years[-1]}"
    pools = build_pool_layer(cands)
    n_dup = int(pools["same_year_duplicate"].sum())
    out = OUT_DIR / f"pools_{span}.gpkg"
    pools.to_file(out, driver="GPKG")
    print(f"Pools: {len(pools)}  (seen in 2+ years: {(pools['n_periods'] > 1).sum()}, "
          f"same-year duplicates flagged with *: {n_dup})", flush=True)
    print(f"Saved: {out}", flush=True)

    ground_truth_check(pools)


if __name__ == "__main__":
    main()
