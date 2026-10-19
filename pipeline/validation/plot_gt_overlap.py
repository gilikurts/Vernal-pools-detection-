"""
Plot one or more detected pools against the INPA ground-truth (GT) polygon
they overlap, as a simple vector comparison (no satellite basemap): GT
outlined in dashed red, detected pool(s) filled teal.

Usage: python plot_gt_overlap.py <pool_id> [<pool_id> ...]
  <pool_id> is a `pool_id` from step4_cross_year/output/pools_*.gpkg.

Output: validation/output/gt_overlap_<pool_id>.png
"""
import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
GT_DIR = Path(os.environ.get("GT_DIR", "/Volumes/ElementsData/menasim_volonteer/data/reference"))
OUT_DIR = HERE / "output"


def main():
    pool_ids = [int(a) for a in sys.argv[1:]]
    if not pool_ids:
        print("Usage: python plot_gt_overlap.py <pool_id> [<pool_id> ...]")
        sys.exit(1)

    pools_path = sorted((PIPELINE / "step4_cross_year" / "output").glob("pools_*.gpkg"))[-1]
    pools = gpd.read_file(pools_path)
    gt_path = next((f for f in GT_DIR.glob("*.gpkg") if "בריכות" in f.name or "Vernal" in f.name), None)
    gt = gpd.read_file(gt_path).to_crs(pools.crs)

    pair = pools[pools["pool_id"].isin(pool_ids)]
    gt_poly = gt[gt.geometry.buffer(4).intersects(pair.geometry.buffer(4).union_all())]

    fig, ax = plt.subplots(figsize=(6, 6))
    gt_poly.plot(ax=ax, facecolor="none", edgecolor="crimson", linewidth=2, linestyle="--")
    pair.plot(ax=ax, facecolor="teal", edgecolor="darkslategray", alpha=0.6)

    minx, miny, maxx, maxy = gt_poly.total_bounds
    pad = 15
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Line2D([0], [0], color="crimson", linestyle="--", linewidth=2, label="INPA ground truth"),
        Patch(facecolor="teal", edgecolor="darkslategray", alpha=0.6, label="Detected pool(s)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    ax.set_title("Detected pool(s) vs. INPA ground-truth polygon", fontsize=12)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"gt_overlap_{'_'.join(str(p) for p in pool_ids)}.png"
    fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
