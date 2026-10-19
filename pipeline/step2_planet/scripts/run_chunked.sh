#!/bin/bash
# Runs 01_extract_ndwi.py in small chunks so each process exits and releases
# its memory before the next one starts, instead of one long-lived process
# accumulating memory until the OS kills it. Then runs steps 2 and 3 normally.
# Usage: ./run_chunked.sh <period> [chunk_size]
set -e
PERIOD="${1:?usage: run_chunked.sh <period> [chunk_size]}"
CHUNK="${2:-30}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== chunked extraction: $PERIOD, chunk size $CHUNK ==="
while true; do
  OUT=$(python3 "$HERE/01_extract_ndwi.py" "$PERIOD" "$CHUNK")
  echo "$OUT"
  if echo "$OUT" | grep -q "ALL_DONE"; then
    echo "=== extraction complete ==="
    break
  fi
done

echo "=== step 2: scoring ==="
python3 "$HERE/02_score_candidates.py" "$PERIOD"
echo "=== step 3: OSM filter ==="
python3 "$HERE/../../step3_osm_filter/03_filter_osm.py" "$PERIOD"
echo "=== step 3b: WorldCover (cropland/built-up) filter ==="
python3 "$HERE/../../step3b_worldcover_filter/03b_filter_worldcover.py" "$PERIOD"
echo "=== done ==="
