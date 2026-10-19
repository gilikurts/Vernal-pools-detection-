#!/bin/bash
# Full pipeline (extract -> score -> OSM filter -> WorldCover/PCG filter) for
# each period in turn,
# unattended. Needed whenever step1's depressions_clean.gpkg changes, since
# that invalidates the NDWI cache (built against the old candidate indices).
# Uses run_chunked.sh so memory is released between chunks instead of one
# long process accumulating until the OS kills it.
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/run_all_periods.log"
PERIODS=("2021-2022" "2022-2023" "2023-2024" "2024-2025" "2025-2026")
CHUNK=40

for PERIOD in "${PERIODS[@]}"; do
  FINAL="$HERE/../../step3b_worldcover_filter/output/$PERIOD/pools_$PERIOD.gpkg"
  echo "$(date) === starting $PERIOD ===" >> "$LOG"
  while [ ! -e "$FINAL" ]; do
    echo "$(date) run_chunked.sh $PERIOD $CHUNK" >> "$LOG"
    "$HERE/run_chunked.sh" "$PERIOD" "$CHUNK" >> "$LOG" 2>&1
    RC=$?
    if [ -e "$FINAL" ]; then
      echo "$(date) $PERIOD: final output exists, done" >> "$LOG"
      break
    fi
    echo "$(date) $PERIOD: run ended (exit $RC), retrying in 15s" >> "$LOG"
    sleep 15
  done
  echo "$(date) === $PERIOD complete ===" >> "$LOG"
done

echo "$(date) === ALL PERIODS COMPLETE ===" >> "$LOG"
