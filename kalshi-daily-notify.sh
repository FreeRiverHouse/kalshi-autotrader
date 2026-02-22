#!/bin/bash
# Generate daily trading summary and create alert for heartbeat pickup
# Run at 23:00 PST (07:00 UTC next day)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUMMARY_SCRIPT="$SCRIPT_DIR/kalshi-daily-summary.py"
ALERT_FILE="$SCRIPT_DIR/kalshi-daily-report.alert"

# Generate summary
python3 "$SUMMARY_SCRIPT" > /dev/null 2>&1

# Create alert file for heartbeat to pick up
if [ -f "$SCRIPT_DIR/kalshi-daily-summary.txt" ]; then
    cp "$SCRIPT_DIR/kalshi-daily-summary.txt" "$ALERT_FILE"
    echo "Daily summary alert created"
fi
