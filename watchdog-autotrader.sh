#!/bin/bash
# Watchdog per Kalshi Autotrader v2
# Features:
# - Checks if trader is running, restarts if not
# - Exponential backoff on repeated crashes
# - Circuit breaker: alerts if >3 restarts in 1 hour
# - Tracks uptime stats
# - Telegram alerts for crashes

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRADER_SCRIPT="$SCRIPT_DIR/kalshi-autotrader.py"
LOG_FILE="$SCRIPT_DIR/watchdog.log"
PID_FILE="/tmp/kalshi-autotrader.pid"
ALERT_FILE="$SCRIPT_DIR/kalshi-autotrader-crash.alert"
CIRCUIT_BREAKER_FILE="$SCRIPT_DIR/kalshi-circuit-breaker.alert"
STATE_FILE="/tmp/kalshi-watchdog-state.json"
COOLDOWN_FILE="/tmp/kalshi-crash-alert-cooldown"
COOLDOWN_SECONDS=1800  # 30 min cooldown between crash alerts

# Backoff settings
BACKOFF_MIN=60       # Start at 1 minute
BACKOFF_MAX=1800     # Max 30 minutes
BACKOFF_FACTOR=2     # Double each time
STABLE_RESET=3600    # Reset backoff after 1h of stable running

# Circuit breaker
MAX_RESTARTS_PER_HOUR=3

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Initialize or load state
load_state() {
    if [ -f "$STATE_FILE" ]; then
        # Load existing state
        RESTART_COUNT=$(jq -r '.restart_count // 0' "$STATE_FILE")
        LAST_RESTART=$(jq -r '.last_restart // 0' "$STATE_FILE")
        CURRENT_BACKOFF=$(jq -r '.current_backoff // 60' "$STATE_FILE")
        RESTARTS_THIS_HOUR=$(jq -r '.restarts_this_hour // "[]"' "$STATE_FILE")
        LAST_STABLE_CHECK=$(jq -r '.last_stable_check // 0' "$STATE_FILE")
        TOTAL_UPTIME=$(jq -r '.total_uptime // 0' "$STATE_FILE")
    else
        # Initialize state
        RESTART_COUNT=0
        LAST_RESTART=0
        CURRENT_BACKOFF=$BACKOFF_MIN
        RESTARTS_THIS_HOUR="[]"
        LAST_STABLE_CHECK=$(date +%s)
        TOTAL_UPTIME=0
    fi
}

# Save state
save_state() {
    local now=$(date +%s)
    cat > "$STATE_FILE" << EOF
{
    "restart_count": $RESTART_COUNT,
    "last_restart": $LAST_RESTART,
    "current_backoff": $CURRENT_BACKOFF,
    "restarts_this_hour": $RESTARTS_THIS_HOUR,
    "last_stable_check": $LAST_STABLE_CHECK,
    "total_uptime": $TOTAL_UPTIME,
    "updated_at": "$now"
}
EOF
}

# Update restarts in the last hour (for circuit breaker)
update_hourly_restarts() {
    local now=$(date +%s)
    local one_hour_ago=$((now - 3600))
    
    # Filter out restarts older than 1 hour and add current
    RESTARTS_THIS_HOUR=$(echo "$RESTARTS_THIS_HOUR" | jq --arg cutoff "$one_hour_ago" --arg now "$now" \
        '[.[] | select(. > ($cutoff | tonumber))] + [($now | tonumber)]')
}

# Count restarts in the last hour
count_hourly_restarts() {
    echo "$RESTARTS_THIS_HOUR" | jq 'length'
}

# Check if circuit breaker should trip
check_circuit_breaker() {
    local count=$(count_hourly_restarts)
    if [ "$count" -gt "$MAX_RESTARTS_PER_HOUR" ]; then
        return 0  # Trip
    fi
    return 1  # OK
}

# Calculate next backoff
calculate_backoff() {
    local next=$((CURRENT_BACKOFF * BACKOFF_FACTOR))
    if [ $next -gt $BACKOFF_MAX ]; then
        next=$BACKOFF_MAX
    fi
    echo $next
}

# Check if we should wait (backoff)
should_wait() {
    local now=$(date +%s)
    local time_since_restart=$((now - LAST_RESTART))
    
    if [ $LAST_RESTART -eq 0 ]; then
        return 1  # No wait needed - first run
    fi
    
    if [ $time_since_restart -lt $CURRENT_BACKOFF ]; then
        return 0  # Wait
    fi
    return 1  # OK to restart
}

# Get wait time remaining
get_wait_remaining() {
    local now=$(date +%s)
    local time_since_restart=$((now - LAST_RESTART))
    local remaining=$((CURRENT_BACKOFF - time_since_restart))
    if [ $remaining -lt 0 ]; then
        remaining=0
    fi
    echo $remaining
}

# Reset backoff after stable period
check_stable_reset() {
    local now=$(date +%s)
    local time_since_restart=$((now - LAST_RESTART))
    
    if [ $LAST_RESTART -gt 0 ] && [ $time_since_restart -gt $STABLE_RESET ]; then
        log "🌟 Stable for >1h - resetting backoff"
        CURRENT_BACKOFF=$BACKOFF_MIN
        RESTART_COUNT=0
        RESTARTS_THIS_HOUR="[]"
        # Update uptime tracking
        local uptime_delta=$((now - LAST_STABLE_CHECK))
        TOTAL_UPTIME=$((TOTAL_UPTIME + uptime_delta))
        LAST_STABLE_CHECK=$now
    fi
}

# Check if we should send alert (cooldown logic)
should_alert() {
    if [ ! -f "$COOLDOWN_FILE" ]; then
        return 0
    fi
    local last_alert=$(cat "$COOLDOWN_FILE")
    local now=$(date +%s)
    local diff=$((now - last_alert))
    if [ $diff -gt $COOLDOWN_SECONDS ]; then
        return 0
    fi
    return 1
}

# Send crash alert
send_crash_alert() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local hourly_count=$(count_hourly_restarts)
    local backoff_min=$((CURRENT_BACKOFF / 60))
    
    local message="🚨 AUTOTRADER CRASH DETECTED!

⏰ Time: $timestamp
🔄 Restart #$RESTART_COUNT (${hourly_count}x this hour)
⏱️ Backoff: ${backoff_min}m before next restart
📊 Total uptime: $((TOTAL_UPTIME / 3600))h

Watchdog riavvierà il trader dopo il backoff."

    # Create alert file for heartbeat pickup
    echo "$message" > "$ALERT_FILE"
    
    # Update cooldown
    date +%s > "$COOLDOWN_FILE"
    
    log "📢 Crash alert created"
}

# Send circuit breaker alert
send_circuit_breaker_alert() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local hourly_count=$(count_hourly_restarts)
    
    local message="🛑 CIRCUIT BREAKER TRIPPED!

⏰ Time: $timestamp
❌ ${hourly_count} restarts in the last hour!
🚫 Autotrader is NOT being restarted
⚠️ Manual intervention required!

Possibili cause:
- API key invalid/expired
- Network issues
- Code bug causing crashes

Check logs: scripts/autotrader-v2.log"

    # Create alert file for heartbeat pickup
    echo "$message" > "$CIRCUIT_BREAKER_FILE"
    
    log "🛑 CIRCUIT BREAKER TRIPPED - ${hourly_count} restarts/hour"
}

# Check if trader is running
is_running() {
    if pgrep -f "kalshi-autotrader" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Start the trader
start_trader() {
    log "🚀 Starting Kalshi Autotrader..."
    # Run from project root (parent of scripts/) so relative paths work
    cd "$SCRIPT_DIR/.."
    nohup /opt/homebrew/bin/python3 -u "$TRADER_SCRIPT" >> "$SCRIPT_DIR/autotrader-v2.log" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"
    log "✅ Autotrader started with PID $pid"
    
    # Update state
    LAST_RESTART=$(date +%s)
    RESTART_COUNT=$((RESTART_COUNT + 1))
    update_hourly_restarts
    CURRENT_BACKOFF=$(calculate_backoff)
    
    save_state
}

# Main watchdog logic
main() {
    load_state
    
    if is_running; then
        log "✅ Autotrader is running"
        # Remove stale alerts if trader is running fine
        [ -f "$ALERT_FILE" ] && rm -f "$ALERT_FILE"
        [ -f "$CIRCUIT_BREAKER_FILE" ] && rm -f "$CIRCUIT_BREAKER_FILE"
        
        # Check if we can reset backoff
        check_stable_reset
        save_state
    else
        log "⚠️ Autotrader NOT running!"
        
        # Check circuit breaker first
        if check_circuit_breaker; then
            send_circuit_breaker_alert
            log "🛑 NOT restarting - circuit breaker active"
            save_state
            return
        fi
        
        # Check backoff
        if should_wait; then
            local wait_remaining=$(get_wait_remaining)
            log "⏳ Backoff active - waiting ${wait_remaining}s before restart"
            save_state
            return
        fi
        
        # Send crash alert (with cooldown)
        if should_alert; then
            send_crash_alert
        else
            log "📢 Crash alert skipped (cooldown active)"
        fi
        
        # Restart
        start_trader
    fi
}

main
