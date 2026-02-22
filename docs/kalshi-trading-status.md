# Kalshi Trading Status

**Ultimo aggiornamento:** 2026-02-03 01:55 PST

## 📊 Stato Attuale

| Metrica | Valore |
|---------|--------|
| **Status** | ✅ Running |
| **Mode** | LIVE (dry_run: false) |
| **Cicli completati** | 32 |
| **Posizioni aperte** | 9 |
| **Trade oggi** | 0 |
| **Win rate oggi** | 0% |
| **PnL oggi** | $0.00 |
| **Circuit breaker** | Inattivo |
| **Perdite consecutive** | 15 |

## ⚠️ Note
- 15 perdite consecutive - da monitorare
- Nessun trade eseguito oggi
- Cash a 0 cents - potrebbe necessitare ricarica

## 📁 File Rilevanti
- `scripts/kalshi-autotrader.py` - Bot principale
- `data/trading/autotrader-health.json` - Health status
- `data/trading/autotrader-uptime.json` - Uptime tracking
- `scripts/watchdog-autotrader.sh` - Watchdog cron

## 🔧 Comandi Utili
```bash
# Check status
pgrep -f kalshi-autotrader

# View logs
tail -f /tmp/kalshi-autotrader.log

# Restart
pkill -f kalshi-autotrader
nohup python3 scripts/kalshi-autotrader.py --live &
```
