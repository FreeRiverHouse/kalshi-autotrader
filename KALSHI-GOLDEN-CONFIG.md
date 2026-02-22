# ⚡ KALSHI GOLDEN CONFIG

**Unica fonte di verità per il trading Kalshi. LEGGI QUESTO E BASTA.**
**Ultimo aggiornamento: 2026-02-16 19:18 PST**

---

## 🚨 REGOLE FONDAMENTALI (NON DIMENTICARE MAI)

1. **PAPER MODE = RACCOLTA DATI.** Non proteggere il bankroll virtuale. Più trade = più dati = algo migliore. Zero conservatismo in paper mode.
2. **MAI lamentarsi di "soldi buttati" in paper mode.** Sono VIRTUALI. L'obiettivo è raccogliere dati per capire il win rate REALE.
3. **Filtri bassi in paper mode.** MIN_EDGE basso, max trades alto, niente skip inutili. Servono trade SETTLEDATI per analizzare.
4. **Segui PROC-002:** ogni task chiuso → receipt a Grok → 2 task migliorativi.
5. **Chiedi a Grok, non a Mattia** per decisioni tecniche sull'algo.

---

## 🧠 STRATEGIA (Grok-validated 2026-02-16)

### Formula Breakeven
```
Breakeven Win Rate = Prezzo Pagato (come frazione)
- Compri a 20¢ → serve 20% WR per breakeven
- Compri a 40¢ → serve 40% WR
- Compri a 50¢ → serve 50% WR
```

### Principio Chiave
**COMPRARE A PREZZI BASSI = BREAKEVEN FACILE**
Target: avg price ≤25¢ per un breakeven raggiungibile (~25% WR)

---

## ⚙️ PARAMETRI GOLDEN (autotrader.py)

### 💰 Prezzi & Edge
| Parametro | Valore | Razionale |
|-----------|--------|-----------|
| `MAX_PRICE_CENTS` | **50** | Breakeven 50% WR max (Grok rec C) |
| `MIN_PRICE_CENTS` | **5** | Sotto 5¢ = troppo illiquido |
| `MIN_EDGE_BUY_NO` | **0.03** (3%) | BUY_NO è il nostro punto di forza (50% WR storico) |
| `MIN_EDGE_BUY_YES` | **0.08** (8%) | BUY_YES più debole (33% WR storico), soglia alta |
| `MIN_EDGE` | **0.03** (3%) | Minimo globale |
| `MAX_NO_PRICE_CENTS` | **80** | Hard cap per NO |

### 📊 Calibrazione
| Parametro | Valore | Razionale |
|-----------|--------|-----------|
| `CALIBRATION_FACTOR` | **0.65** | Shrink prob verso 50% (predicted 71.5% → actual 46.2%) |
| `CRYPTO_FAT_TAIL_MULTIPLIER` | **1.0** | Disabilitato dopo v2 disaster |

### 🛡️ Risk Management
| Parametro | Valore | Razionale |
|-----------|--------|-----------|
| `MAX_BET_CENTS` | **500** ($5) | Max per singolo trade |
| `MAX_POSITIONS` | **15** | Max posizioni aperte |
| `MAX_DAILY_TRADES` | **200** | Hard cap giornaliero |
| `MAX_DAILY_EXPOSURE_USD` | **50.0** | Cap esposizione giornaliera |
| `DAILY_LOSS_LIMIT_CENTS` | **500** ($5) | Stop giornaliero |
| `DRAWDOWN_THRESHOLD_PCT` | **0.15** (15%) | Alert se drawdown > 15% |

### 📈 Exit Strategy
| Parametro | Valore | Razionale |
|-----------|--------|-----------|
| `TRAILING_STOP_ENABLED` | **True** | Trailing stop attivo |
| `PROFIT_TAKE_PCT` | **0.30** (30%) | Take profit a +30% |
| `TRAILING_STOP_PCT` | **0.15** (15%) | Trail del 15% dal picco |
| `MIN_PROFIT_TO_TRAIL` | **0.10** (10%) | Inizia trail dopo +10% |

### 🔄 Loop & Timing
| Parametro | Valore | Razionale |
|-----------|--------|-----------|
| Loop interval | **300s** (5 min) | Ciclo ogni 5 minuti |
| `VIRTUAL_BALANCE` | **100.0** | Balance virtuale per paper mode |

---

## 🚫 FILTRI DISABILITATI (per data collection)

| Filtro | Stato | Motivo |
|--------|-------|--------|
| BTC high-vol skip | **OFF** (log only) | Grok rec C: raccogliere dati in tutti i regimi |
| Choppy+high-vol skip | **OFF** (log only) | Grok rec C: no filter in dry run |

**⚠️ Ri-abilitare dopo 50-100 trades se BTC choppy trascina performance!**

---

## 📊 PERFORMANCE STORICA (pre-fix)

### Stats iniziali (13 trades settled)
- Win rate: 46.2% (6W/7L)
- PnL: -$1.65
- BUY_NO: 50% WR (5/10) ← punto di forza
- BUY_YES: 33% WR (1/3) ← debolezza
- Avg price: 64¢ ← TROPPO ALTO (breakeven 64% WR)
- Profit factor: 0.45

### Dopo Grok fix (2026-02-16)
- Avg price: ~19¢ (da 64¢!)
- Breakeven WR: ~19% (da 64%!)
- Volume: 5 trades/ciclo (da 0.36/ciclo)
- Bankroll: $50 dry run

---

## 🔧 COME MODIFICARE

1. **Cambia parametri in** `scripts/kalshi-autotrader.py`
2. **Aggiorna QUESTO file** con i nuovi valori
3. **Commit + push** entrambi
4. **Riavvia autotrader**: `pkill -f kalshi-autotrader && python3 -u scripts/kalshi-autotrader.py --loop 300 > scripts/kalshi-autotrader.log 2>&1 &`
5. **Verifica primo ciclo**: `grep "trade_executed" scripts/kalshi-autotrader.log`

---

## 📝 CHANGELOG

### v3 — 2026-02-16 (Grok Analysis)
- MAX_PRICE: 95¢ → 50¢ (breakeven più facile)
- Calibrazione 0.65 (fix overconfidence 25%)
- BUY_YES min edge: 5% → 8% (penalizza lato debole)
- Rimosso BTC choppy filter (data collection)
- Bankroll: $50 dry run
- **Risultato: 5 trades/ciclo vs 0.36/ciclo, avg price 19¢ vs 64¢**

### v2 — Pre-2026-02-16
- MAX_PRICE: 95¢
- No calibrazione
- BTC choppy filter attivo
- Risultato: -$1.65, 46.2% WR, 64¢ avg price

---

*Questo file è la GOLDEN CONFIG. Se cambi algo, aggiorna qui. Sempre.*
