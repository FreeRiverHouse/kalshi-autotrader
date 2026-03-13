# Kalshi AutoTrader — ROADMAP & MOP

**Obiettivo:** Migliorare l'algoritmo per raggiungere ROI positivo in paper mode, poi live.

**Principio:** Qualità > Quantità. Bettare meno ma su mercati dove l'algo ha edge reale.

---

## 📋 TASK LIST (Priorità Alta → Bassa)

### 🔴 CRITICO

#### TASK-001: Fix forecaster crypto — log-normal con dati reali
**Status:** IN PROGRESS
**Problema:** `_heuristic_crypto` usa il modello giusto (log-normal/t-distribution) ma spesso fallisce su "No crypto price" quando il context non passa i prezzi correttamente.
**Fix:**
1. Verificare che `context["crypto_prices"]` sia sempre popolato prima di chiamare `heuristic_forecast()`
2. Aggiungere fallback con Binance API diretta se CoinGecko fallisce
3. Validare che il prezzo corrente BTC/ETH sia fresco (< 5 min)
4. Log ogni forecast con: ticker, prezzo attuale, strike, DTE, vol, prob calcolata vs prezzo mercato
**Test:** Il 90%+ dei mercati KXBTCD/KXETHD deve avere forecast diverso da 50.0%

---

#### TASK-002: Integra Grok per mercati non-crypto (politics, economy, sports)
**Status:** TODO
**Problema:** L'heuristic ritorna 50% su tutti i mercati non-crypto (moneyline, totals, spread) perché non ha informazioni esterne.
**Fix:**
1. Usa la Grok API (xAI, key in `~/.clawdbot/clawdbot.json`) per mercati non-crypto
2. Prompt conciso: ticker + titolo + YES_PRICE + domanda "qual è la vera probabilità?"
3. Parsare risposta JSON con probability + reasoning
4. Fallback a heuristic se Grok fallisce
5. Rate limit: max 50 chiamate/ciclo (Grok è veloce e cheap)
**File da modificare:** `kalshi-autotrader.py` → `forecast_market_llm()` — sostituisci Anthropic con Grok
**Nota:** Modello da usare: `grok-3-fast` via `https://api.x.ai/v1/chat/completions`

---

#### TASK-003: Traccia win rate per tipo di mercato
**Status:** TODO
**Problema:** Non sappiamo su quali tipi di mercato l'algo vince. Stiamo bettando alla cieca.
**Fix:**
1. Aggiungere campo `market_type` al trade log (crypto/politics/sports/economy/other)
2. Dashboard: mostra tabella win rate per tipo
3. Ogni settimana: disabilita tipi con WR < 40%, boost quelli con WR > 60%
4. Salva in `data/trading/win-rate-by-type.json`
**Priorità:** Alta — senza questi dati non possiamo ottimizzare

---

#### TASK-004: Settlement automatico posizioni scadute
**Status:** TODO  
**Problema:** Le posizioni paper rimangono "open" per sempre, il balance non si aggiorna mai, max_positions si riempie.
**Fix:**
1. Ogni ciclo: controlla tutte le posizioni aperte nel paper state
2. Per ogni posizione: chiama `GET /trade-api/v2/markets/{ticker}` — se `result = "yes"/"no"`, settle
3. Aggiorna balance: se risultato = azione (es. BUY_YES + result=yes) → guadagno, altrimenti → perdita
4. Log ogni settlement in `closed_trades`
5. Aggiungere `manage_paper_settlements()` chiamata all'inizio di ogni ciclo
**File:** `kalshi-autotrader.py` vicino a `manage_positions()`

---

### 🟠 ALTO

#### TASK-005: Escludi mercati 50/50 puri (YES=50 NO=50 Vol=0)
**Status:** DONE (parziale) — MIN_VOLUME=0 li include, ma il forecaster li bypassa con edge~0
**Fix migliorato:**
1. In `filter_markets()`: skip mercati con `yes_price == 50 AND volume == 0 AND open_interest == 0`
2. Eccezione: mantieni mercati 50/50 se DTE < 2 ore (possono muoversi veloce)
3. Questo riduce i mercati da 5255 → ~500 mercati con segnale reale

---

#### TASK-006: Calibra Kelly sull'edge reale
**Status:** TODO
**Problema:** Kelly 0.50 è applicato a tutto, anche su edge=2%. Troppo aggressivo su edge basso.
**Fix:**
1. `kelly_frac = min(0.50, edge * 3.0)` — bet 50% solo se edge > 16%, scale lineare sotto
2. Esempio: edge=5% → kelly=15%, edge=10% → kelly=30%, edge=20% → kelly=50%
3. Minimum bet: 1% se edge > 0 (raccolta dati), 0 se edge ≤ 0 (skip)

---

#### TASK-007: Dashboard — fix ROI e bankroll chart
**Status:** DONE (ROI fix) — bankroll chart ancora mostra dati Feb
**Fix rimanente:**
1. Bankroll series: leggi da `paper-trade-state.json` (closed_trades) invece di SQLite
2. Aggiungi colonna "oggi vs ieri" per vedere trend giornaliero
3. Mostra breakdown per tipo: crypto/politics/sports

---

### 🟡 MEDIO

#### TASK-008: Auto-tune parametri basato su performance
**Status:** TODO
**Idea:** Ogni 24h, analizza ultime 100 trade chiuse e aggiusta:
- Se WR crypto > 60%: aumenta MAX_POSITION_PCT su crypto
- Se WR politics < 40%: abbassa peso o disabilita
- Se drawdown > 20%: riduci Kelly temporaneamente
**File:** nuovo script `kalshi-auto-tune.py` con LaunchAgent giornaliero

---

#### TASK-009: News feed per mercati politics/economy
**Status:** TODO
**Idea:** Per mercati come "Fed rate cut?", "Trump approval > 45%?", fetcha headline da:
- Google News RSS
- Kalshi blog/forum
- Usa Grok per interpretare impatto
**Nota:** Solo per mercati con DTE > 1 giorno

---

#### TASK-010: Backtesting su dati storici
**Status:** TODO
**Idea:** Usa i trade del periodo Feb 23 — Mar 12 (896 settled) per:
1. Re-run heuristic su quei mercati
2. Confronta previsione heuristic vs risultato reale
3. Calcola calibration error
4. Usa output per migliorare i coefficienti del modello
**File:** `kalshi-backtest.py` (già esiste, da estendere)

---

## 📏 MOP — COME IMPLEMENTARE UN TASK

1. **Leggi il task** completo prima di toccare codice
2. **Non modificare manualmente** script complessi — usa Grok API per snippet complessi
3. **Test locale** su M1 prima di riavviare il trader
4. **Commit e push** dopo ogni task completato
5. **Aggiorna questo file** — cambia Status da TODO a IN PROGRESS a DONE
6. **Non disabilitare** trading durante fix — usa `--dry-run` è già attivo

---

## 📊 METRICHE TARGET

| Metrica | Attuale | Target 1 settimana | Target 1 mese |
|---------|---------|-------------------|---------------|
| Win Rate | 54.5% | 58% | 62% |
| ROI | -8.5% | 0% | +15% |
| Trade/giorno | 200+ | 300+ | 400+ |
| Edge medio | ~0% | 5% | 10% |

---

## ⚙️ CONFIG ATTUALE (2026-03-12)

```
--kelly 0.50 --min-edge -1.0 --max-trades 200 --markets 500 --loop 60
Paper bankroll: $10,000
Forecaster: HEURISTIC (LLM Anthropic key invalid)
MIN_VOLUME: 0, MIN_LIQUIDITY: 0
MAX_POSITION_PCT: 15%
```

---

## 🚫 REGOLE FERREE

- **MAI** modificare chiavi API senza backup
- **MAI** passare a LIVE prima di WR > 55% su almeno 500 settled trades paper
- **MAI** Kelly > 0.25 su LIVE
- **SEMPRE** testare in paper prima di ogni cambio di strategia
- **SEMPRE** committare prima di modificare parametri critici
