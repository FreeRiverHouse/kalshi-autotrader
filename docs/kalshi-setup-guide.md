# Kalshi AI Trading Bot - Setup Guide

**Created**: 2025-01-27
**Project**: `/Users/mattia/Projects/kalshi-ai-trading-bot`
**Status**: Setup completato, pronto per configurazione

---

## 🎯 Overview

Bot di trading AI per Kalshi prediction markets usando:
- **Grok-4** (xAI) per analisi multi-agent (Forecaster → Critic → Trader)
- **Kelly Criterion** per position sizing ottimale
- **Risk management** integrato con exit strategies dinamiche

---

## 📋 Step 1: Registrazione Kalshi

### 1.1 Crea Account
1. Vai su https://kalshi.com
2. Clicca "Sign Up"
3. **Requisiti**: 
   - Residenza USA (o VPN con verifica)
   - Età 18+
   - SSN o documento per KYC

### 1.2 Verifica Account
- Upload documento ID
- Verifica indirizzo
- Tempo: 1-3 giorni lavorativi

### 1.3 Deposita Fondi
- **Minimo consigliato**: $50-100 per iniziare
- Metodi: ACH, Wire, Debit Card
- ACH è gratuito (2-3 giorni)

### 1.4 Genera API Key
1. Login → **Settings** → **API**
2. Clicca "Generate API Key"
3. **SALVA LA PRIVATE KEY** - mostrata una sola volta!
4. Copia in `.env` come `KALSHI_API_KEY`

---

## 📋 Step 2: xAI API Key (Grok-4)

### 2.1 Registrazione
1. Vai su https://console.x.ai/
2. Crea account o login con X (Twitter)

### 2.2 Genera API Key
1. Dashboard → **API Keys**
2. "Create new key"
3. Copia in `.env` come `XAI_API_KEY`

### 2.3 Pricing xAI
- Grok-4: ~$0.002/1K input, ~$0.008/1K output (stima)
- Budget giornaliero consigliato: $5-15
- Il bot ha limiti integrati (`DAILY_AI_COST_LIMIT`)

---

## 📋 Step 3: Setup Ambiente

```bash
# 1. Vai nella directory
cd /Users/mattia/Projects/kalshi-ai-trading-bot

# 2. Crea virtual environment
python -m venv venv
source venv/bin/activate

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Copia e configura .env
cp .env.example .env
# Edita .env con le tue API keys

# 5. Inizializza database
python init_database.py
```

---

## 💰 Step 4: Configurazione per $50-100

### Parametri Consigliati per Capitale Limitato

Modifica `src/config/settings.py`:

```python
# === POSITION SIZING CONSERVATIVO ===
max_position_size_pct: float = 3.0     # Max 3% per trade ($1.50-3 su $50-100)
max_positions: int = 5                  # Max 5 posizioni simultanee
min_balance: float = 20.0               # Stop trading sotto $20

# === RISK MANAGEMENT STRETTO ===
max_daily_loss_pct: float = 10.0        # Stop dopo -10% giornaliero
kelly_fraction: float = 0.25            # Kelly conservativo (1/4)
max_single_position: float = 0.03       # Max 3% per posizione

# === AI SPENDING ===
daily_ai_budget: float = 3.0            # Max $3/giorno in AI
max_ai_cost_per_decision: float = 0.05  # Max $0.05 per analisi

# === CONFIDENCE THRESHOLDS ===
min_confidence_to_trade: float = 0.65   # Solo trade ad alta confidenza
min_trade_edge: float = 0.12            # Edge minimo 12%
```

### Variabili .env per Start Conservativo

```bash
LIVE_TRADING_ENABLED=false   # INIZIA SEMPRE IN PAPER MODE!
PAPER_TRADING_MODE=true
DAILY_AI_COST_LIMIT=5.0
```

---

## 🚀 Step 5: Primo Avvio

### Paper Trading (Consigliato per 1-2 settimane)
```bash
# Assicurati che .env abbia:
# LIVE_TRADING_ENABLED=false
# PAPER_TRADING_MODE=true

python beast_mode_bot.py
```

### Monitoraggio
```bash
# Dashboard web
python launch_dashboard.py

# Performance analysis
python performance_analysis.py
```

---

## ⚠️ Checklist Pre-Trading Reale

- [ ] API keys configurate e testate
- [ ] Paper trading per almeno 1 settimana
- [ ] Verificato performance simulata positiva
- [ ] Compreso i rischi (puoi perdere tutto)
- [ ] Depositato SOLO soldi che puoi perdere
- [ ] Configurato `max_daily_loss_pct` 
- [ ] Backup delle API keys in posto sicuro

---

## 📊 Comandi Utili

```bash
# Check posizioni attuali
python get_positions.py

# Analisi performance
python quick_performance_analysis.py

# Health check portfolio
python portfolio_health_check.py

# Sync posizioni da Kalshi
python sync_positions.py
```

---

## 🔴 WARNING: Rischi

1. **Perdita capitale**: Il trading comporta rischi sostanziali
2. **Costi AI**: Grok-4 ha costi per chiamata - monitora spese
3. **Mercati prediction**: Meno liquidi dei mercati tradizionali
4. **Bug software**: È software sperimentale
5. **Regulatory**: Kalshi è regolato CFTC ma le regole possono cambiare

---

## 📝 Note

- Repository: `kalshi-ai-trading-bot`
- Multi-agent: Forecaster → Critic → Trader (Grok-4)
- Database: SQLite locale (`trading_system.db`)
- Logs: `logs/trading_system.log`
