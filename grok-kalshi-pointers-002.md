# Grok Kalshi Pointers #002 — 2026-02-16

## Source: Grok via Mattia, based on Grok 4.20 Prediction Arena performance

### Key Stats from Grok 4.20
- +10-35% returns in weeks
- +34.59% run as of Feb 15, 2026
- Average Kalshi contract lost -22%
- 299+ trades
- Uses weighted LLM ensemble (Grok 30% lead + Claude Sonnet + GPT-4o)

### Reference Repo
- `ryanfrigo/kalshi-ai-trading-bot` — public GitHub repo with Grok-integrated algo
- Key file: `src/agents/forecaster.py`
- Run: `python beast_mode_bot.py --paper`

---

## 5 POINTERS (Priority Order)

### 1. Multi-Agent Ensemble Forecasting
- Grok API as primary (30% weight)
- Claude for sentiment/bias check (20-25%)
- GPT-4o for regime detection (20-25%)
- Consensus vote: skip if confidence < 0.50
- Target: edge lift from 8.8% → 10-12%

### 2. Diversify Beyond BUY_YES
- 60% directional, 30% market-making, 10% arb
- Market-making: scan for wide spreads (>2-3¢)
- Arb: compare correlated contracts (BTC hourly vs daily)
- Force 30% BUY_NO on F&G < 30
- In choppy: shift 20% from directional to market-making

### 3. Fractional Kelly + Caps
- 0.75x Kelly (like Grok), cap 5% per trade
- Max 15 open positions
- Max 90% sector concentration
- Choppy: multiply size by 0.5
- High confidence (>0.7): up to 1.25x
- First live: 0.5-1% of bankroll ($0.50-1 per trade on $100)

### 4. Risk Controls
- 15% per-position stop loss
- 20% trailing take profit
- Time exit: 10 days or confidence decay
- 15% daily loss → pause 24h
- 50% max drawdown → full stop
- No averaging down — exit losers fully

### 5. Validation Plan
- Wait 20-30 settled paper trades
- Win rate > 28% AND edge holds → micro-live $100
- Run parallel: our bot vs forked repo
- After 20 settled live: scale 20-50%
- Target: Grok's +10% monthly benchmark
