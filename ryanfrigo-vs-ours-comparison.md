# Ryan Frigo Repo vs Our Autotrader — Comparison

## Architecture Differences

### Ryan Frigo (ryanfrigo/kalshi-ai-trading-bot)
- **6 LLM agents** in ensemble: Forecaster (Grok 30%), News Analyst (Claude 20%), Bull Researcher (GPT 20%), Bear Researcher (Gemini 15%), Risk Manager (DeepSeek 15%), Trader (Grok - final decision)
- **Debate protocol**: Bull/Bear present cases, Risk Manager evaluates, Trader decides
- Confidence threshold: 0.50
- Max ensemble cost per decision: $0.50
- RSS news feeds for sentiment
- Parallel agent execution
- Calibration tracking over time

### Ours (kalshi-autotrader.py)
- **Heuristic forecaster** (no LLM calls): log-normal/t-distribution model
- Fear & Greed index for sentiment
- Regime detection (trending/choppy/sideways)
- Calibration factor 0.65
- Single-pass: compute probability → compare to market → trade

## Key Params Comparison

| Parameter | Ryan Frigo | Ours |
|-----------|-----------|------|
| Kelly fraction | not explicit (agents decide) | 0.75x |
| Max per trade | 5% of capital | $5 (5% of $100) |
| Max positions | 15 | 15 |
| Min confidence | 0.50 | N/A (edge-based) |
| Min edge | 10% (EV threshold) | 4% YES / 2% NO |
| Daily loss limit | 15% | not implemented yet |
| Scan interval | 30s | 300s |

## What We're Missing (Priority)
1. ✅ Kelly sizing — done (0.75x)
2. ⚠️ Daily loss limit (15%) — IMPLEMENT
3. ⚠️ Per-position stop loss (15%) — IMPLEMENT
4. ⚠️ Trailing take profit (20%) — IMPLEMENT
5. 🔲 Higher scan frequency (300s → 60s?)
6. 🔲 Market-making strategy (wide spreads)
7. 🔲 Arb detection (correlated contracts)
8. 🔲 RSS news feeds
9. 🔲 Multi-agent ensemble (needs API keys or Grok web workaround)

## Strategy: Use Grok Web as Code Reviewer
Since we don't have xAI API key, plan is:
1. Implement all heuristic improvements
2. Create public repo with just the algo
3. Send link to Grok via x.com/i/grok for code review
4. Iterate based on feedback
