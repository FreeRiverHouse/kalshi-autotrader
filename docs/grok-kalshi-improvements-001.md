# Grok Kalshi Autotrader Improvements - 2026-02-16

## Context
- 13 trades, 6W/7L (46% win rate), -$1.65 PnL
- BTC losing (-$1.55), ETH roughly breakeven
- Current: MIN_EDGE 1-5%, Kelly 0.15

## Grok's 5 Recommendations

### 1. Increase Minimum Edge Threshold
- Raise MIN_EDGE to 7-10% to filter marginal trades
- Current 1-5% allows too many low-confidence trades
- Should reduce frequency but improve win rate to 60-70%

### 2. Reduce Kelly Fraction
- Cut from 0.15 to 0.05 (one-third)
- Crypto too volatile for aggressive sizing
- Reduces drawdowns 50-75%
- Formula: f* = (edge / odds) * fraction

### 3. Enhance Forecaster with Sentiment
- Add real-time social media sentiment
- Weighted: forecast = 0.7 * heuristic + 0.3 * sentiment
- 10-20% accuracy gains in studies

### 4. Regime-Specific Filters for BTC vs ETH
- BTC: only trade in low-vol regimes (1h ATR < 1% of price)
- ETH: broader regimes but cap exposure to 50%
- Use RSI > 50 for bullish regime triggers

### 5. Add Early Profit-Taking
- Close at 70-80% of max payout if momentum weakens
- Detect via SMA crossover / regime shift
- Locks in gains before reversals

## Action Plan
1. ✅ Raise MIN_EDGE_BUY_YES to 8%, MIN_EDGE_BUY_NO to 5%
2. ✅ Reduce KELLY_FRACTION to 0.05
3. Add ATR-based BTC filter
4. Improve sentiment integration weight
5. Consider take-profit mechanism (if Kalshi supports early exit)
