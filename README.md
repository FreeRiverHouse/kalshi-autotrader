# Kalshi Crypto Autotrader

Heuristic-based prediction market autotrader for Kalshi crypto markets (BTC/ETH).

## Features
- **No LLM dependency**: Pure heuristic forecaster (log-normal/t-distribution model)
- **Fear & Greed sentiment**: CoinMarketCap F&G index integration
- **Regime detection**: Trending/choppy/sideways with dynamic edge adjustment
- **Student-t distribution**: Heavy tails for crypto volatility (df=4)
- **Calibration factor**: 0.65 (adjustable)
- **Paper trading mode**: Full simulation before going live
- **Position management**: Quarter-Kelly sizing, max $2/trade, 15 position limit
- **Risk controls**: Daily loss limit, max drawdown circuit breaker

## Architecture
Single-file Python script (~3100 lines) with:
- Kalshi API client (REST + settlement checking)
- Crypto price fetcher (CoinGecko)
- Heuristic probability forecaster
- Position sizing (fractional Kelly criterion)
- Paper trade state management

## Inspired By
- [ryanfrigo/kalshi-ai-trading-bot](https://github.com/ryanfrigo/kalshi-ai-trading-bot) - Grok 4.20 ensemble architecture
- Grok's Prediction Arena performance (+34.59% ROI)

## Current Status
- Paper trading mode, collecting settlement data
- Target: prove profitability on $100 paper bankroll before going live

## Usage
```bash
# Paper mode (default)
python kalshi-autotrader.py --loop 300

# With custom parameters
python kalshi-autotrader.py --loop 60 --kelly 0.25 --min-edge 0.04
```

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Kelly fraction | 0.25 | Quarter-Kelly (conservative) |
| Max bet | $2 | 2% of $100 bankroll |
| Max positions | 15 | Concurrent open positions |
| Min edge (YES) | 4% | Minimum edge to buy YES |
| Min edge (NO) | 2% | Minimum edge to buy NO |
| Calibration | 0.65 | Shrink raw probabilities toward 50% |
| Daily loss limit | $15 | 15% of bankroll |

## License
MIT
