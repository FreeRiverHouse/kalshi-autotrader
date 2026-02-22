# Grok Kalshi Probability Calibration - 2026-02-16

## Problem
- 0% win rate on 43 settled trades (all BUY_NO)
- Log-normal model underestimates upside probability
- Prices hitting targets more often than predicted

## Grok's Key Recommendations

### 1. Use Student's t-Distribution (NOT log-normal)
- df (degrees of freedom) = 3-6 for crypto hourly
- Captures fat tails that log-normal misses
- 20-50% improvement in forecasting vs normal
- Model: log(S_{t+1}/S_t) ~ t(μ≈0, σ, df)

### 2. Volatility: Use Realized Volatility (RV)
- Compute from 5-min returns within each hour
- RV_h = sum of squared log-returns
- HAR model for forecasting: RV_{t+1} = β0 + β1*RV_t + β2*RV_week + β3*RV_month
- Current BTC hourly std dev: ~0.5-0.7%

### 3. Fat Tail Handling
- Student's t with low df (3-6) captures most fat tail behavior
- Alternatives: Alpha-Stable, GH, NIG distributions
- NEVER use pure log-normal for crypto

### 4. Implementation
- Use t-CDF to compute P(S_{t+1} < K | S_t)
- For probability: scipy.stats.t.cdf() with fitted df
- Recent 30-90 days of data for calibration

## Action Items
1. Replace log-normal CDF with Student's t CDF in heuristic_forecast
2. Estimate df from recent hourly returns data
3. Use realized volatility instead of static BTC_HOURLY_VOL constant
4. Add asymmetry adjustment (higher vol on downsides)
