# Grok Win Rate Analysis - 2026-02-16

## 1. Breakeven Win Rate
**Formula: breakeven WR = purchase price (as fraction)**
- At 64¢ avg → need **64% WR** to break even
- At 50¢ → need 50% WR
- At 40¢ → need 40% WR  
- At 30¢ → need 30% WR
- Current: 46.2% WR at 64¢ avg → LOSING

## 2. KEY INSIGHT: Trade at LOWER prices!
- Se compri a 30¢ → breakeven è solo 30% WR
- Se compri a 25¢ → breakeven è solo 25% WR
- Attualmente compriamo troppo caro (64¢ avg)

## 3. Calibration Fixes (Overconfident by 25%)
1. Build calibration curve (need 100+ trades)
2. Platt scaling or isotonic regression
3. Quick fix: multiply predicted probs by 0.65 (ratio 46.2/71.5)
4. Diagnose directional bias (YES vs NO separately)
5. More data needed (13 trades is too few)

## 4. Realistic Target Win Rate
- 52-60% for automated crypto binary options
- 55%+ to ensure profitability after fees
- 80%+ is unrealistic/unsustainable

## 5. Focus BUY_NO?
- YES for now (50% WR vs 33% YES)
- Set higher edge threshold for BUY_YES
- Allocate 70-80% to NO while monitoring
- Crypto upward bias makes downside predictions easier

## ACTION ITEMS
1. Lower avg trade price (target <40¢ for easier breakeven)
2. Apply 0.65 scaling factor to predicted probs
3. Bias toward BUY_NO (higher edge threshold for YES)
4. Collect 100+ settled trades for proper calibration
