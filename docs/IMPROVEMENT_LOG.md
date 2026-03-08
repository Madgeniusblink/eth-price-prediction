# ETH Prediction — Improvement Log

Tracks each self-improvement cycle: date, what changed, before/after metrics.

---

## 2026-03-08 — Regime Detection + Mean-Reversion Ensemble

**Cycle:** QUANT daily self-improvement (Sunday 2026-03-08, ~20:15 UTC)

### Problem diagnosed
The baseline `walk_forward_backtest()` used **LinearRegression on raw prices** with a
300-candle training window. Linear trend extrapolation creates a **bullish bias** — the
model always predicts the recent trend continues.

Market structure analysis on the 721-row Kraken 1h dataset revealed:
- Direction continuation (lag-1 autocorrelation): **46.3%** → market is MEAN-REVERTING
- Pure momentum strategies: 46–48% accuracy (worse than random)
- Pure 1h mean-reversion: **53.7%** accuracy — correct model class identified

### Change implemented
**`src/validate.py` — `walk_forward_backtest()` rewritten:**

Replaced `LinearRegression` with a **vote-based mean-reversion ensemble**:

| Signal | Weight | Logic |
|--------|--------|-------|
| 1h mean-reversion | ×2 | If last 1h UP → predict DOWN; if DOWN → predict UP |
| 2h mean-reversion | ×1 | Same, 2h lag |
| RSI-14 extreme    | ×1 | RSI ≥ 65 → DOWN; RSI ≤ 35 → UP; neutral → abstain |
| 200-MA regime     | ×1 | BULL (price > MA×1.005) → -1; BEAR (< MA×0.995) → +1; NEUTRAL → abstain |

No ML model training per step — O(1) signal computation, CI-fast (<1 sec).

**`main.py` — bug fix:** Backtest now reads `data/eth_1h_historical_6mo.csv` (Kraken 1h
data) instead of the missing `eth_1m_data.csv`.

### Results

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Directional accuracy | 51.43% | **53.57%** | +2.14pp |
| RMSE | $71.42 | $22.57 | ↓ (synthetic price artifact) |
| Gate pass (≥57%) | ❌ FAIL | ❌ FAIL | — |
| Regime accuracy: BULL | — | 56.4% | ✓ |
| Regime accuracy: NEUTRAL | — | 56.2% | ✓ |
| Regime accuracy: BEAR | — | 51.1% | ← needs work |

### Next steps (priority order)
1. **Improve BEAR regime accuracy** (51.1% → 55%+): BEAR regime needs stronger DOWN signals;
   consider adding volume-weighted momentum or gas fee data as bearish confirmer.
2. **Add Fear & Greed Index** (Alternative.me API, sleep 10s) — macro sentiment signal.
3. **Tune signal thresholds** (RSI extreme: try 60/40 vs 65/35 cutoffs).
4. Gate target: 57% directional accuracy.

---
