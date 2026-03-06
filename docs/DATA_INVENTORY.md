# Data Inventory — ETH Price Prediction System

**Last Updated:** 2026-03-06  
**Maintained by:** QUANT (agent67-atlas)

---

## 1. Current Data Assets

| File | Timeframe | Rows | First Timestamp | Last Timestamp | Span | Status |
|------|-----------|------|----------------|---------------|------|--------|
| `eth_1m_data.csv` | 1-minute | 721 | 2026-01-27 04:02 | 2026-01-27 16:02 | ~12 hrs | ⚠️ STALE |
| `eth_5m_data.csv` | 5-minute | 721 | 2026-01-27 04:00 | 2026-01-27 16:00 | ~12 hrs | ⚠️ STALE |
| `eth_15m_data.csv` | 15-minute | 721 | 2026-01-27 04:00 | 2026-01-27 16:00 | ~12 hrs | ⚠️ STALE |
| `eth_4h_data.csv` | 4-hour | 721 | 2026-01-27 04:00 | 2026-01-27 16:00 | ~12 hrs | ⚠️ STALE |

**⚠️ Critical:** All static CSV files are from January 27, 2026 — over 5 weeks old. The live system must fetch fresh data via Binance API at runtime. These files are reference/seed data only.

---

## 2. Data Source

| Source | API | Auth Required | Rate Limit | Cost |
|--------|-----|--------------|------------|------|
| Binance | `https://api.binance.com/api/v3/klines` | No | 1200 req/min | Free |
| CoinGecko Fear & Greed (planned) | `https://api.alternative.me/fng/` | No | 100 req/day | Free |
| Glassnode on-chain (planned) | REST API | Yes (free tier) | 10 req/min | Free tier available |
| LunarCrush social (planned) | REST API | Yes | Varies | Paid |

---

## 3. What the Model Currently Uses

| Data Type | Used? | Source | Lookback |
|-----------|-------|--------|---------|
| 1m OHLCV candles | ✅ Yes | Binance (live fetch) | 500 candles (~8.3 hrs) |
| SMA 5/10/20/50 | ✅ Yes | Derived | — |
| EMA 5/10/20 | ✅ Yes | Derived | — |
| RSI (14) | ✅ Yes | Derived | — |
| MACD (12,26,9) | ✅ Yes | Derived | — |
| Bollinger Bands (20) | ✅ Yes | Derived | — |
| Volume / Volume SMA | ✅ Yes | Derived | — |
| Momentum (10) | ✅ Yes | Derived | — |
| 4h candles | ❌ No | Available | — |
| On-chain data | ❌ No | Not integrated | — |
| Fear & Greed Index | ❌ No | Not integrated | — |
| Social volume | ❌ No | Not integrated | — |
| Order book depth | ❌ No | Not integrated | — |

---

## 4. Minimum Data Requirements

| Requirement | Current | Target | Gap |
|-------------|---------|--------|-----|
| Training lookback | 500 candles (8.3h) | 1000 candles (16.7h) | Need 2x more |
| Backtest out-of-sample | 0 days (none!) | 30 days minimum | **Critical gap** |
| Historical archive | ~12 hours | 6+ months | **Critical gap** |
| Directional ground truth | Not tracked | Rolling 7-day log | Need outcome logger |

---

## 5. Planned Data Additions (Priority Order)

### High Priority
1. **Historical ETH 1m data — 6 months** via Binance API (paginated fetch)
   - Script needed: `scripts/fetch_historical.py`
   - Estimated size: ~260,000 rows
   - Purpose: Proper backtesting and model validation

2. **Fear & Greed Index** (free, daily)
   - API: `https://api.alternative.me/fng/?limit=30`
   - Signals regime (extreme fear = buy zone, extreme greed = caution)

3. **Outcome logger** — track every prediction vs actual
   - Needed for real directional accuracy measurement

### Medium Priority
4. **4h candle regime detection** (already have file, not integrated)
5. **Binance order book snapshot** (bid/ask imbalance signal)

### Low Priority
6. Glassnode on-chain: exchange netflow, whale transactions
7. LunarCrush: social volume, sentiment score

---

## 6. Data Quality Rules

- [ ] Freshness: Data must be < 1 hour old before generating predictions
- [ ] Completeness: < 1% missing candles acceptable; > 1% = abort and alert
- [ ] Outlier detection: Single-candle moves > 5% flagged for review
- [ ] Timestamps: Must be monotonically increasing, no duplicates
- [ ] Minimum rows: 200 rows required before model training; < 200 = error, not warning

---

## 7. Data Audit Command

Run at any time to get a full data health report:
```bash
python src/data_audit.py
```
Output saved to: `data/data_audit_report.json`
