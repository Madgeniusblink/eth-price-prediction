# Anti-Patterns — Things NOT To Do

**Purpose:** Hard-won lessons from this codebase. Every entry here represents a bug, bad practice, or failure mode we actually hit. Read before touching model code.

**Rule:** Before any PR is merged, the author must confirm none of these patterns are present.

---

## 🔴 Critical (Will Cause Silent Failures)

### AP-001: Stub `main()` that never writes output
**What happened:** `predict_rl.py` had a `main()` that only printed a message and exited. `generate_report.py` called it as a subprocess, assumed `predictions_summary.json` was written, then read the stale January 2026 file. The GitHub Action "succeeded" for months while broadcasting 2-month-old data.

**Rule:** Every script with a `main()` that is called by the pipeline **must**:
1. Write its output file before exiting
2. Verify the output file exists and is fresh after writing
3. Exit with non-zero code on failure (never silent success)

```python
# ❌ WRONG
def main():
    print("Reinforcement learning predictor - coming soon!")

# ✅ CORRECT
def main():
    predictions = run_predictions()
    output_path = Path(BASE_DIR) / "predictions_summary.json"
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    assert output_path.exists(), "FATAL: output file not written"
    print(f"Predictions written to {output_path}")
```

---

### AP-002: Random train/test split on time series data
**Problem:** Using `train_test_split(shuffle=True)` on time series causes **data leakage** — the model sees future data during training, inflating metrics by 10–30%.

**Rule:** ALWAYS split chronologically. Test set = last N% of data. Never shuffle time series.

```python
# ❌ WRONG — data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# ✅ CORRECT — chronological split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

---

### AP-003: Using R² as the primary accuracy metric
**Problem:** R² of 0.99 on Random Forest looked like a win — it was actually severe overfitting. R² measures how well predictions fit the *training* curve, not whether the model predicts *direction* correctly. A model can have R²=0.98 and still be wrong about direction 55% of the time (barely above coin flip).

**Rule:** Primary metrics must be:
1. **Directional Accuracy** — % of times predicted direction (up/down) matches actual. Target: >57%
2. **Out-of-sample RMSE** — error on held-out data, not training data
3. **Sharpe Ratio** — risk-adjusted return if signals were traded

R² is logged for reference only, never used for go/no-go decisions.

---

### AP-004: Deploying without out-of-sample backtest
**Problem:** Models were evaluated only on training data or a tiny holdout. No 30-day out-of-sample test was run before deployment.

**Rule:** Before any model change ships:
- Hold out the last 30 days of data
- Run walk-forward backtest (never look-ahead)
- Directional accuracy on that holdout must be ≥ 57%
- Document results in `data/backtest_results.json`

---

## 🟡 Important (Degrades Quality)

### AP-005: Using stale training data without freshness check
**Problem:** The pipeline ran with CSV files from January when the model needed live data.

**Rule:** Before training, check data freshness:
```python
last_ts = pd.to_datetime(df['timestamp'].max())
age_hours = (datetime.now() - last_ts).total_seconds() / 3600
if age_hours > 1:
    raise ValueError(f"Data is {age_hours:.1f} hours old — fetch fresh data first")
```

---

### AP-006: Single-model overconfidence
**Problem:** The Random Forest got 99% R² and dominated the ensemble. When it overfit, the ensemble failed with it.

**Rule:** No single model should exceed 60% weight in the ensemble. Cap weights:
```python
weights = np.clip(weights, 0.10, 0.60)  # min 10%, max 60% per model
```

---

### AP-007: Not logging prediction outcomes
**Problem:** We have no ground truth record of whether our predictions were right. Can't measure real-world accuracy or trigger model retraining.

**Rule:** Every prediction run must log to `data/prediction_log.csv`:
- timestamp, predicted_price, actual_price (filled in next run), predicted_direction, actual_direction, model_version

---

### AP-008: Hardcoded 500-row lookback without validation
**Problem:** If Binance returns fewer rows (API issue, new listing), the model trains on insufficient data without warning.

**Rule:** Validate minimum row count before any training:
```python
MIN_ROWS = 200
if len(df) < MIN_ROWS:
    raise ValueError(f"Insufficient data: {len(df)} rows (need {MIN_ROWS})")
```

---

## 🟢 Best Practices (Always Follow)

### BP-001: Walk-forward validation is mandatory
Use `walk_forward_backtest()` in `validate.py` — not `rolling_window_backtest()` with random splits.

### BP-002: Every feature must have economic justification
Before adding a feature, ask: "Why would this be a leading indicator for ETH price?" If no answer, don't add it.

### BP-003: Version your models
Store model metadata (training date, data range, metrics) alongside saved models so you can roll back.

### BP-004: Test the pipeline end-to-end, not just the model
Run `python src/generate_report.py` and verify `predictions_summary.json` has a timestamp from the last 5 minutes before declaring success.

---

## Changelog
| Date | Entry | Added By |
|------|-------|---------|
| 2026-03-06 | AP-001 through AP-008, BP-001 through BP-004 | QUANT |
