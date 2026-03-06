# Verification Checklist — Before Any Model Change Goes Live

**Purpose:** Every PR that touches model code, feature engineering, or the prediction pipeline must complete this checklist. No exceptions.

**How to use:** Copy this checklist into your PR description and check each item before requesting review.

---

## Pre-Flight: Data Quality

- [ ] **Data freshness:** Confirmed training data is < 1 hour old
- [ ] **Row count:** Minimum 200 rows present before training
- [ ] **No gaps > 5 minutes** in 1m candle data (run `python src/data_audit.py`)
- [ ] **No outliers flagged** (single-candle moves > 5%)
- [ ] **Timestamp monotonic:** No duplicates, no out-of-order candles

---

## Model Validation

- [ ] **Walk-forward backtest run** (NOT random split — see ANTI_PATTERNS.md AP-002)
- [ ] **Directional accuracy ≥ 57%** on out-of-sample holdout (last 30 days)
- [ ] **RMSE is lower or equal to previous baseline** (check `data/backtest_results.json`)
- [ ] **Sharpe ratio > 0.5** on simulated signals from backtest
- [ ] **No model weight > 60%** in ensemble (ANTI_PATTERNS.md AP-006)
- [ ] **R² documented** for reference (but NOT used as go/no-go metric — AP-003)

---

## Pipeline Integrity

- [ ] **main() writes output file** and verifies it exists (ANTI_PATTERNS.md AP-001)
- [ ] **`predictions_summary.json` timestamp** is within last 5 minutes after pipeline run
- [ ] **End-to-end test:** Run `python src/generate_report.py` locally — no errors
- [ ] **GitHub Action dry-run:** Triggered workflow on branch, all steps green
- [ ] **Outcome logger active:** `data/prediction_log.csv` is being appended to (AP-007)

---

## Documentation

- [ ] **METHODOLOGY.md updated** if modeling approach changed
- [ ] **DATA_INVENTORY.md updated** if new data sources added/removed
- [ ] **ANTI_PATTERNS.md updated** if a new failure mode was discovered
- [ ] **This checklist** copied into PR description and all items checked

---

## Anti-Pattern Review

Confirm NONE of the following are present in your changes:

- [ ] No stub `main()` functions (AP-001)
- [ ] No `shuffle=True` in any train/test split (AP-002)
- [ ] R² is NOT the primary success metric (AP-003)
- [ ] Out-of-sample backtest was run (AP-004)
- [ ] Data freshness validated at runtime (AP-005)
- [ ] No single model > 60% ensemble weight (AP-006)
- [ ] Predictions being logged to `data/prediction_log.csv` (AP-007)
- [ ] Minimum row count validated before training (AP-008)

---

## Live Trading Gate (additional — only when trading real funds)

- [ ] **Paper trading ran for ≥ 14 days** with directional accuracy ≥ 57%
- [ ] **Max drawdown < 15%** in paper trading period
- [ ] **Sharpe ratio > 0.8** over paper trading period
- [ ] **Win rate > 50%** over paper trading period
- [ ] **Position sizing rules confirmed:** Max 20% per trade, max 10% daily loss limit
- [ ] **Stop-loss logic verified** in code (not just documented)
- [ ] **QUANT + Cris approval** before first live trade

---

## Sign-Off

| Role | Name | Date | Notes |
|------|------|------|-------|
| Author | | | |
| Reviewer | cristian-bloclabs | | |

---

*Checklist version: 1.0 — 2026-03-06*  
*If this checklist feels excessive, re-read ANTI_PATTERNS.md AP-001.*
