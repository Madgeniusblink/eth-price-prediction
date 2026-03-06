#!/usr/bin/env python3
"""
Data Audit Script — ETH Price Prediction System
Run this to get a full inventory and health check of all data assets.

Usage:
    python src/data_audit.py

Output:
    - Console: human-readable report
    - data/data_audit_report.json: machine-readable audit results
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Resolve BASE_DIR relative to this script
BASE_DIR = Path(__file__).parent.parent


def audit_csv_file(filepath: Path) -> dict:
    """Audit a single CSV data file."""
    result = {
        "file": str(filepath.name),
        "path": str(filepath),
        "exists": filepath.exists(),
    }

    if not filepath.exists():
        result["status"] = "MISSING"
        return result

    try:
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        result["rows"] = len(df)
        result["columns"] = list(df.columns)
        result["first_timestamp"] = str(df["timestamp"].min())
        result["last_timestamp"] = str(df["timestamp"].max())

        # Staleness
        last_ts = df["timestamp"].max()
        if hasattr(last_ts, "to_pydatetime"):
            last_ts = last_ts.to_pydatetime()
        age_hours = (datetime.now() - last_ts).total_seconds() / 3600
        result["age_hours"] = round(age_hours, 1)
        result["stale"] = age_hours > 1.0

        # Span
        span_hours = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
        result["span_hours"] = round(span_hours, 1)

        # Gap detection (for 1m files: gaps > 2 minutes)
        if len(df) > 1:
            diffs = df["timestamp"].diff().dt.total_seconds().dropna()
            median_interval = diffs.median()
            gap_threshold = median_interval * 3
            gaps = (diffs > gap_threshold).sum()
            result["gaps_detected"] = int(gaps)
            result["median_interval_seconds"] = round(median_interval, 0)
        else:
            result["gaps_detected"] = 0

        # Outlier detection (close price moves > 5% in one candle)
        pct_change = df["close"].pct_change().abs()
        outliers = (pct_change > 0.05).sum()
        result["outliers_detected"] = int(outliers)

        # Quality score (0-100)
        score = 100
        if result["stale"]:
            score -= 40
        if result["gaps_detected"] > 5:
            score -= 20
        elif result["gaps_detected"] > 0:
            score -= 10
        if result["outliers_detected"] > 3:
            score -= 10
        if result["rows"] < 200:
            score -= 30
        result["quality_score"] = max(0, score)

        if result["stale"]:
            result["status"] = "STALE"
        elif result["quality_score"] >= 80:
            result["status"] = "OK"
        elif result["quality_score"] >= 50:
            result["status"] = "WARNING"
        else:
            result["status"] = "CRITICAL"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


def check_prediction_log() -> dict:
    """Check if prediction outcome logging is active."""
    log_path = BASE_DIR / "data" / "prediction_log.csv"
    result = {
        "file": "data/prediction_log.csv",
        "exists": log_path.exists(),
    }
    if log_path.exists():
        df = pd.read_csv(log_path)
        result["rows"] = len(df)
        result["status"] = "ACTIVE" if len(df) > 0 else "EMPTY"
    else:
        result["status"] = "MISSING"
        result["recommendation"] = "Outcome logging not set up — cannot measure real directional accuracy"
    return result


def check_predictions_freshness() -> dict:
    """Check if predictions_summary.json is fresh."""
    pred_path = BASE_DIR / "predictions_summary.json"
    result = {"file": "predictions_summary.json", "exists": pred_path.exists()}
    if pred_path.exists():
        with open(pred_path) as f:
            data = json.load(f)
        generated_at = data.get("generated_at", "unknown")
        result["generated_at"] = generated_at
        if generated_at != "unknown":
            try:
                gen_dt = datetime.fromisoformat(generated_at)
                age_minutes = (datetime.now() - gen_dt).total_seconds() / 60
                result["age_minutes"] = round(age_minutes, 1)
                result["stale"] = age_minutes > 60
                result["status"] = "STALE" if result["stale"] else "FRESH"
            except Exception:
                result["status"] = "UNKNOWN"
    else:
        result["status"] = "MISSING"
    return result


def generate_recommendations(audits: list, pred_status: dict, log_status: dict) -> list:
    """Generate actionable recommendations from audit results."""
    recs = []

    stale_files = [a for a in audits if a.get("stale")]
    if stale_files:
        recs.append({
            "priority": "HIGH",
            "action": f"{len(stale_files)} data file(s) are stale. Run: python src/fetch_data.py",
        })

    gap_files = [a for a in audits if a.get("gaps_detected", 0) > 5]
    if gap_files:
        recs.append({
            "priority": "MEDIUM",
            "action": f"Data gaps found in: {[a['file'] for a in gap_files]}. Refetch or backfill.",
        })

    if pred_status.get("stale") or pred_status.get("status") == "MISSING":
        recs.append({
            "priority": "HIGH",
            "action": "predictions_summary.json is stale/missing. Run: python src/generate_report.py",
        })

    if log_status.get("status") == "MISSING":
        recs.append({
            "priority": "HIGH",
            "action": "Prediction outcome logger not set up. Real directional accuracy cannot be measured. Implement src/outcome_logger.py",
        })

    all_rows = [a.get("rows", 0) for a in audits]
    if any(r < 500 for r in all_rows if r > 0):
        recs.append({
            "priority": "MEDIUM",
            "action": "Some files have < 500 rows. Run: python scripts/fetch_historical.py to build 6-month archive.",
        })

    if not recs:
        recs.append({"priority": "INFO", "action": "All data assets look healthy."})

    return recs


def print_report(audits: list, pred_status: dict, log_status: dict, recs: list):
    """Print human-readable audit report."""
    print("\n" + "=" * 60)
    print("  ETH PRICE PREDICTION — DATA AUDIT REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n📁 DATA FILES\n")
    print(f"  {'File':<25} {'Rows':>6} {'Span':>8} {'Age(h)':>7} {'Gaps':>5} {'Score':>6} {'Status'}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*7} {'-'*5} {'-'*6} {'-'*8}")
    for a in audits:
        if a.get("exists"):
            print(
                f"  {a['file']:<25} {a.get('rows', 'N/A'):>6} "
                f"{str(a.get('span_hours','?'))+'h':>8} "
                f"{str(a.get('age_hours','?'))+'h':>7} "
                f"{a.get('gaps_detected', 0):>5} "
                f"{a.get('quality_score', 0):>6} "
                f"{a.get('status', 'UNKNOWN')}"
            )
        else:
            print(f"  {a['file']:<25} {'MISSING':>6}")

    print(f"\n📊 PREDICTIONS\n")
    print(f"  predictions_summary.json: {pred_status.get('status')} "
          f"(age: {pred_status.get('age_minutes', 'N/A')} min)")

    print(f"\n📝 OUTCOME LOGGING\n")
    print(f"  prediction_log.csv: {log_status.get('status')} "
          f"({log_status.get('rows', 0)} entries)")

    print(f"\n⚡ RECOMMENDATIONS\n")
    for r in recs:
        icon = "🔴" if r["priority"] == "HIGH" else "🟡" if r["priority"] == "MEDIUM" else "🟢"
        print(f"  {icon} [{r['priority']}] {r['action']}")

    print("\n" + "=" * 60 + "\n")


def main():
    print("Running data audit...")

    # Audit all CSV files
    csv_files = [
        BASE_DIR / "eth_1m_data.csv",
        BASE_DIR / "eth_5m_data.csv",
        BASE_DIR / "eth_15m_data.csv",
        BASE_DIR / "eth_4h_data.csv",
    ]
    audits = [audit_csv_file(f) for f in csv_files]

    # Check predictions freshness
    pred_status = check_predictions_freshness()

    # Check outcome logging
    log_status = check_prediction_log()

    # Generate recommendations
    recs = generate_recommendations(audits, pred_status, log_status)

    # Print report
    print_report(audits, pred_status, log_status, recs)

    # Save JSON report
    report = {
        "generated_at": datetime.now().isoformat(),
        "data_files": audits,
        "predictions_status": pred_status,
        "outcome_log_status": log_status,
        "recommendations": recs,
        "overall_health": "OK" if all(a.get("status") in ("OK", "WARNING") for a in audits if a.get("exists")) else "DEGRADED",
    }

    output_path = BASE_DIR / "data" / "data_audit_report.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Audit saved to: {output_path}")

    return report


if __name__ == "__main__":
    main()
