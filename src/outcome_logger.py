"""
Outcome Logger — tracks prediction accuracy over time.
Logs predictions to data/prediction_log.csv and validates outcomes.
"""

import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

# Resolve paths relative to repo root
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SRC_DIR)
_LOG_PATH = os.path.join(_BASE_DIR, "data", "prediction_log.csv")

_COLUMNS = [
    "timestamp_made",
    "target_time",
    "predicted_price",
    "confidence",
    "model_weights",
    "actual_price",
    "direction_correct",
    "pct_error",
]


def _load_log() -> pd.DataFrame:
    """Load the prediction log, returning an empty DataFrame if missing."""
    if os.path.exists(_LOG_PATH):
        try:
            df = pd.read_csv(_LOG_PATH, parse_dates=["timestamp_made", "target_time"])
            # Ensure all columns exist
            for col in _COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan
            return df[_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=_COLUMNS)


def _save_log(df: pd.DataFrame) -> None:
    """Persist the prediction log to disk."""
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    df.to_csv(_LOG_PATH, index=False)


def _fetch_binance_price(target_time: datetime) -> float | None:
    """
    Fetch the closing price of the 1-hour candle that contains target_time.
    Returns None on any error.
    """
    try:
        # Convert to milliseconds epoch
        ts_ms = int(target_time.timestamp() * 1000)
        url = (
            "https://api.binance.com/api/v3/klines"
            f"?symbol=ETHUSDT&interval=1h&startTime={ts_ms}&limit=1"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0][4])  # close price
    except Exception:
        pass
    return None


def log_prediction(prediction_data: dict) -> None:
    """
    Append a new prediction row to the log.

    Expected keys in prediction_data:
        timestamp_made  – datetime (UTC) when prediction was made
        target_time     – datetime (UTC) the prediction is for
        predicted_price – float
        confidence      – float 0-1
        model_weights   – dict or str (serialised as JSON string)
    """
    df = _load_log()

    weights = prediction_data.get("model_weights", {})
    if isinstance(weights, dict):
        weights = json.dumps(weights)

    row = {
        "timestamp_made": prediction_data.get("timestamp_made", datetime.now(timezone.utc)),
        "target_time": prediction_data.get("target_time"),
        "predicted_price": prediction_data.get("predicted_price"),
        "confidence": prediction_data.get("confidence"),
        "model_weights": weights,
        "actual_price": np.nan,
        "direction_correct": np.nan,
        "pct_error": np.nan,
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_log(df)


def validate_past_predictions(current_price: float, current_time: datetime) -> int:
    """
    For every unvalidated row whose target_time <= current_time, fetch the
    actual price from Binance and fill outcome columns.

    Returns the number of rows validated.
    """
    df = _load_log()
    if df.empty:
        return 0

    # Ensure datetime columns are tz-aware for comparison
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    df["timestamp_made"] = pd.to_datetime(df["timestamp_made"], utc=True)

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    mask = (df["target_time"] <= current_time) & (df["actual_price"].isna())
    validated = 0

    for idx in df[mask].index:
        target_time = df.at[idx, "target_time"]
        predicted = df.at[idx, "predicted_price"]
        made_price = predicted  # we use the predicted as the baseline direction ref

        # Try to use the most-recent actual price for "now" predictions
        if abs((target_time - current_time).total_seconds()) < 3600:
            actual = current_price
        else:
            actual = _fetch_binance_price(target_time)

        if actual is None:
            continue

        df.at[idx, "actual_price"] = actual
        df.at[idx, "pct_error"] = abs(actual - predicted) / actual * 100 if actual else np.nan

        # Direction: compare predicted move to actual move vs price at timestamp_made
        # We don't store base_price so we infer direction from predicted vs actual
        # direction_correct = 1 if (predicted > actual) == (predicted > actual) – trivially true
        # Instead: direction is whether predicted_price > some reference.
        # Since we don't store the price at timestamp_made, we use a proxy:
        # if predicted >= actual → bull prediction; actual >= prior actual → bull outcome
        # Best we can do without base: mark as NULL unless we have prior rows
        df.at[idx, "direction_correct"] = np.nan  # filled below

        validated += 1

    # Second pass: compute direction_correct using prior row's actual_price as reference
    df_sorted = df.sort_values("timestamp_made")
    for idx in df_sorted[mask].index:
        if pd.isna(df.at[idx, "actual_price"]):
            continue
        # Find the row just before this one that has an actual_price
        prior = df_sorted[
            (df_sorted["timestamp_made"] < df.at[idx, "timestamp_made"])
            & (~df_sorted["actual_price"].isna())
        ]
        if prior.empty:
            continue
        base_price = prior.iloc[-1]["actual_price"]
        predicted = df.at[idx, "predicted_price"]
        actual = df.at[idx, "actual_price"]
        predicted_up = predicted >= base_price
        actual_up = actual >= base_price
        df.at[idx, "direction_correct"] = int(predicted_up == actual_up)

    _save_log(df)
    return validated


def get_accuracy_summary() -> dict:
    """
    Return a summary dict with:
        total_predictions, validated_count,
        directional_accuracy_7d, directional_accuracy_30d,
        avg_pct_error, sharpe_ratio
    """
    df = _load_log()
    now = datetime.now(timezone.utc)

    if df.empty:
        return {
            "total_predictions": 0,
            "validated_count": 0,
            "directional_accuracy_7d": None,
            "directional_accuracy_30d": None,
            "avg_pct_error": None,
            "sharpe_ratio": None,
        }

    df["timestamp_made"] = pd.to_datetime(df["timestamp_made"], utc=True)
    validated = df[~df["actual_price"].isna()]

    def _directional_accuracy(days: int) -> float | None:
        cutoff = now - timedelta(days=days)
        subset = validated[validated["timestamp_made"] >= cutoff]
        subset = subset[~subset["direction_correct"].isna()]
        if subset.empty:
            return None
        return float(subset["direction_correct"].mean())

    def _sharpe(errors: pd.Series) -> float | None:
        if len(errors) < 2:
            return None
        # treat negative pct_error as gains, positive as losses for a simple proxy
        returns = -errors  # lower error = better return proxy
        mean_r = returns.mean()
        std_r = returns.std(ddof=1)
        if std_r == 0:
            return None
        return float(mean_r / std_r)

    avg_pct_error = float(validated["pct_error"].mean()) if not validated.empty else None
    sharpe = _sharpe(validated["pct_error"].dropna()) if not validated.empty else None

    return {
        "total_predictions": int(len(df)),
        "validated_count": int(len(validated)),
        "directional_accuracy_7d": _directional_accuracy(7),
        "directional_accuracy_30d": _directional_accuracy(30),
        "avg_pct_error": avg_pct_error,
        "sharpe_ratio": sharpe,
    }
