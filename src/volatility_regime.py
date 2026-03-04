#!/usr/bin/env python3
"""
Volatility Regime Classifier
Calculates realized volatility and classifies market regime for LP optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def calculate_realized_vol(df: pd.DataFrame, window_hours: int, trading_hours_per_year: int = 8760) -> float:
    """Calculate annualized realized volatility from close prices over a window."""
    if df is None or len(df) < 2:
        return None
    # Assume 1-minute bars; convert window_hours to candle count
    candles = window_hours * 60
    subset = df.tail(candles)
    if len(subset) < 10:
        return None
    log_returns = np.log(subset['close'] / subset['close'].shift(1)).dropna()
    # Annualize: multiply by sqrt(minutes_per_year / 1)
    minutes_per_year = 365 * 24 * 60
    realized_vol = log_returns.std() * np.sqrt(minutes_per_year)
    return float(realized_vol * 100)  # as percentage


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR for the last `period` candles."""
    if df is None or len(df) < period + 1:
        return None
    tail = df.tail(period + 1).copy()
    high_low = tail['high'] - tail['low']
    high_close = np.abs(tail['high'] - tail['close'].shift(1))
    low_close = np.abs(tail['low'] - tail['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(tr.tail(period).mean())


def classify_regime(annualized_vol_pct: float) -> str:
    """Classify volatility regime from annualized vol percentage."""
    if annualized_vol_pct is None:
        return "UNKNOWN"
    if annualized_vol_pct < 40:
        return "LOW"
    elif annualized_vol_pct < 80:
        return "MEDIUM"
    elif annualized_vol_pct < 150:
        return "HIGH"
    else:
        return "EXTREME"


def get_volatility_regime(df: pd.DataFrame) -> dict:
    """
    Main entry point. Returns full volatility regime dict.

    Args:
        df: DataFrame with 1-minute OHLC bars (columns: timestamp, open, high, low, close, volume)

    Returns:
        dict with keys: regime, vol_1h, vol_4h, vol_24h, atr, timestamp
    """
    try:
        vol_1h = calculate_realized_vol(df, window_hours=1)
        vol_4h = calculate_realized_vol(df, window_hours=4)
        vol_24h = calculate_realized_vol(df, window_hours=24)
        atr = calculate_atr(df, period=14)

        # Use 24h as primary, fallback to 4h or 1h
        primary_vol = vol_24h or vol_4h or vol_1h
        regime = classify_regime(primary_vol)

        return {
            "regime": regime,
            "vol_1h_pct": round(vol_1h, 2) if vol_1h else None,
            "vol_4h_pct": round(vol_4h, 2) if vol_4h else None,
            "vol_24h_pct": round(vol_24h, 2) if vol_24h else None,
            "primary_vol_pct": round(primary_vol, 2) if primary_vol else None,
            "atr": round(atr, 2) if atr else None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "regime": "UNKNOWN",
            "vol_1h_pct": None,
            "vol_4h_pct": None,
            "vol_24h_pct": None,
            "primary_vol_pct": None,
            "atr": None,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


if __name__ == "__main__":
    import json
    # Quick smoke test with synthetic data
    dates = pd.date_range(end=datetime.utcnow(), periods=1500, freq="1min")
    np.random.seed(42)
    close = 2000 * np.exp(np.cumsum(np.random.normal(0, 0.001, 1500)))
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close * (1 - 0.0005),
        "high": close * (1 + 0.001),
        "low": close * (1 - 0.001),
        "close": close,
        "volume": np.random.uniform(10, 100, 1500),
    })
    result = get_volatility_regime(df)
    print(json.dumps(result, indent=2))
