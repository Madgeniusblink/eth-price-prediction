#!/usr/bin/env python3
"""
Lightweight walk-forward backtest on 90 days of Kraken OHLC data.
Outputs reports/latest/backtest_summary.json.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR


def load_ohlc(days: int = 90) -> pd.DataFrame | None:
    """Load Kraken OHLC data; prefer 1h or 4h for 90-day coverage."""
    for fname in ['eth_4h_data.csv', 'eth_1m_data.csv', 'eth_5m_data.csv', 'eth_15m_data.csv']:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                cutoff = df['timestamp'].max() - pd.Timedelta(days=days)
                df = df[df['timestamp'] >= cutoff].copy()
                if len(df) >= 50:
                    print(f"  Loaded {len(df)} rows from {fname}")
                    return df
            except Exception as e:
                print(f"  Could not load {fname}: {e}")
    return None


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, EMA, MACD to dataframe."""
    d = df.copy()
    delta = d['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    d['rsi'] = 100 - (100 / (1 + rs))
    d['ema_12'] = d['close'].ewm(span=12).mean()
    d['ema_26'] = d['close'].ewm(span=26).mean()
    d['macd'] = d['ema_12'] - d['ema_26']
    d['macd_signal'] = d['macd'].ewm(span=9).mean()
    return d


def simple_signal(row) -> int:
    """Simple signal: +1 = BUY, -1 = SELL, 0 = HOLD."""
    if pd.isna(row['rsi']) or pd.isna(row['macd']):
        return 0
    buy = (row['rsi'] > 50) and (row['macd'] > row['macd_signal']) and (row['ema_12'] > row['ema_26'])
    sell = (row['rsi'] < 50) and (row['macd'] < row['macd_signal']) and (row['ema_12'] < row['ema_26'])
    if buy:
        return 1
    if sell:
        return -1
    return 0


def run_backtest(df: pd.DataFrame) -> dict:
    """
    Walk-forward backtest:
    - Generate signal at bar t
    - Measure 1-bar-ahead return
    - Track direction accuracy, Sharpe, max drawdown
    """
    d = compute_indicators(df)
    d = d.dropna().reset_index(drop=True)

    signals = d.apply(simple_signal, axis=1)
    returns = d['close'].pct_change().shift(-1)  # next-bar return

    # Direction accuracy
    pred_dir = np.sign(signals)
    actual_dir = np.sign(returns)
    mask = pred_dir != 0
    if mask.sum() == 0:
        direction_accuracy = 0.0
    else:
        direction_accuracy = float((pred_dir[mask] == actual_dir[mask]).mean() * 100)

    # MAE (as % of price)
    active = signals[mask]
    strategy_returns = returns[mask] * pred_dir[mask]

    # Sharpe (annualized — assume hourly bars → 8760 bars/year)
    if len(strategy_returns) > 2:
        bars_per_year = 8760 if 'h' in str(df['timestamp'].diff().median()) else 365 * 24 * 4
        mean_r = strategy_returns.mean()
        std_r = strategy_returns.std()
        sharpe = float((mean_r / std_r) * np.sqrt(bars_per_year)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    cum = (1 + strategy_returns.fillna(0)).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_drawdown = float(dd.min() * 100)

    # Mean absolute error (average |predicted direction signal * actual return|)
    mae = float(np.abs(returns[mask]).mean() * 100)

    return {
        'direction_accuracy_pct': round(direction_accuracy, 2),
        'mean_absolute_error_pct': round(mae, 4),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown_pct': round(max_drawdown, 2),
        'total_signals': int(mask.sum()),
        'total_bars': len(d),
        'backtest_period_days': 90,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }


def main():
    print("=" * 60)
    print("  ETH SIGNAL BACKTEST (90 days)")
    print("=" * 60)

    df = load_ohlc(days=90)
    if df is None:
        print("✗ No OHLC data available for backtest")
        results = {
            'error': 'no_data',
            'generated_at': datetime.now(timezone.utc).isoformat(),
        }
    else:
        results = run_backtest(df)
        print(f"\n{'Metric':<30} {'Value':>12}")
        print("-" * 44)
        print(f"{'Direction Accuracy':<30} {results['direction_accuracy_pct']:>11.2f}%")
        print(f"{'Sharpe Ratio':<30} {results['sharpe_ratio']:>12.4f}")
        print(f"{'Max Drawdown':<30} {results['max_drawdown_pct']:>11.2f}%")
        print(f"{'MAE (% price)':<30} {results['mean_absolute_error_pct']:>11.4f}%")
        print(f"{'Total Signals':<30} {results['total_signals']:>12}")
        print(f"{'Total Bars':<30} {results['total_bars']:>12}")

    # Write output
    out_dir = os.path.join(BASE_DIR, 'reports', 'latest')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'backtest_summary.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Wrote {out_path}")


if __name__ == '__main__':
    main()
