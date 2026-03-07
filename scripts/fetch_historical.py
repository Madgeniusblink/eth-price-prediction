"""
Fetch 6 months of 1-hour OHLCV data for ETHUSDT from Binance public API.
Saves to data/eth_1h_historical_6mo.csv.
No API key required.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT_PATH = os.path.join(_BASE_DIR, "data", "eth_1h_historical_6mo.csv")

_BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
_SYMBOL = "ETHUSDT"
_INTERVAL = "1h"
_LIMIT = 500  # max per request
_TARGET_ROWS = 4380  # ~6 months × 30 days × 24h

_COLUMNS = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch_all() -> pd.DataFrame:
    """Paginate through Binance klines to collect ~6 months of hourly data."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    six_months_ago = datetime.now(timezone.utc) - timedelta(days=183)
    start_ms = int(six_months_ago.timestamp() * 1000)

    all_rows: list[list] = []
    current_start = start_ms

    print(f"Fetching {_TARGET_ROWS} rows of {_INTERVAL} ETHUSDT data from Binance…")

    while len(all_rows) < _TARGET_ROWS and current_start < now_ms:
        params = {
            "symbol": _SYMBOL,
            "interval": _INTERVAL,
            "startTime": current_start,
            "limit": _LIMIT,
        }
        try:
            resp = requests.get(_BINANCE_KLINES, params=params, timeout=15)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:
            print(f"  ⚠ Request failed: {exc} — retrying in 3s")
            time.sleep(3)
            continue

        if not batch:
            break

        all_rows.extend(batch)
        current_start = batch[-1][6] + 1  # close_time + 1 ms → next candle
        print(f"  fetched {len(all_rows):,} rows so far…", end="\r")

        # Respect Binance rate limits (1200 req/min = 1 req/50ms)
        time.sleep(0.05)

    print()  # newline after \r progress
    return pd.DataFrame(all_rows, columns=_COLUMNS)


def main():
    os.makedirs(os.path.join(_BASE_DIR, "data"), exist_ok=True)

    df = fetch_all()

    if df.empty:
        print("✗ No data fetched. Check connectivity and try again.")
        return

    # Convert timestamps from ms epoch to human-readable UTC
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Cast price/volume columns to float
    for col in ["open", "high", "low", "close", "volume",
                "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)

    df["num_trades"] = df["num_trades"].astype(int)
    df.drop(columns=["ignore"], inplace=True)
    df.drop_duplicates(subset=["open_time"], inplace=True)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(_OUT_PATH, index=False)

    first_ts = df["open_time"].iloc[0].strftime("%Y-%m-%d %H:%M UTC")
    last_ts = df["open_time"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
    print(f"✓ Saved {len(df):,} rows to {_OUT_PATH}")
    print(f"  Date range: {first_ts} → {last_ts}")


if __name__ == "__main__":
    main()
