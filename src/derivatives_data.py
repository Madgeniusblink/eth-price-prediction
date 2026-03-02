"""
Derivatives Data Module
Fetches ETH perpetual derivatives data from OKX + Bybit (geo-restriction free on GitHub Actions).
Falls back gracefully if any endpoint fails.
"""

import requests
import time
from typing import Dict, Optional


HEADERS = {"User-Agent": "eth-prediction-bot/1.0"}
TIMEOUT = 10


def fetch_derivatives_data(max_retries: int = 3) -> Dict[str, Optional[float]]:
    """
    Fetch derivatives market data.

    Returns dict with:
      funding_rate              float | None   (decimal, e.g. 0.0001 = 0.01%)
      open_interest             float | None   (USD notional)
      long_short_ratio          float | None
      fear_greed_index          int   | None   (0-100)
      fear_greed_classification str   | None
    """
    result: Dict[str, Optional[float]] = {
        'funding_rate': None,
        'open_interest': None,
        'long_short_ratio': None,
        'fear_greed_index': None,
        'fear_greed_classification': None,
    }

    result['funding_rate']     = _fetch_funding_rate_okx(max_retries)
    result['open_interest']    = _fetch_open_interest_okx(max_retries)
    result['long_short_ratio'] = _fetch_lsr_okx(max_retries)

    fg = _fetch_fear_greed(max_retries)
    if fg:
        result['fear_greed_index']          = fg['value']
        result['fear_greed_classification'] = fg['classification']

    return result


# ── OKX endpoints (no geo-restrictions) ───────────────────────────────────────

def _fetch_funding_rate_okx(max_retries: int) -> Optional[float]:
    """OKX: GET /api/v5/public/funding-rate?instId=ETH-USDT-SWAP"""
    url = "https://www.okx.com/api/v5/public/funding-rate"
    params = {"instId": "ETH-USDT-SWAP"}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            rate = data["data"][0]["fundingRate"]
            return float(rate)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error fetching funding rate after {max_retries} attempts: {e}")
            else:
                time.sleep(1)
    return None


def _fetch_open_interest_okx(max_retries: int) -> Optional[float]:
    """OKX: GET /api/v5/public/open-interest?instType=SWAP&instId=ETH-USDT-SWAP"""
    url = "https://www.okx.com/api/v5/public/open-interest"
    params = {"instType": "SWAP", "instId": "ETH-USDT-SWAP"}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            oi_contracts = float(data["data"][0]["oi"])       # contracts (ETH)
            # Approximate USD: multiply by mark price from tickers endpoint
            ticker_url = "https://www.okx.com/api/v5/market/ticker"
            tr = requests.get(ticker_url, params={"instId": "ETH-USDT-SWAP"}, headers=HEADERS, timeout=TIMEOUT)
            mark = float(tr.json()["data"][0]["last"])
            return oi_contracts * mark
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error fetching open interest after {max_retries} attempts: {e}")
            else:
                time.sleep(1)
    return None


def _fetch_lsr_okx(max_retries: int) -> Optional[float]:
    """OKX: GET /api/v5/rubik/stat/contracts/long-short-account-ratio?ccy=ETH&period=1H"""
    url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
    params = {"ccy": "ETH", "period": "1H"}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            # Returns list sorted newest first
            latest = data["data"][0]
            long_ratio  = float(latest[1])
            short_ratio = float(latest[2])
            return long_ratio / short_ratio if short_ratio else None
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error fetching long/short ratio after {max_retries} attempts: {e}")
            else:
                time.sleep(1)
    return None


# ── Fear & Greed (alternative.me — no geo-restriction) ────────────────────────

def _fetch_fear_greed(max_retries: int) -> Optional[Dict]:
    url = "https://api.alternative.me/fng/"
    params = {"limit": 1}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            entry = r.json()["data"][0]
            return {
                "value":          int(entry["value"]),
                "classification": entry["value_classification"],
            }
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error fetching Fear & Greed after {max_retries} attempts: {e}")
            else:
                time.sleep(1)
    return None


# ── Also used by signal_accuracy_tracker for historical spot prices ────────────

def fetch_historical_price_kraken(timestamp_utc: str) -> Optional[float]:
    """
    Fetch closest 1-minute OHLC close from Kraken for a given UTC timestamp string.
    Used by accuracy tracker (Kraken has no geo-restrictions on GH Actions).
    """
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
        since = int(dt.timestamp())
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "ETHUSD", "interval": 1, "since": since}
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        pair_key = [k for k in data["result"] if k != "last"][0]
        rows = data["result"][pair_key]
        if rows:
            return float(rows[0][4])   # close price
    except Exception as e:
        print(f"Error fetching historical price for {timestamp_utc}: {e}")
    return None


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_derivatives_data(), indent=2))
