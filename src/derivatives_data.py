"""
Derivatives Data Module
Fetches derivatives market data from Binance Futures API (public, no auth required)
"""

import requests
import time
from typing import Dict, Optional


def fetch_derivatives_data(max_retries: int = 3) -> Dict[str, Optional[float]]:
    """
    Fetch derivatives market data from Binance Futures public API

    Returns:
        dict: Dictionary containing:
            - funding_rate: Current funding rate for ETHUSDT perpetual
            - open_interest: Current open interest in USDT
            - long_short_ratio: Ratio of long to short positions
            - fear_greed_index: Fear & Greed Index (0-100)
            - fear_greed_classification: Text classification (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)
    """

    result = {
        'funding_rate': None,
        'open_interest': None,
        'long_short_ratio': None,
        'fear_greed_index': None,
        'fear_greed_classification': None
    }

    # Fetch funding rate
    result['funding_rate'] = _fetch_funding_rate(max_retries)

    # Fetch open interest
    result['open_interest'] = _fetch_open_interest(max_retries)

    # Fetch long/short ratio
    result['long_short_ratio'] = _fetch_long_short_ratio(max_retries)

    # Fetch Fear & Greed Index
    fear_greed_data = _fetch_fear_greed(max_retries)
    if fear_greed_data:
        result['fear_greed_index'] = fear_greed_data['value']
        result['fear_greed_classification'] = fear_greed_data['classification']

    return result


def _fetch_funding_rate(max_retries: int = 3) -> Optional[float]:
    """Fetch current funding rate from Binance Futures"""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {
        'symbol': 'ETHUSDT',
        'limit': 1
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and len(data) > 0:
                return float(data[0]['fundingRate'])
            return None

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error fetching funding rate after {max_retries} attempts: {e}")
                return None
            time.sleep(1)

    return None


def _fetch_open_interest(max_retries: int = 3) -> Optional[float]:
    """Fetch current open interest from Binance Futures"""
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    params = {
        'symbol': 'ETHUSDT'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and 'openInterest' in data:
                return float(data['openInterest'])
            return None

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error fetching open interest after {max_retries} attempts: {e}")
                return None
            time.sleep(1)

    return None


def _fetch_long_short_ratio(max_retries: int = 3) -> Optional[float]:
    """Fetch long/short ratio from Binance Futures"""
    url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
    params = {
        'symbol': 'ETHUSDT',
        'period': '1h',
        'limit': 3
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and len(data) > 0:
                # Use the most recent ratio
                return float(data[0]['longShortRatio'])
            return None

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error fetching long/short ratio after {max_retries} attempts: {e}")
                return None
            time.sleep(1)

    return None


def _fetch_fear_greed(max_retries: int = 3) -> Optional[Dict[str, any]]:
    """Fetch Fear & Greed Index from alternative.me API"""
    url = "https://api.alternative.me/fng/"
    params = {
        'limit': 1
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and 'data' in data and len(data['data']) > 0:
                fng_data = data['data'][0]
                return {
                    'value': int(fng_data['value']),
                    'classification': fng_data['value_classification']
                }
            return None

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error fetching Fear & Greed Index after {max_retries} attempts: {e}")
                return None
            time.sleep(1)

    return None


def main():
    """Test the derivatives data module"""
    print("=== Fetching Derivatives Market Data ===\n")

    data = fetch_derivatives_data()

    print(f"Funding Rate: {data['funding_rate']}")
    print(f"Open Interest: {data['open_interest']} USDT")
    print(f"Long/Short Ratio: {data['long_short_ratio']}")
    print(f"Fear & Greed Index: {data['fear_greed_index']} ({data['fear_greed_classification']})")

    print("\n✓ Derivatives data fetched successfully")


if __name__ == '__main__':
    main()
