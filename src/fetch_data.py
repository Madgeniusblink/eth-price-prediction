#!/usr/bin/env python3
"""
Fetch Ethereum price data from Kraken API (reliable, free, no auth required)
PRODUCTION VERSION - Replaces Binance (blocked) with Kraken (works!)
"""

import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from config import BASE_DIR
import sys
from logger import setup_logger, log_error_with_context

# Setup logger
logger = setup_logger(__name__)

def fetch_kraken_ohlc(pair='ETHUSD', interval=1, since=None):
    """
    Fetch OHLC data from Kraken
    
    Args:
        pair: Trading pair (ETHUSD, ETHUSDT)
        interval: Time frame interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        since: Return committed OHLC data since given ID (optional)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, vwap, volume, count
    """
    url = 'https://api.kraken.com/0/public/OHLC'
    params = {
        'pair': pair,
        'interval': interval
    }
    if since:
        params['since'] = since
    
    try:
        logger.info(f"Fetching {interval}-minute OHLC data from Kraken...")
        print(f"Fetching {interval}-minute OHLC data from Kraken...")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data['error']:
            error_msg = f"Kraken API error: {data['error']}"
            logger.error(error_msg)
            print(f"✗ {error_msg}")
            return None
        
        # Extract the pair data (key varies: XETHZUSD, ETHUSD, etc.)
        pair_key = list(data['result'].keys())[0]  # Get first key (the pair)
        ohlc_data = data['result'][pair_key]
        
        # Convert to DataFrame
        # Format: [timestamp, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(ohlc_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
            df[col] = df[col].astype(float)
        
        print(f"✓ Fetched {len(df)} candles from Kraken")
        print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Current price: ${df['close'].iloc[-1]:,.2f}")
        
        # CRITICAL: Validate data freshness
        latest_time = df['timestamp'].max()
        age = datetime.now() - latest_time.replace(tzinfo=None)
        age_minutes = age.total_seconds() / 60
        
        print(f"  Data age: {age_minutes:.1f} minutes")
        
        if age_minutes > 60:  # Data older than 1 hour is stale
            logger.critical(f"Data is {age_minutes:.1f} minutes old (STALE!)")
            logger.critical("Refusing to use stale data for predictions!")
            print(f"✗ CRITICAL: Data is {age_minutes:.1f} minutes old (STALE!)")
            print(f"✗ Refusing to use stale data for predictions!")
            return None
        
        return df
        
    except Exception as e:
        log_error_with_context(logger, e, {'pair': pair, 'interval': interval})
        print(f"✗ Error fetching Kraken OHLC data: {e}")
        return None

def fetch_coinbase_price():
    """
    Fetch current ETH price from Coinbase API
    """
    try:
        url = 'https://api.coinbase.com/v2/prices/ETH-USD/spot'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        price = float(data['data']['amount'])
        logger.info(f"Fetched price from Coinbase: ${price:,.2f}")
        return price
    except Exception as e:
        logger.warning(f"Coinbase API failed: {e}")
        return None

def fetch_coingecko_price():
    """
    Fetch current ETH price from CoinGecko API
    """
    try:
        url = 'https://api.coingecko.com/api/v3/simple/price'
        params = {'ids': 'ethereum', 'vs_currencies': 'usd'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        price = float(data['ethereum']['usd'])
        logger.info(f"Fetched price from CoinGecko: ${price:,.2f}")
        return price
    except Exception as e:
        logger.warning(f"CoinGecko API failed: {e}")
        return None

def fetch_current_price():
    """
    Fetch current Ethereum price from multiple sources with fallback
    Tries Kraken -> Coinbase -> CoinGecko
    """
    prices = {}
    sources = [
        ('CoinGecko', fetch_coingecko_price),
        ('Coinbase', fetch_coinbase_price),
        ('Kraken', lambda: fetch_kraken_spot_price()),
    ]
    
    # Try each source in order
    for source_name, fetch_func in sources:
        try:
            price = fetch_func()
            if price and price > 0:
                prices[source_name] = price
                logger.info(f"Successfully fetched price from {source_name}: ${price:,.2f}")
                # Return first successful price
                return price
        except Exception as e:
            logger.warning(f"{source_name} failed: {e}")
            continue
    
    # If all sources failed, log critical error
    logger.critical("All price sources failed!")
    return None

def fetch_kraken_spot_price():
    """
    Fetch current ETH spot price from Kraken
    """
    try:
        url = 'https://api.kraken.com/0/public/Ticker'
        params = {'pair': 'ETHUSD'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data['error']:
            pair_key = list(data['result'].keys())[0]
            price = float(data['result'][pair_key]['c'][0])  # Last trade closed
            logger.info(f"Fetched price from Kraken: ${price:,.2f}")
            return price
    except Exception as e:
        logger.warning(f"Kraken spot price failed: {e}")
        return None

def main():
    """
    Main function to fetch and save all required data
    """
    print("=" * 80)
    print("ETHEREUM DATA FETCHER (KRAKEN API)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    # Fetch current price first
    current_price = fetch_current_price()
    if current_price is None:
        print("\n✗ CRITICAL ERROR: Could not fetch current price!")
        print("✗ Cannot proceed without current price data.")
        sys.exit(1)
    
    print()
    
    # Fetch 1-minute data (last 720 candles = 12 hours)
    print("Fetching 1-minute candle data...")
    df_1m = fetch_kraken_ohlc(pair='ETHUSD', interval=1)
    
    if df_1m is None or len(df_1m) < 100:
        print("\n✗ CRITICAL ERROR: Could not fetch sufficient 1-minute data!")
        print(f"✗ Need at least 100 candles, got {len(df_1m) if df_1m is not None else 0}")
        sys.exit(1)
    
    # Save 1-minute data
    output_file_1m = f"{BASE_DIR}/eth_1m_data.csv"
    df_1m.to_csv(output_file_1m, index=False)
    print(f"✓ Saved {len(df_1m)} candles to: {output_file_1m}")
    
    print()
    
    # Fetch 5-minute data (last 720 candles = 60 hours)
    print("Fetching 5-minute candle data...")
    df_5m = fetch_kraken_ohlc(pair='ETHUSD', interval=5)
    
    if df_5m is not None and len(df_5m) >= 50:
        output_file_5m = f"{BASE_DIR}/eth_5m_data.csv"
        df_5m.to_csv(output_file_5m, index=False)
        print(f"✓ Saved {len(df_5m)} candles to: {output_file_5m}")
    else:
        print("⚠ Could not fetch 5-minute data (non-critical)")
    
    print()
    
    # Fetch 15-minute data (last 720 candles = 7.5 days)
    print("Fetching 15-minute candle data...")
    df_15m = fetch_kraken_ohlc(pair='ETHUSD', interval=15)
    
    if df_15m is not None and len(df_15m) >= 50:
        output_file_15m = f"{BASE_DIR}/eth_15m_data.csv"
        df_15m.to_csv(output_file_15m, index=False)
        print(f"✓ Saved {len(df_15m)} candles to: {output_file_15m}")
    else:
        print("⚠ Could not fetch 15-minute data (non-critical)")
    
    print()
    
    # Fetch 4-hour data (last 500 candles = ~83 days)
    print("Fetching 4-hour candle data...")
    df_4h = fetch_kraken_ohlc(pair='ETHUSD', interval=240)  # 240 minutes = 4 hours
    
    if df_4h is not None and len(df_4h) >= 50:
        output_file_4h = f"{BASE_DIR}/eth_4h_data.csv"
        df_4h.to_csv(output_file_4h, index=False)
        print(f"✓ Saved {len(df_4h)} candles to: {output_file_4h}")
    else:
        print("⚠ Could not fetch 4-hour data (non-critical)")
    
    print()
    print("=" * 80)
    print("DATA COLLECTION COMPLETE")
    print("=" * 80)
    print(f"✓ Current price: ${current_price:,.2f}")
    print(f"✓ 1-minute candles: {len(df_1m)}")
    print(f"✓ Latest data point: {df_1m['timestamp'].max()}")
    print(f"✓ Data is FRESH and ready for predictions!")
    print()

def fetch_coingecko_spot_extended():
    """
    Fetch ETH spot price with 24h vol and change from CoinGecko.
    Returns dict with price, change_24h, volume_24h.
    """
    try:
        url = 'https://api.coingecko.com/api/v3/simple/price'
        params = {
            'ids': 'ethereum',
            'vs_currencies': 'usd',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()['ethereum']
        return {
            'price': float(data['usd']),
            'change_24h': float(data.get('usd_24h_change', 0)),
            'volume_24h': float(data.get('usd_24h_vol', 0)),
        }
    except Exception as e:
        logger.warning(f"CoinGecko extended price fetch failed: {e}")
        return None


def fetch_coingecko_ohlc(days=7):
    """
    Fetch OHLC data from CoinGecko (1 API call).
    Returns DataFrame with columns: timestamp, open, high, low, close.
    Uses delta fetch: only returns new candles since last_fetch.json.
    """
    import os
    last_fetch_file = os.path.join(BASE_DIR, 'data', 'last_fetch.json')

    try:
        url = 'https://api.coingecko.com/api/v3/coins/ethereum/ohlc'
        params = {'vs_currency': 'usd', 'days': str(days)}
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        raw = response.json()  # [[timestamp_ms, open, high, low, close], ...]

        df = pd.DataFrame(raw, columns=['timestamp_ms', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
        df = df.drop(columns=['timestamp_ms'])
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Delta fetch: filter to only new candles
        last_ts = None
        if os.path.exists(last_fetch_file):
            try:
                with open(last_fetch_file) as f:
                    meta = json.load(f)
                last_ts = meta.get('last_ohlc_timestamp')
            except Exception:
                pass

        if last_ts:
            df = df[df['timestamp'] > pd.Timestamp(last_ts)]

        # Save new last fetch timestamp
        if not df.empty:
            os.makedirs(os.path.dirname(last_fetch_file), exist_ok=True)
            with open(last_fetch_file, 'w') as f:
                json.dump({'last_ohlc_timestamp': str(df['timestamp'].max())}, f)

        logger.info(f"CoinGecko OHLC: {len(df)} new candles")
        return df

    except Exception as e:
        logger.warning(f"CoinGecko OHLC fetch failed: {e}")
        return None


def fetch_defillama_uniswap():
    """
    Fetch Uniswap protocol data from DeFiLlama.
    Returns dict with tvl and recent volume estimate.
    """
    try:
        url = 'https://api.llama.fi/protocol/uniswap'
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        tvl = data.get('currentChainTvls', {}).get('Ethereum', 0) or data.get('tvl', 0)
        # tvl may be a list (time series) or a number
        if isinstance(tvl, list) and len(tvl) > 0:
            tvl = tvl[-1].get('totalLiquidityUSD', 0)
        return {'tvl': float(tvl) if tvl else 0}
    except Exception as e:
        logger.warning(f"DeFiLlama fetch failed: {e}")
        return None


def fetch_quant_data():
    """
    Fetch all data needed for quant DeFi report.
    Returns dict with: spot, ohlc_df, defillama
    Uses 2 CoinGecko calls max.
    """
    spot = fetch_coingecko_spot_extended()
    ohlc_df = fetch_coingecko_ohlc(days=7)
    defillama = fetch_defillama_uniswap()
    return {
        'spot': spot,
        'ohlc_df': ohlc_df,
        'defillama': defillama,
    }


if __name__ == '__main__':
    main()
