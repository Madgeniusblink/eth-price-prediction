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
        print(f"Fetching {interval}-minute OHLC data from Kraken...")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data['error']:
            print(f"✗ Kraken API error: {data['error']}")
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
            print(f"✗ CRITICAL: Data is {age_minutes:.1f} minutes old (STALE!)")
            print(f"✗ Refusing to use stale data for predictions!")
            return None
        
        return df
        
    except Exception as e:
        print(f"✗ Error fetching Kraken OHLC data: {e}")
        import traceback
        traceback.print_exc()
        return None

def fetch_current_price():
    """
    Fetch current Ethereum price from multiple sources
    """
    prices = {}
    
    # Try Kraken first
    try:
        url = 'https://api.kraken.com/0/public/Ticker'
        params = {'pair': 'ETHUSD'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data['error']:
            pair_key = list(data['result'].keys())[0]
            prices['kraken'] = float(data['result'][pair_key]['c'][0])  # Last trade closed
    except Exception as e:
        print(f"⚠ Could not fetch from Kraken: {e}")
    
    # Try CoinGecko as backup
    try:
        url = 'https://api.coingecko.com/api/v3/simple/price'
        params = {
            'ids': 'ethereum',
            'vs_currencies': 'usd',
            'include_24hr_change': 'true'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        prices['coingecko'] = data['ethereum']['usd']
        change_24h = data['ethereum'].get('usd_24h_change', 0)
    except Exception as e:
        print(f"⚠ Could not fetch from CoinGecko: {e}")
    
    if not prices:
        print("✗ Could not fetch current price from any source!")
        return None
    
    # Use average if multiple sources
    avg_price = sum(prices.values()) / len(prices)
    
    print(f"✓ Current ETH price: ${avg_price:,.2f}")
    if len(prices) > 1:
        print(f"  Sources: {', '.join(f'{k}=${v:.2f}' for k, v in prices.items())}")
    if 'coingecko' in locals():
        print(f"  24h change: {change_24h:+.2f}%")
    
    return avg_price

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
    print("=" * 80)
    print("DATA COLLECTION COMPLETE")
    print("=" * 80)
    print(f"✓ Current price: ${current_price:,.2f}")
    print(f"✓ 1-minute candles: {len(df_1m)}")
    print(f"✓ Latest data point: {df_1m['timestamp'].max()}")
    print(f"✓ Data is FRESH and ready for predictions!")
    print()

if __name__ == '__main__':
    main()
