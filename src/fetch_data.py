import os
#!/usr/bin/env python3
"""
Fetch Ethereum price data from multiple sources for swing trading (4-hour timeframe)
"""

import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from config import BASE_DIR

def fetch_binance_data(symbol='ETHUSDT', interval='1m', limit=500):
    """
    Fetch historical kline/candlestick data from Binance
    Interval options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        print(f"✓ Fetched {len(df)} candles from Binance")
        print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Current price: ${df['close'].iloc[-1]:.2f}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error fetching Binance data: {e}")
        return None

def fetch_coingecko_data(days=1):
    """
    Fetch historical market data from CoinGecko
    Returns price, market cap, and volume data
    """
    url = f'https://api.coingecko.com/api/v3/coins/ethereum/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'minute' if days <= 1 else 'hourly'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        
        print(f"✓ Fetched {len(prices)} price points from CoinGecko")
        print(f"  Time range: {prices['timestamp'].min()} to {prices['timestamp'].max()}")
        
        return prices
        
    except Exception as e:
        print(f"✗ Error fetching CoinGecko data: {e}")
        return None

def fetch_current_price():
    """
    Fetch current Ethereum price from multiple sources
    """
    prices = {}
    
    # Binance
    try:
        url = 'https://api.binance.com/api/v3/ticker/price'
        params = {'symbol': 'ETHUSDT'}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        prices['binance'] = float(response.json()['price'])
    except:
        pass
    
    # CoinGecko
    try:
        url = 'https://api.coingecko.com/api/v3/simple/price'
        params = {'ids': 'ethereum', 'vs_currencies': 'usd'}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        prices['coingecko'] = float(response.json()['ethereum']['usd'])
    except:
        pass
    
    if prices:
        avg_price = sum(prices.values()) / len(prices)
        print(f"\n=== Current Ethereum Price ===")
        for source, price in prices.items():
            print(f"  {source.capitalize()}: ${price:.2f}")
        print(f"  Average: ${avg_price:.2f}")
        return avg_price
    
    return None

def main():
    print("=== Ethereum Price Data Fetcher ===\n")
    
    # Fetch current price
    current_price = fetch_current_price()
    
    print("\n=== Fetching Historical Data ===\n")
    
    # Fetch 4-hour data from Binance (last ~83 days with 500 candles)
    df_4h = fetch_binance_data(symbol='ETHUSDT', interval='4h', limit=500)
    if df_4h is not None:
        df_4h.to_csv(os.path.join(BASE_DIR, 'eth_4h_data.csv'), index=False)
        print(f"  Saved to: {os.path.join(BASE_DIR, 'eth_4h_data.csv')}")
    
    print()
    
    # Fetch 1-hour data from Binance (last ~20 days)
    df_1h = fetch_binance_data(symbol='ETHUSDT', interval='1h', limit=500)
    if df_1h is not None:
        df_1h.to_csv(os.path.join(BASE_DIR, 'eth_1h_data.csv'), index=False)
        print(f"  Saved to: {os.path.join(BASE_DIR, 'eth_1h_data.csv')}")
    
    print()
    
    # Fetch 15-minute data from Binance (last ~5 days for additional context)
    df_15m = fetch_binance_data(symbol='ETHUSDT', interval='15m', limit=500)
    if df_15m is not None:
        df_15m.to_csv(os.path.join(BASE_DIR, 'eth_15m_data.csv'), index=False)
        print(f"  Saved to: {os.path.join(BASE_DIR, 'eth_15m_data.csv')}")
    
    print()
    
    # Fetch daily data from CoinGecko for longer-term context
    df_cg = fetch_coingecko_data(days=90)
    if df_cg is not None:
        df_cg.to_csv(os.path.join(BASE_DIR, 'eth_coingecko_data.csv'), index=False)
        print(f"  Saved to: {os.path.join(BASE_DIR, 'eth_coingecko_data.csv')}")
    
    print("\n=== Data Collection Complete ===")
    
    # Save metadata
    metadata = {
        'fetch_time': datetime.now().isoformat(),
        'current_price': current_price,
        'data_files': {
            '4h': os.path.join(BASE_DIR, 'eth_4h_data.csv'),
            '1h': os.path.join(BASE_DIR, 'eth_1h_data.csv'),
            '15m': os.path.join(BASE_DIR, 'eth_15m_data.csv'),
            'coingecko': os.path.join(BASE_DIR, 'eth_coingecko_data.csv')
        }
    }
    
    with open(os.path.join(BASE_DIR, 'eth_data_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {os.path.join(BASE_DIR, 'eth_data_metadata.json')}")

if __name__ == '__main__':
    main()
