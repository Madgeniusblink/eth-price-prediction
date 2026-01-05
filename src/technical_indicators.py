#!/usr/bin/env python3
"""
Technical Indicators Module
Calculates various technical analysis indicators for price prediction
"""

import pandas as pd
import numpy as np

def calculate_sma(data, periods):
    """Calculate Simple Moving Average"""
    return data.rolling(window=periods).mean()

def calculate_ema(data, periods):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=periods, adjust=False).mean()

def calculate_rsi(data, periods=14):
    """
    Calculate Relative Strength Index
    
    Args:
        data: Price series
        periods: RSI period (default 14)
    
    Returns:
        RSI values
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, periods=20, std_dev=2):
    """
    Calculate Bollinger Bands
    
    Args:
        data: Price series
        periods: Period for moving average
        std_dev: Number of standard deviations
    
    Returns:
        tuple: (Upper band, Middle band, Lower band)
    """
    middle = data.rolling(window=periods).mean()
    std = data.rolling(window=periods).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper, middle, lower

def calculate_momentum(data, periods=10):
    """Calculate price momentum"""
    return data.pct_change(periods=periods)

def calculate_volatility(data, periods=20):
    """Calculate price volatility (standard deviation)"""
    return data.rolling(window=periods).std()

def calculate_volume_ratio(volume, periods=20):
    """Calculate volume ratio compared to average"""
    volume_sma = volume.rolling(window=periods).mean()
    return volume / volume_sma

def add_all_indicators(df):
    """
    Add all technical indicators to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added indicator columns
    """
    data = df.copy()
    
    # Moving Averages
    for period in [5, 10, 20, 50]:
        data[f'SMA_{period}'] = calculate_sma(data['close'], period)
    
    for period in [5, 10, 20]:
        data[f'EMA_{period}'] = calculate_ema(data['close'], period)
    
    # RSI
    data['RSI'] = calculate_rsi(data['close'], 14)
    
    # MACD
    macd, signal, hist = calculate_macd(data['close'])
    data['MACD'] = macd
    data['MACD_signal'] = signal
    data['MACD_hist'] = hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data['close'])
    data['BB_upper'] = bb_upper
    data['BB_middle'] = bb_middle
    data['BB_lower'] = bb_lower
    
    # Momentum and Volatility
    data['momentum'] = calculate_momentum(data['close'], 10)
    data['volatility'] = calculate_volatility(data['close'], 20)
    
    # Volume indicators
    data['volume_sma'] = calculate_sma(data['volume'], 20)
    data['volume_ratio'] = calculate_volume_ratio(data['volume'], 20)
    
    return data

def get_indicator_summary(df):
    """
    Get a summary of current indicator values
    
    Args:
        df: DataFrame with indicators
    
    Returns:
        dict: Summary of indicators
    """
    latest = df.iloc[-1]
    
    # Determine trend
    if latest['close'] > latest['SMA_20'] and latest['SMA_5'] > latest['SMA_20']:
        trend = "BULLISH"
    elif latest['close'] < latest['SMA_20'] and latest['SMA_5'] < latest['SMA_20']:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"
    
    # RSI signal
    if latest['RSI'] > 70:
        rsi_signal = "OVERBOUGHT"
    elif latest['RSI'] < 30:
        rsi_signal = "OVERSOLD"
    else:
        rsi_signal = "NEUTRAL"
    
    # MACD signal
    if latest['MACD'] > latest['MACD_signal']:
        macd_signal = "BULLISH"
    elif latest['MACD'] < latest['MACD_signal']:
        macd_signal = "BEARISH"
    else:
        macd_signal = "NEUTRAL"
    
    # Bollinger Bands position
    if latest['close'] > latest['BB_upper']:
        bb_position = "ABOVE_UPPER"
    elif latest['close'] < latest['BB_lower']:
        bb_position = "BELOW_LOWER"
    else:
        bb_position = "MIDDLE"
    
    return {
        'trend': trend,
        'rsi': latest['RSI'],
        'rsi_signal': rsi_signal,
        'macd': latest['MACD'],
        'macd_signal': macd_signal,
        'bb_position': bb_position,
        'current_price': latest['close'],
        'sma_20': latest['SMA_20'],
        'bb_upper': latest['BB_upper'],
        'bb_lower': latest['BB_lower'],
        'volatility': latest['volatility']
    }

if __name__ == '__main__':
    # Test the module
    print("Technical Indicators Module")
    print("This module provides functions for calculating technical indicators.")
    print("Import it in other scripts to use the indicators.")
