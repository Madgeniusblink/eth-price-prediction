import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from config import BASE_DIR

def plot_predictions_overview():
    """
    Overview of all prediction timeframes
    """
    # Load predictions
    with open(os.path.join(BASE_DIR, 'predictions_summary.json'), 'r') as f:
        predictions = json.load(f)
    
    current_price = predictions['current_price']
    pred_4h = predictions['predictions']['4h']['price']
    pred_8h = predictions['predictions']['8h']['price']
    pred_24h = predictions['predictions']['24h']['price']
    pred_48h = predictions['predictions']['48h']['price']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Price predictions
    timeframes = ['Current', '4h', '8h', '24h', '48h']
    prices = [current_price, pred_4h, pred_8h, pred_24h, pred_48h]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    ax1.plot(timeframes, prices, marker='o', linewidth=3, markersize=12, color='#2E86AB')
    for i, (tf, price, color) in enumerate(zip(timeframes, prices, colors)):
        ax1.scatter([i], [price], s=300, color=color, zorder=5, edgecolors='black', linewidths=2)
        change_pct = ((price / current_price) - 1) * 100 if i > 0 else 0
        ax1.annotate(f'${price:.2f}\n({change_pct:+.2f}%)',
                    xy=(i, price), xytext=(0, 15),
                    textcoords='offset points', fontsize=11, fontweight='bold',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    ax1.set_xlabel('Timeframe', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=14, fontweight='bold')
    ax1.set_title('Ethereum Price Predictions - 4-Hour Timeframe Analysis', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Percentage changes
    changes = [0, ((pred_4h / current_price) - 1) * 100,
               ((pred_8h / current_price) - 1) * 100,
               ((pred_24h / current_price) - 1) * 100,
               ((pred_48h / current_price) - 1) * 100]
    
    bar_colors = ['green' if c >= 0 else 'red' for c in changes]
    bars = ax2.bar(timeframes, changes, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, change in zip(bars, changes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{change:+.2f}%',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=12, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax2.set_xlabel('Timeframe', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price Change (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Predicted Price Changes from Current', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'eth_predictions_overview.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'eth_predictions_overview.png')}")
    plt.close()

def plot_technical_indicators():
    """
    Technical indicators chart
    """
    # Load 4-hour data
    df = pd.read_csv(os.path.join(BASE_DIR, 'eth_4h_data.csv'))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Use last 30 periods (5 days of 4-hour candles)
    df_recent = df.tail(30)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Plot 1: Price and Moving Averages
    ax1.plot(df_recent['timestamp'], df_recent['close'], label='Close Price', color='#2E86AB', linewidth=2)
    ax1.plot(df_recent['timestamp'], df_recent['sma_20'], label='SMA 20', color='orange', linewidth=1.5, linestyle='--')
    ax1.plot(df_recent['timestamp'], df_recent['ema_12'], label='EMA 12', color='green', linewidth=1.5, linestyle='--')
    ax1.fill_between(df_recent['timestamp'], df_recent['bb_upper'], df_recent['bb_lower'], 
                     alpha=0.2, color='gray', label='Bollinger Bands')
    ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Ethereum Technical Indicators - 4-Hour Timeframe (Last 5 Days)', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RSI
    ax2.plot(df_recent['timestamp'], df_recent['rsi'], label='RSI', color='purple', linewidth=2)
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (30)')
    ax2.fill_between(df_recent['timestamp'], 30, 70, alpha=0.1, color='gray')
    ax2.set_ylabel('RSI', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MACD
    ax3.plot(df_recent['timestamp'], df_recent['macd'], label='MACD', color='blue', linewidth=2)
    ax3.plot(df_recent['timestamp'], df_recent['macd_signal'], label='Signal', color='red', linewidth=2)
    ax3.bar(df_recent['timestamp'], df_recent['macd_hist'], label='Histogram', 
           color=['green' if x >= 0 else 'red' for x in df_recent['macd_hist']], alpha=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MACD', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'eth_technical_indicators.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'eth_technical_indicators.png')}")
    plt.close()

def plot_4hour_prediction():
    """
    Focused view on 48-hour prediction (12 periods of 4-hour candles)
    """
    # Load 4-hour data
    df_4h = pd.read_csv(os.path.join(BASE_DIR, 'eth_4h_data.csv'))
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    
    # Load predictions
    with open(os.path.join(BASE_DIR, 'predictions_summary.json'), 'r') as f:
        predictions = json.load(f)
    
    current_price = predictions['current_price']
    pred_4h = predictions['predictions']['4h']['price']
    pred_8h = predictions['predictions']['8h']['price']
    pred_24h = predictions['predictions']['24h']['price']
    pred_48h = predictions['predictions']['48h']['price']
    
    # Use last 24 periods (4 days) of historical data
    df_recent = df_4h.tail(24)
    
    # Create prediction timeline
    last_timestamp = df_recent['timestamp'].iloc[-1]
    pred_timestamps = [
        last_timestamp + timedelta(hours=4),
        last_timestamp + timedelta(hours=8),
        last_timestamp + timedelta(hours=24),
        last_timestamp + timedelta(hours=48)
    ]
    pred_prices = [pred_4h, pred_8h, pred_24h, pred_48h]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot historical
    ax.plot(df_recent['timestamp'], df_recent['close'], 
            label='Historical Price', color='#2E86AB', linewidth=2.5, marker='o', markersize=4)
    
    # Plot predictions
    all_timestamps = list(df_recent['timestamp']) + pred_timestamps
    all_prices = list(df_recent['close']) + pred_prices
    
    # Connect last historical to first prediction
    ax.plot([df_recent['timestamp'].iloc[-1], pred_timestamps[0]], 
            [df_recent['close'].iloc[-1], pred_prices[0]],
            color='#A23B72', linewidth=2, linestyle='--', alpha=0.5)
    
    # Plot prediction line
    ax.plot(pred_timestamps, pred_prices, 
            label='Predicted Price', color='#A23B72', linewidth=3, linestyle='--', marker='s', markersize=6)
    
    # Mark current time
    current_time = df_recent['timestamp'].iloc[-1]
    ax.axvline(x=current_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current Time')
    ax.scatter([current_time], [current_price], color='red', s=200, zorder=5, marker='o', edgecolors='black', linewidths=2)
    
    # Add annotations for predictions
    for i, (ts, price) in enumerate(zip(pred_timestamps, pred_prices)):
        change_pct = ((price / current_price) - 1) * 100
        timeframe = ['4h', '8h', '24h', '48h'][i]
        ax.annotate(f'{timeframe}: ${price:.2f}\n({change_pct:+.2f}%)',
                   xy=(ts, price), xytext=(10, 10 + i*15),
                   textcoords='offset points', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Annotate current price
    ax.annotate(f'Current: ${current_price:.2f}',
               xy=(current_time, current_price), xytext=(20, -30),
               textcoords='offset points', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax.set_title('Ethereum 48-Hour Price Prediction\nLast 4 Days (Historical) + Next 48 Hours (Predicted)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'eth_48hour_prediction.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'eth_48hour_prediction.png')}")
    plt.close()

def main():
    print("=== Creating Visualizations ===")
    
    plot_predictions_overview()
    # plot_technical_indicators()  # Temporarily disabled - needs technical indicators in data file
    plot_4hour_prediction()
    
    print("\n=== All Visualizations Created ===")

if __name__ == '__main__':
    main()
