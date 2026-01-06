import os
#!/usr/bin/env python3
"""
Visualize Ethereum price predictions with trend lines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
from config import BASE_DIR

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def plot_predictions_overview():
    """
    Create comprehensive visualization with historical data and predictions
    """
    # Load data
    df_4h = pd.read_csv(os.path.join(BASE_DIR, 'eth_4h_data.csv'))
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    
    pred_48h = pd.read_csv(os.path.join(BASE_DIR, 'predictions_48h.csv'))
    pred_48h['timestamp'] = pd.to_datetime(pred_48h['timestamp'])
    
    pred_96h = pd.read_csv(os.path.join(BASE_DIR, 'predictions_96h.csv'))
    pred_96h['timestamp'] = pd.to_datetime(pred_96h['timestamp'])
    
    with open(os.path.join(BASE_DIR, 'predictions_summary.json'), 'r') as f:
        summary = json.load(f)
    
    # Use last 30 periods (120 hours / 5 days) of historical data
    df_recent = df_4h.tail(30)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Main plot: Historical + Predictions
    ax1 = fig.add_subplot(gs[0:2, :])
    
    # Plot historical data
    ax1.plot(df_recent['timestamp'], df_recent['close'], 
             label='Historical Price', color='#2E86AB', linewidth=2, alpha=0.8)
    
    # Plot predictions
    ax1.plot(pred_96h['timestamp'], pred_96h['ensemble'], 
             label='Ensemble Prediction', color='#A23B72', linewidth=2.5, linestyle='--')
    ax1.plot(pred_96h['timestamp'], pred_96h['linear'], 
             label='Linear Trend', color='#F18F01', linewidth=1.5, alpha=0.6, linestyle=':')
    ax1.plot(pred_96h['timestamp'], pred_96h['polynomial'], 
             label='Polynomial Trend', color='#C73E1D', linewidth=1.5, alpha=0.6, linestyle=':')
    ax1.plot(pred_96h['timestamp'], pred_96h['ml_features'], 
             label='ML Features', color='#6A994E', linewidth=1.5, alpha=0.6, linestyle=':')
    
    # Mark current time
    current_time = df_recent['timestamp'].iloc[-1]
    current_price = df_recent['close'].iloc[-1]
    ax1.axvline(x=current_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current Time')
    ax1.scatter([current_time], [current_price], color='red', s=100, zorder=5, marker='o')
    
    # Mark key prediction points
    for time_label, pred_data in summary['predictions'].items():
        pred_time = pd.to_datetime(pred_data['timestamp'])
        pred_price = pred_data['price']
        ax1.scatter([pred_time], [pred_price], s=80, zorder=5, marker='D')
        ax1.annotate(f"{time_label}\n${pred_price:.2f}\n({pred_data['change_pct']:+.2f}%)",
                    xy=(pred_time, pred_price), xytext=(10, 10),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add confidence band (simplified)
    std_dev = df_recent['close'].std()
    ax1.fill_between(pred_96h['timestamp'], 
                     pred_96h['ensemble'] - std_dev, 
                     pred_96h['ensemble'] + std_dev,
                     alpha=0.2, color='purple', label='Confidence Band')
    
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Ethereum Price Prediction - Next 4 Days (96 Hours)\nHistorical Data + Multi-Model Ensemble Forecast', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add text box with current analysis
    textstr = f"""Current Market Analysis:
Trend: {summary['trend_analysis']['trend']}
RSI: {summary['trend_analysis']['rsi']:.2f} ({summary['trend_analysis']['rsi_signal']})
MACD: {summary['trend_analysis']['macd_signal']}
Current Price: ${summary['current_price']:.2f}
20-SMA: ${summary['trend_analysis']['sma_20']:.2f}"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Bottom left: Model comparison
    ax2 = fig.add_subplot(gs[2, 0])
    models = list(summary['model_scores'].keys())
    scores = list(summary['model_scores'].values())
    colors = ['#F18F01', '#C73E1D', '#6A994E']
    
    bars = ax2.bar(models, scores, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax2.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Bottom right: Prediction summary table
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.axis('off')
    
    table_data = []
    for time_label in ['4h', '8h', '24h', '48h']:
        pred_data = summary['predictions'][time_label]
        table_data.append([
            time_label,
            f"${pred_data['price']:.2f}",
            f"{pred_data['change_pct']:+.2f}%"
        ])
    
    table = ax3.table(cellText=table_data,
                     colLabels=['Time', 'Predicted Price', 'Change'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.4, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax3.set_title('Price Predictions Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(BASE_DIR, 'eth_prediction_overview.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'eth_prediction_overview.png')}")
    plt.close()

def plot_technical_indicators():
    """
    Plot technical indicators with predictions
    """
    # Load data
    df_4h = pd.read_csv(os.path.join(BASE_DIR, 'eth_4h_data.csv'))
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    
    # Calculate indicators
    df_4h['SMA_20'] = df_4h['close'].rolling(window=20).mean()
    df_4h['EMA_10'] = df_4h['close'].ewm(span=10, adjust=False).mean()
    
    # RSI
    delta = df_4h['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_4h['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df_4h['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_4h['close'].ewm(span=26, adjust=False).mean()
    df_4h['MACD'] = exp1 - exp2
    df_4h['MACD_signal'] = df_4h['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df_4h['BB_middle'] = df_4h['close'].rolling(window=20).mean()
    bb_std = df_4h['close'].rolling(window=20).std()
    df_4h['BB_upper'] = df_4h['BB_middle'] + (bb_std * 2)
    df_4h['BB_lower'] = df_4h['BB_middle'] - (bb_std * 2)
    
    # Use last 30 periods (5 days)
    df_recent = df_4h.tail(30)
    
    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    # Price with Bollinger Bands
    axes[0].plot(df_recent['timestamp'], df_recent['close'], label='Price', color='#2E86AB', linewidth=2)
    axes[0].plot(df_recent['timestamp'], df_recent['BB_upper'], label='BB Upper', color='red', linestyle='--', alpha=0.5)
    axes[0].plot(df_recent['timestamp'], df_recent['BB_middle'], label='BB Middle', color='gray', linestyle='--', alpha=0.5)
    axes[0].plot(df_recent['timestamp'], df_recent['BB_lower'], label='BB Lower', color='green', linestyle='--', alpha=0.5)
    axes[0].fill_between(df_recent['timestamp'], df_recent['BB_lower'], df_recent['BB_upper'], alpha=0.1, color='gray')
    axes[0].set_ylabel('Price (USD)', fontweight='bold')
    axes[0].set_title('Ethereum Technical Indicators - Last 5 Days (30 periods)', fontsize=14, fontweight='bold', pad=20)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Moving Averages
    axes[1].plot(df_recent['timestamp'], df_recent['close'], label='Price', color='#2E86AB', linewidth=2)
    axes[1].plot(df_recent['timestamp'], df_recent['SMA_20'], label='SMA 20', color='orange', linestyle='--', linewidth=1.5)
    axes[1].plot(df_recent['timestamp'], df_recent['EMA_10'], label='EMA 10', color='purple', linestyle='--', linewidth=1.5)
    axes[1].set_ylabel('Price (USD)', fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # RSI
    axes[2].plot(df_recent['timestamp'], df_recent['RSI'], label='RSI', color='#6A994E', linewidth=2)
    axes[2].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    axes[2].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    axes[2].fill_between(df_recent['timestamp'], 30, 70, alpha=0.1, color='gray')
    axes[2].set_ylabel('RSI', fontweight='bold')
    axes[2].set_ylim(0, 100)
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    # MACD
    axes[3].plot(df_recent['timestamp'], df_recent['MACD'], label='MACD', color='#A23B72', linewidth=2)
    axes[3].plot(df_recent['timestamp'], df_recent['MACD_signal'], label='Signal', color='orange', linestyle='--', linewidth=1.5)
    axes[3].bar(df_recent['timestamp'], df_recent['MACD'] - df_recent['MACD_signal'], 
                label='Histogram', color='gray', alpha=0.3)
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[3].set_ylabel('MACD', fontweight='bold')
    axes[3].set_xlabel('Time', fontweight='bold')
    axes[3].legend(loc='upper left')
    axes[3].grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'eth_technical_indicators.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'eth_technical_indicators.png')}")
    plt.close()

def plot_4hour_prediction():
    """
    Focused view on next 1 hour prediction
    """
    # Load data
    df_1m = pd.read_csv(os.path.join(BASE_DIR, 'eth_1m_data.csv'))
    df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
    
    pred_60m = pd.read_csv(os.path.join(BASE_DIR, 'predictions_60m.csv'))
    pred_60m['timestamp'] = pd.to_datetime(pred_60m['timestamp'])
    
    # Use last 60 minutes of historical + 60 minutes prediction
    df_recent = df_1m.tail(60)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot historical
    ax.plot(df_recent['timestamp'], df_recent['close'], 
            label='Historical Price', color='#2E86AB', linewidth=2.5, marker='o', markersize=3)
    
    # Plot ensemble prediction
    ax.plot(pred_60m['timestamp'], pred_60m['ensemble'], 
            label='Predicted Price (Ensemble)', color='#A23B72', linewidth=3, linestyle='--', marker='s', markersize=3)
    
    # Mark current time
    current_time = df_recent['timestamp'].iloc[-1]
    current_price = df_recent['close'].iloc[-1]
    ax.axvline(x=current_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current Time')
    ax.scatter([current_time], [current_price], color='red', s=200, zorder=5, marker='o', edgecolors='black', linewidths=2)
    
    # Add trend line for historical data
    x_hist = np.arange(len(df_recent))
    z = np.polyfit(x_hist, df_recent['close'], 1)
    p = np.poly1d(z)
    trend_line = p(x_hist)
    ax.plot(df_recent['timestamp'], trend_line, 
            label=f'Historical Trend Line (slope: {z[0]:.4f})', 
            color='green', linestyle=':', linewidth=2, alpha=0.7)
    
    # Add prediction trend line
    x_pred = np.arange(len(pred_60m))
    z_pred = np.polyfit(x_pred, pred_60m['ensemble'], 1)
    p_pred = np.poly1d(z_pred)
    pred_trend_line = p_pred(x_pred)
    ax.plot(pred_60m['timestamp'], pred_trend_line, 
            label=f'Prediction Trend Line (slope: {z_pred[0]:.4f})', 
            color='orange', linestyle=':', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax.set_title('Ethereum 1-Hour Price Prediction with Trend Lines\nLast 60 Minutes (Historical) + Next 60 Minutes (Predicted)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add annotations
    final_pred_price = pred_48h['ensemble'].iloc[-1]
    final_pred_time = pred_48h['timestamp'].iloc[-1]
    change_pct = ((final_pred_price / current_price) - 1) * 100
    
    ax.annotate(f'Predicted: ${final_pred_price:.2f}\n({change_pct:+.2f}%)',
               xy=(final_pred_time, final_pred_price), xytext=(-80, 20),
               textcoords='offset points', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.annotate(f'Current: ${current_price:.2f}',
               xy=(current_time, current_price), xytext=(20, -30),
               textcoords='offset points', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'eth_1hour_prediction.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'eth_1hour_prediction.png')}")
    plt.close()

def main():
    print("=== Creating Visualizations ===\n")
    
    plot_predictions_overview()
    plot_technical_indicators()
    plot_4hour_prediction()
    
    print("\n=== All Visualizations Created ===")

if __name__ == '__main__':
    main()
