"""
Enhanced Automated Report Generation Script
Generates comprehensive prediction reports with trading signals
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime, timezone
import json
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_signals import TradingSignals
from glossary import GLOSSARY
from config import BASE_DIR, DATA_DIR

def create_report_folders():
    """Create folder structure for reports"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, 'reports')
    
    now = datetime.now(timezone.utc)
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')
    time_folder = now.strftime('%H-%M')  # e.g., "07-53"
    
    # Create individual report folder: reports/YYYY/MM/DD/HH-MM/
    dated_dir = os.path.join(reports_dir, year, month, day, time_folder)
    os.makedirs(dated_dir, exist_ok=True)
    
    # Create/update latest folder
    latest_dir = os.path.join(reports_dir, 'latest')
    os.makedirs(latest_dir, exist_ok=True)
    
    return dated_dir, latest_dir, now

def generate_report_filename(timestamp):
    """Generate standardized report filename"""
    return timestamp.strftime('%Y-%m-%d_%H-%M')

def run_prediction_pipeline():
    """Run the complete prediction pipeline"""
    print("=" * 70)
    print("  AUTOMATED ETHEREUM PRICE PREDICTION REPORT")
    print("=" * 70)
    print(f"\nGeneration Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Fetch Data
    print("Step 1/4: Fetching latest market data...")
    result = subprocess.run([sys.executable, os.path.join(src_dir, 'fetch_data.py')], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"✗ Error fetching data: {result.stderr}")
        return False
    print("✓ Data fetched successfully\n")
    
    # Step 2: Generate Predictions
    print("Step 2/4: Generating predictions...")
    result = subprocess.run([sys.executable, os.path.join(src_dir, 'predict.py')], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"✗ Error generating predictions: {result.stderr}")
        return False
    print("✓ Predictions generated successfully\n")
    
    # Step 3: Create Visualizations
    print("Step 3/4: Creating visualizations...")
    result = subprocess.run([sys.executable, os.path.join(src_dir, 'visualize.py')], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"✗ Error creating visualizations: {result.stderr}")
        return False
    print("✓ Visualizations created successfully\n")
    
    # Step 4: Generate Trading Signals
    print("Step 4/4: Analyzing trading signals...")
    try:
        signals_data = generate_trading_signals()
        if signals_data:
            # Save trading signals
            with open(os.path.join(BASE_DIR, 'trading_signals.json'), 'w') as f:
                json.dump(signals_data, f, indent=2)
            print("✓ Trading signals generated successfully\n")
        else:
            print("⚠ Trading signals could not be generated (insufficient data)\n")
    except Exception as e:
        print(f"⚠ Warning: Trading signals generation failed: {e}\n")
    
    return True

def generate_trading_signals():
    """Generate trading signals from market data"""
    try:
        # Load market data
        data_file = os.path.join(BASE_DIR, 'eth_1m_data.csv')
        if not os.path.exists(data_file):
            return None
        
        df = pd.read_csv(data_file)
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Initialize trading signals
        signals = TradingSignals(df)
        
        # Get all analysis
        trend = signals.detect_trend()
        levels = signals.find_support_resistance()
        entry_signals = signals.generate_entry_signals()
        
        # Combine into single structure
        return {
            'trend_analysis': trend,
            'support_resistance': levels,
            'trading_signal': entry_signals,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        print(f"Error generating trading signals: {e}")
        return None

def copy_outputs_to_report_folder(dated_dir, latest_dir, timestamp):
    """Copy prediction outputs to report folders"""
    
    # Files to copy - use simple names since each report is in its own folder
    files_to_copy = {
        'predictions_summary.json': 'prediction.json',
        'trading_signals.json': 'signals.json',
        'eth_prediction_overview.png': 'overview.png',
        'eth_1hour_prediction.png': '1hour.png',
        'eth_technical_indicators.png': 'indicators.png',
        'eth_1m_data.csv': 'data.csv'
    }
    
    copied_files = []
    
    for source_name, dest_name in files_to_copy.items():
        source_path = os.path.join(BASE_DIR, source_name)
        
        if os.path.exists(source_path):
            # Copy to dated folder
            dated_dest = os.path.join(dated_dir, dest_name)
            shutil.copy2(source_path, dated_dest)
            
            # Copy to latest folder (with generic name)
            latest_dest = os.path.join(latest_dir, source_name)
            shutil.copy2(source_path, latest_dest)
            
            copied_files.append(dest_name)
            print(f"  ✓ Copied {source_name}")
        else:
            if source_name != 'trading_signals.json':  # Optional file
                print(f"  ✗ Warning: {source_name} not found")
    
    return copied_files

def generate_comprehensive_report(dated_dir, timestamp, copied_files):
    """Generate comprehensive report with all components"""
    
    # Load prediction data (using simple filenames)
    prediction_file = os.path.join(dated_dir, 'prediction.json')
    signals_file = os.path.join(dated_dir, 'signals.json')
    
    try:
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
    except:
        print("  ✗ Warning: Could not load prediction data")
        return None
    
    # Load trading signals if available
    trading_signals = None
    try:
        if os.path.exists(signals_file):
            with open(signals_file, 'r') as f:
                trading_signals = json.load(f)
    except:
        print("  ⚠ Warning: Could not load trading signals")
    
    # Generate comprehensive markdown report
    report = generate_markdown_report(predictions, trading_signals, timestamp, copied_files)
    
    # Save report
    report_file = os.path.join(dated_dir, 'README.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"  ✓ Generated comprehensive report")
    
    return report

def generate_markdown_report(predictions, trading_signals, timestamp, copied_files):
    """Generate the complete markdown report"""
    
    report = f"""# Ethereum Price Prediction Report

**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Current Price:** ${predictions['current_price']:.2f}

---

## Executive Summary

"""
    
    # Add trading signal if available
    if trading_signals and 'trading_signal' in trading_signals:
        signal = trading_signals['trading_signal']
        trend = trading_signals['trend_analysis']
        
        report += f"""### Trading Signal: {signal['signal']}

**Action:** {signal['action']}  
**Confidence:** {signal['confidence']}  
**Market Trend:** {trend['trend']}

"""
        
        if signal['signal'] != 'WAIT':
            report += f"""**Trade Setup:**
- Entry: ${signal['entry']:.2f}
- Stop Loss: ${signal['stop_loss']:.2f}
- Target: ${signal['target']:.2f}
- Risk/Reward: 1:{signal['risk_reward']:.2f}

"""
    
    # Quick metrics table
    report += f"""### Market Metrics

| Metric | Value |
|:-------|:------|
| **Current Price** | ${predictions['current_price']:.2f} |
| **Trend** | {predictions['trend_analysis']['trend']} |
| **RSI (14)** | {predictions['trend_analysis']['rsi']:.2f} ({predictions['trend_analysis']['rsi_signal']}) |
| **MACD** | {predictions['trend_analysis']['macd_signal']} |
| **BB Position** | {predictions['trend_analysis']['bb_position']} |

---

## Price Predictions

| Time Horizon | Predicted Price | Change | Target Time |
|:-------------|:----------------|:-------|:------------|
"""
    
    for time_label, pred_data in predictions['predictions'].items():
        report += f"| **{time_label}** | ${pred_data['price']:.2f} | {pred_data['change_pct']:+.2f}% | {pred_data['timestamp']} |\n"
    
    report += "\n---\n\n"
    
    # Trading Signals Section
    if trading_signals:
        report += generate_trading_signals_section(trading_signals)
    
    # Charts Section
    report += f"""## Prediction Charts

### Overview Chart

Complete view of historical data, predictions from all models, and ensemble forecast with confidence intervals.

![Prediction Overview](overview.png)

### 1-Hour Focused Prediction

Detailed near-term view with trend lines and prediction paths.

![1-Hour Prediction](1hour.png)

### Technical Indicators

Comprehensive analysis of all technical indicators.

![Technical Indicators](indicators.png)

---

"""
    
    # Model Performance
    report += """## Model Performance

| Model | R² Score | Ensemble Weight | Status |
|:------|:---------|:----------------|:-------|
"""
    
    for model, score in predictions['model_scores'].items():
        weight = predictions['model_weights'].get(model, 0) * 100
        status = "Excellent" if score > 0.90 else "Good" if score > 0.70 else "Fair"
        report += f"| {model.replace('_', ' ').title()} | {score:.4f} | {weight:.1f}% | {status} |\n"
    
    report += f"""
**Ensemble R² Score:** {predictions.get('ensemble_r2', 'N/A')}  
**Prediction Confidence:** {'High' if predictions['model_scores'].get('ml_features', 0) > 0.85 else 'Medium'}

---

"""
    
    # Terminology Guide
    report += generate_terminology_section()
    
    # Files Section
    report += """## Report Files

All data files included in this report:

"""
    for filename in copied_files:
        file_type = "Prediction Data" if 'prediction' in filename else "Trading Signals" if 'signals' in filename else "Market Data" if filename.endswith('.csv') else "Chart" if filename.endswith('.png') else "Document"
        report += f"- {file_type}: `{filename}`\n"
    
    # Disclaimer
    report += """

---

## Disclaimer

This is an automated prediction generated by machine learning models. **Cryptocurrency trading carries substantial risk of loss.** These predictions and trading signals are for informational and educational purposes only and should not be considered financial advice.

### Important Risk Factors

- Past performance does not guarantee future results
- Cryptocurrency markets are extremely volatile
- Model predictions can be incorrect, especially during unexpected market events
- Trading signals are based on technical analysis only and do not account for fundamental factors
- Always use proper risk management and never invest more than you can afford to lose
- Consider consulting with qualified financial advisors before making trading decisions

### Model Limitations

- Optimized for 1-2 hour prediction horizons
- Performance degrades for longer timeframes
- Does not account for major news events or market manipulation
- Requires sufficient historical data for accuracy

---

*Report generated by [Ethereum Price Prediction System](https://github.com/Madgeniusblink/eth-price-prediction)*  
*Model Version: 1.0 | Data Source: Binance API*
"""
    
    return report

def generate_trading_signals_section(trading_signals):
    """Generate the trading signals section of the report"""
    
    trend = trading_signals['trend_analysis']
    levels = trading_signals['support_resistance']
    signal = trading_signals['trading_signal']
    
    section = f"""## Trading Analysis

### Market Trend Assessment

**Overall Trend:** {trend['trend']} (Confidence: {trend['confidence']})

**Trend Components:**
- Moving Average Alignment: {trend['ma_alignment'].title()}
- Price Action: {trend['price_action'].title()}
- Momentum: {trend['momentum'].title()}
- RSI: {trend['rsi']:.2f}
- MACD: {trend['macd_signal'].title()}

### Support and Resistance Levels

**Current Price:** ${levels['current_price']:.2f}

"""
    
    if levels['resistance']:
        section += "**Resistance Levels:**\n"
        for i, level in enumerate(levels['resistance'], 1):
            distance = ((level - levels['current_price']) / levels['current_price']) * 100
            section += f"- R{i}: ${level:.2f} (+{distance:.2f}%)\n"
        section += "\n"
    
    if levels['support']:
        section += "**Support Levels:**\n"
        for i, level in enumerate(levels['support'], 1):
            distance = ((levels['current_price'] - level) / levels['current_price']) * 100
            section += f"- S{i}: ${level:.2f} (-{distance:.2f}%)\n"
        section += "\n"
    
    section += f"""### Trading Signal

**Signal:** {signal['signal']}  
**Action:** {signal['action']}  
**Confidence:** {signal['confidence']}

"""
    
    if signal['signal'] != 'WAIT':
        section += f"""**Recommended Trade Setup:**

| Parameter | Value |
|:----------|:------|
| Entry Price | ${signal['entry']:.2f} |
| Stop Loss | ${signal['stop_loss']:.2f} |
| Target Price | ${signal['target']:.2f} |
| Risk/Reward Ratio | 1:{signal['risk_reward']:.2f} |

"""
    
    section += "**Analysis Reasoning:**\n\n"
    for reason in signal['reasoning']:
        section += f"- {reason}\n"
    
    section += "\n**Signal Strength Scores:**\n\n"
    section += f"- Buy Score: {signal['scores']['buy']}\n"
    section += f"- Sell Score: {signal['scores']['sell']}\n"
    section += f"- Short Score: {signal['scores']['short']}\n"
    
    section += "\n---\n\n"
    
    return section

def generate_terminology_section():
    """Generate terminology guide section"""
    
    section = """## Terminology Guide

Understanding the technical terms used in this report:

"""
    
    # Select key terms to include
    key_terms = ['RSI', 'MACD', 'Bollinger Bands', 'R² Score', 'Ensemble Model', 
                 'Support', 'Resistance', 'Trend', 'Stop Loss', 'Risk/Reward Ratio']
    
    for term in key_terms:
        if term in GLOSSARY:
            info = GLOSSARY[term]
            section += f"### {term}\n\n"
            section += f"**Description:** {info['description']}\n\n"
            section += f"**Interpretation:** {info['interpretation']}\n\n"
            if 'range' in info:
                section += f"**Range:** {info['range']}\n\n"
    
    section += "---\n\n"
    
    return section

def update_latest_readme(latest_dir, report):
    """Update the latest folder README with proper image paths"""
    import re
    
    # Replace timestamped filenames with generic names for latest folder
    report_latest = re.sub(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_overview\.png', 'eth_prediction_overview.png', report)
    report_latest = re.sub(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_1hour\.png', 'eth_1hour_prediction.png', report_latest)
    report_latest = re.sub(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_indicators\.png', 'eth_technical_indicators.png', report_latest)
    
    readme_path = os.path.join(latest_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(report_latest)
    print(f"  ✓ Updated latest README")

def generate_index(reports_dir):
    """Generate an index of all reports"""
    index_content = """# Ethereum Price Prediction Reports

This directory contains automated prediction reports with trading signals, generated every 4 hours.

## Latest Report

See the [`latest/`](latest/) folder for the most recent prediction and trading analysis.

## What's Included in Each Report

Every report contains:

1. **Executive Summary** - Quick overview of market status and trading signal
2. **Price Predictions** - 15-minute, 30-minute, 1-hour, and 2-hour forecasts
3. **Trading Analysis** - Trend assessment, support/resistance levels, and entry/exit signals
4. **Prediction Charts** - Three comprehensive visualizations
5. **Model Performance** - Accuracy metrics and ensemble weights
6. **Terminology Guide** - Explanations of technical terms
7. **Raw Data Files** - JSON predictions, trading signals, and market data CSV

## Report Schedule

Reports are generated at the following times (UTC):

- **00:00** - Asian session opening
- **04:00** - Asian session peak  
- **08:00** - European session opening
- **12:00** - European/US overlap
- **16:00** - US session peak
- **20:00** - US session closing

## Archive Structure

Reports are organized by date: `YYYY/MM/DD/`

Each report folder contains:
- `README.md` - Complete report with all analysis
- `*_prediction.json` - Price prediction data
- `*_signals.json` - Trading signals and trend analysis
- `*_overview.png` - Prediction overview chart
- `*_1hour.png` - 1-hour focused chart
- `*_indicators.png` - Technical indicators chart
- `*_data.csv` - Raw market data

## How to Use These Reports

### For Quick Decisions
1. Open the `latest/` folder
2. Check the **Trading Signal** at the top
3. Review the **Executive Summary**
4. Look at the charts for visual confirmation

### For Deep Analysis
1. Navigate to a specific date
2. Read the complete **Trading Analysis** section
3. Study the **Support/Resistance Levels**
4. Review the **Model Performance** metrics
5. Download the JSON/CSV files for your own analysis

### For Learning
1. Read the **Terminology Guide** in each report
2. Compare predictions vs actual prices over time
3. Study how different market conditions affect signals
4. Track model performance across different market regimes

## Important Notes

- Trading signals are based on technical analysis only
- Always use proper risk management
- These are educational tools, not financial advice
- Cryptocurrency trading carries substantial risk

---

*Generated by [Ethereum Price Prediction System](https://github.com/Madgeniusblink/eth-price-prediction)*
"""
    
    index_path = os.path.join(reports_dir, 'README.md')
    with open(index_path, 'w') as f:
        f.write(index_content)

def main():
    """Main report generation function"""
    try:
        # Create folder structure
        dated_dir, latest_dir, timestamp = create_report_folders()
        print(f"Report folders created:")
        print(f"  Dated: {dated_dir}")
        print(f"  Latest: {latest_dir}\n")
        
        # Run prediction pipeline (including trading signals)
        success = run_prediction_pipeline()
        
        if not success:
            print("\n✗ Report generation failed")
            return 1
        
        # Copy outputs to report folders
        print("Organizing report files...")
        copied_files = copy_outputs_to_report_folder(dated_dir, latest_dir, timestamp)
        
        # Generate comprehensive report
        print("\nGenerating comprehensive report...")
        report = generate_comprehensive_report(dated_dir, timestamp, copied_files)
        
        if report:
            update_latest_readme(latest_dir, report)
        
        # Update main index
        base_reports = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
        generate_index(base_reports)
        print("  ✓ Updated reports index")
        
        print("\n" + "=" * 70)
        print("  REPORT GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nReport Location: {dated_dir}")
        print(f"Latest Report: {latest_dir}")
        print(f"\nFiles Generated: {len(copied_files)}")
        print("\nReport includes:")
        print("  ✓ Price predictions (15m, 30m, 1h, 2h)")
        print("  ✓ Trading signals and trend analysis")
        print("  ✓ Support/resistance levels")
        print("  ✓ Three comprehensive charts")
        print("  ✓ Model performance metrics")
        print("  ✓ Terminology guide")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
