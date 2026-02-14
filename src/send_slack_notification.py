#!/usr/bin/env python3
"""
Send formatted Slack notifications with ETH price predictions and charts.
Clean, professional design without emojis, includes chart images.
"""

import os
import sys
import json
import requests
from datetime import datetime

def send_slack_notification(predictions_file, signals_file, report_url):
    """Send a clean, professional Slack notification with predictions and charts."""
    
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if not webhook_url:
        print("Error: SLACK_WEBHOOK_URL environment variable not set")
        sys.exit(1)
    
    # Load prediction and signal data
    try:
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        with open(signals_file, 'r') as f:
            signals = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
        sys.exit(1)
    
    # Extract data with safe fallbacks
    # Current prediction engine uses 15m/30m/60m/120m timeframes
    current_price = predictions.get('current_price', 0)
    preds = predictions.get('predictions', {})
    pred_15m = preds.get('15m', preds.get('15min', {}))
    pred_30m = preds.get('30m', preds.get('30min', {}))
    pred_60m = preds.get('60m', preds.get('60min', {}))
    pred_120m = preds.get('120m', preds.get('120min', {}))
    
    # Extract trading signal data from correct structure
    trading_signal = signals.get('trading_signal', {})
    signal = trading_signal.get('signal', 'HOLD')
    action = trading_signal.get('action', 'Monitor position')
    confidence = trading_signal.get('confidence', 'MEDIUM')
    entry = trading_signal.get('entry', current_price)
    stop_loss = trading_signal.get('stop_loss', 0)
    target = trading_signal.get('target', 0)
    risk_reward = trading_signal.get('risk_reward', 0)
    
    # Extract trend analysis
    trend_analysis = signals.get('trend_analysis', {})
    trend = trend_analysis.get('trend', 'NEUTRAL')
    rsi_value = trend_analysis.get('rsi', 0)
    macd_status = trend_analysis.get('macd_signal', 'NEUTRAL').upper()
    
    # Extract support/resistance
    support_resistance = signals.get('support_resistance', {})
    support = support_resistance.get('nearest_support', 0)
    resistance = support_resistance.get('nearest_resistance', 0)
    
    # Calculate position size (2% risk rule)
    if stop_loss > 0:
        risk_per_coin = abs(entry - stop_loss)
        account_size = 10000  # Default account size
        risk_amount = account_size * 0.02
        position_size = f"${risk_amount / risk_per_coin:,.2f}"
    else:
        position_size = "Calculate based on your risk tolerance"
    
    # Calculate volume trend and volatility from recent price action
    volume_trend = "INCREASING" if trend == "BULL MARKET" else "DECREASING" if trend == "BEAR MARKET" else "STABLE"
    change_120m = abs(pred_120m.get('change_pct', pred_120m.get('change_percent', 0)))
    volatility_level = "HIGH" if change_120m > 5 else "MEDIUM" if change_120m > 2 else "LOW"
    
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # Determine signal color
    signal_colors = {
        'BUY': '#2ecc71',    # Green
        'SELL': '#e74c3c',   # Red
        'SHORT': '#e67e22',  # Orange
        'HOLD': '#95a5a6'    # Gray
    }
    color = signal_colors.get(signal, '#95a5a6')
    
    # GitHub raw URLs for chart images
    repo_url = "https://raw.githubusercontent.com/Madgeniusblink/eth-price-prediction/main/reports/latest"
    overview_img = f"{repo_url}/eth_predictions_overview.png"
    hour_img = f"{repo_url}/eth_2hour_prediction.png"
    indicators_img = f"{repo_url}/eth_technical_indicators.png"
    
    # Build the message payload
    payload = {
        "text": f"ETH Price Prediction Report - {timestamp}",
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": "ETH Price Prediction Report"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Generated:*\n{timestamp}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Current Price:*\n${current_price:,.2f}"
                            }
                        ]
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*TRADING SIGNAL: {signal}*"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Action:*\n{action}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Confidence:*\n{confidence}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Market Trend:*\n{trend}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Risk/Reward:*\n{risk_reward:.2f}:1" if risk_reward > 0 else "*Risk/Reward:*\nCalculate based on entry/exit"
                            }
                        ]
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*PRICE PREDICTIONS*"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*15 min:*\n${pred_15m.get('price', 0):,.2f} ({pred_15m.get('change_pct', pred_15m.get('change_percent', 0)):+.2f}%)"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*30 min:*\n${pred_30m.get('price', 0):,.2f} ({pred_30m.get('change_pct', pred_30m.get('change_percent', 0)):+.2f}%)"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*60 min:*\n${pred_60m.get('price', 0):,.2f} ({pred_60m.get('change_pct', pred_60m.get('change_percent', 0)):+.2f}%)"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*120 min:*\n${pred_120m.get('price', 0):,.2f} ({pred_120m.get('change_pct', pred_120m.get('change_percent', 0)):+.2f}%)"
                            }
                        ]
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*TECHNICAL INDICATORS*"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*RSI (14):*\n{rsi_value:.2f}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*MACD:*\n{macd_status}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Support:*\n${support:,.2f}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Resistance:*\n${resistance:,.2f}"
                            }
                        ]
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*TRADE SETUP*"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Entry Price:*\n${entry:,.2f}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Stop Loss:*\n${stop_loss:,.2f}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Target Price:*\n${target:,.2f}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Position Size:*\n{position_size}"
                            }
                        ]
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*MARKET ANALYSIS*"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Volume Trend:*\n{volume_trend}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Volatility:*\n{volatility_level}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Trend Strength:*\n{confidence}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Market Phase:*\n{trend}"
                            }
                        ]
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*PREDICTION CHARTS*"
                        }
                    },
                    {
                        "type": "image",
                        "image_url": overview_img,
                        "alt_text": "Prediction Overview Chart"
                    },
                    {
                        "type": "image",
                        "image_url": hour_img,
                        "alt_text": "2-Hour Prediction Chart"
                    },
                    {
                        "type": "image",
                        "image_url": indicators_img,
                        "alt_text": "Technical Indicators Chart"
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"<{report_url}|View Full Report on GitHub>"
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": "⚠️ _This is an automated prediction for educational purposes. Not financial advice. Trade at your own risk._"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    # Send the notification
    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        print("✓ Slack notification sent successfully")
    except requests.exceptions.RequestException as e:
        print(f"Error sending Slack notification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: send_slack_notification.py <predictions_file> <signals_file> <report_url>")
        sys.exit(1)
    
    send_slack_notification(sys.argv[1], sys.argv[2], sys.argv[3])
