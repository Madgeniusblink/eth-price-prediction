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
    current_price = predictions.get('current_price', 0)
    preds = predictions.get('predictions', {})
    pred_15m = preds.get('15min', preds.get('15m', {}))
    pred_30m = preds.get('30min', preds.get('30m', {}))
    pred_1h = preds.get('1hour', preds.get('1h', {}))
    pred_2h = preds.get('2hour', preds.get('2h', {}))
    
    signal = signals.get('signal', 'HOLD')
    action = signals.get('action', 'Monitor position')
    confidence = signals.get('confidence', 'MEDIUM')
    trend = signals.get('market_trend', signals.get('trend', 'NEUTRAL'))
    
    trade_setup = signals.get('trade_setup', {})
    entry = trade_setup.get('entry_price', trade_setup.get('entry', current_price))
    stop_loss = trade_setup.get('stop_loss', 0)
    target = trade_setup.get('target_price', trade_setup.get('target', 0))
    risk_reward = trade_setup.get('risk_reward', '0:0')
    position_size = trade_setup.get('position_size', 'N/A')
    
    indicators = signals.get('technical_indicators', signals)
    rsi_data = indicators.get('rsi', {})
    rsi_value = rsi_data.get('value', 0)
    rsi_status = rsi_data.get('status', 'NEUTRAL')
    
    macd_data = indicators.get('macd', {})
    macd_status = macd_data.get('status', 'NEUTRAL')
    
    bb_data = indicators.get('bollinger_bands', indicators.get('bollinger', {}))
    bb_position = bb_data.get('position', 'MIDDLE')
    
    trend_data = indicators.get('trend', {})
    trend_direction = trend_data.get('direction', trend)
    
    volume_data = indicators.get('volume', {})
    volume_trend = volume_data.get('trend', 'N/A')
    
    volatility_data = indicators.get('volatility', {})
    volatility_level = volatility_data.get('level', 'N/A')
    
    support = indicators.get('support', 0)
    resistance = indicators.get('resistance', 0)
    
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    
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
    overview_img = f"{repo_url}/eth_prediction_overview.png"
    hour_img = f"{repo_url}/eth_1hour_prediction.png"
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
                                "text": f"*Risk/Reward:*\n{risk_reward}"
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
                                "text": f"*15 min:*\n${pred_15m.get('price', 0):,.2f} ({pred_15m.get('change_percent', 0):+.2f}%)"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*30 min:*\n${pred_30m.get('price', 0):,.2f} ({pred_30m.get('change_percent', 0):+.2f}%)"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*1 hour:*\n${pred_1h.get('price', 0):,.2f} ({pred_1h.get('change_percent', 0):+.2f}%)"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*2 hours:*\n${pred_2h.get('price', 0):,.2f} ({pred_2h.get('change_percent', 0):+.2f}%)"
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
                                "text": f"*RSI (14):*\n{rsi_value:.2f} - {rsi_status}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*MACD:*\n{macd_status}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Bollinger Bands:*\n{bb_position}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Trend Direction:*\n{trend_direction}"
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
                                "text": f"*Entry Price:*\n${entry:,.2f}" if entry > 0 else "*Entry Price:*\nN/A"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Stop Loss:*\n${stop_loss:,.2f}" if stop_loss > 0 else "*Stop Loss:*\nN/A"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Target Price:*\n${target:,.2f}" if target > 0 else "*Target Price:*\nN/A"
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
                                "text": f"*Support Level:*\n${support:,.2f}" if support > 0 else "*Support Level:*\nN/A"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Resistance Level:*\n${resistance:,.2f}" if resistance > 0 else "*Resistance Level:*\nN/A"
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
                        "alt_text": "1-Hour Prediction Chart"
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
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "View Full Report"
                                },
                                "url": report_url,
                                "style": "primary"
                            }
                        ]
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": "_Automated prediction for educational purposes only. Not financial advice. Trade at your own risk._"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    # Send the notification
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✓ Slack notification sent successfully!")
        print(f"  Signal: {signal}")
        print(f"  Price: ${current_price:,.2f}")
        print(f"  Confidence: {confidence}")
        print(f"  Charts included: 3 images")
    except requests.exceptions.RequestException as e:
        print(f"Error sending Slack notification: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python send_slack_notification.py <predictions_file> <signals_file> <report_url>")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    signals_file = sys.argv[2]
    report_url = sys.argv[3]
    
    send_slack_notification(predictions_file, signals_file, report_url)
