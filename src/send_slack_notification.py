#!/usr/bin/env python3
"""
Send Slack notifications for ETH price prediction reports.
"""

import json
import os
import sys
import requests
from datetime import datetime


def load_json_file(filepath):
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def format_price(price):
    """Format price with comma separators and 2 decimal places."""
    return f"${price:,.2f}"


def format_percentage(value):
    """Format percentage with sign and 2 decimal places."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def get_signal_emoji(signal):
    """Get emoji for trading signal."""
    signal_upper = signal.upper()
    if signal_upper == "BUY":
        return "🟢"
    elif signal_upper == "SELL":
        return "🔴"
    elif signal_upper == "SHORT":
        return "🔻"
    else:
        return "⚪"


def get_confidence_emoji(confidence):
    """Get emoji for confidence level."""
    confidence_upper = confidence.upper()
    if confidence_upper == "HIGH":
        return "⭐⭐⭐"
    elif confidence_upper == "MEDIUM":
        return "⭐⭐"
    elif confidence_upper == "LOW":
        return "⭐"
    else:
        return "⭐"


def get_trend_emoji(trend):
    """Get emoji for market trend."""
    trend_upper = trend.upper()
    if "BULL" in trend_upper:
        return "📈"
    elif "BEAR" in trend_upper:
        return "📉"
    else:
        return "➡️"


def create_slack_message(predictions, signals, report_url):
    """Create a formatted Slack message."""
    
    # Extract key information
    current_price = predictions.get('current_price', 0)
    timestamp = predictions.get('timestamp', datetime.utcnow().isoformat())
    
    # Get predictions
    pred_15m = predictions.get('predictions', {}).get('15m', {})
    pred_30m = predictions.get('predictions', {}).get('30m', {})
    pred_1h = predictions.get('predictions', {}).get('1h', {})
    pred_2h = predictions.get('predictions', {}).get('2h', {})
    
    # Get trading signals
    signal = signals.get('signal', 'HOLD')
    confidence = signals.get('confidence', 'MEDIUM')
    trend = signals.get('trend', 'NEUTRAL')
    action = signals.get('action', 'Monitor position')
    
    # Get trade setup
    trade_setup = signals.get('trade_setup', {})
    entry = trade_setup.get('entry', current_price)
    stop_loss = trade_setup.get('stop_loss', 0)
    target = trade_setup.get('target', 0)
    risk_reward = trade_setup.get('risk_reward', '0:0')
    
    # Get technical indicators
    rsi = signals.get('rsi', {}).get('value', 0)
    rsi_status = signals.get('rsi', {}).get('status', 'NEUTRAL')
    macd = signals.get('macd', {}).get('status', 'NEUTRAL')
    bb_position = signals.get('bollinger', {}).get('position', 'MIDDLE')
    
    # Build Slack message
    message = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🔮 ETH Price Prediction Report",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Generated:*\n{timestamp.replace('T', ' ').split('.')[0]} UTC"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Current Price:*\n{format_price(current_price)}"
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
                    "text": f"*{get_signal_emoji(signal)} Trading Signal: {signal.upper()}*\n"
                           f"*Action:* {action}\n"
                           f"*Confidence:* {confidence.upper()} {get_confidence_emoji(confidence)}\n"
                           f"*Market Trend:* {trend.upper()} {get_trend_emoji(trend)}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Entry Price:*\n{format_price(entry)}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Stop Loss:*\n{format_price(stop_loss)}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Target Price:*\n{format_price(target)}"
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
                    "text": "*📊 Price Predictions*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*15 min:*\n{format_price(pred_15m.get('price', 0))} ({format_percentage(pred_15m.get('change_percent', 0))})"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*30 min:*\n{format_price(pred_30m.get('price', 0))} ({format_percentage(pred_30m.get('change_percent', 0))})"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*1 hour:*\n{format_price(pred_1h.get('price', 0))} ({format_percentage(pred_1h.get('change_percent', 0))})"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*2 hours:*\n{format_price(pred_2h.get('price', 0))} ({format_percentage(pred_2h.get('change_percent', 0))})"
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
                    "text": "*📈 Technical Indicators*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*RSI (14):*\n{rsi:.2f} ({rsi_status})"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*MACD:*\n{macd}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Bollinger Bands:*\n{bb_position}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Trend:*\n{trend.upper()}"
                    }
                ]
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
                            "text": "📄 View Full Report",
                            "emoji": True
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
                        "text": "⚠️ _This is an automated prediction for educational purposes. Not financial advice. Trade at your own risk._"
                    }
                ]
            }
        ]
    }
    
    return message


def send_slack_notification(webhook_url, predictions_file, signals_file, report_url):
    """Send notification to Slack."""
    
    # Load data files
    predictions = load_json_file(predictions_file)
    signals = load_json_file(signals_file)
    
    if not predictions or not signals:
        print("Error: Could not load required data files")
        return False
    
    # Create message
    message = create_slack_message(predictions, signals, report_url)
    
    # Send to Slack
    try:
        response = requests.post(
            webhook_url,
            json=message,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            print("✓ Slack notification sent successfully!")
            return True
        else:
            print(f"✗ Failed to send Slack notification: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error sending Slack notification: {e}")
        return False


def main():
    """Main function."""
    
    # Get webhook URL from environment
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if not webhook_url:
        print("Error: SLACK_WEBHOOK_URL environment variable not set")
        sys.exit(1)
    
    # Get file paths from arguments or use defaults
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    predictions_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, 'predictions_summary.json')
    signals_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base_dir, 'trading_signals.json')
    report_url = sys.argv[3] if len(sys.argv) > 3 else 'https://github.com/Madgeniusblink/eth-price-prediction/tree/main/reports/latest'
    
    # Send notification
    success = send_slack_notification(webhook_url, predictions_file, signals_file, report_url)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
