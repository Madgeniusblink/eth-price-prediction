"""
Alert System for ETH Price Prediction System
Sends notifications via Slack for critical errors, warnings, and successes
"""

import requests
import json
import os
from datetime import datetime
from logger import setup_logger

logger = setup_logger(__name__)

class AlertSystem:
    """Send alerts via Slack webhook"""
    
    def __init__(self):
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.enabled = self.slack_webhook is not None
        
        if not self.enabled:
            logger.warning("Slack webhook not configured - alerts will be logged only")
    
    def send_slack(self, message, level='info'):
        """
        Send message to Slack
        
        Args:
            message: Message text
            level: Message level (critical, error, warning, info, success)
        """
        if not self.enabled:
            logger.info(f"[ALERT - {level.upper()}] {message}")
            return False
        
        # Color coding based on level
        colors = {
            'critical': '#FF0000',  # Red
            'error': '#FF6B6B',     # Light red
            'warning': '#FFA500',   # Orange
            'info': '#4A90E2',      # Blue
            'success': '#00C851'    # Green
        }
        
        # Emoji based on level
        emojis = {
            'critical': '🔴',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅'
        }
        
        color = colors.get(level, '#808080')
        emoji = emojis.get(level, '📢')
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f"{emoji} ETH Prediction System Alert",
                'text': message,
                'footer': 'ETH Price Prediction System',
                'ts': int(datetime.now().timestamp())
            }]
        }
        
        try:
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Alert sent successfully: {message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def send_critical_alert(self, message, context=None):
        """
        Send critical alert (system failure, data issues)
        
        Args:
            message: Alert message
            context: Additional context dict (optional)
        """
        full_message = f"🔴 *CRITICAL*: {message}"
        
        if context:
            full_message += f"\n\n*Context:*\n```{json.dumps(context, indent=2)}```"
        
        logger.critical(message)
        if context:
            logger.critical(f"Context: {context}")
        
        return self.send_slack(full_message, 'critical')
    
    def send_error_alert(self, message, error=None):
        """
        Send error alert (recoverable errors, API failures)
        
        Args:
            message: Alert message
            error: Exception object (optional)
        """
        full_message = f"❌ *ERROR*: {message}"
        
        if error:
            full_message += f"\n\n*Error:* `{str(error)}`"
        
        logger.error(message)
        if error:
            logger.error(f"Error details: {error}")
        
        return self.send_slack(full_message, 'error')
    
    def send_warning_alert(self, message):
        """
        Send warning alert (performance degradation, stale data)
        
        Args:
            message: Alert message
        """
        full_message = f"⚠️ *WARNING*: {message}"
        
        logger.warning(message)
        return self.send_slack(full_message, 'warning')
    
    def send_success_alert(self, message, metrics=None):
        """
        Send success alert (predictions generated, system healthy)
        
        Args:
            message: Alert message
            metrics: Performance metrics dict (optional)
        """
        full_message = f"✅ *SUCCESS*: {message}"
        
        if metrics:
            metrics_text = "\n".join([f"  • {k}: {v}" for k, v in metrics.items()])
            full_message += f"\n\n*Metrics:*\n{metrics_text}"
        
        logger.info(message)
        if metrics:
            logger.info(f"Metrics: {metrics}")
        
        return self.send_slack(full_message, 'success')
    
    def send_info_alert(self, message):
        """
        Send informational alert
        
        Args:
            message: Alert message
        """
        full_message = f"ℹ️ *INFO*: {message}"
        
        logger.info(message)
        return self.send_slack(full_message, 'info')
    
    def send_prediction_alert(self, current_price, predictions, signal):
        """
        Send prediction summary alert
        
        Args:
            current_price: Current ETH price
            predictions: Predictions dict
            signal: Trading signal dict
        """
        # Format predictions
        pred_lines = []
        for timeframe, pred in predictions.items():
            price = pred.get('price', 0)
            change = pred.get('change_pct', 0)
            pred_lines.append(f"  • {timeframe}: ${price:,.2f} ({change:+.2f}%)")
        
        predictions_text = "\n".join(pred_lines)
        
        # Format signal
        signal_type = signal.get('signal', 'WAIT')
        confidence = signal.get('confidence', 'UNKNOWN')
        entry = signal.get('entry', 0)
        target = signal.get('target', 0)
        stop_loss = signal.get('stop_loss', 0)
        
        # Choose emoji based on signal
        signal_emoji = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '🟡',
            'WAIT': '⏸️'
        }.get(signal_type, '❓')
        
        message = f"""*New ETH Price Prediction*

*Current Price:* ${current_price:,.2f}

*Predictions:*
{predictions_text}

*Trading Signal:* {signal_emoji} {signal_type}
*Confidence:* {confidence}
*Entry:* ${entry:,.2f}
*Target:* ${target:,.2f}
*Stop Loss:* ${stop_loss:,.2f}

View full report: https://github.com/Madgeniusblink/eth-price-prediction/tree/main/reports/latest
"""
        
        logger.info(f"Sending prediction alert: {signal_type} signal with {confidence} confidence")
        return self.send_slack(message, 'info')
    
    def send_health_check_alert(self, status, metrics):
        """
        Send health check status alert
        
        Args:
            status: Health status (healthy, degraded, critical)
            metrics: Health metrics dict
        """
        status_emoji = {
            'healthy': '💚',
            'degraded': '💛',
            'critical': '🔴'
        }.get(status, '❓')
        
        level = {
            'healthy': 'success',
            'degraded': 'warning',
            'critical': 'critical'
        }.get(status, 'info')
        
        metrics_text = "\n".join([f"  • {k}: {v}" for k, v in metrics.items()])
        
        message = f"""*System Health Check*

*Status:* {status_emoji} {status.upper()}

*Metrics:*
{metrics_text}
"""
        
        logger.info(f"Sending health check alert: {status}")
        return self.send_slack(message, level)

# Create global alert system instance
alert_system = AlertSystem()

if __name__ == '__main__':
    # Test the alert system
    print("Testing Alert System...")
    print(f"Slack webhook configured: {alert_system.enabled}")
    print()
    
    # Test different alert levels
    alert_system.send_info_alert("Testing alert system")
    alert_system.send_success_alert("System test successful", {'test_metric': '100%'})
    alert_system.send_warning_alert("This is a test warning")
    alert_system.send_error_alert("This is a test error", ValueError("Test error"))
    alert_system.send_critical_alert("This is a test critical alert", {'component': 'test'})
    
    # Test prediction alert
    alert_system.send_prediction_alert(
        current_price=3154.32,
        predictions={
            '15m': {'price': 3170.50, 'change_pct': 0.5},
            '30m': {'price': 3185.25, 'change_pct': 1.0},
            '60m': {'price': 3200.00, 'change_pct': 1.5}
        },
        signal={
            'signal': 'BUY',
            'confidence': 'HIGH',
            'entry': 3155.00,
            'target': 3200.00,
            'stop_loss': 3130.00
        }
    )
    
    # Test health check alert
    alert_system.send_health_check_alert(
        status='healthy',
        metrics={
            'uptime': '99.9%',
            'accuracy': '69.5%',
            'last_run': 'Success'
        }
    )
    
    print("\n✓ Alert system test complete")
