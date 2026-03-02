"""
Signal Accuracy Tracker
Tracks prediction accuracy by comparing past predictions with actual prices
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests


def load_predictions_history() -> List[Dict]:
    """Load historical predictions from predictions_summary.json"""
    try:
        with open('predictions_summary.json', 'r') as f:
            prediction = json.load(f)
            return [prediction]  # Return as list for consistency
    except FileNotFoundError:
        print("No predictions_summary.json found")
        return []
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return []


def fetch_historical_price(timestamp: str) -> Optional[float]:
    """
    Fetch historical price from Binance at a specific timestamp

    Args:
        timestamp: ISO format timestamp string

    Returns:
        Price at that timestamp, or None if unable to fetch
    """
    try:
        # Convert ISO timestamp to milliseconds
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        start_time = int(dt.timestamp() * 1000)
        end_time = start_time + 60000  # +1 minute window

        # Use Kraken (no geo-restrictions on GitHub Actions)
        since_sec = int(dt.timestamp())
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            'pair': 'ETHUSD',
            'interval': 1,
            'since': since_sec,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        pair_key = [k for k in data["result"] if k != "last"][0]
        rows = data["result"][pair_key]

        if rows:
            # Kraken OHLC: [time, open, high, low, close, vwap, volume, count]
            return float(rows[0][4])  # close price
        return None

    except Exception as e:
        print(f"Error fetching historical price for {timestamp}: {e}")
        return None


def calculate_prediction_accuracy(predictions: List[Dict]) -> Dict:
    """
    Calculate accuracy statistics for predictions

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Dictionary with accuracy statistics
    """
    accuracy_records = []
    stats_by_timeframe = {
        '15m': {'errors': [], 'correct_direction': 0, 'total': 0},
        '30m': {'errors': [], 'correct_direction': 0, 'total': 0},
        '60m': {'errors': [], 'correct_direction': 0, 'total': 0},
        '120m': {'errors': [], 'correct_direction': 0, 'total': 0}
    }

    for pred in predictions:
        generated_at = pred.get('generated_at')
        current_price = pred.get('current_price')
        predictions_data = pred.get('predictions', {})

        for timeframe, pred_data in predictions_data.items():
            predicted_price = pred_data.get('price')
            predicted_timestamp = pred_data.get('timestamp')
            predicted_change_pct = pred_data.get('change_pct')

            # Fetch actual price at predicted timestamp
            actual_price = fetch_historical_price(predicted_timestamp)

            if actual_price and predicted_price and current_price:
                # Calculate errors
                price_error = predicted_price - actual_price
                price_error_pct = (price_error / actual_price) * 100
                mae = abs(price_error)
                mape = abs(price_error_pct)

                # Check if direction was correct
                actual_change_pct = ((actual_price - current_price) / current_price) * 100
                direction_correct = (predicted_change_pct > 0 and actual_change_pct > 0) or \
                                  (predicted_change_pct < 0 and actual_change_pct < 0) or \
                                  (predicted_change_pct == 0 and actual_change_pct == 0)

                # Record accuracy
                record = {
                    'generated_at': generated_at,
                    'timeframe': timeframe,
                    'predicted_timestamp': predicted_timestamp,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'actual_price': actual_price,
                    'price_error': price_error,
                    'price_error_pct': price_error_pct,
                    'mae': mae,
                    'mape': mape,
                    'direction_correct': direction_correct,
                    'predicted_change_pct': predicted_change_pct,
                    'actual_change_pct': actual_change_pct
                }

                accuracy_records.append(record)

                # Update stats by timeframe
                stats_by_timeframe[timeframe]['errors'].append(price_error_pct)
                stats_by_timeframe[timeframe]['total'] += 1
                if direction_correct:
                    stats_by_timeframe[timeframe]['correct_direction'] += 1

    return {
        'records': accuracy_records,
        'stats_by_timeframe': stats_by_timeframe
    }


def generate_accuracy_summary(stats_by_timeframe: Dict) -> Dict:
    """Generate summary statistics"""
    summary = {}

    for timeframe, data in stats_by_timeframe.items():
        if data['total'] > 0:
            errors = data['errors']
            mean_error = sum(errors) / len(errors) if errors else 0
            mae = sum(abs(e) for e in errors) / len(errors) if errors else 0
            direction_accuracy = (data['correct_direction'] / data['total']) * 100

            summary[timeframe] = {
                'total_predictions': data['total'],
                'mean_error_pct': round(mean_error, 4),
                'mean_absolute_error_pct': round(mae, 4),
                'direction_accuracy_pct': round(direction_accuracy, 2),
                'correct_direction': data['correct_direction'],
                'incorrect_direction': data['total'] - data['correct_direction']
            }

    return summary


def save_accuracy_data(accuracy_data: Dict):
    """Save accuracy data to JSON files"""
    os.makedirs('data', exist_ok=True)

    # Save detailed accuracy log
    log_file = 'data/accuracy_log.json'
    with open(log_file, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'records': accuracy_data['records']
        }, f, indent=2)
    print(f"✓ Accuracy log saved to {log_file}")

    # Generate and save summary stats
    summary = generate_accuracy_summary(accuracy_data['stats_by_timeframe'])
    stats_file = 'data/accuracy_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'total_evaluations': sum(s['total_predictions'] for s in summary.values())
        }, f, indent=2)
    print(f"✓ Accuracy stats saved to {stats_file}")


def get_accuracy_summary() -> Dict:
    """
    Load and return accuracy summary statistics

    Returns:
        Dictionary with accuracy stats, or empty dict if not available
    """
    try:
        with open('data/accuracy_stats.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'error': 'No accuracy stats available. Run track_accuracy() first.',
            'summary': {}
        }
    except Exception as e:
        return {
            'error': f'Error loading accuracy stats: {e}',
            'summary': {}
        }


def track_accuracy():
    """Main function to track and update accuracy statistics"""
    print("=== Signal Accuracy Tracker ===\n")

    # Load predictions
    print("Loading predictions...")
    predictions = load_predictions_history()

    if not predictions:
        print("No predictions found to track")
        return

    print(f"Found {len(predictions)} prediction(s) to evaluate\n")

    # Calculate accuracy
    print("Calculating accuracy (fetching historical prices)...")
    accuracy_data = calculate_prediction_accuracy(predictions)

    if not accuracy_data['records']:
        print("No accuracy data could be calculated (predictions may be too recent)")
        return

    # Save results
    print("\nSaving results...")
    save_accuracy_data(accuracy_data)

    # Display summary
    summary = generate_accuracy_summary(accuracy_data['stats_by_timeframe'])
    print("\n=== Accuracy Summary ===")
    for timeframe, stats in summary.items():
        print(f"\n{timeframe}:")
        print(f"  Total Predictions: {stats['total_predictions']}")
        print(f"  Direction Accuracy: {stats['direction_accuracy_pct']}%")
        print(f"  Mean Absolute Error: {stats['mean_absolute_error_pct']}%")

    print("\n✓ Accuracy tracking complete")


def main():
    """Run accuracy tracker"""
    track_accuracy()


if __name__ == '__main__':
    main()
