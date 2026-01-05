"""
Prediction Accuracy Tracking System
Compares past predictions with actual prices to measure model performance over time
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
from config import BASE_DIR
matplotlib.use('Agg')

class AccuracyTracker:
    """Track and analyze prediction accuracy over time"""
    
    def __init__(self, reports_dir=os.path.join(BASE_DIR, 'reports')):
        self.reports_dir = reports_dir
        self.history_file = os.path.join(reports_dir, 'accuracy_history.json')
        self.load_history()
    
    def load_history(self):
        """Load historical accuracy data"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                'predictions': [],
                'actuals': [],
                'summary': {
                    'total_predictions': 0,
                    'avg_error': 0,
                    'avg_error_pct': 0,
                    'directional_accuracy': 0
                }
            }
    
    def save_history(self):
        """Save accuracy history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record_prediction(self, timestamp, predictions, current_price):
        """
        Record a new prediction for future validation
        
        Args:
            timestamp: When the prediction was made
            predictions: Dict with time horizons and predicted prices
            current_price: Current price when prediction was made
        """
        record = {
            'prediction_time': timestamp.isoformat(),
            'current_price': current_price,
            'predictions': predictions,
            'validated': False
        }
        self.history['predictions'].append(record)
        self.save_history()
    
    def validate_predictions(self, current_time, current_price):
        """
        Check past predictions and compare with actual prices
        
        Args:
            current_time: Current datetime
            current_price: Current actual price
        """
        validated_count = 0
        
        for pred_record in self.history['predictions']:
            if pred_record['validated']:
                continue
            
            pred_time = datetime.fromisoformat(pred_record['prediction_time'])
            
            # Check each prediction horizon
            for horizon, pred_data in pred_record['predictions'].items():
                target_time = datetime.fromisoformat(pred_data['timestamp'])
                
                # If we've reached or passed the target time, validate
                if current_time >= target_time:
                    # Calculate error
                    predicted_price = pred_data['price']
                    error = abs(predicted_price - current_price)
                    error_pct = (error / current_price) * 100
                    
                    # Check directional accuracy
                    predicted_direction = 'up' if predicted_price > pred_record['current_price'] else 'down'
                    actual_direction = 'up' if current_price > pred_record['current_price'] else 'down'
                    direction_correct = predicted_direction == actual_direction
                    
                    # Record actual result
                    actual_record = {
                        'prediction_time': pred_record['prediction_time'],
                        'target_time': target_time.isoformat(),
                        'horizon': horizon,
                        'predicted_price': predicted_price,
                        'actual_price': current_price,
                        'error': error,
                        'error_pct': error_pct,
                        'direction_correct': direction_correct,
                        'validation_time': current_time.isoformat()
                    }
                    
                    self.history['actuals'].append(actual_record)
                    pred_record['validated'] = True
                    validated_count += 1
        
        if validated_count > 0:
            self.update_summary()
            self.save_history()
        
        return validated_count
    
    def update_summary(self):
        """Update summary statistics"""
        if not self.history['actuals']:
            return
        
        df = pd.DataFrame(self.history['actuals'])
        
        self.history['summary'] = {
            'total_predictions': len(df),
            'avg_error': float(df['error'].mean()),
            'avg_error_pct': float(df['error_pct'].mean()),
            'directional_accuracy': float(df['direction_correct'].mean() * 100),
            'best_horizon': df.groupby('horizon')['error_pct'].mean().idxmin(),
            'last_updated': datetime.now().isoformat()
        }
    
    def generate_accuracy_report(self):
        """Generate a detailed accuracy report"""
        if not self.history['actuals']:
            return "No validated predictions yet. Run the system for at least one prediction cycle."
        
        df = pd.DataFrame(self.history['actuals'])
        
        report = f"""# Prediction Accuracy Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Total Predictions Validated:** {len(df)}

## Overall Performance

| Metric | Value |
|:-------|:------|
| **Average Error** | ${self.history['summary']['avg_error']:.2f} |
| **Average Error %** | {self.history['summary']['avg_error_pct']:.2f}% |
| **Directional Accuracy** | {self.history['summary']['directional_accuracy']:.1f}% |
| **Best Time Horizon** | {self.history['summary']['best_horizon']} |

## Performance by Time Horizon

"""
        
        # Group by horizon
        horizon_stats = df.groupby('horizon').agg({
            'error': 'mean',
            'error_pct': 'mean',
            'direction_correct': 'mean'
        }).round(2)
        
        report += "| Horizon | Avg Error ($) | Avg Error (%) | Direction Accuracy |\n"
        report += "|:--------|:--------------|:--------------|:-------------------|\n"
        
        for horizon, row in horizon_stats.iterrows():
            report += f"| **{horizon}** | ${row['error']:.2f} | {row['error_pct']:.2f}% | {row['direction_correct']*100:.1f}% |\n"
        
        report += f"""

## Recent Predictions (Last 10)

"""
        
        recent = df.tail(10)[['prediction_time', 'horizon', 'predicted_price', 'actual_price', 'error_pct', 'direction_correct']]
        report += recent.to_markdown(index=False)
        
        report += """

## Interpretation

- **Average Error**: How far off predictions are on average (in dollars)
- **Average Error %**: Percentage difference between predicted and actual price
- **Directional Accuracy**: How often we correctly predict if price will go up or down
- **Best Time Horizon**: Which prediction timeframe is most accurate

## Recommendations

"""
        
        if self.history['summary']['directional_accuracy'] > 60:
            report += "- Directional accuracy is good (>60%). The model correctly identifies trends.\n"
        else:
            report += "- Directional accuracy needs improvement. Consider adjusting technical indicators.\n"
        
        if self.history['summary']['avg_error_pct'] < 1.0:
            report += "- Price predictions are highly accurate (<1% error).\n"
        elif self.history['summary']['avg_error_pct'] < 2.0:
            report += "- Price predictions are good (<2% error).\n"
        else:
            report += "- Price prediction error is high. Model may need retraining or parameter tuning.\n"
        
        return report
    
    def plot_accuracy_over_time(self, output_path):
        """Create visualization of accuracy trends"""
        if not self.history['actuals']:
            return False
        
        df = pd.DataFrame(self.history['actuals'])
        df['validation_time'] = pd.to_datetime(df['validation_time'])
        df = df.sort_values('validation_time')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Error over time
        ax1.plot(df['validation_time'], df['error'], 'b-', alpha=0.6)
        ax1.set_title('Prediction Error Over Time')
        ax1.set_ylabel('Error ($)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error percentage by horizon
        for horizon in df['horizon'].unique():
            horizon_data = df[df['horizon'] == horizon]
            ax2.plot(horizon_data['validation_time'], horizon_data['error_pct'], 
                    label=horizon, marker='o', alpha=0.7)
        ax2.set_title('Error % by Time Horizon')
        ax2.set_ylabel('Error (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Directional accuracy (rolling)
        window = min(10, len(df))
        rolling_accuracy = df['direction_correct'].rolling(window=window).mean() * 100
        ax3.plot(df['validation_time'], rolling_accuracy, 'g-', linewidth=2)
        ax3.axhline(y=50, color='r', linestyle='--', label='Random (50%)')
        ax3.set_title(f'Directional Accuracy (Rolling {window}-period)')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error distribution
        ax4.hist(df['error_pct'], bins=20, edgecolor='black', alpha=0.7)
        ax4.axvline(x=df['error_pct'].mean(), color='r', linestyle='--', 
                   label=f'Mean: {df["error_pct"].mean():.2f}%')
        ax4.set_title('Error Distribution')
        ax4.set_xlabel('Error (%)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True

def main():
    """Test the accuracy tracker"""
    tracker = AccuracyTracker()
    
    # Example: Record a prediction
    predictions = {
        '15m': {'price': 3170.0, 'timestamp': (datetime.now() + timedelta(minutes=15)).isoformat()},
        '30m': {'price': 3175.0, 'timestamp': (datetime.now() + timedelta(minutes=30)).isoformat()},
        '60m': {'price': 3185.0, 'timestamp': (datetime.now() + timedelta(minutes=60)).isoformat()},
        '120m': {'price': 3210.0, 'timestamp': (datetime.now() + timedelta(minutes=120)).isoformat()}
    }
    
    tracker.record_prediction(datetime.now(), predictions, 3167.61)
    
    # Example: Validate predictions (would be called later)
    # validated = tracker.validate_predictions(datetime.now(), 3172.50)
    # print(f"Validated {validated} predictions")
    
    # Generate report
    report = tracker.generate_accuracy_report()
    print(report)

if __name__ == '__main__':
    main()
