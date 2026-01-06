"""
Enhanced Prediction Accuracy Tracking System
Tracks per-model accuracy, market conditions, and enables reinforcement learning
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
from config import BASE_DIR
matplotlib.use('Agg')

class EnhancedAccuracyTracker:
    """Track and analyze prediction accuracy with per-model granularity"""
    
    def __init__(self, reports_dir=os.path.join(BASE_DIR, 'reports')):
        self.reports_dir = reports_dir
        self.history_file = os.path.join(reports_dir, 'accuracy_history.json')
        self.model_performance_file = os.path.join(reports_dir, 'model_performance.json')
        self.load_history()
        self.load_model_performance()
    
    def load_history(self):
        """Load historical accuracy data"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                'predictions': [],
                'validations': [],
                'summary': {
                    'total_predictions': 0,
                    'total_validations': 0,
                    'ensemble_avg_error_pct': 0,
                    'linear_avg_error_pct': 0,
                    'polynomial_avg_error_pct': 0,
                    'random_forest_avg_error_pct': 0,
                    'directional_accuracy': 0,
                    'last_updated': None
                }
            }
    
    def load_model_performance(self):
        """Load per-model performance by market condition"""
        if os.path.exists(self.model_performance_file):
            with open(self.model_performance_file, 'r') as f:
                self.model_performance = json.load(f)
        else:
            self.model_performance = {}
    
    def save_history(self):
        """Save accuracy history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_model_performance(self):
        """Save model performance data"""
        with open(self.model_performance_file, 'w') as f:
            json.dump(self.model_performance, f, indent=2)
    
    def record_prediction(self, timestamp, predictions, current_price, market_condition=None, model_weights=None):
        """
        Record a new prediction with per-model details
        
        Args:
            timestamp: When the prediction was made
            predictions: Dict with time horizons and predicted prices (including per-model predictions)
            current_price: Current price when prediction was made
            market_condition: Dict with market state (trend, volatility, etc.)
            model_weights: Dict with weights used for each model
        """
        prediction_id = timestamp.isoformat()
        
        record = {
            'id': prediction_id,
            'prediction_time': timestamp.isoformat(),
            'current_price': current_price,
            'market_condition': market_condition or 'unknown',
            'predictions': {},
            'model_weights': model_weights or {},
            'validated': False
        }
        
        # Store predictions with per-model details
        for horizon, pred_data in predictions.items():
            record['predictions'][horizon] = {
                'timestamp': pred_data['timestamp'],
                'ensemble_price': pred_data['price'],
                'models': pred_data.get('models', {}),
                'weights': pred_data.get('weights', {})
            }
        
        self.history['predictions'].append(record)
        self.save_history()
        
        return prediction_id
    
    def validate_predictions(self, current_time, current_price):
        """
        Validate past predictions and calculate per-model errors
        
        Args:
            current_time: Current datetime
            current_price: Current actual price
        
        Returns:
            Number of predictions validated
        """
        validated_count = 0
        
        for pred_record in self.history['predictions']:
            if pred_record.get('validated', False):
                continue
            
            pred_time = datetime.fromisoformat(pred_record['prediction_time'])
            all_validated = True
            
            # Check each prediction horizon
            for horizon, pred_data in pred_record['predictions'].items():
                target_time = datetime.fromisoformat(pred_data['timestamp'])
                
                # If we've reached or passed the target time, validate
                if current_time >= target_time:
                    # Calculate ensemble error
                    ensemble_price = pred_data['ensemble_price']
                    ensemble_error = abs(ensemble_price - current_price)
                    ensemble_error_pct = (ensemble_error / current_price) * 100
                    
                    # Check directional accuracy
                    predicted_direction = 'up' if ensemble_price > pred_record['current_price'] else 'down'
                    actual_direction = 'up' if current_price > pred_record['current_price'] else 'down'
                    direction_correct = predicted_direction == actual_direction
                    
                    # Calculate per-model errors
                    model_errors = {}
                    for model_name, model_price in pred_data.get('models', {}).items():
                        error = abs(model_price - current_price)
                        error_pct = (error / current_price) * 100
                        model_direction = 'up' if model_price > pred_record['current_price'] else 'down'
                        model_direction_correct = model_direction == actual_direction
                        
                        model_errors[model_name] = {
                            'absolute': error,
                            'percentage': error_pct,
                            'direction_correct': model_direction_correct
                        }
                    
                    # Record validation
                    validation_record = {
                        'prediction_id': pred_record['id'],
                        'prediction_time': pred_record['prediction_time'],
                        'validation_time': current_time.isoformat(),
                        'target_time': target_time.isoformat(),
                        'horizon': horizon,
                        'actual_price': current_price,
                        'market_condition': pred_record.get('market_condition', 'unknown'),
                        'errors': {
                            'ensemble': {
                                'absolute': ensemble_error,
                                'percentage': ensemble_error_pct,
                                'direction_correct': direction_correct
                            },
                            **model_errors
                        }
                    }
                    
                    self.history['validations'].append(validation_record)
                    
                    # Update model performance by condition
                    self._update_model_performance(validation_record)
                    
                    validated_count += 1
                else:
                    all_validated = False
            
            # Mark prediction as validated if all horizons are validated
            if all_validated:
                pred_record['validated'] = True
        
        if validated_count > 0:
            self.update_summary()
            self.save_history()
            self.save_model_performance()
        
        return validated_count
    
    def _update_model_performance(self, validation_record):
        """Update per-model performance statistics by market condition"""
        condition = validation_record['market_condition']
        
        if condition not in self.model_performance:
            self.model_performance[condition] = {}
        
        # Update stats for each model
        for model_name, error_data in validation_record['errors'].items():
            if model_name not in self.model_performance[condition]:
                self.model_performance[condition][model_name] = {
                    'total_predictions': 0,
                    'total_error_pct': 0,
                    'total_direction_correct': 0,
                    'avg_error_pct': 0,
                    'directional_accuracy': 0
                }
            
            stats = self.model_performance[condition][model_name]
            stats['total_predictions'] += 1
            stats['total_error_pct'] += error_data['percentage']
            stats['total_direction_correct'] += (1 if error_data['direction_correct'] else 0)
            
            # Calculate averages
            stats['avg_error_pct'] = stats['total_error_pct'] / stats['total_predictions']
            stats['directional_accuracy'] = (stats['total_direction_correct'] / stats['total_predictions']) * 100
    
    def update_summary(self):
        """Update summary statistics"""
        if not self.history['validations']:
            return
        
        df = pd.DataFrame(self.history['validations'])
        
        # Calculate ensemble stats
        ensemble_errors = [v['errors']['ensemble']['percentage'] for v in self.history['validations']]
        ensemble_directions = [v['errors']['ensemble']['direction_correct'] for v in self.history['validations']]
        
        # Calculate per-model stats
        model_stats = {}
        for model_name in ['linear', 'polynomial', 'random_forest']:
            model_errors = []
            for v in self.history['validations']:
                if model_name in v['errors']:
                    model_errors.append(v['errors'][model_name]['percentage'])
            
            if model_errors:
                model_stats[f'{model_name}_avg_error_pct'] = float(np.mean(model_errors))
            else:
                model_stats[f'{model_name}_avg_error_pct'] = 0
        
        self.history['summary'] = {
            'total_predictions': len(self.history['predictions']),
            'total_validations': len(self.history['validations']),
            'ensemble_avg_error_pct': float(np.mean(ensemble_errors)),
            **model_stats,
            'directional_accuracy': float(np.mean(ensemble_directions) * 100),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_model_weights_for_condition(self, market_condition, recent_window=20):
        """
        Calculate optimal model weights based on historical performance in similar conditions
        
        Args:
            market_condition: Current market condition
            recent_window: Number of recent predictions to consider
        
        Returns:
            Dict with optimal weights for each model
        """
        # Get recent validations for this condition
        recent_validations = [
            v for v in self.history['validations'][-recent_window:]
            if v['market_condition'] == market_condition
        ]
        
        if len(recent_validations) < 5:
            # Not enough data, use equal weights
            return {
                'linear': 1/3,
                'polynomial': 1/3,
                'random_forest': 1/3
            }
        
        # Calculate accuracy scores (inverse of error percentage)
        model_scores = {}
        for model_name in ['linear', 'polynomial', 'random_forest']:
            errors = []
            for v in recent_validations:
                if model_name in v['errors']:
                    # Use inverse of error as score (lower error = higher score)
                    error_pct = v['errors'][model_name]['percentage']
                    score = 1.0 / (1.0 + error_pct)  # Normalize to 0-1
                    errors.append(score)
            
            if errors:
                # Apply exponential decay (more recent = more weight)
                decay_factor = 0.95
                weighted_score = 0
                total_weight = 0
                
                for i, score in enumerate(reversed(errors)):
                    weight = decay_factor ** i
                    weighted_score += score * weight
                    total_weight += weight
                
                model_scores[model_name] = weighted_score / total_weight if total_weight > 0 else 0
            else:
                model_scores[model_name] = 1/3
        
        # Normalize to sum to 1.0
        total_score = sum(model_scores.values())
        if total_score > 0:
            weights = {k: v/total_score for k, v in model_scores.items()}
        else:
            weights = {'linear': 1/3, 'polynomial': 1/3, 'random_forest': 1/3}
        
        return weights
    
    def generate_performance_report(self):
        """Generate detailed performance report with per-model breakdown"""
        if not self.history['validations']:
            return "No validated predictions yet. Run the system for at least one prediction cycle."
        
        report = f"""# Model Performance Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Total Predictions:** {self.history['summary']['total_predictions']}  
**Total Validations:** {self.history['summary']['total_validations']}

## Overall Performance

| Model | Avg Error % | Status |
|:------|:------------|:-------|
| **Ensemble** | {self.history['summary']['ensemble_avg_error_pct']:.2f}% | {'✅ Excellent' if self.history['summary']['ensemble_avg_error_pct'] < 1.0 else '✓ Good' if self.history['summary']['ensemble_avg_error_pct'] < 2.0 else '⚠️ Needs Improvement'} |
| **Linear** | {self.history['summary']['linear_avg_error_pct']:.2f}% | {'✅ Best' if self.history['summary']['linear_avg_error_pct'] < min(self.history['summary']['polynomial_avg_error_pct'], self.history['summary']['random_forest_avg_error_pct']) else '✓'} |
| **Polynomial** | {self.history['summary']['polynomial_avg_error_pct']:.2f}% | {'✅ Best' if self.history['summary']['polynomial_avg_error_pct'] < min(self.history['summary']['linear_avg_error_pct'], self.history['summary']['random_forest_avg_error_pct']) else '✓'} |
| **Random Forest** | {self.history['summary']['random_forest_avg_error_pct']:.2f}% | {'✅ Best' if self.history['summary']['random_forest_avg_error_pct'] < min(self.history['summary']['linear_avg_error_pct'], self.history['summary']['polynomial_avg_error_pct']) else '✓'} |

**Directional Accuracy:** {self.history['summary']['directional_accuracy']:.1f}%

## Performance by Market Condition

"""
        
        if self.model_performance:
            for condition, models in self.model_performance.items():
                report += f"\n### {condition.replace('_', ' ').title()}\n\n"
                report += "| Model | Avg Error % | Direction Accuracy | Predictions |\n"
                report += "|:------|:------------|:-------------------|:------------|\n"
                
                for model_name, stats in models.items():
                    report += f"| {model_name.title()} | {stats['avg_error_pct']:.2f}% | {stats['directional_accuracy']:.1f}% | {stats['total_predictions']} |\n"
        
        report += f"""

## Recent Performance (Last 10 Validations)

"""
        
        recent_validations = self.history['validations'][-10:]
        report += "| Time | Horizon | Ensemble Error | Best Model | Worst Model |\n"
        report += "|:-----|:--------|:---------------|:-----------|:------------|\n"
        
        for v in recent_validations:
            ensemble_error = v['errors']['ensemble']['percentage']
            
            # Find best and worst models
            model_errors = {k: v['percentage'] for k, v in v['errors'].items() if k != 'ensemble'}
            if model_errors:
                best_model = min(model_errors, key=model_errors.get)
                worst_model = max(model_errors, key=model_errors.get)
                best_error = model_errors[best_model]
                worst_error = model_errors[worst_model]
            else:
                best_model = worst_model = 'N/A'
                best_error = worst_error = 0
            
            pred_time = datetime.fromisoformat(v['prediction_time']).strftime('%m-%d %H:%M')
            report += f"| {pred_time} | {v['horizon']} | {ensemble_error:.2f}% | {best_model} ({best_error:.2f}%) | {worst_model} ({worst_error:.2f}%) |\n"
        
        report += """

## Insights & Recommendations

"""
        
        # Analyze which model is best
        model_errors = {
            'Linear': self.history['summary']['linear_avg_error_pct'],
            'Polynomial': self.history['summary']['polynomial_avg_error_pct'],
            'Random Forest': self.history['summary']['random_forest_avg_error_pct']
        }
        best_model = min(model_errors, key=model_errors.get)
        
        report += f"- **Best Performing Model:** {best_model} with {model_errors[best_model]:.2f}% average error\n"
        report += f"- **Ensemble Benefit:** Ensemble reduces error by {abs(self.history['summary']['ensemble_avg_error_pct'] - model_errors[best_model]):.2f}% compared to best single model\n"
        
        if self.history['summary']['directional_accuracy'] > 60:
            report += f"- **Strong Directional Accuracy:** {self.history['summary']['directional_accuracy']:.1f}% (above 60% threshold)\n"
        else:
            report += f"- **Weak Directional Accuracy:** {self.history['summary']['directional_accuracy']:.1f}% (below 60% threshold) - Consider adjusting indicators\n"
        
        return report

def main():
    """Test the enhanced accuracy tracker"""
    tracker = EnhancedAccuracyTracker()
    
    # Example: Record a prediction with per-model details
    timestamp = datetime.now()
    predictions = {
        '15min': {
            'price': 3252.50,
            'timestamp': (timestamp + timedelta(minutes=15)).isoformat(),
            'models': {
                'linear': 3251.00,
                'polynomial': 3253.00,
                'random_forest': 3253.50
            },
            'weights': {
                'linear': 0.23,
                'polynomial': 0.33,
                'random_forest': 0.44
            }
        }
    }
    
    market_condition = 'bull_medium_volatility'
    model_weights = {'linear': 0.23, 'polynomial': 0.33, 'random_forest': 0.44}
    
    pred_id = tracker.record_prediction(timestamp, predictions, 3248.16, market_condition, model_weights)
    print(f"Recorded prediction: {pred_id}")
    
    # Generate report
    report = tracker.generate_performance_report()
    print(report)

if __name__ == '__main__':
    main()
