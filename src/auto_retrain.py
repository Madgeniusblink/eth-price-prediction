"""
Automatic Model Retraining System
Monitors model performance and triggers retraining when needed
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from logger import setup_logger, log_error_with_context
from alert_system import alert_system
from model_manager import ModelManager
from track_accuracy_enhanced import EnhancedAccuracyTracker
from config import BASE_DIR

logger = setup_logger(__name__)

class AutoRetrainer:
    """Automatic model retraining system"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.accuracy_tracker = EnhancedAccuracyTracker()
        self.min_accuracy_threshold = 0.50  # 50% directional accuracy minimum
        self.max_model_age_days = 7
        self.min_training_samples = 200
    
    def check_retraining_needed(self):
        """
        Check if any models need retraining
        
        Returns:
            Dict with retraining recommendations
        """
        recommendations = {
            'retrain_needed': False,
            'reasons': [],
            'models_to_retrain': []
        }
        
        # Get recent accuracy
        summary = self.accuracy_tracker.history.get('summary', {})
        if not summary:
            logger.info("No accuracy history yet - skipping retraining check")
            return recommendations
        
        total_validations = summary.get('total_validations', 0)
        if total_validations < 10:
            logger.info(f"Only {total_validations} validations - need at least 10 for retraining")
            return recommendations
        
        # Check directional accuracy
        direction_accuracy = summary.get('directional_accuracy', 100) / 100
        if direction_accuracy < self.min_accuracy_threshold:
            recommendations['retrain_needed'] = True
            recommendations['reasons'].append(
                f"Directional accuracy below threshold: {direction_accuracy:.1%} < {self.min_accuracy_threshold:.1%}"
            )
            recommendations['models_to_retrain'] = ['linear', 'polynomial', 'random_forest']
        
        # Check per-model performance
        for model_name in ['linear', 'polynomial', 'random_forest']:
            error_key = f'{model_name}_avg_error_pct'
            avg_error = summary.get(error_key, 0)
            
            # If error > 5%, consider retraining
            if avg_error > 5.0:
                if model_name not in recommendations['models_to_retrain']:
                    recommendations['models_to_retrain'].append(model_name)
                    recommendations['retrain_needed'] = True
                    recommendations['reasons'].append(
                        f"{model_name} model has high error: {avg_error:.2f}%"
                    )
        
        # Check model age
        for model_name in ['random_forest']:  # Focus on complex models
            should_retrain, reason = self.model_manager.should_retrain(
                model_name, 
                recent_accuracy=direction_accuracy
            )
            
            if should_retrain and model_name not in recommendations['models_to_retrain']:
                recommendations['models_to_retrain'].append(model_name)
                recommendations['retrain_needed'] = True
                recommendations['reasons'].append(reason)
        
        return recommendations
    
    def prepare_training_data(self, lookback_days=30):
        """
        Prepare training data from historical data
        
        Args:
            lookback_days: Number of days of historical data to use
        
        Returns:
            Tuple of (X, y) training data or None if insufficient data
        """
        try:
            # Load 4-hour data
            data_file = os.path.join(BASE_DIR, 'eth_4h_data.csv')
            if not os.path.exists(data_file):
                logger.error("No 4-hour data file found for training")
                return None
            
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter to recent data
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            df = df[df['timestamp'] >= cutoff_date].copy()
            
            if len(df) < self.min_training_samples:
                logger.warning(f"Insufficient training data: {len(df)} < {self.min_training_samples}")
                return None
            
            # Calculate technical indicators
            from predict_rl import calculate_technical_indicators
            df = calculate_technical_indicators(df)
            
            # Drop NaN values
            df = df.dropna()
            
            if len(df) < self.min_training_samples:
                logger.warning(f"Insufficient data after cleaning: {len(df)} < {self.min_training_samples}")
                return None
            
            # Prepare features
            feature_cols = [
                'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
                'EMA_5', 'EMA_10', 'EMA_20',
                'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                'BB_upper', 'BB_middle', 'BB_lower',
                'momentum', 'volatility', 'volume_ratio'
            ]
            
            X = df[feature_cols].values
            y = df['close'].values
            
            logger.info(f"Prepared training data: {len(X)} samples with {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            log_error_with_context(logger, e, {'function': 'prepare_training_data'})
            return None
    
    def retrain_models(self, models_to_retrain):
        """
        Retrain specified models
        
        Args:
            models_to_retrain: List of model names to retrain
        
        Returns:
            Dict with retraining results
        """
        results = {
            'success': False,
            'models_retrained': [],
            'errors': []
        }
        
        logger.info(f"Starting retraining for models: {models_to_retrain}")
        alert_system.send_info_alert(f"Starting automatic model retraining: {', '.join(models_to_retrain)}")
        
        # Prepare training data
        training_data = self.prepare_training_data(lookback_days=30)
        if training_data is None:
            error_msg = "Failed to prepare training data"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
        
        X, y = training_data
        
        # Retrain each model
        for model_name in models_to_retrain:
            try:
                logger.info(f"Retraining {model_name} model...")
                
                if model_name == 'random_forest':
                    # Retrain random forest with optimization
                    model, metrics = self.model_manager.train_random_forest(
                        X, y, 
                        accuracy_tracker=self.accuracy_tracker
                    )
                    
                    # Save new model version
                    self.model_manager.save_model(
                        model, 
                        model_name,
                        performance_metrics=metrics
                    )
                    
                    logger.info(f"Random Forest retrained - R²: {metrics.get('r2_score', 0):.4f}")
                    results['models_retrained'].append(model_name)
                
                else:
                    # For linear and polynomial, we don't persist them
                    # They're retrained on each prediction
                    logger.info(f"{model_name} model will be retrained on next prediction")
                    results['models_retrained'].append(model_name)
                
            except Exception as e:
                error_msg = f"Failed to retrain {model_name}: {str(e)}"
                logger.error(error_msg)
                log_error_with_context(logger, e, {'model': model_name})
                results['errors'].append(error_msg)
        
        if results['models_retrained']:
            results['success'] = True
            success_msg = f"Successfully retrained models: {', '.join(results['models_retrained'])}"
            logger.info(success_msg)
            alert_system.send_success_alert(success_msg)
        
        if results['errors']:
            alert_system.send_error_alert(
                f"Retraining completed with errors: {'; '.join(results['errors'])}"
            )
        
        return results
    
    def run_retraining_check(self):
        """
        Main function to check and execute retraining if needed
        
        Returns:
            Dict with check and retraining results
        """
        logger.info("Running automatic retraining check...")
        
        # Check if retraining is needed
        recommendations = self.check_retraining_needed()
        
        if not recommendations['retrain_needed']:
            logger.info("No retraining needed - models performing well")
            return {
                'retraining_needed': False,
                'message': 'Models performing within acceptable parameters'
            }
        
        # Log reasons for retraining
        logger.warning("Retraining recommended:")
        for reason in recommendations['reasons']:
            logger.warning(f"  - {reason}")
        
        # Send alert about retraining decision
        alert_system.send_warning_alert(
            f"Model retraining triggered:\n" + "\n".join(f"• {r}" for r in recommendations['reasons'])
        )
        
        # Execute retraining
        retrain_results = self.retrain_models(recommendations['models_to_retrain'])
        
        return {
            'retraining_needed': True,
            'reasons': recommendations['reasons'],
            'retrain_results': retrain_results
        }

def main():
    """Main function for standalone execution"""
    logger.info("=" * 70)
    logger.info("AUTOMATIC MODEL RETRAINING CHECK")
    logger.info("=" * 70)
    
    retrainer = AutoRetrainer()
    results = retrainer.run_retraining_check()
    
    if results['retraining_needed']:
        print("\n✓ Retraining check complete - retraining was performed")
        print(f"\nReasons:")
        for reason in results['reasons']:
            print(f"  - {reason}")
        
        if results['retrain_results']['success']:
            print(f"\n✓ Successfully retrained: {', '.join(results['retrain_results']['models_retrained'])}")
        
        if results['retrain_results']['errors']:
            print(f"\n✗ Errors occurred:")
            for error in results['retrain_results']['errors']:
                print(f"  - {error}")
    else:
        print("\n✓ Retraining check complete - no retraining needed")
        print(f"  {results['message']}")
    
    logger.info("Retraining check completed")
    return 0

if __name__ == '__main__':
    sys.exit(main())
