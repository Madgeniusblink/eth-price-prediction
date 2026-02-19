"""
Model Manager
Handles model persistence, versioning, and smart retraining decisions
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from config import BASE_DIR

class ModelManager:
    """Manage ML model lifecycle - save, load, and retrain intelligently"""
    
    def __init__(self, models_dir=os.path.join(BASE_DIR, 'models')):
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, 'model_metadata.json')
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        self.load_metadata()
    
    def load_metadata(self):
        """Load model metadata"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'models': {},
                'last_updated': None
            }
    
    def save_metadata(self):
        """Save model metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_model(self, model, model_name, version=None, hyperparameters=None, performance_metrics=None):
        """
        Save a trained model to disk
        
        Args:
            model: Trained sklearn model
            model_name: Name of the model (e.g., 'random_forest')
            version: Version string (auto-generated if None)
            hyperparameters: Dict of model hyperparameters
            performance_metrics: Dict of performance metrics
        
        Returns:
            Version string of saved model
        """
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_filename = f"{model_name}_v{version}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # Save model
        joblib.dump(model, model_path)
        
        # Update metadata
        if model_name not in self.metadata['models']:
            self.metadata['models'][model_name] = {
                'versions': [],
                'current_version': None
            }
        
        version_info = {
            'version': version,
            'filename': model_filename,
            'created_at': datetime.now().isoformat(),
            'hyperparameters': hyperparameters or {},
            'performance_metrics': performance_metrics or {},
            'file_size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }
        
        self.metadata['models'][model_name]['versions'].append(version_info)
        self.metadata['models'][model_name]['current_version'] = version
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        self.save_metadata()
        
        print(f"✓ Saved {model_name} model version {version}")
        print(f"  File: {model_path}")
        print(f"  Size: {version_info['file_size_mb']:.2f} MB")
        
        return version
    
    def load_model(self, model_name, version=None):
        """
        Load a trained model from disk
        
        Args:
            model_name: Name of the model
            version: Specific version to load (loads current if None)
        
        Returns:
            Loaded model or None if not found
        """
        if model_name not in self.metadata['models']:
            return None
        
        if version is None:
            version = self.metadata['models'][model_name]['current_version']
        
        if version is None:
            return None
        
        # Find version info
        version_info = None
        for v in self.metadata['models'][model_name]['versions']:
            if v['version'] == version:
                version_info = v
                break
        
        if version_info is None:
            return None
        
        model_path = os.path.join(self.models_dir, version_info['filename'])
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            return None
        
        model = joblib.load(model_path)
        print(f"✓ Loaded {model_name} model version {version}")
        
        return model
    
    def should_retrain(self, model_name, recent_accuracy=None, accuracy_threshold=0.5):
        """
        Determine if a model should be retrained
        
        Args:
            model_name: Name of the model
            recent_accuracy: Recent directional accuracy (0-1)
            accuracy_threshold: Minimum acceptable accuracy
        
        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        if model_name not in self.metadata['models']:
            return True, "Model not found - initial training required"
        
        current_version = self.metadata['models'][model_name]['current_version']
        if current_version is None:
            return True, "No current version - initial training required"
        
        # Find current version info
        version_info = None
        for v in self.metadata['models'][model_name]['versions']:
            if v['version'] == current_version:
                version_info = v
                break
        
        if version_info is None:
            return True, "Version info not found - retraining required"
        
        # Check age
        created_at = datetime.fromisoformat(version_info['created_at'])
        age_days = (datetime.now() - created_at).days
        
        if age_days >= 7:
            return True, f"Model is {age_days} days old (threshold: 7 days)"
        
        # Check performance
        if recent_accuracy is not None and recent_accuracy < accuracy_threshold:
            return True, f"Poor performance: {recent_accuracy:.1%} (threshold: {accuracy_threshold:.1%})"
        
        return False, "Model is recent and performing well"
    
    def train_random_forest(self, X, y, hyperparameters=None):
        """
        Train a Random Forest model with given or default hyperparameters
        
        Args:
            X: Training features
            y: Training targets
            hyperparameters: Dict of hyperparameters (uses defaults if None)
        
        Returns:
            Trained model
        """
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        
        print(f"Training Random Forest with hyperparameters: {hyperparameters}")
        
        model = RandomForestRegressor(**hyperparameters)
        model.fit(X, y)
        
        # Calculate training score
        train_score = model.score(X, y)
        
        performance_metrics = {
            'train_r2_score': float(train_score),
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        print(f"✓ Training complete - R² score: {train_score:.4f}")
        
        return model, performance_metrics
    
    def optimize_hyperparameters(self, X, y, accuracy_tracker=None):
        """
        Optimize Random Forest hyperparameters based on historical performance
        
        Args:
            X: Training features
            y: Training targets
            accuracy_tracker: AccuracyTracker instance for historical data
        
        Returns:
            Dict of optimized hyperparameters
        """
        # Start with defaults
        base_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # If we have accuracy history, adjust parameters
        if accuracy_tracker and len(accuracy_tracker.history.get('validations', [])) > 20:
            # Analyze recent performance
            recent_validations = accuracy_tracker.history['validations'][-20:]
            rf_errors = [v['errors'].get('random_forest', {}).get('percentage', 0) 
                        for v in recent_validations if 'random_forest' in v['errors']]
            
            if rf_errors:
                avg_error = np.mean(rf_errors)
                
                # If error is high, try deeper trees
                if avg_error > 2.0:
                    base_params['max_depth'] = 15
                    base_params['n_estimators'] = 150
                    print("  Increasing model complexity due to high error")
                
                # If error is very low, we might be overfitting - simplify
                elif avg_error < 0.5:
                    base_params['max_depth'] = 8
                    base_params['min_samples_split'] = 5
                    print("  Reducing model complexity to prevent overfitting")
        
        return base_params
    
    def get_model_info(self, model_name):
        """
        Get information about a model
        
        Args:
            model_name: Name of the model
        
        Returns:
            Dict with model information or None
        """
        if model_name not in self.metadata['models']:
            return None
        
        model_info = self.metadata['models'][model_name]
        current_version = model_info['current_version']
        
        if current_version is None:
            return {
                'name': model_name,
                'status': 'Not trained',
                'versions': []
            }
        
        # Find current version details
        current_version_info = None
        for v in model_info['versions']:
            if v['version'] == current_version:
                current_version_info = v
                break
        
        return {
            'name': model_name,
            'status': 'Trained',
            'current_version': current_version,
            'created_at': current_version_info['created_at'] if current_version_info else None,
            'age_days': (datetime.now() - datetime.fromisoformat(current_version_info['created_at'])).days if current_version_info else None,
            'hyperparameters': current_version_info['hyperparameters'] if current_version_info else {},
            'performance': current_version_info['performance_metrics'] if current_version_info else {},
            'total_versions': len(model_info['versions'])
        }
    
    def cleanup_old_versions(self, model_name, keep_latest=3):
        """
        Remove old model versions to save disk space
        
        Args:
            model_name: Name of the model
            keep_latest: Number of latest versions to keep
        """
        if model_name not in self.metadata['models']:
            return
        
        versions = self.metadata['models'][model_name]['versions']
        
        if len(versions) <= keep_latest:
            return
        
        # Sort by creation date
        versions_sorted = sorted(versions, key=lambda x: x['created_at'], reverse=True)
        
        # Keep latest N, delete the rest
        to_delete = versions_sorted[keep_latest:]
        
        deleted_count = 0
        for version_info in to_delete:
            model_path = os.path.join(self.models_dir, version_info['filename'])
            if os.path.exists(model_path):
                os.remove(model_path)
                deleted_count += 1
            
            # Remove from metadata
            versions.remove(version_info)
        
        self.save_metadata()
        
        print(f"✓ Cleaned up {deleted_count} old versions of {model_name}")

def main():
    """Test the model manager"""
    manager = ModelManager()
    
    # Example: Check if model needs retraining
    should_train, reason = manager.should_retrain('random_forest')
    print(f"Should retrain: {should_train}")
    print(f"Reason: {reason}")
    
    # Get model info
    info = manager.get_model_info('random_forest')
    if info:
        print(f"\nModel Info: {json.dumps(info, indent=2)}")

if __name__ == '__main__':
    main()
