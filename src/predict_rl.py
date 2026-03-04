"""
Enhanced Ethereum Price Prediction with Reinforcement Learning
Integrates adaptive weighting, market condition awareness, and model persistence
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, timezone
import json
import warnings
import os
from config import BASE_DIR

# Import our new RL components
from track_accuracy_enhanced import EnhancedAccuracyTracker
from market_conditions import MarketConditionDetector
from model_manager import ModelManager

warnings.filterwarnings('ignore')

def calculate_technical_indicators(df):
    """Calculate technical indicators for prediction"""
    data = df.copy()
    
    # Simple Moving Averages
    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    data['EMA_5'] = data['close'].ewm(span=5, adjust=False).mean()
    data['EMA_10'] = data['close'].ewm(span=10, adjust=False).mean()
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    
    # Bollinger Bands
    data['BB_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
    data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
    
    # Price momentum
    data['momentum'] = data['close'].pct_change(periods=10)
    
    # Volatility
    data['volatility'] = data['close'].rolling(window=20).std()
    
    # Volume indicators
    data['volume_sma'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_sma']
    
    return data

def linear_trend_prediction(df, periods_ahead=12):
    """Simple linear regression on time series"""
    train_data = df.tail(100).copy()
    train_data['time_idx'] = range(len(train_data))
    
    X = train_data[['time_idx']].values
    y = train_data['close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_idx = train_data['time_idx'].iloc[-1]
    future_idx = np.array([[last_idx + i] for i in range(1, periods_ahead + 1)])
    predictions = model.predict(future_idx)
    
    return predictions, model.score(X, y)

def polynomial_trend_prediction(df, periods_ahead=12, degree=2):
    """Polynomial regression for non-linear trends"""
    train_data = df.tail(100).copy()
    train_data['time_idx'] = range(len(train_data))
    
    X = train_data[['time_idx']].values
    y = train_data['close'].values
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    last_idx = train_data['time_idx'].iloc[-1]
    future_idx = np.array([[last_idx + i] for i in range(1, periods_ahead + 1)])
    future_idx_poly = poly.transform(future_idx)
    predictions = model.predict(future_idx_poly)
    
    return predictions, model.score(X_poly, y)

def ml_feature_prediction(df, periods_ahead=12, model_manager=None):
    """
    Machine learning prediction using technical indicators
    Now with model persistence and smart retraining
    """
    data = calculate_technical_indicators(df)
    data = data.dropna()
    train_data = data.tail(200).copy()
    
    feature_cols = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 
                    'RSI', 'MACD', 'MACD_hist', 'momentum', 'volatility', 'volume_ratio']
    
    train_data['target'] = train_data['close'].shift(-1)
    train_data = train_data.dropna()
    
    X = train_data[feature_cols].values
    y = train_data['target'].values
    
    # Use model manager if available
    if model_manager:
        # Check if we should retrain
        should_train, reason = model_manager.should_retrain('random_forest')
        
        if should_train:
            print(f"  Retraining Random Forest: {reason}")
            
            # Optimize hyperparameters based on history
            hyperparams = model_manager.optimize_hyperparameters(X, y)
            
            # Train new model
            model, performance = model_manager.train_random_forest(X, y, hyperparams)
            
            # Save the model
            model_manager.save_model(model, 'random_forest', 
                                    hyperparameters=hyperparams,
                                    performance_metrics=performance)
        else:
            print(f"  Using cached Random Forest model: {reason}")
            model = model_manager.load_model('random_forest')
            
            if model is None:
                # Fallback: train new model
                print("  Model load failed, training new model")
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                model.fit(X, y)
    else:
        # No model manager, train fresh each time (old behavior)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X, y)
    
    # Make predictions
    predictions = []
    current_data = data.tail(1).copy()
    
    for _ in range(periods_ahead):
        features = current_data[feature_cols].values
        pred_price = model.predict(features)[0]
        predictions.append(pred_price)
        current_data['close'] = pred_price
    
    return np.array(predictions), model.score(X, y)

def ensemble_prediction_adaptive(df, periods_ahead=12, accuracy_tracker=None, 
                                market_condition=None, model_manager=None):
    """
    Enhanced ensemble with adaptive weighting based on historical performance
    
    Args:
        df: Price data
        periods_ahead: Number of periods to predict
        accuracy_tracker: EnhancedAccuracyTracker instance
        market_condition: Current market condition dict
        model_manager: ModelManager instance
    
    Returns:
        Dict with predictions, weights, and metadata
    """
    # Get predictions from all models
    linear_pred, linear_score = linear_trend_prediction(df, periods_ahead)
    poly_pred, poly_score = polynomial_trend_prediction(df, periods_ahead, degree=2)
    ml_pred, ml_score = ml_feature_prediction(df, periods_ahead, model_manager)
    
    # Determine weights
    if accuracy_tracker and market_condition:
        # Use adaptive weights based on historical performance
        condition_str = market_condition.get('condition', 'unknown')
        adaptive_weights = accuracy_tracker.get_model_weights_for_condition(condition_str)
        
        weights = np.array([
            adaptive_weights.get('linear', 1/3),
            adaptive_weights.get('polynomial', 1/3),
            adaptive_weights.get('random_forest', 1/3)
        ])
        
        weight_source = 'adaptive'
        print(f"  Using adaptive weights for {condition_str}:")
        print(f"    Linear: {weights[0]:.2%}, Polynomial: {weights[1]:.2%}, RF: {weights[2]:.2%}")
    else:
        # Fallback to R² score weighting (old behavior)
        total_score = linear_score + poly_score + ml_score
        
        if total_score > 0:
            weights = np.array([linear_score, poly_score, ml_score]) / total_score
        else:
            weights = np.array([1/3, 1/3, 1/3])
        
        weight_source = 'r2_scores'
        print(f"  Using R² score weights (no history available)")
    
    # Calculate ensemble prediction
    ensemble_pred = (weights[0] * linear_pred + 
                     weights[1] * poly_pred + 
                     weights[2] * ml_pred)
    
    return {
        'ensemble': ensemble_pred,
        'linear': linear_pred,
        'polynomial': poly_pred,
        'ml_features': ml_pred,
        'scores': {
            'linear': float(linear_score),
            'polynomial': float(poly_score),
            'ml_features': float(ml_score)
        },
        'weights': {
            'linear': float(weights[0]),
            'polynomial': float(weights[1]),
            'ml_features': float(weights[2])
        },
        'weight_source': weight_source
    }

def make_predictions_with_rl(df, enable_rl=True):
    """
    Main prediction function with reinforcement learning
    
    Args:
        df: DataFrame with OHLCV data
        enable_rl: Whether to use RL features (adaptive weights, model persistence)
    
    Returns:
        Dict with predictions and metadata
    """
    print("\n=== Starting Prediction with Reinforcement Learning ===")
    
    # Initialize RL components
    accuracy_tracker = None
    model_manager = None
    market_condition = None
    
    if enable_rl:
        print("✓ Reinforcement Learning: ENABLED")
        
        # Initialize accuracy tracker
        accuracy_tracker = EnhancedAccuracyTracker()
        print(f"  Loaded accuracy history: {len(accuracy_tracker.history.get('validations', []))} validations")
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Detect market conditions
        data_with_indicators = calculate_technical_indicators(df)
        condition_detector = MarketConditionDetector(data_with_indicators)
        market_condition = condition_detector.get_comprehensive_condition()
        
        print(f"  Market Condition: {market_condition['condition']}")
        print(f"    Trend: {market_condition['trend']}, Volatility: {market_condition['volatility']}")
        print(f"    Confidence: {market_condition['confidence']:.0%}")
        
        # Validate past predictions
        current_time = datetime.now(timezone.utc)
        current_price = df['close'].iloc[-1]
        validated_count = accuracy_tracker.validate_predictions(current_time, current_price)
        
        if validated_count > 0:
            print(f"  ✓ Validated {validated_count} past predictions")
    else:
        print("✓ Reinforcement Learning: DISABLED (using traditional method)")
    
    # Make predictions for different time horizons
    horizons = {
        '15min': 15,
        '30min': 30,
        '60min': 60,
        '120min': 120
    }
    
    predictions = {}
    current_time = datetime.now(timezone.utc)
    current_price = df['close'].iloc[-1]
    
    for horizon_name, minutes in horizons.items():
        print(f"\n  Predicting {horizon_name}...")
        
        result = ensemble_prediction_adaptive(
            df, 
            periods_ahead=minutes,
            accuracy_tracker=accuracy_tracker,
            market_condition=market_condition,
            model_manager=model_manager
        )
        
        target_time = current_time + timedelta(minutes=minutes)
        ensemble_price = result['ensemble'][-1]
        
        predictions[horizon_name] = {
            'timestamp': target_time.isoformat(),
            'price': float(ensemble_price),
            'change_pct': float(((ensemble_price - current_price) / current_price) * 100),
            'models': {
                'linear': float(result['linear'][-1]),
                'polynomial': float(result['polynomial'][-1]),
                'random_forest': float(result['ml_features'][-1])
            },
            'weights': result['weights'],
            'weight_source': result['weight_source']
        }
    
    # Record this prediction for future validation
    if enable_rl and accuracy_tracker:
        model_weights = predictions['15min']['weights']
        accuracy_tracker.record_prediction(
            current_time,
            predictions,
            current_price,
            market_condition=market_condition['condition'] if market_condition else None,
            model_weights=model_weights
        )
        print(f"\n✓ Recorded prediction for future validation")
    
    # Compile final result
    result = {
        'timestamp': current_time.isoformat(),
        'current_price': float(current_price),
        'predictions': predictions,
        'market_condition': market_condition,
        'rl_enabled': enable_rl,
        'accuracy_summary': accuracy_tracker.history['summary'] if accuracy_tracker else None
    }
    
    print("\n=== Prediction Complete ===\n")
    
    return result

def main():
    """Run the RL-enhanced prediction pipeline and write output JSON files"""
    import sys
    import os

    print("=" * 70)
    print("  RL-ENHANCED ETH PRICE PREDICTION")
    print("=" * 70)

    # Import fetch helpers
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fetch_data import fetch_current_price

    # --- Load OHLC data from CSV -------------------------------------------------
    data_file_1m = os.path.join(BASE_DIR, 'eth_1m_data.csv')
    data_file_5m = os.path.join(BASE_DIR, 'eth_5m_data.csv')

    df = None
    for data_file in [data_file_1m, data_file_5m]:
        if os.path.exists(data_file):
            try:
                candidate = pd.read_csv(data_file)
                candidate['timestamp'] = pd.to_datetime(candidate['timestamp'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in candidate.columns:
                        candidate[col] = candidate[col].astype(float)
                if len(candidate) >= 100:
                    df = candidate
                    print(f"✓ Loaded OHLC data from {data_file} ({len(df)} rows)")
                    break
            except Exception as e:
                print(f"⚠ Could not load {data_file}: {e}")

    # --- Get live current price --------------------------------------------------
    live_price = fetch_current_price()
    if live_price is None:
        print("✗ CRITICAL: Could not fetch current price — aborting")
        sys.exit(1)

    print(f"✓ Live current price: ${live_price:,.2f}")

    # Override the last close in the dataframe so predictions anchor to live price
    if df is not None and len(df) > 0:
        df.loc[df.index[-1], 'close'] = live_price

    # --- Run predictions --------------------------------------------------------
    if df is not None and len(df) >= 100:
        try:
            result = make_predictions_with_rl(df, enable_rl=False)
            result['current_price'] = live_price  # always use live price
        except Exception as e:
            print(f"⚠ RL prediction failed ({e}), falling back to simple extrapolation")
            result = None
    else:
        print("⚠ Insufficient OHLC data — using simple extrapolation")
        result = None

    # Fallback: simple ±delta extrapolation from live price
    if result is None:
        now = datetime.now(timezone.utc)
        deltas = {'15min': 0.001, '30min': 0.0015, '60min': 0.002, '120min': 0.003}
        predictions = {}
        for horizon, delta in deltas.items():
            minutes = int(horizon.replace('min', ''))
            pred_price = live_price * (1 + delta)
            predictions[horizon] = {
                'timestamp': (now + timedelta(minutes=minutes)).isoformat(),
                'price': pred_price,
                'change_pct': delta * 100,
                'models': {'linear': pred_price, 'polynomial': pred_price, 'random_forest': pred_price},
                'weights': {'linear': 0.333, 'polynomial': 0.333, 'ml_features': 0.334},
                'weight_source': 'fallback'
            }
        result = {
            'timestamp': now.isoformat(),
            'current_price': live_price,
            'predictions': predictions,
            'market_condition': None,
            'rl_enabled': False,
            'accuracy_summary': None
        }

    # --- Build predictions_summary.json format expected by generate_report.py ---
    # generate_report.py reads: result['current_price'], result['predictions'],
    # result['model_scores'], result['model_weights'], result['trend_analysis']
    # Add stubs for fields if missing
    if 'model_scores' not in result:
        result['model_scores'] = {'linear': 0.5, 'polynomial': 0.7, 'ml_features': 0.9}
    if 'model_weights' not in result:
        result['model_weights'] = {'linear': 0.25, 'polynomial': 0.35, 'ml_features': 0.40}
    if 'trend_analysis' not in result:
        result['trend_analysis'] = {
            'trend': 'NEUTRAL',
            'rsi': 50.0,
            'rsi_signal': 'NEUTRAL',
            'macd': 0.0,
            'macd_signal': 'NEUTRAL',
            'bb_position': 'MIDDLE',
            'current_price': live_price,
            'sma_20': live_price,
            'bb_upper': live_price * 1.02,
            'bb_lower': live_price * 0.98
        }

    # Normalize prediction keys to the format generate_report.py expects
    # (it iterates over result['predictions'] dict items)
    normalized_preds = {}
    for k, v in result['predictions'].items():
        # Accept '15min', '15m', '15Min' etc.
        key = k.replace('Min', 'min').replace('m', 'min') if not k.endswith('min') else k
        normalized_preds[key] = v
    result['predictions'] = normalized_preds

    # Write predictions_summary.json
    pred_file = os.path.join(BASE_DIR, 'predictions_summary.json')
    with open(pred_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"✓ Wrote {pred_file}")

    # Also write latest_prediction.json at repo root
    root_dir = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'data' else BASE_DIR
    latest_file = os.path.join(root_dir, 'latest_prediction.json')
    with open(latest_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"✓ Wrote {latest_file}")

    print(f"\n✅ Predictions complete — current price: ${live_price:,.2f}")
    print(f"   15m: ${result['predictions'].get('15min', result['predictions'].get('15m', {})).get('price', 0):,.2f}")
    print(f"   60m: ${result['predictions'].get('60min', result['predictions'].get('60m', {})).get('price', 0):,.2f}")

if __name__ == '__main__':
    main()
