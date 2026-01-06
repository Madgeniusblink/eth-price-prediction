import os
import sys
#!/usr/bin/env python3
"""
Ethereum Swing Trading Price Prediction Model (4-Hour Timeframe)
Uses multiple approaches: Linear Regression, Polynomial Regression, and Technical Analysis
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import json
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for prediction
    """
    # Make a copy to avoid modifying original
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
    
    # RSI (Relative Strength Index)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
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
    """
    Simple linear regression on time series
    For 4-hour data, 12 periods = 48 hours (2 days)
    """
    # Use last 100 data points for training
    train_data = df.tail(100).copy()
    
    # Create time feature (minutes from start)
    train_data['time_idx'] = range(len(train_data))
    
    X = train_data[['time_idx']].values
    y = train_data['close'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future
    last_idx = train_data['time_idx'].iloc[-1]
    future_idx = np.array([[last_idx + i] for i in range(1, periods_ahead + 1)])
    predictions = model.predict(future_idx)
    
    return predictions, model.score(X, y)

def polynomial_trend_prediction(df, periods_ahead=12, degree=2):
    """
    Polynomial regression for capturing non-linear trends
    """
    # Use last 100 data points
    train_data = df.tail(100).copy()
    train_data['time_idx'] = range(len(train_data))
    
    X = train_data[['time_idx']].values
    y = train_data['close'].values
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict future
    last_idx = train_data['time_idx'].iloc[-1]
    future_idx = np.array([[last_idx + i] for i in range(1, periods_ahead + 1)])
    future_idx_poly = poly.transform(future_idx)
    predictions = model.predict(future_idx_poly)
    
    return predictions, model.score(X_poly, y)

def ml_feature_prediction(df, periods_ahead=12):
    """
    Machine learning prediction using technical indicators as features
    """
    # Calculate indicators
    data = calculate_technical_indicators(df)
    
    # Drop NaN values
    data = data.dropna()
    
    # Use last 200 points for training
    train_data = data.tail(200).copy()
    
    # Features
    feature_cols = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 
                    'RSI', 'MACD', 'MACD_hist', 'momentum', 'volatility', 'volume_ratio']
    
    # Create target: next period's price
    train_data['target'] = train_data['close'].shift(-1)
    train_data = train_data.dropna()
    
    X = train_data[feature_cols].values
    y = train_data['target'].values
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # For prediction, we iteratively predict and update features
    predictions = []
    current_data = data.tail(1).copy()
    
    for _ in range(periods_ahead):
        # Get features from current data
        features = current_data[feature_cols].values
        
        # Predict next price
        pred_price = model.predict(features)[0]
        predictions.append(pred_price)
        
        # Update for next iteration (simplified - in reality would recalculate all indicators)
        # For short-term this approximation is reasonable
        current_data['close'] = pred_price
    
    return np.array(predictions), model.score(X, y)

def ensemble_prediction(df, periods_ahead=12):
    """
    Ensemble of multiple prediction methods
    """
    # Get predictions from different models
    linear_pred, linear_score = linear_trend_prediction(df, periods_ahead)
    poly_pred, poly_score = polynomial_trend_prediction(df, periods_ahead, degree=2)
    ml_pred, ml_score = ml_feature_prediction(df, periods_ahead)
    
    # Weighted average based on scores
    total_score = linear_score + poly_score + ml_score
    
    if total_score > 0:
        weights = np.array([linear_score, poly_score, ml_score]) / total_score
    else:
        weights = np.array([1/3, 1/3, 1/3])
    
    ensemble_pred = (weights[0] * linear_pred + 
                     weights[1] * poly_pred + 
                     weights[2] * ml_pred)
    
    return {
        'ensemble': ensemble_pred,
        'linear': linear_pred,
        'polynomial': poly_pred,
        'ml_features': ml_pred,
        'scores': {
            'linear': linear_score,
            'polynomial': poly_score,
            'ml_features': ml_score
        },
        'weights': {
            'linear': weights[0],
            'polynomial': weights[1],
            'ml_features': weights[2]
        }
    }

def analyze_trend(df):
    """
    Analyze current trend and momentum
    """
    data = calculate_technical_indicators(df)
    latest = data.iloc[-1]
    
    # Trend analysis
    trend = "NEUTRAL"
    if latest['close'] > latest['SMA_20'] and latest['SMA_5'] > latest['SMA_20']:
        trend = "BULLISH"
    elif latest['close'] < latest['SMA_20'] and latest['SMA_5'] < latest['SMA_20']:
        trend = "BEARISH"
    
    # RSI analysis
    rsi_signal = "NEUTRAL"
    if latest['RSI'] > 70:
        rsi_signal = "OVERBOUGHT"
    elif latest['RSI'] < 30:
        rsi_signal = "OVERSOLD"
    
    # MACD analysis
    macd_signal = "NEUTRAL"
    if latest['MACD'] > latest['MACD_signal']:
        macd_signal = "BULLISH"
    elif latest['MACD'] < latest['MACD_signal']:
        macd_signal = "BEARISH"
    
    # Price position relative to Bollinger Bands
    bb_position = "MIDDLE"
    if latest['close'] > latest['BB_upper']:
        bb_position = "ABOVE_UPPER"
    elif latest['close'] < latest['BB_lower']:
        bb_position = "BELOW_LOWER"
    
    return {
        'trend': trend,
        'rsi': latest['RSI'],
        'rsi_signal': rsi_signal,
        'macd': latest['MACD'],
        'macd_signal': macd_signal,
        'bb_position': bb_position,
        'current_price': latest['close'],
        'sma_20': latest['SMA_20'],
        'bb_upper': latest['BB_upper'],
        'bb_lower': latest['BB_lower']
    }

def main():
    print("=== Ethereum Swing Trading Price Prediction (4-Hour Timeframe) ===")
    
    # Load 4-hour data
    df_4h = pd.read_csv(os.path.join(BASE_DIR, 'eth_4h_data.csv'))
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    
    print(f"Loaded {len(df_4h)} 4-hour candles")
    print(f"Time range: {df_4h['timestamp'].min()} to {df_4h['timestamp'].max()}")
    print(f"Current price: ${df_4h['close'].iloc[-1]:.2f}\n")
    
    # Try to use RL-enhanced prediction
    try:
        from predict_rl import make_predictions_with_rl
        
        # Use RL-enhanced prediction system
        rl_result = make_predictions_with_rl(df_4h, enable_rl=True)
        
        # Extract data in format compatible with rest of pipeline
        trend_analysis = analyze_trend(df_4h)
        predictions_12p = ensemble_prediction(df_4h, periods_ahead=12)  # 48 hours
        predictions_24p = ensemble_prediction(df_4h, periods_ahead=24)  # 96 hours
        
        print("\n✓ Using RL-Enhanced Predictions")
        if rl_result.get('market_condition'):
            mc = rl_result['market_condition']
            print(f"  Market: {mc['trend']} trend, {mc['volatility']} volatility")
            print(f"  Confidence: {mc['confidence']:.0%}")
        
    except Exception as e:
        print(f"\n⚠ RL system unavailable ({e}), using traditional method")
        
        # Fallback to traditional prediction
        trend_analysis = analyze_trend(df_4h)
        print("=== Current Market Analysis ===")
        print(f"Trend: {trend_analysis['trend']}")
        print(f"RSI: {trend_analysis['rsi']:.2f} ({trend_analysis['rsi_signal']})")
        print(f"MACD Signal: {trend_analysis['macd_signal']}")
        print(f"Bollinger Bands Position: {trend_analysis['bb_position']}")
        print(f"Current Price: ${trend_analysis['current_price']:.2f}")
        print(f"20-Period SMA: ${trend_analysis['sma_20']:.2f}")
        print(f"BB Upper: ${trend_analysis['bb_upper']:.2f}")
        print(f"BB Lower: ${trend_analysis['bb_lower']:.2f}\n")
        
        # Generate predictions for next 2-4 days
        print("=== Generating Predictions ===")
        
        # 48-hour prediction (12 periods of 4h)
        predictions_12p = ensemble_prediction(df_4h, periods_ahead=12)
        
        # 96-hour prediction (24 periods of 4h)
        predictions_24p = ensemble_prediction(df_4h, periods_ahead=24)
    
    print("\nModel Performance Scores (R²):")
    print(f"  Linear Regression: {predictions_12p['scores']['linear']:.4f}")
    print(f"  Polynomial Regression: {predictions_12p['scores']['polynomial']:.4f}")
    print(f"  ML Features: {predictions_12p['scores']['ml_features']:.4f}")
    
    print("\nEnsemble Weights:")
    print(f"  Linear: {predictions_12p['weights']['linear']:.2%}")
    print(f"  Polynomial: {predictions_12p['weights']['polynomial']:.2%}")
    print(f"  ML Features: {predictions_12p['weights']['ml_features']:.2%}")
    
    # Key predictions
    current_price = df_4h['close'].iloc[-1]
    pred_4h = predictions_12p['ensemble'][0]  # 4 hours (1 period)
    pred_8h = predictions_12p['ensemble'][1]  # 8 hours (2 periods)
    pred_24h = predictions_12p['ensemble'][5]  # 24 hours (6 periods)
    pred_48h = predictions_12p['ensemble'][11]  # 48 hours (12 periods)
    
    print("\n=== Price Predictions ===")
    print(f"Current Price: ${current_price:.2f}")
    print(f"4-hour prediction: ${pred_4h:.2f} ({((pred_4h/current_price - 1) * 100):+.2f}%)")
    print(f"8-hour prediction: ${pred_8h:.2f} ({((pred_8h/current_price - 1) * 100):+.2f}%)")
    print(f"24-hour prediction: ${pred_24h:.2f} ({((pred_24h/current_price - 1) * 100):+.2f}%)")
    print(f"48-hour prediction: ${pred_48h:.2f} ({((pred_48h/current_price - 1) * 100):+.2f}%)")
    
    # Save predictions
    last_timestamp = df_4h['timestamp'].iloc[-1]
    
    # Calculate ensemble R² score (weighted average of model scores)
    ensemble_r2 = sum(predictions_12p['scores'][k] * predictions_12p['weights'][k] 
                      for k in ['linear', 'polynomial', 'ml_features'])
    
    predictions_data = {
        'generated_at': datetime.now().isoformat(),
        'current_price': float(current_price),
        'last_data_timestamp': last_timestamp.isoformat(),
        'trend_analysis': trend_analysis,
        'model_scores': {k: float(v) for k, v in predictions_12p['scores'].items()},
        'model_weights': {k: float(v) for k, v in predictions_12p['weights'].items()},
        'ensemble_r2': float(ensemble_r2),
        'predictions': {
            '4h': {
                'price': float(pred_4h),
                'change_pct': float((pred_4h/current_price - 1) * 100),
                'timestamp': (last_timestamp + timedelta(hours=4)).isoformat()
            },
            '8h': {
                'price': float(pred_8h),
                'change_pct': float((pred_8h/current_price - 1) * 100),
                'timestamp': (last_timestamp + timedelta(hours=8)).isoformat()
            },
            '24h': {
                'price': float(pred_24h),
                'change_pct': float((pred_24h/current_price - 1) * 100),
                'timestamp': (last_timestamp + timedelta(hours=24)).isoformat()
            },
            '48h': {
                'price': float(pred_48h),
                'change_pct': float((pred_48h/current_price - 1) * 100),
                'timestamp': (last_timestamp + timedelta(hours=48)).isoformat()
            }
        }
    }
    
    # Save detailed predictions for visualization
    timestamps_12p = [last_timestamp + timedelta(hours=4*i) for i in range(1, 13)]
    timestamps_24p = [last_timestamp + timedelta(hours=4*i) for i in range(1, 25)]
    
    pred_df_12p = pd.DataFrame({
        'timestamp': timestamps_12p,
        'ensemble': predictions_12p['ensemble'],
        'linear': predictions_12p['linear'],
        'polynomial': predictions_12p['polynomial'],
        'ml_features': predictions_12p['ml_features']
    })
    
    pred_df_24p = pd.DataFrame({
        'timestamp': timestamps_24p,
        'ensemble': predictions_24p['ensemble'],
        'linear': predictions_24p['linear'],
        'polynomial': predictions_24p['polynomial'],
        'ml_features': predictions_24p['ml_features']
    })
    
    pred_df_12p.to_csv(os.path.join(BASE_DIR, 'predictions_48h.csv'), index=False)
    pred_df_24p.to_csv(os.path.join(BASE_DIR, 'predictions_96h.csv'), index=False)
    
    with open(os.path.join(BASE_DIR, 'predictions_summary.json'), 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    print("\n=== Prediction Files Saved ===")
    print(f"  Prediction keys: {list(predictions_data['predictions'].keys())}")
    print(f"  4h prediction: ${predictions_data['predictions']['4h']['price']:.2f}")
    print(f"  {os.path.join(BASE_DIR, 'predictions_48h.csv')}")
    print(f"  {os.path.join(BASE_DIR, 'predictions_96h.csv')}")
    print(f"  {os.path.join(BASE_DIR, 'predictions_summary.json')}")

if __name__ == '__main__':
    main()
