#!/usr/bin/env python3
"""
Model Validation and Backtesting Module
Implements rigorous validation techniques for the prediction system
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive performance metrics
    
    Args:
        y_true: Actual prices
        y_pred: Predicted prices
    
    Returns:
        dict: Dictionary of metrics
    """
    # Basic metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional Accuracy
    actual_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Max error
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'directional_accuracy': directional_accuracy,
        'max_error': max_error,
        'n_samples': len(y_true)
    }

def rolling_window_backtest(df, model_func, window_size=100, forecast_horizon=1):
    """
    Perform rolling-window backtesting
    
    Args:
        df: DataFrame with price data
        model_func: Function that takes training data and returns predictions
        window_size: Size of training window
        forecast_horizon: Number of periods ahead to predict
    
    Returns:
        dict: Backtest results with predictions and actuals
    """
    predictions = []
    actuals = []
    timestamps = []
    
    # Ensure we have enough data
    if len(df) < window_size + forecast_horizon:
        raise ValueError(f"Insufficient data: need at least {window_size + forecast_horizon} points")
    
    # Rolling window
    for i in range(window_size, len(df) - forecast_horizon):
        # Training window
        train_data = df.iloc[i-window_size:i].copy()
        
        # Actual future value
        actual = df.iloc[i + forecast_horizon - 1]['close']
        timestamp = df.iloc[i + forecast_horizon - 1]['timestamp']
        
        # Make prediction
        try:
            pred = model_func(train_data, forecast_horizon)
            predictions.append(pred)
            actuals.append(actual)
            timestamps.append(timestamp)
        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue
    
    return {
        'predictions': np.array(predictions),
        'actuals': np.array(actuals),
        'timestamps': timestamps
    }

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe Ratio for a series of returns
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        float: Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    
    # Annualize (assuming minute-level data)
    sharpe_annualized = sharpe * np.sqrt(525600)  # minutes in a year
    
    return sharpe_annualized

def simulate_trading_strategy(predictions, actuals, transaction_cost=0.001):
    """
    Simulate a simple trading strategy based on predictions
    
    Args:
        predictions: Predicted prices
        actuals: Actual prices
        transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
    
    Returns:
        dict: Trading performance metrics
    """
    returns = []
    positions = []  # 1 for long, -1 for short, 0 for neutral
    
    # Simple strategy: go long if prediction > current, short if prediction < current
    for i in range(len(predictions) - 1):
        current_price = actuals[i]
        predicted_price = predictions[i]
        next_actual_price = actuals[i + 1]
        
        # Determine position
        if predicted_price > current_price * 1.001:  # Threshold to avoid noise
            position = 1  # Long
        elif predicted_price < current_price * 0.999:
            position = -1  # Short
        else:
            position = 0  # Neutral
        
        positions.append(position)
        
        # Calculate return
        price_return = (next_actual_price - current_price) / current_price
        strategy_return = position * price_return - abs(position) * transaction_cost
        returns.append(strategy_return)
    
    returns = np.array(returns)
    
    # Calculate metrics
    total_return = np.sum(returns)
    avg_return = np.mean(returns)
    sharpe = calculate_sharpe_ratio(returns)
    win_rate = np.mean(returns > 0) * 100
    
    return {
        'total_return': total_return * 100,  # As percentage
        'avg_return_per_trade': avg_return * 100,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'num_trades': len(returns),
        'long_trades': np.sum(np.array(positions) == 1),
        'short_trades': np.sum(np.array(positions) == -1)
    }

def validate_model_stability(backtest_results, window=20):
    """
    Assess model stability over time by analyzing rolling performance
    
    Args:
        backtest_results: Results from rolling_window_backtest
        window: Window size for rolling metrics
    
    Returns:
        dict: Stability metrics
    """
    predictions = backtest_results['predictions']
    actuals = backtest_results['actuals']
    
    # Calculate rolling R² scores
    rolling_r2 = []
    rolling_rmse = []
    
    for i in range(window, len(predictions)):
        y_true = actuals[i-window:i]
        y_pred = predictions[i-window:i]
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        rolling_r2.append(r2)
        rolling_rmse.append(rmse)
    
    rolling_r2 = np.array(rolling_r2)
    rolling_rmse = np.array(rolling_rmse)
    
    return {
        'mean_rolling_r2': np.mean(rolling_r2),
        'std_rolling_r2': np.std(rolling_r2),
        'min_rolling_r2': np.min(rolling_r2),
        'max_rolling_r2': np.max(rolling_r2),
        'mean_rolling_rmse': np.mean(rolling_rmse),
        'std_rolling_rmse': np.std(rolling_rmse)
    }

def main():
    """
    Main validation routine
    """
    print("=== Model Validation and Backtesting ===\n")
    
    # Load data
    try:
        df = pd.read_csv('/home/ubuntu/eth_1m_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✓ Loaded {len(df)} data points")
    except FileNotFoundError:
        print("✗ Data file not found. Please run fetch_data.py first.")
        return
    
    # Define a simple prediction function for backtesting
    # (In practice, this would use the actual models from predict.py)
    def simple_linear_predictor(train_data, horizon):
        """Simple linear trend predictor for validation"""
        prices = train_data['close'].values
        x = np.arange(len(prices))
        
        # Fit linear regression
        coeffs = np.polyfit(x, prices, 1)
        
        # Predict
        pred_x = len(prices) + horizon - 1
        pred_price = coeffs[0] * pred_x + coeffs[1]
        
        return pred_price
    
    print("\n=== Running Rolling-Window Backtest ===")
    print("Window size: 100 periods")
    print("Forecast horizon: 1 period (1 minute)")
    
    # Run backtest
    backtest_results = rolling_window_backtest(
        df, 
        simple_linear_predictor, 
        window_size=100, 
        forecast_horizon=1
    )
    
    print(f"✓ Completed {len(backtest_results['predictions'])} predictions\n")
    
    # Calculate metrics
    print("=== Performance Metrics ===")
    metrics = calculate_metrics(
        backtest_results['actuals'], 
        backtest_results['predictions']
    )
    
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"RMSE: ${metrics['rmse']:.2f}")
    print(f"MAE: ${metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    print(f"Max Error: ${metrics['max_error']:.2f}")
    
    # Assess stability
    print("\n=== Model Stability Analysis ===")
    stability = validate_model_stability(backtest_results, window=20)
    
    print(f"Mean Rolling R²: {stability['mean_rolling_r2']:.4f}")
    print(f"Std Dev Rolling R²: {stability['std_rolling_r2']:.4f}")
    print(f"R² Range: [{stability['min_rolling_r2']:.4f}, {stability['max_rolling_r2']:.4f}]")
    print(f"Mean Rolling RMSE: ${stability['mean_rolling_rmse']:.2f}")
    
    # Simulate trading
    print("\n=== Trading Strategy Simulation ===")
    trading_results = simulate_trading_strategy(
        backtest_results['predictions'],
        backtest_results['actuals'],
        transaction_cost=0.001
    )
    
    print(f"Total Return: {trading_results['total_return']:.2f}%")
    print(f"Avg Return per Trade: {trading_results['avg_return_per_trade']:.4f}%")
    print(f"Sharpe Ratio: {trading_results['sharpe_ratio']:.2f}")
    print(f"Win Rate: {trading_results['win_rate']:.2f}%")
    print(f"Number of Trades: {trading_results['num_trades']}")
    print(f"  Long: {trading_results['long_trades']}, Short: {trading_results['short_trades']}")
    
    # Save results
    validation_report = {
        'timestamp': datetime.now().isoformat(),
        'data_points': len(df),
        'backtest_predictions': len(backtest_results['predictions']),
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in metrics.items()},
        'stability': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                     for k, v in stability.items()},
        'trading': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in trading_results.items()}
    }
    
    output_path = '/home/ubuntu/validation_report.json'
    with open(output_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"\n✓ Validation report saved to: {output_path}")
    
    # Performance assessment
    print("\n=== Performance Assessment ===")
    
    if metrics['r2_score'] >= 0.70:
        print("✓ R² Score: PASS (≥0.70)")
    else:
        print("✗ R² Score: FAIL (<0.70)")
    
    if metrics['rmse'] <= 10.0:
        print("✓ RMSE: PASS (≤$10)")
    else:
        print("✗ RMSE: FAIL (>$10)")
    
    if metrics['directional_accuracy'] >= 55:
        print("✓ Directional Accuracy: PASS (≥55%)")
    else:
        print("✗ Directional Accuracy: FAIL (<55%)")
    
    if trading_results['sharpe_ratio'] >= 0.5:
        print("✓ Sharpe Ratio: PASS (≥0.5)")
    else:
        print("✗ Sharpe Ratio: FAIL (<0.5)")

if __name__ == '__main__':
    main()
