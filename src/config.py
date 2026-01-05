"""
Configuration settings for Ethereum Price Prediction System
"""

import os

# ============================================================================
# DATA SETTINGS
# ============================================================================

# Binance API settings (no API key required for public endpoints)
BINANCE_BASE_URL = 'https://api.binance.com/api/v3'
SYMBOL = 'ETHUSDT'
INTERVAL = '1m'  # Options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
DATA_POINTS = 500  # Number of historical candles to fetch

# Alternative intervals for multi-timeframe analysis
INTERVALS = {
    '1m': 500,   # ~8.3 hours
    '5m': 500,   # ~41 hours
    '15m': 500,  # ~5 days
}

# Data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Ensemble configuration
ENSEMBLE_WEIGHTS = 'auto'  # 'auto' for performance-based, or dict with manual weights
# Example manual weights: {'linear': 0.3, 'polynomial': 0.3, 'ml_features': 0.4}

# Linear Regression
LINEAR_TRAINING_WINDOW = 100  # Number of recent periods to train on

# Polynomial Regression
POLYNOMIAL_DEGREE = 2  # Degree of polynomial (2 = quadratic, 3 = cubic)
POLYNOMIAL_TRAINING_WINDOW = 100

# Random Forest (ML Features)
RF_N_ESTIMATORS = 100  # Number of trees
RF_MAX_DEPTH = 10  # Maximum tree depth
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2
RF_RANDOM_STATE = 42
RF_TRAINING_WINDOW = 200  # Need more data for feature-based model

# ============================================================================
# TECHNICAL INDICATORS SETTINGS
# ============================================================================

# Moving Averages
SMA_PERIODS = [5, 10, 20, 50]
EMA_PERIODS = [5, 10, 20]

# RSI
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD_DEV = 2

# Momentum
MOMENTUM_PERIOD = 10

# Volume
VOLUME_SMA_PERIOD = 20

# ============================================================================
# PREDICTION SETTINGS
# ============================================================================

# Forecast horizons (in minutes)
FORECAST_PERIODS = [15, 30, 60, 120]

# Confidence intervals
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval

# Prediction thresholds
MIN_DATA_POINTS = 100  # Minimum data points required for prediction
MAX_PREDICTION_CHANGE = 0.10  # Flag predictions with >10% change as anomalies

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

# Backtesting
BACKTEST_PERIODS = 100  # Number of periods to backtest
BACKTEST_STEP = 1  # Step size for rolling window

# Cross-validation
CV_FOLDS = 5  # Number of folds for time series cross-validation
CV_TEST_SIZE = 0.2  # Test set size as proportion

# Performance thresholds
MIN_R2_SCORE = 0.50  # Minimum acceptable R² score
MAX_RMSE = 10.0  # Maximum acceptable RMSE in dollars
MAX_MAE = 7.0  # Maximum acceptable MAE in dollars
MIN_DIRECTIONAL_ACCURACY = 0.55  # Minimum % of correct direction predictions

# Model update frequency
MODEL_UPDATE_FREQUENCY = 10  # Retrain models every N predictions

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Chart settings
FIGURE_DPI = 300
FIGURE_STYLE = 'seaborn-v0_8-darkgrid'

# Colors
COLOR_HISTORICAL = '#2E86AB'
COLOR_PREDICTION = '#A23B72'
COLOR_LINEAR = '#F18F01'
COLOR_POLYNOMIAL = '#C73E1D'
COLOR_ML = '#6A994E'
COLOR_CONFIDENCE = 'purple'

# Chart sizes
OVERVIEW_FIGURE_SIZE = (18, 12)
INDICATORS_FIGURE_SIZE = (16, 12)
FOCUSED_FIGURE_SIZE = (16, 8)

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(BASE_DIR, 'prediction.log')

# ============================================================================
# RISK MANAGEMENT SETTINGS
# ============================================================================

# Trading risk parameters (for reference only - not used in predictions)
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.03  # 3% take profit
MAX_POSITION_SIZE = 0.10  # Maximum 10% of portfolio per trade

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable/disable features
ENABLE_BACKTESTING = True
ENABLE_CROSS_VALIDATION = True
ENABLE_ANOMALY_DETECTION = True
ENABLE_REGIME_DETECTION = True
SAVE_VISUALIZATIONS = True
SAVE_PREDICTIONS = True

# ============================================================================
# API RATE LIMITING
# ============================================================================

# Binance rate limits (to avoid being blocked)
MAX_REQUESTS_PER_MINUTE = 1200
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Regime detection thresholds
TRENDING_THRESHOLD = 0.7  # ADX threshold for trending market
HIGH_VOLATILITY_THRESHOLD = 1.5  # Standard deviation multiplier

# Outlier detection
OUTLIER_STD_THRESHOLD = 3  # Number of standard deviations for outlier

# Feature importance threshold
MIN_FEATURE_IMPORTANCE = 0.01  # Minimum importance to keep feature
