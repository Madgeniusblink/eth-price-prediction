"""
Centralized Logging Infrastructure for ETH Price Prediction System
Provides structured logging with file rotation and multiple log levels
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from config import BASE_DIR

def setup_logger(name, log_level=logging.INFO):
    """
    Setup a logger with both file and console handlers
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation (10MB max, keep 5 backups)
    log_file = os.path.join(log_dir, f'system_{datetime.now():%Y%m%d}.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_error_with_context(logger, error, context=None):
    """
    Log an error with additional context information
    
    Args:
        logger: Logger instance
        error: Exception object
        context: Dict with additional context (optional)
    """
    import traceback
    
    logger.error(f"Error occurred: {str(error)}")
    logger.error(f"Error type: {type(error).__name__}")
    
    if context:
        logger.error(f"Context: {context}")
    
    logger.error(f"Traceback:\n{traceback.format_exc()}")

def log_performance_metrics(logger, metrics):
    """
    Log performance metrics in a structured format
    
    Args:
        logger: Logger instance
        metrics: Dict with performance metrics
    """
    logger.info("=" * 60)
    logger.info("PERFORMANCE METRICS")
    logger.info("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)

def log_prediction_summary(logger, predictions, current_price):
    """
    Log prediction summary in a readable format
    
    Args:
        logger: Logger instance
        predictions: Dict with predictions
        current_price: Current ETH price
    """
    logger.info("=" * 60)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Current Price: ${current_price:,.2f}")
    logger.info("")
    
    for timeframe, pred in predictions.items():
        price = pred.get('price', 0)
        change = pred.get('change_pct', 0)
        logger.info(f"  {timeframe}: ${price:,.2f} ({change:+.2f}%)")
    
    logger.info("=" * 60)

# Create a default system logger
system_logger = setup_logger('eth_system')

if __name__ == '__main__':
    # Test the logger
    test_logger = setup_logger('test')
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    # Test error logging with context
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error_with_context(test_logger, e, {'test_key': 'test_value'})
    
    # Test metrics logging
    log_performance_metrics(test_logger, {
        'accuracy': 0.6954,
        'win_rate': 0.622,
        'profit_factor': 1.55
    })
    
    # Test prediction logging
    log_prediction_summary(test_logger, {
        '15m': {'price': 3170.50, 'change_pct': 0.5},
        '30m': {'price': 3185.25, 'change_pct': 1.0},
        '60m': {'price': 3200.00, 'change_pct': 1.5}
    }, 3154.32)
    
    print(f"\n✓ Logger test complete. Check logs directory: {os.path.join(BASE_DIR, 'logs')}")
