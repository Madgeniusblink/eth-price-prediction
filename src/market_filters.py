"""
Market Filters for ETH Price Prediction System
Implements trend detection and support/resistance level identification
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from logger import setup_logger

logger = setup_logger(__name__)


class MarketFilters:
    """
    Provides trend analysis and support/resistance detection
    to filter trading signals and improve directional accuracy
    """
    
    def __init__(self, ma_short=50, ma_long=200, sr_tolerance=0.015):
        """
        Initialize market filters
        
        Args:
            ma_short: Short-term moving average period (default: 50)
            ma_long: Long-term moving average period (default: 200)
            sr_tolerance: Support/resistance proximity threshold (default: 1.5%)
        """
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.sr_tolerance = sr_tolerance
        self.support_resistance_levels = []
        
    def calculate_trend(self, df, price_col='close'):
        """
        Calculate market trend based on moving averages
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price (default: 'close')
            
        Returns:
            dict with trend info: {
                'trend': 'up'|'down'|'neutral',
                'ma_short': float,
                'ma_long': float,
                'current_price': float,
                'strength': float (0-1)
            }
        """
        if len(df) < self.ma_long:
            logger.warning(f"Insufficient data for MA{self.ma_long}. Need {self.ma_long}, have {len(df)}")
            # Use available data
            ma_long_period = len(df)
        else:
            ma_long_period = self.ma_long
            
        if len(df) < self.ma_short:
            ma_short_period = len(df)
        else:
            ma_short_period = self.ma_short
        
        # Calculate moving averages
        ma_short = df[price_col].rolling(window=ma_short_period).mean().iloc[-1]
        ma_long = df[price_col].rolling(window=ma_long_period).mean().iloc[-1]
        current_price = df[price_col].iloc[-1]
        
        # Determine trend
        if current_price > ma_long:
            trend = 'up'
            # Strength based on distance from MA
            strength = min(1.0, (current_price - ma_long) / ma_long / 0.05)  # 5% = full strength
        elif current_price < ma_long:
            trend = 'down'
            strength = min(1.0, (ma_long - current_price) / ma_long / 0.05)
        else:
            trend = 'neutral'
            strength = 0.0
        
        logger.info(f"Trend: {trend.upper()} | Price: ${current_price:.2f} | MA{ma_short_period}: ${ma_short:.2f} | MA{ma_long_period}: ${ma_long:.2f}")
        
        return {
            'trend': trend,
            'ma_short': ma_short,
            'ma_long': ma_long,
            'current_price': current_price,
            'strength': strength
        }
    
    def identify_support_resistance(self, df, price_col='close', num_levels=10):
        """
        Identify support and resistance levels using price clustering
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            num_levels: Maximum number of S/R levels to identify
            
        Returns:
            list of support/resistance price levels
        """
        prices = df[price_col].values
        
        # Create price histogram
        bins = min(50, len(prices) // 10)
        hist, bin_edges = np.histogram(prices, bins=bins)
        
        # Find peaks in histogram (price levels where price spent more time)
        prominence = max(2, len(prices) // 100)
        peaks, properties = find_peaks(hist, prominence=prominence)
        
        # Convert bin indices to price levels
        levels = []
        for peak in peaks:
            level = (bin_edges[peak] + bin_edges[peak + 1]) / 2
            # Store level with its strength (histogram count)
            levels.append((level, hist[peak]))
        
        # Sort by strength and take top N
        levels.sort(key=lambda x: x[1], reverse=True)
        self.support_resistance_levels = [level for level, _ in levels[:num_levels]]
        
        logger.info(f"Identified {len(self.support_resistance_levels)} S/R levels")
        for level in sorted(self.support_resistance_levels):
            logger.info(f"  S/R Level: ${level:.2f}")
        
        return self.support_resistance_levels
    
    def check_near_support_resistance(self, price):
        """
        Check if price is near a support/resistance level
        
        Args:
            price: Current price to check
            
        Returns:
            dict with proximity info: {
                'near_sr': bool,
                'closest_level': float or None,
                'distance_pct': float
            }
        """
        if not self.support_resistance_levels:
            return {
                'near_sr': False,
                'closest_level': None,
                'distance_pct': 0.0
            }
        
        # Find closest S/R level
        distances = [abs(price - level) for level in self.support_resistance_levels]
        min_distance = min(distances)
        closest_level = self.support_resistance_levels[distances.index(min_distance)]
        distance_pct = min_distance / price
        
        near_sr = distance_pct < self.sr_tolerance
        
        if near_sr:
            logger.info(f"Price ${price:.2f} is near S/R level ${closest_level:.2f} ({distance_pct*100:.2f}%)")
        
        return {
            'near_sr': near_sr,
            'closest_level': closest_level,
            'distance_pct': distance_pct
        }
    
    def should_take_trade(self, predicted_direction, trend_info, sr_info):
        """
        Determine if a trade signal should be taken based on filters
        
        Args:
            predicted_direction: 1 for up, -1 for down
            trend_info: dict from calculate_trend()
            sr_info: dict from check_near_support_resistance()
            
        Returns:
            dict with decision: {
                'take_trade': bool,
                'confidence': float (0-1),
                'reasons': list of str
            }
        """
        reasons = []
        confidence = 0.5  # Base confidence
        
        # Check trend alignment
        trend = trend_info['trend']
        trend_aligned = (
            (trend == 'up' and predicted_direction == 1) or
            (trend == 'down' and predicted_direction == -1)
        )
        
        if not trend_aligned:
            reasons.append(f"Signal conflicts with {trend} trend")
            return {
                'take_trade': False,
                'confidence': 0.0,
                'reasons': reasons
            }
        
        reasons.append(f"Aligned with {trend} trend")
        confidence += 0.2 * trend_info['strength']
        
        # Check support/resistance proximity
        if sr_info['near_sr']:
            reasons.append(f"Near S/R level ${sr_info['closest_level']:.2f} - reduced confidence")
            confidence -= 0.15
        else:
            reasons.append("Clear of S/R levels")
            confidence += 0.1
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        take_trade = confidence >= 0.5
        
        logger.info(f"Trade decision: {'TAKE' if take_trade else 'SKIP'} | Confidence: {confidence:.2%}")
        for reason in reasons:
            logger.info(f"  - {reason}")
        
        return {
            'take_trade': take_trade,
            'confidence': confidence,
            'reasons': reasons
        }
    
    def get_market_context(self, df, current_price, price_col='close'):
        """
        Get complete market context for decision making
        
        Args:
            df: DataFrame with OHLCV data
            current_price: Current price
            price_col: Column name for price
            
        Returns:
            dict with complete market context
        """
        # Calculate trend
        trend_info = self.calculate_trend(df, price_col)
        
        # Identify S/R levels if not already done
        if not self.support_resistance_levels:
            self.identify_support_resistance(df, price_col)
        
        # Check S/R proximity
        sr_info = self.check_near_support_resistance(current_price)
        
        return {
            'trend': trend_info,
            'support_resistance': sr_info,
            'sr_levels': self.support_resistance_levels
        }


def apply_filters_to_signals(signals, market_context):
    """
    Apply market filters to trading signals
    
    Args:
        signals: dict with trading signals from generate_signals()
        market_context: dict from get_market_context()
        
    Returns:
        dict with filtered signals
    """
    filters = MarketFilters()
    
    # Determine predicted direction from signal
    signal_type = signals.get('signal', 'WAIT')
    if signal_type == 'BUY':
        predicted_direction = 1
    elif signal_type == 'SELL':
        predicted_direction = -1
    else:
        # HOLD or WAIT - no filtering needed
        return signals
    
    # Check if trade should be taken
    decision = filters.should_take_trade(
        predicted_direction,
        market_context['trend'],
        market_context['support_resistance']
    )
    
    # Update signals based on filter decision
    if not decision['take_trade']:
        logger.info(f"Trade signal {signal_type} filtered out")
        signals['signal'] = 'WAIT'
        signals['confidence'] = 'FILTERED'
        signals['filter_reasons'] = decision['reasons']
    else:
        # Adjust confidence based on filters
        original_confidence = signals.get('confidence', 'MEDIUM')
        filter_confidence = decision['confidence']
        
        if filter_confidence >= 0.7:
            signals['confidence'] = 'HIGH'
        elif filter_confidence >= 0.5:
            signals['confidence'] = 'MEDIUM'
        else:
            signals['confidence'] = 'LOW'
        
        signals['filter_confidence'] = filter_confidence
        signals['filter_reasons'] = decision['reasons']
    
    return signals
