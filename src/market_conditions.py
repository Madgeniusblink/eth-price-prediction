"""
Market Condition Detector
Analyzes current market state to enable context-aware model weighting
"""

import pandas as pd
import numpy as np
from datetime import datetime

class MarketConditionDetector:
    """Detect and classify current market conditions"""
    
    def __init__(self, df):
        """
        Initialize with price data
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
        """
        self.df = df
    
    def detect_trend(self, window=50):
        """
        Detect overall market trend
        
        Returns:
            'BULL', 'BEAR', or 'NEUTRAL'
        """
        if len(self.df) < window:
            return 'NEUTRAL'
        
        recent = self.df.tail(window)
        current_price = recent['close'].iloc[-1]
        sma_50 = recent['close'].mean()
        
        # Calculate trend strength
        price_change_pct = ((current_price - recent['close'].iloc[0]) / recent['close'].iloc[0]) * 100
        
        # Price above SMA and rising = BULL
        if current_price > sma_50 and price_change_pct > 2:
            return 'BULL'
        # Price below SMA and falling = BEAR
        elif current_price < sma_50 and price_change_pct < -2:
            return 'BEAR'
        else:
            return 'NEUTRAL'
    
    def detect_volatility(self, window=20):
        """
        Detect volatility level
        
        Returns:
            'HIGH', 'MEDIUM', or 'LOW'
        """
        if len(self.df) < window:
            return 'MEDIUM'
        
        recent = self.df.tail(window)
        
        # Calculate volatility as percentage of price
        volatility = recent['close'].std()
        avg_price = recent['close'].mean()
        volatility_pct = (volatility / avg_price) * 100
        
        # Classify volatility
        if volatility_pct > 3.0:
            return 'HIGH'
        elif volatility_pct > 1.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def detect_volume_trend(self, window=20):
        """
        Detect volume trend
        
        Returns:
            'INCREASING', 'DECREASING', or 'STABLE'
        """
        if len(self.df) < window or 'volume' not in self.df.columns:
            return 'STABLE'
        
        recent = self.df.tail(window)
        
        # Compare recent volume to average
        recent_avg = recent['volume'].tail(5).mean()
        overall_avg = recent['volume'].mean()
        
        change_pct = ((recent_avg - overall_avg) / overall_avg) * 100
        
        if change_pct > 20:
            return 'INCREASING'
        elif change_pct < -20:
            return 'DECREASING'
        else:
            return 'STABLE'
    
    def detect_momentum(self, window=10):
        """
        Detect price momentum
        
        Returns:
            'STRONG_UP', 'WEAK_UP', 'NEUTRAL', 'WEAK_DOWN', or 'STRONG_DOWN'
        """
        if len(self.df) < window:
            return 'NEUTRAL'
        
        recent = self.df.tail(window)
        
        # Calculate momentum as rate of change
        momentum_pct = ((recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]) * 100
        
        if momentum_pct > 2:
            return 'STRONG_UP'
        elif momentum_pct > 0.5:
            return 'WEAK_UP'
        elif momentum_pct < -2:
            return 'STRONG_DOWN'
        elif momentum_pct < -0.5:
            return 'WEAK_DOWN'
        else:
            return 'NEUTRAL'
    
    def detect_rsi_state(self):
        """
        Detect RSI-based market state
        
        Returns:
            'OVERBOUGHT', 'OVERSOLD', or 'NEUTRAL'
        """
        if 'RSI' not in self.df.columns or len(self.df) < 1:
            return 'NEUTRAL'
        
        rsi = self.df['RSI'].iloc[-1]
        
        if pd.isna(rsi):
            return 'NEUTRAL'
        
        if rsi > 70:
            return 'OVERBOUGHT'
        elif rsi < 30:
            return 'OVERSOLD'
        else:
            return 'NEUTRAL'
    
    def detect_macd_signal(self):
        """
        Detect MACD signal
        
        Returns:
            'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if 'MACD' not in self.df.columns or 'MACD_signal' not in self.df.columns:
            return 'NEUTRAL'
        
        if len(self.df) < 2:
            return 'NEUTRAL'
        
        current_macd = self.df['MACD'].iloc[-1]
        current_signal = self.df['MACD_signal'].iloc[-1]
        prev_macd = self.df['MACD'].iloc[-2]
        prev_signal = self.df['MACD_signal'].iloc[-2]
        
        if pd.isna(current_macd) or pd.isna(current_signal):
            return 'NEUTRAL'
        
        # Bullish crossover
        if prev_macd <= prev_signal and current_macd > current_signal:
            return 'BULLISH'
        # Bearish crossover
        elif prev_macd >= prev_signal and current_macd < current_signal:
            return 'BEARISH'
        # Continuation
        elif current_macd > current_signal:
            return 'BULLISH'
        elif current_macd < current_signal:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def get_comprehensive_condition(self):
        """
        Get comprehensive market condition analysis
        
        Returns:
            Dict with all condition metrics and a combined condition string
        """
        trend = self.detect_trend()
        volatility = self.detect_volatility()
        volume_trend = self.detect_volume_trend()
        momentum = self.detect_momentum()
        rsi_state = self.detect_rsi_state()
        macd_signal = self.detect_macd_signal()
        
        # Create combined condition string for model weight lookup
        # Format: {trend}_{volatility}
        combined_condition = f"{trend.lower()}_{volatility.lower()}_volatility"
        
        # Calculate confidence score based on indicator agreement
        confidence_signals = []
        
        # Trend and momentum should agree
        if (trend == 'BULL' and momentum in ['STRONG_UP', 'WEAK_UP']) or \
           (trend == 'BEAR' and momentum in ['STRONG_DOWN', 'WEAK_DOWN']) or \
           (trend == 'NEUTRAL' and momentum == 'NEUTRAL'):
            confidence_signals.append(1)
        else:
            confidence_signals.append(0)
        
        # RSI and trend should align
        if (trend == 'BULL' and rsi_state != 'OVERSOLD') or \
           (trend == 'BEAR' and rsi_state != 'OVERBOUGHT') or \
           (trend == 'NEUTRAL'):
            confidence_signals.append(1)
        else:
            confidence_signals.append(0)
        
        # MACD and trend should align
        if (trend == 'BULL' and macd_signal == 'BULLISH') or \
           (trend == 'BEAR' and macd_signal == 'BEARISH') or \
           (trend == 'NEUTRAL'):
            confidence_signals.append(1)
        else:
            confidence_signals.append(0)
        
        confidence = sum(confidence_signals) / len(confidence_signals)
        
        return {
            'condition': combined_condition,
            'trend': trend,
            'volatility': volatility,
            'volume_trend': volume_trend,
            'momentum': momentum,
            'rsi_state': rsi_state,
            'macd_signal': macd_signal,
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_condition_description(self, condition_dict):
        """
        Generate human-readable description of market conditions
        
        Args:
            condition_dict: Dict from get_comprehensive_condition()
        
        Returns:
            String description
        """
        desc = f"Market is in a **{condition_dict['trend']}** trend with **{condition_dict['volatility']}** volatility. "
        desc += f"Momentum is **{condition_dict['momentum'].replace('_', ' ')}** and volume is **{condition_dict['volume_trend']}**. "
        
        if condition_dict['rsi_state'] != 'NEUTRAL':
            desc += f"RSI indicates **{condition_dict['rsi_state']}** conditions. "
        
        if condition_dict['macd_signal'] != 'NEUTRAL':
            desc += f"MACD is **{condition_dict['macd_signal']}**. "
        
        desc += f"(Confidence: {int(condition_dict['confidence']*100)}%)"
        
        return desc

def main():
    """Test the market condition detector"""
    # This would normally use real data
    print("Market Condition Detector - Ready to integrate")
    print("Use with real market data from fetch_data.py")

if __name__ == '__main__':
    main()
