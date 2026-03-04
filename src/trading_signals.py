"""
Trading Signals Module
Generates buy/sell/short signals with multi-timeframe trend analysis
Includes derivatives data integration and advanced filters
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
try:
    from derivatives_data import fetch_derivatives_data
except ImportError:
    from src.derivatives_data import fetch_derivatives_data

class TradingSignals:
    """Generate trading signals based on technical indicators and trend analysis"""
    
    def __init__(self, df):
        """
        Initialize with price data
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self.df = df.copy()
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        self.calculate_all_indicators()
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        # Moving Averages
        self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['sma_50'] = self.df['close'].rolling(window=50).mean()
        self.df['sma_200'] = self.df['close'].rolling(window=200).mean()
        self.df['ema_12'] = self.df['close'].ewm(span=12).mean()
        self.df['ema_26'] = self.df['close'].ewm(span=26).mean()
        
        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        self.df['macd'] = self.df['ema_12'] - self.df['ema_26']
        self.df['macd_signal'] = self.df['macd'].ewm(span=9).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
        
        # Bollinger Bands
        self.df['bb_middle'] = self.df['close'].rolling(window=20).mean()
        bb_std = self.df['close'].rolling(window=20).std()
        self.df['bb_upper'] = self.df['bb_middle'] + (bb_std * 2)
        self.df['bb_lower'] = self.df['bb_middle'] - (bb_std * 2)
        self.df['bb_position'] = (self.df['close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
        
        # ATR (Average True Range) for volatility
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.df['atr'] = true_range.rolling(window=14).mean()
        self.df['atr_14'] = self.df['atr']  # alias for predict_rl features
        
        # Volume indicators
        self.df['volume_sma'] = self.df['volume'].rolling(window=20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma']

        # ── NEW: VWAP (24-hour rolling) ──────────────────────────────────────
        window_vwap = min(24, len(self.df))
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        self.df['vwap'] = (
            (typical_price * self.df['volume']).rolling(window=window_vwap).sum()
            / self.df['volume'].rolling(window=window_vwap).sum()
        )

        # ── NEW: OBV (On-Balance Volume) ─────────────────────────────────────
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > self.df['close'].iloc[i - 1]:
                obv.append(obv[-1] + self.df['volume'].iloc[i])
            elif self.df['close'].iloc[i] < self.df['close'].iloc[i - 1]:
                obv.append(obv[-1] - self.df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        self.df['obv'] = obv
        self.df['obv_sma'] = self.df['obv'].rolling(window=20).mean()

        # ── NEW: Stochastic RSI ───────────────────────────────────────────────
        rsi_min = self.df['rsi'].rolling(window=14).min()
        rsi_max = self.df['rsi'].rolling(window=14).max()
        rsi_range = rsi_max - rsi_min
        self.df['stoch_rsi'] = np.where(
            rsi_range > 0,
            (self.df['rsi'] - rsi_min) / rsi_range,
            0.5,
        )

        # ── NEW: Volume Profile (high-volume nodes) ───────────────────────────
        # Store rolling 50-period volume-weighted price clusters (top 3 nodes)
        self._compute_volume_profile(lookback=50)

        # ── NEW: RSI Divergence ───────────────────────────────────────────────
        self.df['rsi_divergence'] = self._compute_rsi_divergence()
    
    def detect_trend(self, timeframe='current'):
        """
        Detect market trend
        
        Args:
            timeframe: 'short' (last 20 periods), 'medium' (last 50), 'long' (last 200), 'current' (latest)
        
        Returns:
            dict with trend info
        """
        latest = self.df.iloc[-1]
        
        if timeframe == 'current':
            # Current trend based on latest data
            ma_trend = self._check_ma_trend(latest)
            price_trend = self._check_price_trend()
            momentum = self._check_momentum(latest)
            
            # Combine signals
            bullish_signals = sum([
                ma_trend == 'bullish',
                price_trend == 'bullish',
                momentum == 'bullish',
                latest['rsi'] > 50,
                latest['macd'] > latest['macd_signal']
            ])
            
            if bullish_signals >= 4:
                trend = 'BULL MARKET'
                confidence = 'HIGH'
            elif bullish_signals >= 3:
                trend = 'BULLISH'
                confidence = 'MEDIUM'
            elif bullish_signals == 2:
                trend = 'NEUTRAL'
                confidence = 'LOW'
            elif bullish_signals == 1:
                trend = 'BEARISH'
                confidence = 'MEDIUM'
            else:
                trend = 'BEAR MARKET'
                confidence = 'HIGH'
        
        return {
            'trend': trend,
            'confidence': confidence,
            'ma_alignment': ma_trend,
            'price_action': price_trend,
            'momentum': momentum,
            'rsi': latest['rsi'],
            'macd_signal': 'bullish' if latest['macd'] > latest['macd_signal'] else 'bearish'
        }
    
    def _check_ma_trend(self, latest):
        """Check moving average alignment"""
        if pd.isna(latest['sma_200']):
            return 'insufficient_data'
        
        # Golden Cross / Death Cross
        if latest['sma_50'] > latest['sma_200'] and latest['close'] > latest['sma_50']:
            return 'bullish'
        elif latest['sma_50'] < latest['sma_200'] and latest['close'] < latest['sma_50']:
            return 'bearish'
        else:
            return 'neutral'
    
    def _check_price_trend(self):
        """Check if price is making higher lows (bullish) or lower highs (bearish)"""
        recent = self.df.tail(50)
        
        # Find local lows and highs
        lows = recent[recent['low'] == recent['low'].rolling(window=5, center=True).min()]['low']
        highs = recent[recent['high'] == recent['high'].rolling(window=5, center=True).max()]['high']
        
        if len(lows) >= 2:
            # Check if lows are rising
            if lows.iloc[-1] > lows.iloc[-2]:
                return 'bullish'
            elif lows.iloc[-1] < lows.iloc[-2]:
                return 'bearish'
        
        return 'neutral'
    
    def _check_momentum(self, latest):
        """Check momentum indicators"""
        bullish_momentum = sum([
            latest['rsi'] > 50,
            latest['macd_histogram'] > 0,
            latest['close'] > latest['sma_20']
        ])
        
        if bullish_momentum >= 2:
            return 'bullish'
        elif bullish_momentum == 1:
            return 'neutral'
        else:
            return 'bearish'
    
    def find_support_resistance(self, lookback=100):
        """
        Find key support and resistance levels
        
        Returns:
            dict with support and resistance levels
        """
        recent = self.df.tail(lookback)
        current_price = self.df.iloc[-1]['close']
        
        # Find local highs and lows
        local_highs = recent[recent['high'] == recent['high'].rolling(window=5, center=True).max()]['high']
        local_lows = recent[recent['low'] == recent['low'].rolling(window=5, center=True).min()]['low']
        
        # Cluster nearby levels
        resistance_levels = self._cluster_levels(local_highs[local_highs > current_price])
        support_levels = self._cluster_levels(local_lows[local_lows < current_price])
        
        return {
            'current_price': current_price,
            'resistance': sorted(resistance_levels)[:3],  # Top 3 resistance
            'support': sorted(support_levels, reverse=True)[:3],  # Top 3 support
            'nearest_resistance': min(resistance_levels) if resistance_levels else None,
            'nearest_support': max(support_levels) if support_levels else None
        }
    
    def _cluster_levels(self, levels, threshold=0.01):
        """Cluster nearby price levels"""
        if len(levels) == 0:
            return []
        
        levels = sorted(levels.values)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    def generate_entry_signals(self):
        """
        Generate buy/sell/short entry signals with user context awareness
        Integrates derivatives data and advanced filters

        Returns:
            dict with signal information
        """
        # Load user context
        user_context = self._load_user_context()

        # Fetch derivatives data
        derivatives_data = fetch_derivatives_data()

        latest = self.df.iloc[-1]
        trend = self.detect_trend()
        levels = self.find_support_resistance()

        # Calculate signal scores
        buy_score = 0
        sell_score = 0
        short_score = 0
        
        # Trend-based scoring
        if trend['trend'] in ['BULL MARKET', 'BULLISH']:
            buy_score += 3
        elif trend['trend'] in ['BEAR MARKET', 'BEARISH']:
            short_score += 3
        
        # RSI signals
        if latest['rsi'] < 30:  # Oversold
            buy_score += 2
        elif latest['rsi'] > 70:  # Overbought
            sell_score += 2
            if trend['trend'] in ['BEAR MARKET', 'BEARISH']:
                short_score += 2
        
        # MACD signals
        if latest['macd'] > latest['macd_signal'] and latest['macd_histogram'] > 0:
            buy_score += 2
        elif latest['macd'] < latest['macd_signal'] and latest['macd_histogram'] < 0:
            sell_score += 2
            short_score += 1
        
        # Bollinger Band signals
        if latest['bb_position'] < 0.2:  # Near lower band
            buy_score += 2
        elif latest['bb_position'] > 0.8:  # Near upper band
            sell_score += 2
            short_score += 1
        
        # Support/Resistance proximity
        current_price = latest['close']
        if levels['nearest_support']:
            distance_to_support = (current_price - levels['nearest_support']) / current_price
            if distance_to_support < 0.01:  # Within 1% of support
                buy_score += 2
        
        if levels['nearest_resistance']:
            distance_to_resistance = (levels['nearest_resistance'] - current_price) / current_price
            if distance_to_resistance < 0.01:  # Within 1% of resistance
                sell_score += 2
                short_score += 1
        
        # Volume confirmation
        if latest['volume_ratio'] > 1.5:  # High volume
            # Amplify strongest signal
            if buy_score > sell_score and buy_score > short_score:
                buy_score += 1
            elif sell_score > buy_score:
                sell_score += 1
            elif short_score > buy_score:
                short_score += 1

        # Apply derivatives data filters
        funding_rate = derivatives_data.get('funding_rate')
        fear_greed = derivatives_data.get('fear_greed_index')

        # Funding rate filter
        if funding_rate is not None:
            if funding_rate > 0.0001:  # Positive funding (longs pay shorts)
                buy_score -= 2  # Reduce bullish bias
            elif funding_rate < -0.0001:  # Negative funding (shorts pay longs)
                short_score -= 2  # Reduce bearish bias

        # Fear & Greed filter
        if fear_greed is not None:
            if fear_greed < 20:  # Extreme Fear
                buy_score += 1  # Contrarian buy opportunity
            elif fear_greed > 80:  # Extreme Greed
                sell_score += 1  # Contrarian sell/short opportunity
                short_score += 1
        
        # Determine primary signal
        max_score = max(buy_score, sell_score, short_score)
        
        if max_score < 5:
            signal = 'WAIT'
            action = 'No clear signal - wait for better setup'
            confidence = 'LOW'
        elif buy_score == max_score:
            signal = 'BUY'
            action = 'Long entry opportunity'
            confidence = 'HIGH' if buy_score >= 7 else 'MEDIUM'
        elif sell_score == max_score:
            signal = 'SELL'
            action = 'Take profit / Exit long'
            confidence = 'HIGH' if sell_score >= 7 else 'MEDIUM'
        else:
            signal = 'SHORT'
            action = 'Short entry opportunity'
            confidence = 'HIGH' if short_score >= 7 else 'MEDIUM'
        
        # Calculate entry, stop loss, and target
        entry_levels = self._calculate_entry_levels(signal, latest, levels, trend)

        # Apply Risk:Reward filter (minimum 1.5)
        if entry_levels['risk_reward'] < 1.5 and signal in ['BUY', 'SHORT']:
            original_signal = signal
            signal = 'WAIT'
            action = f'Risk:Reward too low ({entry_levels["risk_reward"]:.2f}) - wait for better setup'
            confidence = 'LOW'
            reasoning_suffix_rr = f"\n\n⚠️ Original {original_signal} signal filtered out due to poor Risk:Reward ratio ({entry_levels['risk_reward']:.2f} < 1.5 minimum)"
        else:
            reasoning_suffix_rr = ""

        # Apply user context to refine action recommendation
        action, reasoning_suffix = self._apply_user_context(signal, action, user_context, latest['close'])

        return {
            'signal': signal,
            'action': action,
            'confidence': confidence,
            'scores': {
                'buy': buy_score,
                'sell': sell_score,
                'short': short_score
            },
            'trend_context': trend['trend'],
            'entry': entry_levels['entry'],
            'stop_loss': entry_levels['stop_loss'],
            'target': entry_levels['target'],
            'risk_reward': entry_levels['risk_reward'],
            'reasoning': '. '.join(self._generate_reasoning(signal, latest, trend, levels)) + reasoning_suffix_rr + reasoning_suffix,
            'user_context': user_context,
            'derivatives_data': derivatives_data
        }
    
    def _calculate_entry_levels(self, signal, latest, levels, trend):
        """Calculate entry, stop loss, and target levels"""
        current_price = latest['close']
        atr = latest['atr']
        
        if signal == 'BUY':
            entry = current_price
            stop_loss = levels['nearest_support'] if levels['nearest_support'] else current_price - (2 * atr)
            target = levels['nearest_resistance'] if levels['nearest_resistance'] else current_price + (3 * atr)
        
        elif signal == 'SHORT':
            entry = current_price
            stop_loss = levels['nearest_resistance'] if levels['nearest_resistance'] else current_price + (2 * atr)
            target = levels['nearest_support'] if levels['nearest_support'] else current_price - (3 * atr)
        
        elif signal == 'SELL':
            entry = current_price
            stop_loss = current_price + (1.5 * atr)
            target = current_price  # Already at target (exit signal)
        
        else:  # WAIT
            entry = levels['nearest_support'] if levels['nearest_support'] else current_price * 0.98
            stop_loss = entry * 0.97
            target = levels['nearest_resistance'] if levels['nearest_resistance'] else entry * 1.03
        
        # Calculate risk/reward
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        risk_reward = reward / risk if risk > 0 else 0
        
        return {
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'risk_reward': round(risk_reward, 2)
        }
    
    def _generate_reasoning(self, signal, latest, trend, levels):
        """Generate human-readable reasoning for the signal"""
        reasons = []
        
        # Trend
        reasons.append(f"Market trend: {trend['trend']}")
        
        # RSI
        if latest['rsi'] < 30:
            reasons.append(f"RSI oversold at {latest['rsi']:.1f}")
        elif latest['rsi'] > 70:
            reasons.append(f"RSI overbought at {latest['rsi']:.1f}")
        else:
            reasons.append(f"RSI neutral at {latest['rsi']:.1f}")
        
        # MACD
        if latest['macd'] > latest['macd_signal']:
            reasons.append("MACD bullish crossover")
        else:
            reasons.append("MACD bearish crossover")
        
        # Price position
        if latest['bb_position'] < 0.3:
            reasons.append("Price near lower Bollinger Band")
        elif latest['bb_position'] > 0.7:
            reasons.append("Price near upper Bollinger Band")
        
        # Support/Resistance
        if levels['nearest_support']:
            distance = ((latest['close'] - levels['nearest_support']) / latest['close']) * 100
            reasons.append(f"Support at ${levels['nearest_support']:.2f} ({distance:.1f}% below)")
        
        if levels['nearest_resistance']:
            distance = ((levels['nearest_resistance'] - latest['close']) / latest['close']) * 100
            reasons.append(f"Resistance at ${levels['nearest_resistance']:.2f} ({distance:.1f}% above)")
        
        return reasons
    
    def _load_user_context(self):
        """Load user context from configuration file"""
        import os
        from config import BASE_DIR
        
        context_file = os.path.join(BASE_DIR, 'user_context.json')
        
        try:
            with open(context_file, 'r') as f:
                context = json.load(f)
            return context
        except FileNotFoundError:
            # Return default context if file doesn't exist
            return {
                'trade_status': 'OUT_OF_TRADE',
                'position': {
                    'entry_price': 0,
                    'position_size': 0,
                    'entry_time': None,
                    'type': 'LONG'
                }
            }
    
    def _apply_user_context(self, signal, action, user_context, current_price):
        """Apply user context to refine action recommendations"""
        trade_status = user_context.get('trade_status', 'OUT_OF_TRADE')
        position = user_context.get('position', {})
        entry_price = position.get('entry_price', 0)
        position_type = position.get('type', 'LONG')
        
        reasoning_suffix = ""
        
        if trade_status == 'IN_TRADE':
            # User is currently in a trade
            if position_type == 'LONG':
                pnl_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                
                if signal == 'BUY':
                    action = f"Hold current long position (P/L: {pnl_percent:+.2f}%) or add to position"
                    reasoning_suffix = f"\n\nYou are currently IN A LONG TRADE at ${entry_price:.2f}. Consider holding or adding to your position."
                elif signal == 'SELL':
                    action = f"EXIT LONG POSITION NOW (P/L: {pnl_percent:+.2f}%)"
                    reasoning_suffix = f"\n\n⚠️ You are currently IN A LONG TRADE at ${entry_price:.2f}. Strong exit signal detected. Consider taking profits or cutting losses."
                elif signal == 'SHORT':
                    action = f"EXIT LONG and consider SHORT entry (Current P/L: {pnl_percent:+.2f}%)"
                    reasoning_suffix = f"\n\n⚠️ You are currently IN A LONG TRADE at ${entry_price:.2f}. Market turning bearish - consider exiting and potentially reversing to short."
                elif signal == 'WAIT' or signal == 'HOLD':
                    action = f"Hold current long position (P/L: {pnl_percent:+.2f}%)"
                    reasoning_suffix = f"\n\nYou are currently IN A LONG TRADE at ${entry_price:.2f}. No clear signal - continue holding."
            
            elif position_type == 'SHORT':
                pnl_percent = ((entry_price - current_price) / entry_price) * 100 if entry_price > 0 else 0
                
                if signal == 'SHORT':
                    action = f"Hold current short position (P/L: {pnl_percent:+.2f}%)"
                    reasoning_suffix = f"\n\nYou are currently IN A SHORT TRADE at ${entry_price:.2f}. Consider holding your position."
                elif signal == 'BUY':
                    action = f"COVER SHORT POSITION NOW (P/L: {pnl_percent:+.2f}%)"
                    reasoning_suffix = f"\n\n⚠️ You are currently IN A SHORT TRADE at ${entry_price:.2f}. Strong bullish signal - consider covering your short."
                elif signal == 'SELL':
                    action = f"Cover short and exit (Current P/L: {pnl_percent:+.2f}%)"
                    reasoning_suffix = f"\n\nYou are currently IN A SHORT TRADE at ${entry_price:.2f}. Exit signal detected."
                elif signal == 'WAIT' or signal == 'HOLD':
                    action = f"Hold current short position (P/L: {pnl_percent:+.2f}%)"
                    reasoning_suffix = f"\n\nYou are currently IN A SHORT TRADE at ${entry_price:.2f}. No clear signal - continue holding."
        
        else:  # OUT_OF_TRADE
            # User is not currently in a trade
            if signal == 'BUY':
                action = "ENTER LONG POSITION - Good entry opportunity"
                reasoning_suffix = "\n\nYou are currently OUT OF TRADE. This is a good opportunity to enter a long position."
            elif signal == 'SHORT':
                action = "ENTER SHORT POSITION - Good entry opportunity"
                reasoning_suffix = "\n\nYou are currently OUT OF TRADE. This is a good opportunity to enter a short position."
            elif signal == 'SELL':
                action = "STAY OUT - Wait for better entry"
                reasoning_suffix = "\n\nYou are currently OUT OF TRADE. Exit signal detected but you have no position. Stay on the sidelines."
            elif signal == 'WAIT' or signal == 'HOLD':
                action = "WAIT - No clear entry signal"
                reasoning_suffix = "\n\nYou are currently OUT OF TRADE. No clear entry opportunity. Wait for a better setup."
        
        return action, reasoning_suffix

    def _compute_volume_profile(self, lookback: int = 50):
        """Identify high-volume price nodes (simplified volume profile)."""
        self.volume_nodes = []
        try:
            recent = self.df.tail(lookback)
            # Bin prices into 10 buckets, find highest volume bucket
            price_range = recent['close'].max() - recent['close'].min()
            if price_range <= 0:
                return
            buckets = 10
            bucket_size = price_range / buckets
            low = recent['close'].min()
            vol_buckets = {}
            for _, row in recent.iterrows():
                bucket = int((row['close'] - low) / bucket_size)
                bucket = min(bucket, buckets - 1)
                price_node = low + bucket * bucket_size + bucket_size / 2
                vol_buckets[price_node] = vol_buckets.get(price_node, 0) + row['volume']
            # Top 3 by volume
            self.volume_nodes = sorted(
                vol_buckets.items(), key=lambda x: x[1], reverse=True
            )[:3]
        except Exception:
            self.volume_nodes = []

    def _compute_rsi_divergence(self) -> pd.Series:
        """Detect RSI divergence (bullish/bearish) over rolling 20-period window."""
        divergence = pd.Series('NONE', index=self.df.index)
        try:
            for i in range(20, len(self.df)):
                window_price = self.df['close'].iloc[i - 20:i]
                window_rsi = self.df['rsi'].iloc[i - 20:i].dropna()
                if len(window_rsi) < 10:
                    continue
                price_trend = window_price.iloc[-1] - window_price.iloc[0]
                rsi_trend = window_rsi.iloc[-1] - window_rsi.iloc[0]
                if price_trend < 0 and rsi_trend > 0:
                    divergence.iloc[i] = 'BULLISH'
                elif price_trend > 0 and rsi_trend < 0:
                    divergence.iloc[i] = 'BEARISH'
        except Exception:
            pass
        return divergence

    def compute_signal_score(self) -> dict:
        """
        Compute 0-100 signal score from institutional indicators.
        BUY if score > 65, SELL if score < 35, WAIT otherwise.
        """
        latest = self.df.iloc[-1]
        score = 50  # neutral start

        # EMA momentum (+15 bull / -15 bear)
        if latest['ema_12'] > latest['ema_26']:
            score += 15
        else:
            score -= 15

        # RSI momentum (+10) + not overbought (+5) / oversold penalty
        rsi = latest['rsi'] if not pd.isna(latest['rsi']) else 50
        if rsi > 50:
            score += 10
        else:
            score -= 10
        if rsi < 70:
            score += 5
        else:
            score -= 5

        # MACD crossover (+15 / -15)
        macd = latest['macd'] if not pd.isna(latest['macd']) else 0
        macd_sig = latest['macd_signal'] if not pd.isna(latest['macd_signal']) else 0
        if macd > macd_sig:
            score += 15
        else:
            score -= 15

        # OBV trend (+15 / -15)
        obv_now = latest['obv'] if not pd.isna(latest['obv']) else 0
        obv_sma = latest['obv_sma'] if not pd.isna(latest['obv_sma']) else 0
        if obv_now > obv_sma:
            score += 15
            obv_trend = 'UP'
        else:
            score -= 15
            obv_trend = 'DOWN'

        # Stoch RSI not overextended (+10 / -10)
        stoch = latest['stoch_rsi'] if not pd.isna(latest['stoch_rsi']) else 0.5
        if stoch < 0.8:
            score += 10
        else:
            score -= 10

        # Above VWAP (+15 / -15)
        vwap = latest['vwap'] if not pd.isna(latest['vwap']) else latest['close']
        if latest['close'] > vwap:
            score += 15
        else:
            score -= 15

        # RSI divergence (+15 / -15)
        div = latest['rsi_divergence'] if 'rsi_divergence' in self.df.columns else 'NONE'
        if div == 'BULLISH':
            score += 15
        elif div == 'BEARISH':
            score -= 15

        # Clamp to [0, 100]
        score = max(0, min(100, score))

        if score > 65:
            signal = 'BUY'
        elif score < 35:
            signal = 'SELL'
        else:
            signal = 'WAIT'

        # Confidence
        if score > 80 or score < 20:
            confidence = 'HIGH'
        elif score > 65 or score < 35:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        return {
            'score': round(score, 1),
            'signal': signal,
            'confidence': confidence,
            'obv_trend': obv_trend,
            'stoch_rsi': round(float(stoch), 3),
            'vwap': round(float(vwap), 2),
            'rsi_divergence': str(div),
        }

    def compute_kelly_fraction(self, tracker_history: dict | None = None) -> dict:
        """
        Compute Kelly Criterion position sizing fraction.
        Uses backtested win_rate and avg win/loss from tracker history.
        Returns kelly_fraction capped at 0.25.
        """
        try:
            if tracker_history:
                summary = tracker_history.get('summary', {})
                dir_acc = summary.get('directional_accuracy', 55) / 100
                avg_win = summary.get('avg_win_pct', 1.5)
                avg_loss = summary.get('avg_loss_pct', 1.0)
            else:
                # Conservative defaults
                dir_acc = 0.55
                avg_win = 1.5
                avg_loss = 1.0

            win_rate = max(0.01, min(0.99, dir_acc))
            lose_rate = 1 - win_rate
            b = avg_win / max(avg_loss, 0.001)

            kelly = (b * win_rate - lose_rate) / b
            kelly_fraction = max(0.0, min(0.25, kelly))

            return {
                'kelly_fraction': round(kelly_fraction, 4),
                'kelly_pct': round(kelly_fraction * 100, 2),
                'win_rate': round(win_rate, 3),
                'win_loss_ratio': round(b, 3),
            }
        except Exception as e:
            return {'kelly_fraction': 0.05, 'kelly_pct': 5.0, 'win_rate': 0.55, 'win_loss_ratio': 1.5}


def compute_signal_strength(signals_data: dict, regime: str = "MEDIUM") -> dict:
    """
    Compute a 0-100 signal strength score from existing signals data dict.
    Weights: RSI 20% + MACD 20% + BB position 20% + volume 20% + regime 20%

    Args:
        signals_data: dict from trading_signals.json (as written by generate_report.py)
        regime: volatility regime label (LOW/MEDIUM/HIGH/EXTREME)

    Returns:
        dict with score, components, filtered_signal, sushiswap_context
    """
    try:
        trend_a = signals_data.get('trend_analysis', {})
        trading = signals_data.get('trading_signal', {})

        rsi = trend_a.get('rsi', 50)
        macd_signal = str(trend_a.get('macd_signal', 'neutral')).lower()
        bb_pos = trend_a.get('bb_position', 0.5)
        volume_ratio = trend_a.get('volume_ratio', 1.0)
        raw_signal = trading.get('signal', 'WAIT')

        # RSI score (0-20): oversold = max bullish, overbought = max bearish
        if raw_signal in ('BUY',):
            rsi_score = max(0, min(20, (70 - rsi) / 40 * 20)) if rsi < 70 else 0
        elif raw_signal in ('SELL', 'SHORT'):
            rsi_score = max(0, min(20, (rsi - 30) / 40 * 20)) if rsi > 30 else 0
        else:
            rsi_score = 10  # neutral

        # MACD score (0-20)
        if 'bull' in macd_signal:
            macd_score = 20 if raw_signal == 'BUY' else 5
        elif 'bear' in macd_signal:
            macd_score = 20 if raw_signal in ('SELL', 'SHORT') else 5
        else:
            macd_score = 10

        # BB position score (0-20)
        if raw_signal == 'BUY':
            bb_score = max(0, min(20, (1 - bb_pos) * 20))  # lower = better buy
        elif raw_signal in ('SELL', 'SHORT'):
            bb_score = max(0, min(20, bb_pos * 20))  # higher = better sell
        else:
            bb_score = 10

        # Volume score (0-20)
        vol_score = min(20, (volume_ratio or 1.0) * 10)

        # Regime score (0-20): penalise extreme vol for longs
        regime_score = 10  # default neutral
        if regime == 'LOW':
            regime_score = 20 if raw_signal == 'BUY' else 10
        elif regime == 'MEDIUM':
            regime_score = 15
        elif regime == 'HIGH':
            regime_score = 8
        elif regime == 'EXTREME':
            regime_score = 0  # no confidence in extreme vol

        total_score = int(rsi_score + macd_score + bb_score + vol_score + regime_score)
        total_score = max(0, min(100, total_score))

        # Regime filter: suppress BUY in EXTREME vol
        filtered_signal = raw_signal
        if regime == 'EXTREME' and raw_signal == 'BUY':
            filtered_signal = 'WAIT'

        # SushiSwap context
        sushi_context = None
        if filtered_signal == 'BUY':
            sushi_context = "ETH/USDC — consider LP entry if range stable; ETH/WBTC — monitor BTC correlation"
        elif filtered_signal in ('SELL', 'SHORT'):
            sushi_context = "ETH/USDC — reduce LP exposure; ETH/WBTC — correlation sell pressure likely"
        else:
            sushi_context = "ETH/USDC — hold current positions; monitor funding rate"

        return {
            'score': total_score,
            'signal': filtered_signal,
            'raw_signal': raw_signal,
            'regime_filtered': (filtered_signal != raw_signal),
            'components': {
                'rsi': int(rsi_score),
                'macd': int(macd_score),
                'bb': int(bb_score),
                'volume': int(vol_score),
                'regime': int(regime_score),
            },
            'sushiswap_context': sushi_context,
        }
    except Exception as e:
        return {
            'score': 50,
            'signal': 'WAIT',
            'raw_signal': 'WAIT',
            'regime_filtered': False,
            'components': {},
            'sushiswap_context': "Data unavailable",
            'error': str(e),
        }


def main():
    """Test the trading signals module"""
    # This would normally load real data
    print("Trading Signals Module - Ready to integrate")
    print("Use with real market data from fetch_data.py")

if __name__ == '__main__':
    main()
