# -*- coding: utf-8 -*-
"""
improved_entry_logic.py - Enhanced Entry Signal Logic
Cải thiện logic vào lệnh với nhiều điều kiện hơn
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Độ mạnh của tín hiệu"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NO_SIGNAL = 0


@dataclass
class EntrySignal:
    """Container cho entry signal"""
    should_enter: bool
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: int  # 0-100
    strength: SignalStrength
    position_size_multiplier: float  # 0.0 - 1.5
    reasons: list
    warnings: list
    entry_price: float
    stop_loss: float
    take_profit_targets: list


class ImprovedEntryLogic:
    """
    Logic vào lệnh nâng cao với multiple filters:
    
    1. Trend Filter - Chỉ vào lệnh theo xu hướng
    2. Support/Resistance - Vào gần support
    3. Volume Confirmation - Volume tăng khi breakout
    4. Risk/Reward Check - R:R >= 2:1
    5. Market Regime Check - Thị trường phải OK
    6. Volatility Filter - Không vào khi vol quá cao
    """
    
    def __init__(self,
                 min_confidence: int = 60,
                 min_risk_reward: float = 2.0,
                 require_trend_alignment: bool = True,
                 require_volume_confirmation: bool = True):
        """
        Args:
            min_confidence: Confidence tối thiểu để vào lệnh
            min_risk_reward: R:R ratio tối thiểu
            require_trend_alignment: Yêu cầu phải theo trend
            require_volume_confirmation: Yêu cầu volume confirm
        """
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        self.require_trend_alignment = require_trend_alignment
        self.require_volume_confirmation = require_volume_confirmation
    
    def analyze_entry(self,
                     df: pd.DataFrame,
                     ml_signal: Dict,
                     market_regime: Optional[Dict] = None) -> EntrySignal:
        """
        Phân tích đầy đủ để quyết định có nên vào lệnh
        
        Args:
            df: DataFrame với OHLCV + indicators
            ml_signal: Signal từ ML model
            market_regime: Thông tin market regime (optional)
            
        Returns:
            EntrySignal object với đầy đủ thông tin
        """
        
        if df.empty or len(df) < 50:
            return self._no_signal("Không đủ dữ liệu")
        
        latest = df.iloc[-1]
        signal_type = ml_signal.get('signal', 'HOLD')
        base_confidence = ml_signal.get('confidence', 0)
        
        # Skip nếu không phải BUY signal
        if signal_type != 'BUY':
            return self._no_signal(f"Signal = {signal_type}")
        
        # Skip nếu confidence thấp
        if base_confidence < self.min_confidence:
            return self._no_signal(f"Confidence thấp ({base_confidence}%)")
        
        reasons = []
        warnings = []
        adjustments = []
        
        # ===== FILTER 1: MARKET REGIME =====
        if market_regime and not market_regime.get('tradeable', True):
            return self._no_signal(f"Thị trường: {market_regime.get('regime', 'UNKNOWN')}")
        
        # ===== FILTER 2: TREND ALIGNMENT =====
        trend_check = self._check_trend_alignment(df, signal_type)
        if not trend_check['aligned']:
            if self.require_trend_alignment:
                return self._no_signal(trend_check['reason'])
            else:
                warnings.append(f"⚠️ Trend: {trend_check['reason']}")
                adjustments.append(-10)  # Giảm 10 confidence
        else:
            reasons.append(f"✅ Trend: {trend_check['reason']}")
            if trend_check['strength'] > 50:
                adjustments.append(+5)  # Bonus cho trend mạnh
        
        # ===== FILTER 3: SUPPORT/RESISTANCE =====
        sr_check = self._check_support_resistance(df, latest['close'])
        if sr_check['too_close_to_resistance']:
            warnings.append(f"⚠️ Gần resistance: {sr_check['distance_to_resistance']:.1f}%")
            adjustments.append(-15)
        elif sr_check['near_support']:
            reasons.append(f"✅ Gần support (+{sr_check['distance_to_support']:.1f}%)")
            adjustments.append(+10)
        
        # ===== FILTER 4: VOLUME CONFIRMATION =====
        volume_check = self._check_volume_confirmation(df)
        if not volume_check['confirmed']:
            if self.require_volume_confirmation:
                return self._no_signal(volume_check['reason'])
            else:
                warnings.append(f"⚠️ Volume: {volume_check['reason']}")
                adjustments.append(-10)
        else:
            reasons.append(f"✅ Volume: {volume_check['reason']}")
            if volume_check['surge']:
                adjustments.append(+5)
        
        # ===== FILTER 5: VOLATILITY CHECK =====
        volatility_check = self._check_volatility(df)
        if volatility_check['too_high']:
            warnings.append(f"⚠️ Volatility cao: {volatility_check['value']:.2f}%")
            adjustments.append(-15)
        elif volatility_check['optimal']:
            reasons.append("✅ Volatility vừa phải")
            adjustments.append(+5)
        
        # ===== FILTER 6: RSI CHECK =====
        rsi_check = self._check_rsi(df)
        if rsi_check['overbought']:
            warnings.append(f"⚠️ RSI overbought: {rsi_check['value']:.1f}")
            adjustments.append(-10)
        elif rsi_check['optimal']:
            reasons.append(f"✅ RSI: {rsi_check['value']:.1f}")
            adjustments.append(+5)
        
        # ===== FILTER 7: PRICE ACTION =====
        price_action = self._check_price_action(df)
        if price_action['bullish_pattern']:
            reasons.append(f"✅ Pattern: {price_action['pattern']}")
            adjustments.append(+10)
        elif price_action['bearish_pattern']:
            warnings.append(f"⚠️ Pattern: {price_action['pattern']}")
            adjustments.append(-10)
        
        # ===== CALCULATE ADJUSTED CONFIDENCE =====
        adjusted_confidence = base_confidence + sum(adjustments)
        adjusted_confidence = max(0, min(adjusted_confidence, 100))
        
        # Nếu sau adjustments mà confidence < threshold → reject
        if adjusted_confidence < self.min_confidence:
            return self._no_signal(
                f"Confidence sau adjustment: {adjusted_confidence}% < {self.min_confidence}%"
            )
        
        # ===== CALCULATE ENTRY PRICE, SL, TP =====
        entry_price = latest['close']
        atr = latest.get('atr', latest['close'] * 0.02)
        
        # Stop Loss: dựa vào ATR và support
        support_level = sr_check.get('support_level', entry_price - atr * 2)
        stop_loss = max(support_level, entry_price - atr * 2)
        
        # Take Profit targets
        take_profit_targets = [
            entry_price + atr * 1.5,  # TP1: 1.5R
            entry_price + atr * 3.0,  # TP2: 3R
            entry_price + atr * 5.0   # TP3: 5R (moonshot)
        ]
        
        # ===== RISK/REWARD CHECK =====
        risk = entry_price - stop_loss
        reward = take_profit_targets[1] - entry_price  # Use TP2 for R:R calc
        
        if risk <= 0:
            return self._no_signal("Stop loss không hợp lệ")
        
        risk_reward = reward / risk
        
        if risk_reward < self.min_risk_reward:
            return self._no_signal(
                f"R:R ratio thấp: {risk_reward:.2f} < {self.min_risk_reward:.2f}"
            )
        
        reasons.append(f"✅ R:R ratio: {risk_reward:.2f}")
        
        # ===== DETERMINE SIGNAL STRENGTH =====
        strength = self._calculate_signal_strength(adjusted_confidence, risk_reward, warnings)
        
        # ===== POSITION SIZE MULTIPLIER =====
        position_multiplier = self._calculate_position_multiplier(
            strength,
            adjusted_confidence,
            warnings,
            market_regime
        )
        
        # ===== BUILD ENTRY SIGNAL =====
        return EntrySignal(
            should_enter=True,
            signal_type='BUY',
            confidence=int(adjusted_confidence),
            strength=strength,
            position_size_multiplier=position_multiplier,
            reasons=reasons,
            warnings=warnings,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_targets=take_profit_targets
        )
    
    # ========================================================================
    # HELPER METHODS - FILTERS
    # ========================================================================
    
    def _check_trend_alignment(self, df: pd.DataFrame, signal_type: str) -> Dict:
        """
        Check xem signal có align với trend không
        
        Trend = EMA20 vs EMA50 vs EMA200
        """
        if len(df) < 200:
            return {'aligned': True, 'reason': 'Chưa đủ data để check trend', 'strength': 50}
        
        ema20 = df['close'].ewm(span=20).mean()
        ema50 = df['close'].ewm(span=50).mean()
        ema200 = df['close'].ewm(span=200).mean()
        
        latest_price = df['close'].iloc[-1]
        latest_ema20 = ema20.iloc[-1]
        latest_ema50 = ema50.iloc[-1]
        latest_ema200 = ema200.iloc[-1]
        
        if signal_type == 'BUY':
            # Perfect alignment: Price > EMA20 > EMA50 > EMA200
            perfect = (latest_price > latest_ema20 > latest_ema50 > latest_ema200)
            good = (latest_price > latest_ema20 > latest_ema50)
            ok = (latest_price > latest_ema20)
            
            if perfect:
                strength = 100
                return {'aligned': True, 'reason': 'Perfect uptrend', 'strength': strength}
            elif good:
                strength = 75
                return {'aligned': True, 'reason': 'Strong uptrend', 'strength': strength}
            elif ok:
                strength = 50
                return {'aligned': True, 'reason': 'Short-term uptrend', 'strength': strength}
            else:
                return {'aligned': False, 'reason': 'Downtrend or sideway', 'strength': 0}
        
        return {'aligned': True, 'reason': 'Unknown signal type', 'strength': 50}
    
    def _check_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Check vị trí giá so với support/resistance
        
        Support: Low của 20 ngày
        Resistance: High của 20 ngày
        """
        if len(df) < 20:
            return {
                'near_support': False,
                'too_close_to_resistance': False,
                'support_level': 0,
                'resistance_level': 0,
                'distance_to_support': 0,
                'distance_to_resistance': 0
            }
        
        support = df['low'].rolling(20).min().iloc[-1]
        resistance = df['high'].rolling(20).max().iloc[-1]
        
        distance_to_support = ((current_price - support) / support) * 100
        distance_to_resistance = ((resistance - current_price) / current_price) * 100
        
        # Near support = trong vòng 3%
        near_support = distance_to_support <= 3
        
        # Too close to resistance = trong vòng 2%
        too_close = distance_to_resistance <= 2
        
        return {
            'near_support': near_support,
            'too_close_to_resistance': too_close,
            'support_level': support,
            'resistance_level': resistance,
            'distance_to_support': distance_to_support,
            'distance_to_resistance': distance_to_resistance
        }
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> Dict:
        """
        Check xem volume có confirm signal không
        
        Volume >= 1.2x average = confirmed
        """
        if len(df) < 20:
            return {'confirmed': True, 'reason': 'Chưa đủ data', 'surge': False}
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        if avg_volume == 0:
            return {'confirmed': True, 'reason': 'Volume data invalid', 'surge': False}
        
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio >= 1.5:
            return {'confirmed': True, 'reason': f'Volume surge {volume_ratio:.1f}x', 'surge': True}
        elif volume_ratio >= 1.2:
            return {'confirmed': True, 'reason': f'Volume tăng {volume_ratio:.1f}x', 'surge': False}
        else:
            return {'confirmed': False, 'reason': f'Volume thấp {volume_ratio:.1f}x', 'surge': False}
    
    def _check_volatility(self, df: pd.DataFrame) -> Dict:
        """
        Check volatility (ATR/Price)
        
        < 2%: Too low (no momentum)
        2-3%: Optimal
        > 4%: Too high (risky)
        """
        latest = df.iloc[-1]
        atr = latest.get('atr', 0)
        price = latest['close']
        
        if price == 0:
            return {'too_high': False, 'optimal': True, 'value': 0}
        
        volatility = (atr / price) * 100
        
        if volatility > 4:
            return {'too_high': True, 'optimal': False, 'value': volatility}
        elif 2 <= volatility <= 3:
            return {'too_high': False, 'optimal': True, 'value': volatility}
        else:
            return {'too_high': False, 'optimal': False, 'value': volatility}
    
    def _check_rsi(self, df: pd.DataFrame) -> Dict:
        """
        Check RSI
        
        > 70: Overbought
        30-70: Optimal
        < 30: Oversold (for BUY, this is good)
        """
        if 'rsi' not in df.columns:
            return {'overbought': False, 'optimal': True, 'value': 50}
        
        rsi = df['rsi'].iloc[-1]
        
        if pd.isna(rsi):
            return {'overbought': False, 'optimal': True, 'value': 50}
        
        if rsi > 70:
            return {'overbought': True, 'optimal': False, 'value': rsi}
        elif 30 <= rsi <= 60:
            return {'overbought': False, 'optimal': True, 'value': rsi}
        else:
            return {'overbought': False, 'optimal': False, 'value': rsi}
    
    def _check_price_action(self, df: pd.DataFrame) -> Dict:
        """
        Check candlestick patterns (simplified)
        """
        if len(df) < 3:
            return {'bullish_pattern': False, 'bearish_pattern': False, 'pattern': 'None'}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish engulfing
        if (prev['close'] < prev['open'] and  # Prev bearish
            latest['close'] > latest['open'] and  # Current bullish
            latest['close'] > prev['open'] and
            latest['open'] < prev['close']):
            return {'bullish_pattern': True, 'bearish_pattern': False, 'pattern': 'Bullish Engulfing'}
        
        # Hammer (at support)
        body = abs(latest['close'] - latest['open'])
        lower_shadow = latest['open'] - latest['low'] if latest['close'] > latest['open'] else latest['close'] - latest['low']
        
        if lower_shadow > body * 2:
            return {'bullish_pattern': True, 'bearish_pattern': False, 'pattern': 'Hammer'}
        
        # Bearish patterns
        if (prev['close'] > prev['open'] and
            latest['close'] < latest['open'] and
            latest['close'] < prev['open'] and
            latest['open'] > prev['close']):
            return {'bullish_pattern': False, 'bearish_pattern': True, 'pattern': 'Bearish Engulfing'}
        
        return {'bullish_pattern': False, 'bearish_pattern': False, 'pattern': 'None'}
    
    # ========================================================================
    # SCORING & DECISION
    # ========================================================================
    
    def _calculate_signal_strength(self,
                                   confidence: int,
                                   risk_reward: float,
                                   warnings: list) -> SignalStrength:
        """Tính độ mạnh của signal"""
        
        # Base score
        score = confidence / 20  # 0-5
        
        # Bonus for high R:R
        if risk_reward >= 3:
            score += 1
        elif risk_reward >= 2.5:
            score += 0.5
        
        # Penalty for warnings
        score -= len(warnings) * 0.5
        
        # Classify
        if score >= 5:
            return SignalStrength.VERY_STRONG
        elif score >= 4:
            return SignalStrength.STRONG
        elif score >= 3:
            return SignalStrength.MODERATE
        elif score >= 2:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _calculate_position_multiplier(self,
                                      strength: SignalStrength,
                                      confidence: int,
                                      warnings: list,
                                      market_regime: Optional[Dict]) -> float:
        """
        Tính multiplier cho position size
        
        Returns:
            0.3 - 1.5
        """
        # Base multiplier by strength
        base_multipliers = {
            SignalStrength.VERY_STRONG: 1.3,
            SignalStrength.STRONG: 1.1,
            SignalStrength.MODERATE: 1.0,
            SignalStrength.WEAK: 0.7,
            SignalStrength.VERY_WEAK: 0.5
        }
        
        multiplier = base_multipliers.get(strength, 1.0)
        
        # Adjust by market regime
        if market_regime:
            regime = market_regime.get('regime', 'SIDEWAYS')
            if regime == 'BULL':
                multiplier *= 1.1
            elif regime == 'SIDEWAYS':
                multiplier *= 0.9
        
        # Penalize warnings
        multiplier -= len(warnings) * 0.1
        
        # Clamp
        return max(0.3, min(multiplier, 1.5))
    
    def _no_signal(self, reason: str) -> EntrySignal:
        """Return no signal"""
        return EntrySignal(
            should_enter=False,
            signal_type='HOLD',
            confidence=0,
            strength=SignalStrength.NO_SIGNAL,
            position_size_multiplier=0.0,
            reasons=[],
            warnings=[reason],
            entry_price=0,
            stop_loss=0,
            take_profit_targets=[]
        )
    
    def format_signal_message(self, signal: EntrySignal, symbol: str) -> str:
        """Format signal thành message đẹp"""
        
        if not signal.should_enter:
            return f"⏭️ **{symbol}** - Không vào lệnh\n" \
                   f"Lý do: {', '.join(signal.warnings)}"
        
        msg = f"🎯 **{symbol}** - {signal.signal_type}\n"
        msg += f"💪 Strength: {signal.strength.name}\n"
        msg += f"🎲 Confidence: {signal.confidence}%\n"
        msg += f"📊 Position Size: {signal.position_size_multiplier:.1f}x\n\n"
        
        msg += f"💰 **Entry:** {signal.entry_price:,.0f} VNĐ\n"
        msg += f"🛑 **Stop Loss:** {signal.stop_loss:,.0f} VNĐ "
        msg += f"({((signal.stop_loss - signal.entry_price)/signal.entry_price * 100):+.1f}%)\n\n"
        
        msg += f"🎯 **Take Profit:**\n"
        for i, tp in enumerate(signal.take_profit_targets, 1):
            tp_pct = ((tp - signal.entry_price) / signal.entry_price) * 100
            msg += f"  TP{i}: {tp:,.0f} VNĐ (+{tp_pct:.1f}%)\n"
        
        if signal.reasons:
            msg += f"\n✅ **Reasons:**\n"
            for reason in signal.reasons:
                msg += f"  • {reason}\n"
        
        if signal.warnings:
            msg += f"\n⚠️ **Warnings:**\n"
            for warning in signal.warnings:
                msg += f"  • {warning}\n"
        
        return msg


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from data_loader import load_data
    from ml_signals import MLSignalGenerator
    from features import add_ml_features
    
    print("\n" + "="*70)
    print("🧪 TESTING IMPROVED ENTRY LOGIC")
    print("="*70 + "\n")
    
    # Test với 1 mã
    symbol = 'VNM'
    df = load_data(symbol, 200)
    df = add_ml_features(df)
    
    # Get ML signal
    ml_gen = MLSignalGenerator()
    ml_signal = ml_gen.analyze(df)
    
    print(f"📊 ML Signal: {ml_signal['signal']} ({ml_signal['confidence']}%)")
    
    # Analyze entry
    entry_logic = ImprovedEntryLogic(
        min_confidence=60,
        min_risk_reward=2.0,
        require_trend_alignment=True,
        require_volume_confirmation=False  # Relax for testing
    )
    
    signal = entry_logic.analyze_entry(df, ml_signal)
    
    # Print result
    print("\n" + "="*70)
    message = entry_logic.format_signal_message(signal, symbol)
    print(message)
    print("="*70)