# -*- coding: utf-8 -*-
"""
improved_entry_logic.py - Enhanced Entry Signal Logic
Cải thiện logic vào lệnh với nhiều điều kiện hơn
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import pandas as pd

# Import utilities
from src.utils.indicators import IndicatorUtils, StopLossCalculator
from src.utils.validation import DataValidator
from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

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

    def __init__(
        self,
        min_confidence: int = 60,
        min_risk_reward: float = 2.0,
        support_distance_percent: float = 3.0,
        require_trend_alignment: bool = True,
        require_volume_confirmation: bool = True,
        portfolio_manager=None,
    ):
        """
        Args:
            min_confidence: Confidence tối thiểu để vào lệnh
            min_risk_reward: R:R ratio tối thiểu
            support_distance_percent: Khoảng cách tối đa đến support (%)
            require_trend_alignment: Yêu cầu phải theo trend
            require_volume_confirmation: Yêu cầu volume confirm
            portfolio_manager: Portfolio manager for context-aware decisions
        """
        self.min_confidence = min_confidence
        self.base_min_confidence = min_confidence  # Store original for dynamic adjustment
        self.min_risk_reward = min_risk_reward
        self.support_distance_percent = support_distance_percent
        self.require_trend_alignment = require_trend_alignment
        self.require_volume_confirmation = require_volume_confirmation
        self.portfolio_manager = portfolio_manager

    def _validate_initial_signal(
        self, df: pd.DataFrame, ml_signal: Optional[Dict]
    ) -> tuple[bool, str, float, float]:
        """
        Validate initial data and ML signal
        ENHANCED: Allow fallback to technical analysis when ML signal is None

        Returns:
            (is_valid, signal_type, base_confidence, current_price) or
            (False, reason, 0, 0) if invalid
        """
        try:
            DataValidator.validate_dataframe(df, min_rows=50)
        except Exception as e:
            return (False, f"Data validation failed: {str(e)}", 0, 0)

        # Use safe access instead of df.iloc[-1]
        from utils.dataframe_utils import safe_get_latest

        close_price = safe_get_latest(df, "close", 0)

        # ENHANCEMENT: Fallback to technical analysis if ML signal is None
        if ml_signal is None:
            logger.debug("ML signal is None - using technical analysis fallback")
            # Use technical indicators to generate a fallback signal
            base_confidence = self._calculate_technical_confidence(df)

            # Only proceed if technical confidence is reasonable
            if base_confidence < 40:  # Lower threshold for technical-only signals
                return (False, f"Technical confidence thấp ({base_confidence}%)", 0, 0)

            # Determine signal type from technical analysis
            signal_type = self._get_technical_signal(df)
            if signal_type != "BUY":
                return (False, f"Technical signal = {signal_type}", 0, 0)

            return (True, signal_type, base_confidence, close_price)

        signal_type = ml_signal.get("signal", "HOLD")
        base_confidence = ml_signal.get("confidence", 0)

        # Skip if not BUY signal
        if signal_type != "BUY":
            return (False, f"Signal = {signal_type}", 0, 0)

        # Skip if confidence low
        if base_confidence < self.min_confidence:
            return (False, f"Confidence thấp ({base_confidence}%)", 0, 0)

        return (True, signal_type, base_confidence, close_price)

    def _run_all_filters(
        self,
        df: pd.DataFrame,
        signal_type: str,
        current_price: float,
        market_regime: Optional[Dict],
    ) -> tuple[bool, list, list, list]:
        """
        Run all entry filters

        Returns:
            (passed, reasons, warnings, adjustments)
        """
        reasons = []
        warnings = []
        adjustments = []

        # FILTER 1: MARKET REGIME
        if market_regime and not market_regime.get("tradeable", True):
            return (
                False,
                [],
                [],
                [],
            )

        # FILTER 2: TREND ALIGNMENT
        trend_check = self._check_trend_alignment(df, signal_type)
        if not trend_check["aligned"]:
            if self.require_trend_alignment:
                return (False, [], [], [])
            else:
                warnings.append(f"⚠️ Trend: {trend_check['reason']}")
                adjustments.append(-10)
        else:
            reasons.append(f"✅ Trend: {trend_check['reason']}")
            if trend_check["strength"] > 50:
                adjustments.append(+5)

        # FILTER 3: SUPPORT/RESISTANCE
        sr_check = self._check_support_resistance(df, current_price)
        if sr_check["too_close_to_resistance"]:
            warnings.append(f"⚠️ Gần resistance: {sr_check['distance_to_resistance']:.1f}%")
            adjustments.append(-15)
        elif sr_check["near_support"]:
            reasons.append(f"✅ Gần support (+{sr_check['distance_to_support']:.1f}%)")
            adjustments.append(+10)

        # FILTER 4: VOLUME CONFIRMATION
        volume_check = self._check_volume_confirmation(df)
        if not volume_check["confirmed"]:
            if self.require_volume_confirmation:
                return (False, [], [], [])
            else:
                warnings.append(f"⚠️ Volume: {volume_check['reason']}")
                adjustments.append(-10)
        else:
            reasons.append(f"✅ Volume: {volume_check['reason']}")
            if volume_check["surge"]:
                adjustments.append(+5)

        # FILTER 5: VOLATILITY CHECK
        volatility_check = self._check_volatility(df)
        if volatility_check["too_high"]:
            warnings.append(f"⚠️ Volatility cao: {volatility_check['value']:.2f}%")
            adjustments.append(-15)
        elif volatility_check["optimal"]:
            reasons.append("✅ Volatility vừa phải")
            adjustments.append(+5)

        # FILTER 6: RSI CHECK
        rsi_check = self._check_rsi(df)
        if rsi_check["overbought"]:
            warnings.append(f"⚠️ RSI overbought: {rsi_check['value']:.1f}")
            adjustments.append(-10)
        elif rsi_check["optimal"]:
            reasons.append(f"✅ RSI: {rsi_check['value']:.1f}")
            adjustments.append(+5)

        # FILTER 7: PRICE ACTION
        price_action = self._check_price_action(df)
        if price_action["bullish_pattern"]:
            reasons.append(f"✅ Pattern: {price_action['pattern']}")
            adjustments.append(+10)
        elif price_action["bearish_pattern"]:
            warnings.append(f"⚠️ Pattern: {price_action['pattern']}")
            adjustments.append(-10)

        # FILTER 8: SECTOR STRENGTH
        sector_strength_check = self._check_sector_strength(df, market_regime)
        if sector_strength_check["is_leading"]:
            reasons.append(f"✅ Ngành dẫn dắt ({sector_strength_check['sector_perf']:.1f}%)")
            adjustments.append(+10)
        elif sector_strength_check["is_lagging"]:
            warnings.append(f"⚠️ Ngành yếu ({sector_strength_check['sector_perf']:.1f}%)")
            adjustments.append(-15)

        # FILTER 9: PORTFOLIO CORRELATION (NEW)
        correlation_check = self._check_portfolio_correlation(df, getattr(self, '_current_symbol', None))
        if correlation_check["too_high"]:
            warnings.append(f"⚠️ Correlation cao với portfolio: {correlation_check['max_correlation']:.2f}")
            adjustments.append(-20)  # Penalty lớn cho high correlation
        elif correlation_check["good_diversification"]:
            reasons.append(f"✅ Đa dạng hóa tốt (corr: {correlation_check['max_correlation']:.2f})")
            adjustments.append(+5)

        return (True, reasons, warnings, adjustments)

    def _calculate_prices_and_risk(
        self, df: pd.DataFrame, entry_price: float, sr_check: Dict
    ) -> tuple[bool, str, float, float, list, float]:
        """
        Calculate entry price, stop loss, take profit targets, and risk/reward

        Returns:
            (success, error_msg, stop_loss, reward, take_profit_targets,
             risk_reward)
        """
        atr = IndicatorUtils.get_atr(df)
        support_level = sr_check.get("support_level", None)

        # Calculate stop loss
        try:
            stop_loss, sl_reason = StopLossCalculator.calculate_stop_loss(
                entry_price=entry_price,
                atr=atr,
                support_level=support_level,
                atr_multiplier=2.0,
            )
            logger.debug(f"Stop loss calculated: {stop_loss:.0f} ({sl_reason})")
        except ValueError as e:
            return (False, f"Stop loss calculation failed: {str(e)}", 0, 0, [], 0)

        # Calculate take profit targets
        try:
            take_profit_targets = StopLossCalculator.calculate_take_profit_targets(
                entry_price=entry_price, atr=atr, risk_reward_ratios=[1.5, 3.0, 5.0]
            )
        except ValueError as e:
            return (False, f"Take profit calculation failed: {str(e)}", 0, 0, [], 0)

        # Risk/Reward check
        risk = entry_price - stop_loss
        if risk <= 0:
            error_msg = (
                f"Risk calculation error: risk={risk:.0f} "
                f"(entry={entry_price:.0f}, sl={stop_loss:.0f})"
            )
            return (False, error_msg, 0, 0, [], 0)

        reward = take_profit_targets[1] - entry_price  # Use TP2
        if reward <= 0:
            return (False, f"Reward không hợp lệ: {reward:.0f}", 0, 0, [], 0)

        risk_reward = reward / risk
        if risk_reward < self.min_risk_reward:
            error_msg = f"R:R ratio thấp: {risk_reward:.2f} < " f"{self.min_risk_reward:.2f}"
            return (False, error_msg, 0, 0, [], 0)

        return (True, "", stop_loss, reward, take_profit_targets, risk_reward)

    def analyze_entry(
        self, df: pd.DataFrame, ml_signal: Dict, market_regime: Optional[Dict] = None, symbol: Optional[str] = None
    ) -> EntrySignal:
        """
        Phân tích đầy đủ để quyết định có nên vào lệnh

        Args:
            df: DataFrame với OHLCV + indicators
            ml_signal: Signal từ ML model
            market_regime: Thông tin market regime (optional)

        Returns:
            EntrySignal object với đầy đủ thông tin
        """
        # ENHANCEMENT: Adjust thresholds dynamically based on market regime
        self._adjust_thresholds_for_market(market_regime)

        # Step 1: Validate initial signal
        (
            is_valid,
            signal_or_reason,
            base_confidence,
            current_price,
        ) = self._validate_initial_signal(df, ml_signal)
        if not is_valid:
            return self._no_signal(signal_or_reason)

        signal_type = signal_or_reason

        # Step 2: Run all filters
        # Store symbol temporarily for correlation check
        self._current_symbol = symbol
        passed, reasons, warnings, adjustments = self._run_all_filters(
            df, signal_type, current_price, market_regime
        )
        if not passed:
            regime_name = market_regime.get("regime", "UNKNOWN") if market_regime else "N/A"
            return self._no_signal(f"Thị trường: {regime_name}")

        # Step 3: Calculate adjusted confidence
        adjusted_confidence = base_confidence + sum(adjustments)
        adjusted_confidence = max(0, min(adjusted_confidence, 100))

        if adjusted_confidence < self.min_confidence:
            return self._no_signal(
                f"Confidence sau adjustment: {adjusted_confidence}% < " f"{self.min_confidence}%"
            )

        # Step 4: Calculate prices and risk/reward
        # Use safe access instead of df.iloc[-1]
        close_price = safe_get_latest(df, "close", 0)
        entry_price = DataValidator.validate_price(close_price, "entry_price")
        sr_check = self._check_support_resistance(df, current_price)

        (
            success,
            error_msg,
            stop_loss,
            reward,
            take_profit_targets,
            risk_reward,
        ) = self._calculate_prices_and_risk(df, entry_price, sr_check)
        if not success:
            return self._no_signal(error_msg)

        reasons.append(f"✅ R:R ratio: {risk_reward:.2f}")

        # Step 5: Determine signal strength and position multiplier
        strength = self._calculate_signal_strength(adjusted_confidence, risk_reward, warnings)
        position_multiplier = self._calculate_position_multiplier(
            strength, adjusted_confidence, warnings, market_regime
        )

        # Step 6: Build entry signal
        return EntrySignal(
            should_enter=True,
            signal_type="BUY",
            confidence=int(adjusted_confidence),
            strength=strength,
            position_size_multiplier=position_multiplier,
            reasons=reasons,
            warnings=warnings,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_targets=take_profit_targets,
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
            return {
                "aligned": True,
                "reason": "Chưa đủ data để check trend",
                "strength": 50,
            }

        ema20 = df["close"].ewm(span=20).mean()
        ema50 = df["close"].ewm(span=50).mean()
        ema200 = df["close"].ewm(span=200).mean()

        latest_price = safe_get_latest(df, "close", 0)
        latest_ema20 = ema20.iloc[-1]
        latest_ema50 = ema50.iloc[-1]
        latest_ema200 = ema200.iloc[-1]

        if signal_type == "BUY":
            # Perfect alignment: Price > EMA20 > EMA50 > EMA200
            perfect = latest_price > latest_ema20 > latest_ema50 > latest_ema200
            good = latest_price > latest_ema20 > latest_ema50
            ok = latest_price > latest_ema20

            if perfect:
                strength = 100
                return {
                    "aligned": True,
                    "reason": "Perfect uptrend",
                    "strength": strength,
                }
            elif good:
                strength = 75
                return {
                    "aligned": True,
                    "reason": "Strong uptrend",
                    "strength": strength,
                }
            elif ok:
                strength = 50
                return {
                    "aligned": True,
                    "reason": "Short-term uptrend",
                    "strength": strength,
                }
            else:
                return {
                    "aligned": False,
                    "reason": "Downtrend or sideway",
                    "strength": 0,
                }

        return {"aligned": True, "reason": "Unknown signal type", "strength": 50}

    def _check_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Check vị trí giá so với support/resistance

        Support: Low của 20 ngày
        Resistance: High của 20 ngày
        """
        if len(df) < 20:
            return {
                "near_support": False,
                "too_close_to_resistance": False,
                "support_level": 0,
                "resistance_level": 0,
                "distance_to_support": 0,
                "distance_to_resistance": 0,
            }

        support = safe_rolling_operation(df, "low", 20, "min", 0)
        resistance = safe_rolling_operation(df, "high", 20, "max", 0)

        distance_to_support = ((current_price - support) / support) * 100
        distance_to_resistance = ((resistance - current_price) / current_price) * 100

        # Near support = trong vòng config threshold
        near_support = distance_to_support <= self.support_distance_percent

        # Too close to resistance = trong vòng 2%
        too_close = distance_to_resistance <= 2

        return {
            "near_support": near_support,
            "too_close_to_resistance": too_close,
            "support_level": support,
            "resistance_level": resistance,
            "distance_to_support": distance_to_support,
            "distance_to_resistance": distance_to_resistance,
        }

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)

        OBV measures buying/selling pressure by adding volume on up days
        and subtracting on down days.

        Returns:
            Series with OBV values
        """
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])

        return pd.Series(obv, index=df.index)

    def _check_volume_confirmation(self, df: pd.DataFrame) -> Dict:
        """
        ENHANCED: Check volume confirmation với multiple indicators

        Checks:
        1. Volume ratio (current vs average)
        2. Volume trend (5-day vs 20-day MA)
        3. OBV (On-Balance Volume) - accumulation/distribution

        Returns:
            Dict with detailed volume analysis
        """
        if len(df) < 20:
            return {
                "confirmed": True,
                "reason": "Chưa đủ data",
                "surge": False,
                "obv_bullish": True,
                "volume_trending": True,
                "confidence": 0.5,
            }

        current_volume = safe_get_latest(df, "volume", 0)
        avg_volume_20 = safe_rolling_operation(df, "volume", 20, "mean", 0)

        if avg_volume_20 == 0:
            return {
                "confirmed": True,
                "reason": "Volume data invalid",
                "surge": False,
                "obv_bullish": True,
                "volume_trending": True,
                "confidence": 0.5,
            }

        # ============================================================
        # 1. VOLUME RATIO (existing logic)
        # ============================================================
        volume_ratio = current_volume / avg_volume_20

        # ============================================================
        # 2. VOLUME TREND (NEW)
        # ============================================================
        avg_volume_5 = safe_rolling_operation(df, "volume", 5, "mean", 0)
        volume_trending_up = avg_volume_5 > avg_volume_20

        # ============================================================
        # 3. OBV - ACCUMULATION/DISTRIBUTION (NEW)
        # ============================================================
        obv = self._calculate_obv(df)

        # Calculate OBV slope over last 5 days
        if len(obv) >= 5:
            obv_recent = obv.iloc[-5:]
            obv_slope = (obv_recent.iloc[-1] - obv_recent.iloc[0]) / 5

            # Also check OBV moving average
            obv_ma_5 = obv.rolling(5).mean().iloc[-1]
            obv_ma_20 = obv.rolling(20).mean().iloc[-1]

            obv_bullish = (obv_slope > 0) and (obv_ma_5 > obv_ma_20)
        else:
            obv_bullish = True  # Default to True if not enough data

        # ============================================================
        # 4. COMBINE ALL SIGNALS
        # ============================================================

        # Calculate confidence score (0-1)
        confidence_score = 0.0

        # Volume ratio contributes 40%
        if volume_ratio >= 1.5:
            confidence_score += 0.4
        elif volume_ratio >= 1.2:
            confidence_score += 0.3
        elif volume_ratio >= 1.0:
            confidence_score += 0.2

        # Volume trend contributes 30%
        if volume_trending_up:
            confidence_score += 0.3

        # OBV contributes 30%
        if obv_bullish:
            confidence_score += 0.3

        # Determine if confirmed (threshold: 0.6)
        confirmed = confidence_score >= 0.6

        # Generate detailed reason
        reasons = []
        if volume_ratio >= 1.5:
            reasons.append(f"Volume surge {volume_ratio:.1f}x")
        elif volume_ratio >= 1.2:
            reasons.append(f"Volume tăng {volume_ratio:.1f}x")
        else:
            reasons.append(f"Volume {volume_ratio:.1f}x")

        if volume_trending_up:
            reasons.append("Volume trending up")
        else:
            reasons.append("Volume trending down")

        if obv_bullish:
            reasons.append("OBV bullish (accumulation)")
        else:
            reasons.append("OBV bearish (distribution)")

        return {
            "confirmed": confirmed,
            "reason": " | ".join(reasons),
            "surge": volume_ratio >= 1.5,
            "volume_ratio": volume_ratio,
            "volume_trending": volume_trending_up,
            "obv_bullish": obv_bullish,
            "confidence": confidence_score,
        }

    def _check_volatility(self, df: pd.DataFrame) -> Dict:
        """
        Check volatility (ATR/Price)

        < 2%: Too low (no momentum)
        2-3%: Optimal
        > 4%: Too high (risky)
        """
        # Use safe access instead of df.iloc[-1]
        atr = safe_get_latest(df, "atr", 0)
        price = safe_get_latest(df, "close", 0)

        if price == 0:
            return {"too_high": False, "optimal": True, "value": 0}

        volatility = (atr / price) * 100

        if volatility > 4:
            return {"too_high": True, "optimal": False, "value": volatility}
        elif 2 <= volatility <= 3:
            return {"too_high": False, "optimal": True, "value": volatility}
        else:
            return {"too_high": False, "optimal": False, "value": volatility}

    def _check_rsi(self, df: pd.DataFrame) -> Dict:
        """
        Check RSI

        > 70: Overbought
        30-70: Optimal
        < 30: Oversold (for BUY, this is good)
        """
        if "rsi" not in df.columns:
            return {"overbought": False, "optimal": True, "value": 50}

        rsi = safe_get_latest(df, "rsi", 0)

        if pd.isna(rsi):
            return {"overbought": False, "optimal": True, "value": 50}

        if rsi > 70:
            return {"overbought": True, "optimal": False, "value": rsi}
        elif 30 <= rsi <= 60:
            return {"overbought": False, "optimal": True, "value": rsi}
        else:
            return {"overbought": False, "optimal": False, "value": rsi}

    def _check_price_action(self, df: pd.DataFrame) -> Dict:
        """
        Check candlestick patterns (simplified)
        """
        if len(df) < 3:
            return {
                "bullish_pattern": False,
                "bearish_pattern": False,
                "pattern": "None",
            }

        # Use safe access instead of df.iloc[-1]
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Bullish engulfing
        if (
            prev["close"] < prev["open"]  # Prev bearish
            and latest["close"] > latest["open"]  # Current bullish
            and latest["close"] > prev["open"]
            and latest["open"] < prev["close"]
        ):
            return {
                "bullish_pattern": True,
                "bearish_pattern": False,
                "pattern": "Bullish Engulfing",
            }

        # Hammer (at support)
        body = abs(latest["close"] - latest["open"])
        lower_shadow = (
            latest["open"] - latest["low"]
            if latest["close"] > latest["open"]
            else latest["close"] - latest["low"]
        )

        if lower_shadow > body * 2:
            return {
                "bullish_pattern": True,
                "bearish_pattern": False,
                "pattern": "Hammer",
            }

        # Bearish patterns
        if (
            prev["close"] > prev["open"]
            and latest["close"] < latest["open"]
            and latest["close"] < prev["open"]
            and latest["open"] > prev["close"]
        ):
            return {
                "bullish_pattern": False,
                "bearish_pattern": True,
                "pattern": "Bearish Engulfing",
            }

        return {"bullish_pattern": False, "bearish_pattern": False, "pattern": "None"}

    def _check_sector_strength(self, df: pd.DataFrame, market_regime: Optional[Dict]) -> Dict:
        """
        Kiểm tra sức mạnh của ngành so với thị trường chung (VNINDEX).
        Sử dụng RS (Relative Strength)
        """
        if "rs" not in df.columns or df["rs"].isnull().all():
            return {"is_leading": False, "is_lagging": False, "sector_per": 0}

        # RS > 1: Cổ phiếu/ngành mạnh hơn thị trường
        # RS dốc lên: Sức mạnh đang tăng
        latest_rs = safe_get_latest(df, "rs", 0)
        rs_trend = safe_rolling_operation(df, "rs", 10, "mean", 0) > safe_rolling_operation(
            df, "rs", 30, "mean", 0
        )

        is_leading = latest_rs > 1.0 and rs_trend
        is_lagging = latest_rs < 0.95

        # Lấy performance từ market_regime nếu có
        sector_perf = 0
        if market_regime and "sector_performance" in market_regime:
            # Giả sử df có cột 'sector'
            sector = safe_get_latest(df, "sector", 0) if "sector" in df.columns else "UNKNOWN"
            sector_perf = market_regime["sector_performance"].get(sector, 0)

        return {
            "is_leading": is_leading,
            "is_lagging": is_lagging,
            "sector_perf": sector_perf,
        }

    def _check_portfolio_correlation(self, df: pd.DataFrame, symbol: Optional[str]) -> Dict:
        """
        NEW: Kiểm tra correlation với portfolio hiện tại
        
        Returns:
            Dict with correlation analysis
        """
        if not symbol or not self.portfolio_manager:
            return {
                "too_high": False,
                "good_diversification": False,
                "max_correlation": 0.0,
            }
        
        try:
            from src.risk.metrics import calculate_portfolio_correlation_risk
            
            # Lấy danh sách positions hiện tại
            positions = self.portfolio_manager.get_positions()
            if not positions or len(positions) == 0:
                return {
                    "too_high": False,
                    "good_diversification": True,  # Portfolio rỗng = diversification tốt
                    "max_correlation": 0.0,
                }
            
            # Tính correlation với portfolio
            existing_symbols = list(positions.keys())
            all_symbols = existing_symbols + [symbol]
            
            correlation_metrics = calculate_portfolio_correlation_risk(
                all_symbols,
                lookback=60,
                max_avg_correlation=0.70,
            )
            
            max_correlation = correlation_metrics.get("max_correlation", 0.0)
            avg_correlation = correlation_metrics.get("avg_correlation", 0.0)
            
            # Threshold: > 0.7 = quá cao, < 0.3 = diversification tốt
            too_high = max_correlation > 0.70
            good_diversification = max_correlation < 0.30 and avg_correlation < 0.25
            
            return {
                "too_high": too_high,
                "good_diversification": good_diversification,
                "max_correlation": max_correlation,
                "avg_correlation": avg_correlation,
            }
        except Exception as e:
            logger.warning(f"⚠️ Error checking portfolio correlation: {e}")
            return {
                "too_high": False,
                "good_diversification": False,
                "max_correlation": 0.0,
            }

    # ========================================================================
    # SCORING & DECISION
    # ========================================================================

    def _calculate_signal_strength(
        self, confidence: int, risk_reward: float, warnings: list
    ) -> SignalStrength:
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

    def _calculate_position_multiplier(
        self,
        strength: SignalStrength,
        confidence: int,
        warnings: list,
        market_regime: Optional[Dict],
    ) -> float:
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
            SignalStrength.VERY_WEAK: 0.5,
        }

        multiplier = base_multipliers.get(strength, 1.0)

        # Adjust by market regime
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            if regime == "BULL":
                multiplier *= 1.1
            elif regime == "SIDEWAYS":
                multiplier *= 0.9

        # Penalize warnings
        multiplier -= len(warnings) * 0.1

        # Clamp
        return max(0.3, min(multiplier, 1.5))

    def _adjust_thresholds_for_market(self, market_regime: Optional[Dict]):
        """
        ENHANCEMENT: Dynamically adjust confidence thresholds based on market conditions

        Logic:
        - BULL market: Lower threshold (more opportunities)
        - BEAR/HIGH_VOLATILITY: Higher threshold (more selective)
        - Consider portfolio heat
        """
        if not market_regime:
            self.min_confidence = self.base_min_confidence
            return

        regime = market_regime.get("regime", "SIDEWAYS")
        regime_confidence = market_regime.get("confidence", 50)

        # Base adjustment by regime type
        if regime == "BULL" and regime_confidence >= 70:
            # Strong bull market - can be less strict
            adjustment = -5
        elif regime == "BEAR":
            # Bear market - be more selective
            adjustment = +10
        elif regime == "HIGH_VOLATILITY":
            # High volatility - require higher confidence
            adjustment = +15
        else:
            # SIDEWAYS or unknown
            adjustment = 0

        # Portfolio heat adjustment
        if self.portfolio_manager:
            try:
                positions = self.portfolio_manager.get_positions()
                num_positions = len(positions)

                # If portfolio is getting crowded, be more selective
                if num_positions >= 8:
                    adjustment += 10
                    logger.info(
                        f"🔥 Portfolio heat: {num_positions} positions. Raising confidence threshold by +10"
                    )
                elif num_positions >= 5:
                    adjustment += 5
                    logger.info(
                        f"🔥 Portfolio heat: {num_positions} positions. Raising confidence threshold by +5"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Could not check portfolio heat: {e}")

        # Apply adjustment (with limits)
        # Allow lower bound 40 in favorable regimes to increase opportunities
        self.min_confidence = max(40, min(80, self.base_min_confidence + adjustment))

        if adjustment != 0:
            logger.info(
                f"📊 Dynamic threshold adjustment: {self.base_min_confidence} → {self.min_confidence} "
                f"(regime: {regime}, adj: {adjustment:+d})"
            )

    def _calculate_technical_confidence(self, df: pd.DataFrame) -> float:
        """
        Calculate confidence from technical indicators when ML signal is unavailable

        Uses multiple technical factors:
        - RSI position
        - MACD signal
        - Moving average alignment
        - Price action strength
        """
        if len(df) < 50:
            return 0.0

        from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

        confidence = 50.0  # Base confidence

        # RSI check
        if "rsi" in df.columns:
            rsi = safe_get_latest(df, "rsi", 50)
            if not pd.isna(rsi):
                if 30 <= rsi <= 60:  # Good range for entry
                    confidence += 10
                elif 60 < rsi <= 70:
                    confidence += 5

        # Moving average alignment
        if len(df) >= 50:
            ema20 = df["close"].ewm(span=20).mean()
            ema50 = df["close"].ewm(span=50).mean()
            current_price = safe_get_latest(df, "close", 0)
            latest_ema20 = ema20.iloc[-1]
            latest_ema50 = ema50.iloc[-1]

            if current_price > latest_ema20:
                confidence += 10
            if latest_ema20 > latest_ema50:
                confidence += 10

        # Volume confirmation
        if len(df) >= 20:
            current_volume = safe_get_latest(df, "volume", 0)
            avg_volume = safe_rolling_operation(df, "volume", 20, "mean", 0)
            if avg_volume > 0 and current_volume > avg_volume * 1.2:
                confidence += 10

        return min(confidence, 100.0)

    def _get_technical_signal(self, df: pd.DataFrame) -> str:
        """
        Determine signal type from technical analysis
        Returns: "BUY", "SELL", or "HOLD"
        """
        if len(df) < 50:
            return "HOLD"

        from utils.dataframe_utils import safe_get_latest

        current_price = safe_get_latest(df, "close", 0)
        prev_price = df["close"].iloc[-2] if len(df) >= 2 else current_price

        # Simple trend following
        if current_price > prev_price:
            return "BUY"
        elif current_price < prev_price:
            return "SELL"
        else:
            return "HOLD"

    def _no_signal(self, reason: str) -> EntrySignal:
        """Return no signal"""
        return EntrySignal(
            should_enter=False,
            signal_type="HOLD",
            confidence=0,
            strength=SignalStrength.NO_SIGNAL,
            position_size_multiplier=0.0,
            reasons=[],
            warnings=[reason],
            entry_price=0,
            stop_loss=0,
            take_profit_targets=[],
        )

    def format_signal_message(self, signal: EntrySignal, symbol: str) -> str:
        """Format signal thành message đẹp"""

        if not signal.should_enter:
            return f"⏭️ **{symbol}** - Không vào lệnh\n" f"Lý do: {', '.join(signal.warnings)}"

        msg = f"🎯 **{symbol}** - {signal.signal_type}\n"
        msg += f"💪 Strength: {signal.strength.name}\n"
        msg += f"🎲 Confidence: {signal.confidence}%\n"
        msg += f"📊 Position Size: {signal.position_size_multiplier:.1f}x\n\n"

        msg += f"💰 **Entry:** {signal.entry_price:,.0f} VNĐ\n"
        msg += f"🛑 **Stop Loss:** {signal.stop_loss:,.0f} VNĐ "
        msg += f"({((signal.stop_loss - signal.entry_price)/signal.entry_price * 100):+.1f}%)\n\n"

        msg += "🎯 **Take Profit:**\n"
        for i, tp in enumerate(signal.take_profit_targets, 1):
            tp_pct = ((tp - signal.entry_price) / signal.entry_price) * 100
            msg += f"  TP{i}: {tp:,.0f} VNĐ (+{tp_pct:.1f}%)\n"

        if signal.reasons:
            msg += "\n✅ **Reasons:**\n"
            for reason in signal.reasons:
                msg += f"  • {reason}\n"

        if signal.warnings:
            msg += "\n⚠️ **Warnings:**\n"
            for warning in signal.warnings:
                msg += f"  • {warning}\n"

        return msg


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from src.data.loader import load_data
    from src.ml.features.technical import add_ml_features
    from src.ml.signals.generator import MLSignalGenerator
    from utils.dataframe_utils import safe_get_latest

    print("\n" + "=" * 70)
    print("🧪 TESTING IMPROVED ENTRY LOGIC")
    print("=" * 70 + "\n")

    # Test với 1 mã
    symbol = "VNM"
    df = load_data(symbol, 200)
    df = add_ml_features(df)

    # Get ML signal
    ml_gen = MLSignalGenerator()
    ml_signal = ml_gen.analyze(df)

    print("📊 ML Signal: {ml_signal['signal']} ({ml_signal['confidence']}%)")

    # Analyze entry
    entry_logic = ImprovedEntryLogic(
        min_confidence=60,
        min_risk_reward=2.0,
        require_trend_alignment=True,
        require_volume_confirmation=False,  # Relax for testing
    )

    signal = entry_logic.analyze_entry(df, ml_signal)

    # Print result
    print("\n" + "=" * 70)
    message = entry_logic.format_signal_message(signal, symbol)
    print(message)
    print("=" * 70)
