# -*- coding: utf-8 -*-
"""
market_regime.py - Market Regime Detection
Phát hiện tình trạng thị trường để quyết định có nên trade hay không
"""

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from src.data.loader import load_data
from src.config.trading_config import get_config
from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

config = get_config()
logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore

    HMM_AVAILABLE = True

    # Suppress hmmlearn convergence warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="hmmlearn")
except ImportError:  # pragma: no cover - optional dependency
    HMM_AVAILABLE = False
    logger.info("hmmlearn not installed. HMM-based regime detection disabled.")


class MarketRegimeAnalyzer:
    """
    Phân tích tình trạng thị trường (Bull/Bear/Sideways/High Volatility)
    để quyết định có nên trade hay không
    """

    def __init__(
        self,
        bear_threshold=-0.03,  # -3% trong 1 tuần = bear
        high_volatility_threshold=0.03,  # ATR/Price > 3% = high vol
        trend_period=50,
    ):
        self.bear_threshold = bear_threshold
        self.high_volatility_threshold = high_volatility_threshold
        self.trend_period = trend_period
        self.end_date = config.data.end_date
        self.start_date = config.data.start_date

    def analyze_market_regime(self) -> Dict:
        """
        Phân tích tình trạng thị trường VNINDEX

        Returns:
            dict: {
                'regime': 'BULL' | 'BEAR' | 'SIDEWAYS' | 'HIGH_VOLATILITY',
                'tradeable': True/False,
                'confidence': 0-100,
                'details': {...}
            }
        """
        try:
            # Load VNINDEX data
            vnindex = load_data(
                symbol="VNINDEX",
                start_date=self.start_date,
                end_date=self.end_date,
                resolution="1D",
                data_type="index",
            )

            if vnindex.empty or len(vnindex) < 50:
                logger.warning(f"Không đủ dữ liệu VNINDEX: cần ít nhất 50, có {len(vnindex)}")
                return self._default_regime()

            # Tính các chỉ số
            latest = vnindex.iloc[-1]

            # 1. Weekly change (5 trading days)
            weekly_change = self._calculate_weekly_change(vnindex)

            # 2. Trend analysis (SMA)
            trend_direction, trend_strength = self._analyze_trend(vnindex)

            # 3. Volatility
            volatility = self._calculate_volatility(vnindex)

            # 4. Market breadth (nếu có dữ liệu)
            # breadth_score = self._analyze_market_breadth()

            details = {
                "weekly_change": weekly_change,
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "volatility": volatility,
                "vnindex_price": latest["close"],
                "sma20": vnindex["close"].rolling(20).mean().iloc[-1],
                "sma50": vnindex["close"].rolling(50).mean().iloc[-1],
            }

            # Determine regime
            regime = self._determine_regime(
                weekly_change, trend_direction, trend_strength, volatility
            )

            hmm_info = self._detect_regime_hmm(vnindex)
            if hmm_info:
                details["hmm_state"] = hmm_info["state"]
                details["hmm_regime"] = hmm_info["regime"]
                details["hmm_confidence"] = hmm_info["confidence"]
                details["hmm_probabilities"] = hmm_info["probabilities"]
                details["hmm_state_means"] = hmm_info["state_means"]
                details["hmm_state_volatility"] = hmm_info["state_volatility"]

                if hmm_info["confidence"] >= 0.6 and hmm_info["regime"] != regime:
                    details["regime_before_hmm"] = regime
                    regime = hmm_info["regime"]

            # Tradeable decision
            tradeable = self._is_tradeable(regime, volatility, weekly_change)

            # Confidence score
            confidence = self._calculate_confidence(regime, trend_strength, volatility)

            result = {
                "regime": regime,
                "tradeable": tradeable,
                "confidence": confidence,
                "details": details,
                "message": self._generate_message(regime, tradeable, details),
            }

            logger.info(f"Market Regime: {regime} | Tradeable: {tradeable}")
            return result

        except Exception:
            logger.error("Lỗi phân tích market regime")
            return self._default_regime()

    def _calculate_weekly_change(self, df: pd.DataFrame) -> float:
        """Tính % thay đổi trong 1 tuần (5 ngày giao dịch)"""
        if len(df) < 6:
            return 0.0

        current_close = safe_get_latest(df, "close", 0)
        week_ago_close = df["close"].iloc[-6]

        change = (current_close / week_ago_close - 1) * 100
        return change

    def _analyze_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Phân tích xu hướng dựa trên SMA

        Returns:
            direction: 'UP' | 'DOWN' | 'SIDEWAYS'
            strength: 0-100
        """
        sma20 = df["close"].rolling(20).mean()
        sma50 = df["close"].rolling(50).mean()
        current_close = safe_get_latest(df, "close", 0)

        sma20_val = sma20.iloc[-1]
        sma50_val = sma50.iloc[-1]

        # Direction
        if sma20_val > sma50_val and current_close > sma20_val:
            direction = "UP"
            # Strength: khoảng cách giữa SMAs
            strength = ((sma20_val - sma50_val) / sma50_val) * 100
            strength = min(strength * 10, 100)  # Normalize to 0-100
        elif sma20_val < sma50_val and current_close < sma20_val:
            direction = "DOWN"
            strength = ((sma50_val - sma20_val) / sma50_val) * 100
            strength = min(strength * 10, 100)
        else:
            direction = "SIDEWAYS"
            strength = 30  # Low confidence in sideways

        return direction, strength

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        Tính volatility (normalized ATR)

        Returns:
            float: ATR/Price ratio
        """
        if "atr" not in df.columns:
            # Calculate ATR if not exists
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - df["close"].shift())
            low_close = abs(df["low"] - df["close"].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()
            df["atr"] = atr

        current_atr = safe_get_latest(df, "atr", 0)
        current_price = safe_get_latest(df, "close", 0)

        volatility = current_atr / current_price
        return volatility

    def _determine_regime(
        self,
        weekly_change: float,
        trend_direction: str,
        trend_strength: float,
        volatility: float,
    ) -> str:
        """Xác định regime thị trường"""

        # High volatility override
        if volatility > self.high_volatility_threshold:
            return "HIGH_VOLATILITY"

        # Bear market
        if weekly_change < self.bear_threshold:
            return "BEAR"

        # Bull market
        if trend_direction == "UP" and trend_strength > 40:
            return "BULL"

        # Strong downtrend
        if trend_direction == "DOWN" and trend_strength > 40:
            return "BEAR"

        # Default: sideways
        return "SIDEWAYS"

    def _detect_regime_hmm(self, df: pd.DataFrame) -> Optional[Dict]:
        """Sử dụng Hidden Markov Model để phát hiện regime (Bull/Neutral/Bear)."""
        if not HMM_AVAILABLE or len(df) < 60:
            return None

        try:
            returns = df["close"].pct_change().dropna()
            if len(returns) < 50:
                return None

            returns_array = returns.values.reshape(-1, 1)

            # Configure HMM with better convergence parameters
            hmm = GaussianHMM(
                n_components=3,
                covariance_type="diag",  # Changed from "full" for better convergence
                n_iter=100,  # Reduced iterations (was 200)
                tol=1e-2,  # Relaxed tolerance for convergence
                random_state=42,
                verbose=False,
            )

            # Fit with convergence monitoring suppressed
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                hmm.fit(returns_array)

            hidden_states = hmm.predict(returns_array)
            state_probs = hmm.predict_proba(returns_array)
            state_means = hmm.means_.flatten()

            order = np.argsort(state_means)
            regime_map = {
                order[0]: "BEAR",
                order[1]: "SIDEWAYS",
                order[2]: "BULL",
            }

            current_state = hidden_states[-1]
            current_regime = regime_map.get(current_state, "SIDEWAYS")
            confidence = float(state_probs[-1, current_state])

            covariances = hmm.covars_
            if covariances.ndim == 3:
                state_vol = covariances.reshape(covariances.shape[0], -1).mean(axis=1)
            else:
                state_vol = covariances.reshape(-1)

            return {
                "state": int(current_state),
                "regime": current_regime,
                "confidence": confidence,
                "probabilities": state_probs[-1].tolist(),
                "state_means": state_means.tolist(),
                "state_volatility": state_vol.tolist(),
            }
        except Exception:
            logger.debug("HMM regime detection failed")
            return None

    def _is_tradeable(self, regime: str, volatility: float, weekly_change: float) -> bool:
        """
        Quyết định có nên trade hay không

        KHÔNG TRADE khi:
        - Bear market
        - High volatility
        - Weekly change quá âm
        """

        # Rule 1: Không trade trong bear market
        if regime == "BEAR":
            return False

        # Rule 2: Không trade khi volatility quá cao
        if regime == "HIGH_VOLATILITY":
            return False

        # Rule 3: Không trade khi thị trường giảm mạnh
        if weekly_change < -5:  # -5% trong tuần
            return False

        # Rule 4: Chỉ trade trong bull hoặc sideways nhẹ
        if regime in ["BULL", "SIDEWAYS"]:
            return True

        return False

    def _calculate_confidence(self, regime: str, trend_strength: float, volatility: float) -> int:
        """Tính confidence score cho quyết định"""

        base_confidence = {
            "BULL": 80,
            "SIDEWAYS": 50,
            "BEAR": 20,
            "HIGH_VOLATILITY": 10,
        }

        confidence = base_confidence.get(regime, 50)

        # Adjust by trend strength
        if regime == "BULL":
            confidence = min(confidence + (trend_strength * 0.2), 100)

        # Penalize high volatility
        if volatility > 0.025:
            confidence -= 20

        return int(max(0, min(confidence, 100)))

    def _generate_message(self, regime: str, tradeable: bool, details: Dict) -> str:
        """Generate human-readable message"""

        if not tradeable:
            if regime == "BEAR":
                return (
                    "⛔ THỊ TRƯỜNG GIẢM ĐIỂM - KHÔNG NÊN TRADE\n"
                    f"📉 VNINDEX: {details['vnindex_price']:.2f}\n"
                    f"📊 Tuần này: {details['weekly_change']:+.2f}%"
                )

            elif regime == "HIGH_VOLATILITY":
                return (
                    "⚠️ THỊ TRƯỜNG BIẾN ĐỘNG MẠNH - RỦI RO CAO\n"
                    f"📊 Volatility: {details['volatility']*100:.2f}%\n"
                    "💡 Nên chờ ổn định hơn"
                )

            else:
                return (
                    "⏸️ THỊ TRƯỜNG KHÔNG RÕ HƯỚNG\n"
                    f"📊 Regime: {regime}\n"
                    "💡 Đợi tín hiệu rõ ràng hơn"
                )

        else:
            if regime == "BULL":
                return (
                    "✅ THỊ TRƯỜNG TÍCH CỰC - CÓ THỂ TRADE\n"
                    f"📈 VNINDEX: {details['vnindex_price']:.2f}\n"
                    f"🎯 Xu hướng: {details['trend_direction']} ({details['trend_strength']:.0f}%)"
                )

            else:  # SIDEWAYS
                return (
                    "⚡ THỊ TRƯỜNG ĐANG DAO ĐỘNG\n"
                    f"📊 VNINDEX: {details['vnindex_price']:.2f}\n"
                    "💡 Trade thận trọng, chọn mã tốt"
                )

    def _default_regime(self) -> Dict:
        """
        Default response khi không có dữ liệu

        Trả về SIDEWAYS với confidence thấp thay vì UNKNOWN
        để consistent với regime_detector.py và tránh confusion
        """
        return {
            "regime": "SIDEWAYS",
            "tradeable": False,
            "confidence": 30,  # Low confidence
            "details": {
                "reason": "Insufficient data or detection error",
                "warning": "Using default cautious regime"
            },
            "message": "⚠️ Không đủ dữ liệu - sử dụng chế độ thận trọng (SIDEWAYS)",
        }

    def get_position_multiplier(self) -> float:
        """
        Trả về multiplier cho position size dựa trên market regime

        Returns:
            float: 0.0 - 1.2
                1.2 = Bull strong (tăng 20% position)
                1.0 = Normal
                0.5 = Sideways/Cautious (giảm 50%)
                0.0 = Don't trade
        """
        regime_info = self.analyze_market_regime()

        if not regime_info["tradeable"]:
            return 0.0

        regime = regime_info["regime"]
        confidence = regime_info["confidence"]

        if regime == "BULL":
            if confidence >= 80:
                return 1.2  # Strong bull → tăng position
            else:
                return 1.0

        elif regime == "SIDEWAYS":
            return 0.7  # Giảm position trong sideway

        else:
            return 0.5  # Very cautious


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================


def check_market_before_trading() -> Tuple[bool, str]:
    """
    Helper function để check nhanh trước khi trade

    Returns:
        (can_trade, message)
    """
    analyzer = MarketRegimeAnalyzer()
    result = analyzer.analyze_market_regime()

    return result["tradeable"], result["message"]


def get_market_position_adjustment() -> float:
    """
    Helper function để lấy multiplier cho position sizing

    Returns:
        float: multiplier (0.0 - 1.2)
    """
    analyzer = MarketRegimeAnalyzer()
    return analyzer.get_position_multiplier()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test market regime analyzer
    print("\n" + "=" * 70)
    print("🧪 TESTING MARKET REGIME ANALYZER")
    print("=" * 70 + "\n")

    analyzer = MarketRegimeAnalyzer()
    result = analyzer.analyze_market_regime()

    print(f"📊 Regime: {result['regime']}")
    print(f"✅ Tradeable: {result['tradeable']}")
    print("🎯 Confidence: {result['confidence']}%")
    print(f"\n{result['message']}")
    print("\n📈 Details:")
    for key, value in result["details"].items():
        print(f"  • {key}: {value}")

    print("\n💰 Position Multiplier: {analyzer.get_position_multiplier():.2f}x")
    print("\n" + "=" * 70)
