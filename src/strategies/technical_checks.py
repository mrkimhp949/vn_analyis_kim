"""
Technical Checks Module

Contains all _check_* methods for technical analysis.
These perform individual technical validations and return detailed results.
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

logger = logging.getLogger(__name__)

# Vietnam market constants
VIETNAM_PRICE_LIMIT_PERCENT = 0.07  # ±7% daily limit
VN_CEILING_DISTANCE_THRESHOLD = 1.5  # 1.5% from ceiling = warning
VN_FLOOR_DISTANCE_THRESHOLD = 2.0  # 2% from floor = warning
VN_MIN_LIQUIDITY_VALUE = 2_000_000_000  # 2B VND minimum
VN_CRITICAL_LIQUIDITY_VALUE = 500_000_000  # 500M VND critical


class TechnicalChecker:
    """
    Performs technical analysis checks for entry validation.

    All methods return Dict with detailed analysis results.
    """

    def __init__(
        self,
        support_distance_percent: float = 3.0,
        portfolio_manager: Optional[object] = None,
    ):
        """
        Initialize TechnicalChecker.

        Args:
            support_distance_percent: Threshold for "near support" detection
            portfolio_manager: Optional portfolio manager for correlation checks
        """
        self.support_distance_percent = support_distance_percent
        self.portfolio_manager = portfolio_manager

        # Correlation cache
        self._correlation_cache: Optional[Dict] = None
        self._correlation_cache_time: Optional[float] = None
        self._correlation_cache_symbols: Optional[Tuple] = None
        self._correlation_cache_portfolio_hash: Optional[str] = None
        self._correlation_cache_ttl = 300  # 5 minutes

    def check_trend_alignment(self, df: pd.DataFrame, signal_type: str) -> Dict:
        """
        Check xem signal có align với trend không.

        Trend = EMA20 vs EMA50 vs EMA200 (adaptive based on available data)

        Args:
            df: DataFrame with OHLCV data
            signal_type: "BUY" or "SELL"

        Returns:
            Dict with aligned, reason, strength
        """
        data_len = len(df)

        # Tối thiểu cần 20 bars để check trend cơ bản
        if data_len < 20:
            return {
                "aligned": True,
                "reason": "Chưa đủ data để check trend (cần ít nhất 20 bars)",
                "strength": 50,
            }

        ema20 = df["close"].ewm(span=20).mean()
        latest_price = safe_get_latest(df, "close", 0)
        latest_ema20 = ema20.iloc[-1]

        # Adaptive: dùng EMA50 nếu có đủ data
        use_ema50 = data_len >= 50
        if use_ema50:
            ema50 = df["close"].ewm(span=50).mean()
            latest_ema50 = ema50.iloc[-1]
        else:
            ema50 = None
            latest_ema50 = None

        # Adaptive: dùng EMA200 nếu có đủ data
        use_ema200 = data_len >= 200
        if use_ema200:
            ema200 = df["close"].ewm(span=200).mean()
            latest_ema200 = ema200.iloc[-1]
        else:
            ema200 = None
            latest_ema200 = None

        if signal_type == "BUY":
            # Adaptive trend check based on available data
            ok = latest_price > latest_ema20

            # Check good/perfect only if we have enough data
            if use_ema50 and use_ema200:
                # Full data: Perfect alignment: Price > EMA20 > EMA50 > EMA200
                perfect = latest_price > latest_ema20 > latest_ema50 > latest_ema200
                good = latest_price > latest_ema20 > latest_ema50
            elif use_ema50:
                # Medium data: Good alignment: Price > EMA20 > EMA50
                perfect = False
                good = latest_price > latest_ema20 > latest_ema50
            else:
                # Limited data: Only check EMA20
                perfect = False
                good = False

            # Check for early reversal signals
            prev_price = df["close"].iloc[-2] if data_len >= 2 else latest_price
            prev_ema20 = ema20.iloc[-2] if len(ema20) >= 2 else latest_ema20

            # Price just crossed above EMA20 (reversal signal)
            price_cross_ema20 = prev_price <= prev_ema20 and latest_price > latest_ema20

            # EMA20 just crossed above EMA50 (trend turning) - only if we have EMA50
            ema20_cross_ema50 = False
            if use_ema50:
                prev_ema50 = ema50.iloc[-2] if len(ema50) >= 2 else latest_ema50
                ema20_cross_ema50 = prev_ema20 <= prev_ema50 and latest_ema20 > latest_ema50

            if perfect:
                strength = 100
                return {
                    "aligned": True,
                    "reason": "Perfect uptrend (EMA20>50>200)",
                    "strength": strength,
                }
            elif good:
                strength = 75
                reason = "Strong uptrend (EMA20>50)" if not use_ema200 else "Strong uptrend"
                return {
                    "aligned": True,
                    "reason": reason,
                    "strength": strength,
                }
            elif price_cross_ema20 or ema20_cross_ema50:
                # Early reversal signal - can catch trends early
                strength = 60
                reason = "Early reversal signal"
                if price_cross_ema20:
                    reason += " (Price crossed EMA20)"
                if ema20_cross_ema50:
                    reason += " (EMA20 crossed EMA50)"
                return {
                    "aligned": True,
                    "reason": reason,
                    "strength": strength,
                }
            elif ok:
                strength = 50
                data_note = f" (data: {data_len} bars)" if not use_ema50 else ""
                return {
                    "aligned": True,
                    "reason": f"Short-term uptrend{data_note}",
                    "strength": strength,
                }
            else:
                return {
                    "aligned": False,
                    "reason": "Downtrend or sideway",
                    "strength": 0,
                }

        return {"aligned": True, "reason": "Unknown signal type", "strength": 50}

    def check_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Check vị trí giá so với support/resistance.

        Support: Low của 20 ngày
        Resistance: High của 20 ngày
        Enhanced: Check if price is bouncing FROM support (reversal signal)

        Args:
            df: DataFrame with OHLCV data
            current_price: Current stock price

        Returns:
            Dict with support/resistance analysis
        """
        if len(df) < 20:
            return {
                "near_support": False,
                "bouncing_from_support": False,
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

        # Check if price is bouncing FROM support
        bouncing_from_support = False
        if near_support and len(df) >= 5:
            recent_low = safe_rolling_operation(df, "low", 5, "min", 0)
            prev_3_avg = df["close"].iloc[-4:-1].mean() if len(df) >= 4 else current_price

            # Price was near support in last 5 days
            if abs(recent_low - support) / support < 0.02:  # Within 2% of support
                # Check for sustained upward movement (1% above 3-bar average)
                if current_price > prev_3_avg * 1.01:
                    # Volume confirmation: current volume > 1.2x recent average
                    current_volume = safe_get_latest(df, "volume", 0)
                    avg_volume_5 = safe_rolling_operation(df, "volume", 5, "mean", 1)

                    if avg_volume_5 > 0 and current_volume > avg_volume_5 * 1.2:
                        bouncing_from_support = True
                        logger.debug(
                            f"✅ Support bounce detected: price {current_price:.0f} > "
                            f"3-bar avg {prev_3_avg:.0f} (+{((current_price/prev_3_avg - 1)*100):.1f}%), "
                            f"volume {current_volume/avg_volume_5:.1f}x"
                        )

        # Too close to resistance = trong vòng 2%
        too_close = distance_to_resistance <= 2

        return {
            "near_support": near_support,
            "bouncing_from_support": bouncing_from_support,
            "too_close_to_resistance": too_close,
            "support_level": support,
            "resistance_level": resistance,
            "distance_to_support": distance_to_support,
            "distance_to_resistance": distance_to_resistance,
        }

    def check_vietnam_price_limits(self, df: pd.DataFrame, current_price: float, symbol: str = "") -> Dict:
        """
        Check if price is near Vietnam market floor/ceiling limits (±7%).

        Vietnam stock market has daily price limits:
        - Ceiling (trần): +7% from reference price
        - Floor (sàn): -7% from reference price

        Args:
            df: DataFrame with OHLCV data
            current_price: Current stock price
            symbol: Stock symbol for logging

        Returns:
            Dict with near_limit, limit_type, warning, and price levels
        """
        ceiling_mult = 1 + VIETNAM_PRICE_LIMIT_PERCENT
        floor_mult = 1 - VIETNAM_PRICE_LIMIT_PERCENT

        if len(df) < 2:
            return {
                "near_limit": False,
                "limit_type": None,
                "warning": None,
                "reference_price": current_price,
                "ceiling_price": current_price * ceiling_mult,
                "floor_price": current_price * floor_mult,
            }

        reference_price = df["close"].iloc[-2]
        ceiling_price = reference_price * ceiling_mult
        floor_price = reference_price * floor_mult

        distance_to_ceiling = ((ceiling_price - current_price) / reference_price) * 100
        distance_to_floor = ((current_price - floor_price) / reference_price) * 100

        near_ceiling = distance_to_ceiling <= VN_CEILING_DISTANCE_THRESHOLD
        near_floor = distance_to_floor <= VN_FLOOR_DISTANCE_THRESHOLD

        result = {
            "near_limit": False,
            "limit_type": None,
            "warning": None,
            "reference_price": reference_price,
            "ceiling_price": ceiling_price,
            "floor_price": floor_price,
            "distance_to_ceiling": distance_to_ceiling,
            "distance_to_floor": distance_to_floor,
        }

        if near_ceiling:
            result["near_limit"] = True
            result["limit_type"] = "CEILING"
            symbol_tag = f"[{symbol}] " if symbol else ""
            result["warning"] = (
                f"{symbol_tag}Giá gần trần ({current_price:,.0f} / {ceiling_price:,.0f}), "
                f"chỉ còn {distance_to_ceiling:.2f}% - RỦI RO CAO, không thể mua thêm"
            )
            logger.warning(f"🚫 {result['warning']}")

        elif near_floor:
            result["near_limit"] = True
            result["limit_type"] = "FLOOR"
            symbol_tag = f"[{symbol}] " if symbol else ""
            result["warning"] = (
                f"{symbol_tag}Giá gần sàn ({current_price:,.0f} / {floor_price:,.0f}), "
                f"chỉ còn {distance_to_floor:.2f}% - Có thể là panic selling"
            )
            logger.warning(f"⚠️ {result['warning']}")

        return result

    def check_vietnam_market_liquidity(self, df: pd.DataFrame) -> Dict:
        """
        Check minimum liquidity requirements for Vietnam market.

        Vietnam market characteristics:
        - Smaller market cap than US/EU
        - Lower daily trading volumes
        - T+2 settlement requires holding period consideration
        - Minimum 2B VND daily value recommended for institutional trading

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict with sufficient, reason, avg_daily_value, and thresholds
        """
        if "volume" not in df.columns or "close" not in df.columns or len(df) < 5:
            return {
                "sufficient": True,
                "reason": "Insufficient data for liquidity check",
                "avg_daily_value": 0,
                "min_required": VN_MIN_LIQUIDITY_VALUE,
            }

        avg_volume = df["volume"].tail(20).mean()
        avg_price = df["close"].tail(20).mean()
        avg_daily_value = avg_volume * avg_price

        recent_volume = df["volume"].tail(5).mean()
        recent_price = df["close"].tail(5).mean()
        recent_daily_value = recent_volume * recent_price

        effective_value = min(avg_daily_value, recent_daily_value)

        result = {
            "sufficient": True,
            "reason": "",
            "avg_daily_value": avg_daily_value,
            "recent_daily_value": recent_daily_value,
            "effective_value": effective_value,
            "min_required": VN_MIN_LIQUIDITY_VALUE,
            "critical_threshold": VN_CRITICAL_LIQUIDITY_VALUE,
        }

        if effective_value < VN_CRITICAL_LIQUIDITY_VALUE:
            result["sufficient"] = False
            result["reason"] = (
                f"Critical liquidity: {effective_value/1e9:.2f}B VND < "
                f"{VN_CRITICAL_LIQUIDITY_VALUE/1e9:.1f}B VND minimum"
            )
            logger.warning(f"🚫 {result['reason']}")
            return result

        if effective_value < VN_MIN_LIQUIDITY_VALUE:
            result["reason"] = (
                f"Low liquidity: {effective_value/1e9:.2f}B VND < "
                f"{VN_MIN_LIQUIDITY_VALUE/1e9:.1f}B VND recommended"
            )
            logger.debug(f"⚠️ {result['reason']}")
            return result

        result["reason"] = f"Good liquidity: {effective_value/1e9:.2f}B VND"
        return result

    def check_volatility(self, df: pd.DataFrame) -> Dict:
        """
        Check volatility (ATR/Price).

        < 2%: Too low (no momentum)
        2-3%: Optimal
        > 4%: Too high (risky)

        Args:
            df: DataFrame with OHLCV and ATR data

        Returns:
            Dict with too_high, optimal, value
        """
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

    def check_rsi(self, df: pd.DataFrame) -> Dict:
        """
        Check RSI.

        > 70: Overbought (penalty)
        60-70: Neutral
        30-60: Optimal (good for entry)
        < 30: Oversold (strong buy signal)

        Args:
            df: DataFrame with RSI data

        Returns:
            Dict with overbought, optimal, oversold, value
        """
        if "rsi" not in df.columns:
            return {"overbought": False, "optimal": True, "oversold": False, "value": 50}

        rsi = safe_get_latest(df, "rsi", 0)

        if pd.isna(rsi):
            return {"overbought": False, "optimal": True, "oversold": False, "value": 50}

        if rsi > 70:
            return {"overbought": True, "optimal": False, "oversold": False, "value": rsi}
        elif 30 <= rsi <= 60:
            return {"overbought": False, "optimal": True, "oversold": False, "value": rsi}
        elif rsi < 30:
            return {"overbought": False, "optimal": False, "oversold": True, "value": rsi}
        else:  # 60 < rsi <= 70
            return {"overbought": False, "optimal": False, "oversold": False, "value": rsi}

    def check_price_action(self, df: pd.DataFrame) -> Dict:
        """
        Check candlestick patterns (simplified).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict with bullish_pattern, bearish_pattern, pattern name
        """
        if len(df) < 3:
            return {
                "bullish_pattern": False,
                "bearish_pattern": False,
                "pattern": "None",
            }

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

    def check_multi_timeframe_trend(self, df: pd.DataFrame) -> Dict:
        """
        Check multi-timeframe trend alignment.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict with weekly_up, monthly_up, changes
        """
        if len(df) < 20:
            return {
                "weekly_up": True,
                "monthly_up": True,
                "weekly_change": 0.0,
                "monthly_change": 0.0,
            }

        current_close = safe_get_latest(df, "close", 0)
        weekly_close = df["close"].iloc[-5] if len(df) >= 5 else current_close
        monthly_close = df["close"].iloc[-20] if len(df) >= 20 else weekly_close

        weekly_change = ((current_close / weekly_close) - 1) * 100 if weekly_close else 0
        monthly_change = ((current_close / monthly_close) - 1) * 100 if monthly_close else 0

        weekly_up = weekly_change >= 0
        monthly_up = monthly_change >= 0

        return {
            "weekly_up": weekly_up,
            "monthly_up": monthly_up,
            "weekly_change": weekly_change,
            "monthly_change": monthly_change,
        }

    def check_market_breadth(self, market_regime: Optional[Dict]) -> Dict:
        """
        Kiểm tra breadth của thị trường (số mã tăng/giảm).

        Enhanced version that fetches real-time data if not available in regime.

        Args:
            market_regime: Market regime data

        Returns:
            Dict with strong, weak, advance_ratio, and details
        """
        advancers = 0
        decliners = 0
        unchanged = 0
        source = "none"

        # Try to get from regime data first
        if market_regime:
            details = market_regime.get("details", {})
            breadth = market_regime.get("breadth") or details.get("breadth") or {}

            advancers = breadth.get("advancers") or breadth.get("advancing") or 0
            decliners = breadth.get("decliners") or breadth.get("declining") or 0
            unchanged = breadth.get("unchanged", 0)
            source = "regime"

        # If not available, try to fetch from TCBS API
        if advancers == 0 and decliners == 0:
            try:
                import requests
                from src.utils.rate_limiter import tcbs_limiter

                tcbs_limiter.wait()
                url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/top-price-change"
                response = requests.get(url, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        # Count advancing/declining from top movers
                        top_gainers = data.get("topGainers", [])
                        top_losers = data.get("topLosers", [])
                        advancers = len(top_gainers) if top_gainers else 0
                        decliners = len(top_losers) if top_losers else 0
                        source = "tcbs_api"
            except Exception:
                pass

        # If still no data, try SSI API
        if advancers == 0 and decliners == 0:
            try:
                import requests

                url = "https://iboard.ssi.com.vn/dchart/api/1.1/defaultAllStocks"
                response = requests.get(url, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and "data" in data:
                        stocks = data["data"]
                        for stock in stocks:
                            change = stock.get("priceChange", 0)
                            if change > 0:
                                advancers += 1
                            elif change < 0:
                                decliners += 1
                            else:
                                unchanged += 1
                        source = "ssi_api"
            except Exception:
                pass

        # Calculate metrics
        total = advancers + decliners
        if total == 0:
            return {
                "strong": False,
                "weak": False,
                "advance_ratio": 0.5,
                "advancers": 0,
                "decliners": 0,
                "unchanged": 0,
                "source": source,
                "available": False,
            }

        advance_ratio = advancers / total

        # Breadth interpretation:
        # > 60% advancing = Strong (bullish breadth)
        # < 40% advancing = Weak (bearish breadth)
        # 40-60% = Neutral
        strong = advance_ratio >= 0.60
        weak = advance_ratio <= 0.40

        # Additional insights
        breadth_score = (advance_ratio - 0.5) * 2  # -1 to +1 scale

        return {
            "strong": strong,
            "weak": weak,
            "advance_ratio": advance_ratio,
            "breadth_score": breadth_score,
            "advancers": advancers,
            "decliners": decliners,
            "unchanged": unchanged,
            "source": source,
            "available": True,
        }

    def check_sector_strength(
        self, df: pd.DataFrame, market_regime: Optional[Dict], current_symbol: Optional[str] = None
    ) -> Dict:
        """
        Kiểm tra sức mạnh của ngành so với thị trường chung (VNINDEX).

        Uses SectorRotationAnalyzer để xác định:
        - Ngành đang dẫn dắt (leading) hay tụt hậu (lagging)
        - Giai đoạn thị trường (EARLY, MID, LATE, RECESSION)
        - Điều chỉnh confidence dựa trên sector rotation
        - Rotation bonus khi symbol thuộc overweight sector
        - Rotation penalty khi symbol thuộc underweight sector

        Args:
            df: DataFrame with OHLCV data
            market_regime: Market regime data
            current_symbol: Current stock symbol

        Returns:
            Dict with is_leading, is_lagging, sector_perf, sector_id, rotation_phase,
            rotation_bonus, in_overweight_sector, in_underweight_sector, confidence
        """
        result = {
            "is_leading": False,
            "is_lagging": False,
            "sector_perf": 0.0,
            "sector_id": "unknown",
            "rotation_phase": "UNKNOWN",
            "rotation_bonus": 0,  # Bonus/penalty from rotation (-10 to +15)
            "in_overweight_sector": False,
            "in_underweight_sector": False,
            "confidence": 0,  # Rotation signal confidence
            "top_sector_picks": [],  # Best stocks in leading sector
        }

        # Try to use SectorRotationAnalyzer first
        try:
            from src.market.sector_rotation import (
                get_sector_rotation_analyzer,
                get_symbol_sector_info,
            )

            if current_symbol:
                sector_info = get_symbol_sector_info(current_symbol)
                result["sector_id"] = sector_info.get("sector_id", "unknown")
                result["is_leading"] = sector_info.get("is_leading", False)
                result["is_lagging"] = sector_info.get("is_lagging", False)

                analyzer = get_sector_rotation_analyzer()
                rotation = analyzer.get_rotation_signal()

                # Get rotation signal confidence
                result["confidence"] = rotation.confidence

                # Check if current symbol's sector is in overweight/underweight
                sector_upper = result["sector_id"].upper()

                if sector_upper in rotation.overweight:
                    result["in_overweight_sector"] = True
                    result["is_leading"] = True
                    # Bonus scales with confidence (5-15 points)
                    result["rotation_bonus"] = int(5 + (rotation.confidence / 100) * 10)
                    # Get top picks for this sector
                    if sector_upper in rotation.top_picks:
                        result["top_sector_picks"] = rotation.top_picks[sector_upper]

                elif sector_upper in rotation.underweight:
                    result["in_underweight_sector"] = True
                    result["is_lagging"] = True
                    # Penalty scales with confidence (-5 to -15 points)
                    result["rotation_bonus"] = -int(5 + (rotation.confidence / 100) * 10)

                # Determine rotation phase
                if any(s in rotation.overweight for s in ["BANKING", "SECURITIES"]):
                    result["rotation_phase"] = "EARLY"
                elif any(s in rotation.overweight for s in ["TECHNOLOGY", "REAL_ESTATE"]):
                    result["rotation_phase"] = "MID"
                elif any(s in rotation.overweight for s in ["ENERGY", "INDUSTRIAL"]):
                    result["rotation_phase"] = "LATE"
                elif any(s in rotation.overweight for s in ["CONSUMER", "UTILITIES"]):
                    result["rotation_phase"] = "RECESSION"

                momentum_data = analyzer.get_sector_momentum()
                if sector_upper in momentum_data:
                    perf = momentum_data[sector_upper]
                    result["sector_perf"] = perf.return_1m * 100

                    # Additional bonus for strong momentum within overweight sector
                    if result["in_overweight_sector"] and perf.momentum_score > 0.7:
                        result["rotation_bonus"] += 5  # Extra bonus for high momentum
                    elif result["in_underweight_sector"] and perf.momentum_score < -0.5:
                        result["rotation_bonus"] -= 5  # Extra penalty for weak momentum

                logger.debug(
                    f"📊 Sector check for {current_symbol}: "
                    f"sector={result['sector_id']}, leading={result['is_leading']}, "
                    f"lagging={result['is_lagging']}, phase={result['rotation_phase']}, "
                    f"rotation_bonus={result['rotation_bonus']}"
                )
                return result

        except ImportError:
            logger.debug("SectorRotationAnalyzer not available, falling back to RS")
        except Exception as e:
            logger.debug(f"Sector rotation check failed: {e}")

        # Fallback to RS (Relative Strength)
        if "rs" not in df.columns or df["rs"].isnull().all():
            return result

        latest_rs = safe_get_latest(df, "rs", 0)
        rs_trend = safe_rolling_operation(df, "rs", 10, "mean", 0) > safe_rolling_operation(
            df, "rs", 30, "mean", 0
        )

        result["is_leading"] = latest_rs > 1.0 and rs_trend
        result["is_lagging"] = latest_rs < 0.95

        if market_regime and "sector_performance" in market_regime:
            sector = safe_get_latest(df, "sector", 0) if "sector" in df.columns else "UNKNOWN"
            result["sector_perf"] = market_regime["sector_performance"].get(sector, 0)

        return result

    def check_portfolio_correlation(self, df: pd.DataFrame, symbol: Optional[str]) -> Dict:
        """
        Kiểm tra correlation với portfolio hiện tại (with caching).

        Cache strategy:
        - Cache correlation matrix for 5 minutes
        - Invalidate when portfolio symbols change
        - Reduces redundant calculations during parallel scanning

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol to check

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
            import time

            positions = self.portfolio_manager.get_positions()
            if not positions or len(positions) == 0:
                return {
                    "too_high": False,
                    "good_diversification": True,
                    "max_correlation": 0.0,
                }

            existing_symbols = list(positions.keys())
            all_symbols = existing_symbols + [symbol]
            symbols_key = tuple(sorted(existing_symbols))

            current_time = time.time()
            portfolio_hash = hashlib.md5(str(sorted(existing_symbols)).encode()).hexdigest()

            # Check if cache is from the same date
            cache_date_valid = True
            if self._correlation_cache_time is not None:
                cache_date = datetime.fromtimestamp(self._correlation_cache_time).date()
                current_date = datetime.now().date()
                cache_date_valid = cache_date == current_date

            portfolio_changed = (
                self._correlation_cache_portfolio_hash is not None
                and self._correlation_cache_portfolio_hash != portfolio_hash
            )

            cache_valid = (
                self._correlation_cache is not None
                and self._correlation_cache_time is not None
                and self._correlation_cache_symbols == symbols_key
                and (current_time - self._correlation_cache_time) < self._correlation_cache_ttl
                and cache_date_valid
                and not portfolio_changed
            )

            if cache_valid:
                correlation_metrics = self._correlation_cache
                logger.debug(
                    f"✅ Using cached correlation matrix "
                    f"(age: {current_time - self._correlation_cache_time:.0f}s)"
                )
            else:
                correlation_metrics = calculate_portfolio_correlation_risk(
                    all_symbols,
                    lookback=60,
                    max_avg_correlation=0.70,
                )
                self._correlation_cache = correlation_metrics
                self._correlation_cache_time = current_time
                self._correlation_cache_symbols = symbols_key
                self._correlation_cache_portfolio_hash = portfolio_hash
                logger.debug("🔄 Calculated and cached new correlation matrix")

            max_correlation = correlation_metrics.get("max_correlation", 0.0)
            avg_correlation = correlation_metrics.get("avg_correlation", 0.0)

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

    def check_volume_manipulation(self, df: pd.DataFrame) -> Dict:
        """
        Detect potential volume manipulation (abnormal volume spike).

        Signs of manipulation:
        - Volume > 5x average (extreme spike)
        - Volume spike with price barely moving (wash trading)
        - Volume spike at end of day (closing manipulation)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict with is_manipulation, reason, volume_ratio
        """
        if df is None or len(df) < 20 or "volume" not in df.columns:
            return {"is_manipulation": False, "reason": "Insufficient data", "volume_ratio": 1.0}

        try:
            current_volume = safe_get_latest(df, "volume", 0)
            avg_volume = safe_rolling_operation(df, "volume", 20, "mean", 1)

            if avg_volume == 0:
                return {"is_manipulation": False, "reason": "No volume data", "volume_ratio": 1.0}

            volume_ratio = current_volume / avg_volume

            if volume_ratio > 5.0:
                current_close = safe_get_latest(df, "close", 0)
                prev_close = df["close"].iloc[-2] if len(df) >= 2 else current_close
                price_change_pct = (
                    abs((current_close - prev_close) / prev_close * 100) if prev_close > 0 else 0
                )

                if price_change_pct < 2.0:
                    return {
                        "is_manipulation": True,
                        "reason": (
                            f"Volume spike {volume_ratio:.1f}x với giá chỉ thay đổi "
                            f"{price_change_pct:.1f}% - nghi ngờ wash trading"
                        ),
                        "volume_ratio": volume_ratio,
                    }

                if volume_ratio > 8.0:
                    return {
                        "is_manipulation": True,
                        "reason": (
                            f"Volume đột biến bất thường {volume_ratio:.1f}x - "
                            f"nghi ngờ manipulation"
                        ),
                        "volume_ratio": volume_ratio,
                    }

            return {
                "is_manipulation": False,
                "reason": "Volume bình thường",
                "volume_ratio": volume_ratio,
            }

        except Exception as e:
            logger.warning(f"Volume manipulation check error: {e}")
            return {"is_manipulation": False, "reason": f"Error: {e}", "volume_ratio": 1.0}
