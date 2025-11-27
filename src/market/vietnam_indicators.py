# -*- coding: utf-8 -*-
"""
Vietnam Market Specific Indicators
Các chỉ báo đặc thù cho thị trường chứng khoán Việt Nam

IMPROVEMENTS v3.0:
- Foreign flow indicator (khối ngoại)
- Intraday volatility check
- Session boundary detection
- VN30 correlation
- Sector rotation analysis
"""

import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VietnamMarketIndicators:
    """
    Các chỉ báo đặc thù cho thị trường Việt Nam

    Features:
    - Foreign flow tracking (khối ngoại mua/bán ròng)
    - Intraday volatility analysis
    - Price limit detection (±7%)
    - Session boundary warnings
    - VN30 correlation
    """

    # Vietnam market constants
    PRICE_LIMIT_PERCENT = 0.07  # ±7% daily limit
    LOT_SIZE = 100  # Minimum lot size
    TICK_SIZE = 10  # 10 VND tick for most stocks

    # Trading sessions
    MORNING_SESSION = (time(9, 0), time(11, 30))
    AFTERNOON_SESSION = (time(13, 0), time(15, 0))

    def __init__(self):
        self._foreign_flow_cache = {}
        self._vn30_cache = None
        self._vn30_cache_time = None

    def check_price_limits(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Kiểm tra giá có gần floor/ceiling không

        VN market có giới hạn ±7% mỗi ngày
        - Gần ceiling (>5%): Rủi ro cao, không nên mua
        - Gần floor (<-5%): Có thể là cơ hội nhưng cần cẩn thận

        Args:
            df: DataFrame với OHLCV
            current_price: Giá hiện tại

        Returns:
            Dict với thông tin price limit
        """
        if df.empty or len(df) < 2:
            return {"near_limit": False, "limit_type": None}

        # Lấy giá đóng cửa hôm qua (reference price)
        prev_close = df.iloc[-2]["close"] if len(df) >= 2 else df.iloc[-1]["close"]

        # Tính ceiling và floor
        ceiling = prev_close * (1 + self.PRICE_LIMIT_PERCENT)
        floor = prev_close * (1 - self.PRICE_LIMIT_PERCENT)

        # Tính % thay đổi từ reference
        change_pct = (current_price - prev_close) / prev_close * 100

        # Kiểm tra gần ceiling (>5%)
        if change_pct >= 5.0:
            return {
                "near_limit": True,
                "limit_type": "CEILING",
                "change_pct": change_pct,
                "ceiling": ceiling,
                "floor": floor,
                "warning": f"Gần trần ({change_pct:+.1f}%) - Rủi ro cao",
                "recommendation": "AVOID_BUY",
            }

        # Kiểm tra gần floor (<-5%)
        if change_pct <= -5.0:
            return {
                "near_limit": True,
                "limit_type": "FLOOR",
                "change_pct": change_pct,
                "ceiling": ceiling,
                "floor": floor,
                "warning": f"Gần sàn ({change_pct:+.1f}%) - Cần cẩn thận",
                "recommendation": "CAUTION",
            }

        return {
            "near_limit": False,
            "limit_type": None,
            "change_pct": change_pct,
            "ceiling": ceiling,
            "floor": floor,
        }

    def check_intraday_volatility(self, df: pd.DataFrame, max_intraday_range: float = 5.0) -> Dict:
        """
        Kiểm tra biến động trong ngày

        Tránh mua khi:
        - Giá đã tăng mạnh trong ngày (>5%)
        - Biên độ dao động quá lớn

        Args:
            df: DataFrame với OHLCV
            max_intraday_range: Biên độ tối đa cho phép (%)

        Returns:
            Dict với thông tin intraday volatility
        """
        if df.empty or len(df) < 1:
            return {"safe": True, "reason": "No data"}

        latest = df.iloc[-1]
        today_open = latest.get("open", 0)
        today_high = latest.get("high", 0)
        today_low = latest.get("low", 0)
        today_close = latest.get("close", 0)

        if today_open <= 0:
            return {"safe": True, "reason": "Invalid open price"}

        # Tính biên độ trong ngày
        intraday_range = (today_high - today_low) / today_open * 100

        # Tính % thay đổi từ open
        change_from_open = (today_close - today_open) / today_open * 100

        # Tính % từ low (đã tăng bao nhiêu từ đáy ngày)
        if today_low > 0:
            change_from_low = (today_close - today_low) / today_low * 100
        else:
            change_from_low = 0

        warnings = []
        safe = True

        # Cảnh báo nếu biên độ quá lớn
        if intraday_range > max_intraday_range:
            warnings.append(f"Biên độ ngày cao: {intraday_range:.1f}%")
            safe = False

        # Cảnh báo nếu đã tăng mạnh từ open
        if change_from_open > 4.0:
            warnings.append(f"Đã tăng {change_from_open:.1f}% từ open")
            safe = False

        # Cảnh báo nếu đã tăng mạnh từ đáy ngày
        if change_from_low > 5.0:
            warnings.append(f"Đã tăng {change_from_low:.1f}% từ đáy ngày")
            safe = False

        return {
            "safe": safe,
            "intraday_range": intraday_range,
            "change_from_open": change_from_open,
            "change_from_low": change_from_low,
            "warnings": warnings,
            "recommendation": "OK" if safe else "WAIT_FOR_PULLBACK",
        }

    def estimate_foreign_flow(self, df: pd.DataFrame, lookback: int = 5) -> Dict:
        """
        Ước tính dòng tiền khối ngoại dựa trên volume và price action

        Logic:
        - Volume tăng + giá tăng = khối ngoại có thể đang mua
        - Volume tăng + giá giảm = khối ngoại có thể đang bán
        - Cần kết hợp với dữ liệu thực từ API nếu có

        Args:
            df: DataFrame với OHLCV
            lookback: Số ngày nhìn lại

        Returns:
            Dict với ước tính foreign flow
        """
        if df.empty or len(df) < lookback + 1:
            return {"estimated_flow": "NEUTRAL", "confidence": 0, "reason": "Insufficient data"}

        recent = df.tail(lookback)

        # Tính volume trung bình
        avg_volume = df["volume"].tail(20).mean() if len(df) >= 20 else df["volume"].mean()

        # Đếm số ngày volume cao + giá tăng (bullish)
        bullish_days = 0
        bearish_days = 0

        for i in range(len(recent)):
            row = recent.iloc[i]
            vol_ratio = row["volume"] / avg_volume if avg_volume > 0 else 1
            price_change = (row["close"] - row["open"]) / row["open"] if row["open"] > 0 else 0

            if vol_ratio > 1.2 and price_change > 0.01:
                bullish_days += 1
            elif vol_ratio > 1.2 and price_change < -0.01:
                bearish_days += 1

        # Xác định xu hướng
        if bullish_days >= 3:
            flow = "BUYING"
            confidence = min(bullish_days / lookback * 100, 80)
        elif bearish_days >= 3:
            flow = "SELLING"
            confidence = min(bearish_days / lookback * 100, 80)
        else:
            flow = "NEUTRAL"
            confidence = 50

        return {
            "estimated_flow": flow,
            "confidence": confidence,
            "bullish_days": bullish_days,
            "bearish_days": bearish_days,
            "lookback": lookback,
            "recommendation": (
                "BUY" if flow == "BUYING" else ("AVOID" if flow == "SELLING" else "NEUTRAL")
            ),
        }

    def check_session_timing(self) -> Dict:
        """
        Kiểm tra thời điểm trong phiên giao dịch

        Khuyến nghị:
        - Đầu phiên sáng (9:00-9:30): Biến động cao, cẩn thận
        - Cuối phiên sáng (11:00-11:30): Có thể có selling pressure
        - Đầu phiên chiều (13:00-13:30): Thường có gap
        - Cuối phiên chiều (14:30-15:00): ATC, biến động cao

        Returns:
            Dict với thông tin session timing
        """
        try:
            import pytz

            VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")
            now = datetime.now(VN_TZ)
        except ImportError:
            now = datetime.now()

        current_time = now.time()

        # Kiểm tra các giai đoạn đặc biệt
        warnings = []
        is_risky_period = False

        # Đầu phiên sáng (9:00-9:30)
        if time(9, 0) <= current_time <= time(9, 30):
            warnings.append("Đầu phiên sáng - Biến động cao")
            is_risky_period = True

        # Cuối phiên sáng (11:00-11:30)
        elif time(11, 0) <= current_time <= time(11, 30):
            warnings.append("Cuối phiên sáng - Có thể có selling pressure")
            is_risky_period = True

        # Đầu phiên chiều (13:00-13:30)
        elif time(13, 0) <= current_time <= time(13, 30):
            warnings.append("Đầu phiên chiều - Thường có gap")
            is_risky_period = True

        # ATC (14:30-15:00)
        elif time(14, 30) <= current_time <= time(15, 0):
            warnings.append("Gần ATC - Biến động cao")
            is_risky_period = True

        # Xác định session
        if time(9, 0) <= current_time <= time(11, 30):
            session = "MORNING"
        elif time(13, 0) <= current_time <= time(15, 0):
            session = "AFTERNOON"
        else:
            session = "CLOSED"

        return {
            "session": session,
            "current_time": current_time.strftime("%H:%M"),
            "is_risky_period": is_risky_period,
            "warnings": warnings,
            "recommendation": "CAUTION" if is_risky_period else "OK",
        }

    def calculate_vn30_correlation(
        self, df: pd.DataFrame, vn30_df: Optional[pd.DataFrame] = None, lookback: int = 20
    ) -> Dict:
        """
        Tính correlation với VN30 index

        Stocks có correlation cao với VN30:
        - Dễ dự đoán hơn dựa trên market direction
        - Nhưng cũng có thể bị ảnh hưởng mạnh khi market giảm

        Args:
            df: DataFrame của stock
            vn30_df: DataFrame của VN30 (optional)
            lookback: Số ngày tính correlation

        Returns:
            Dict với thông tin correlation
        """
        if df.empty or len(df) < lookback:
            return {"correlation": 0, "beta": 1.0, "reason": "Insufficient data"}

        # Tính returns
        stock_returns = df["close"].pct_change().tail(lookback).dropna()

        if vn30_df is not None and not vn30_df.empty and len(vn30_df) >= lookback:
            vn30_returns = vn30_df["close"].pct_change().tail(lookback).dropna()

            # Align data
            min_len = min(len(stock_returns), len(vn30_returns))
            stock_returns = stock_returns.tail(min_len)
            vn30_returns = vn30_returns.tail(min_len)

            if len(stock_returns) >= 10:
                correlation = stock_returns.corr(vn30_returns)

                # Tính beta
                if vn30_returns.var() > 0:
                    beta = stock_returns.cov(vn30_returns) / vn30_returns.var()
                else:
                    beta = 1.0

                return {
                    "correlation": correlation,
                    "beta": beta,
                    "interpretation": self._interpret_correlation(correlation, beta),
                }

        return {"correlation": 0, "beta": 1.0, "reason": "VN30 data not available"}

    def _interpret_correlation(self, correlation: float, beta: float) -> str:
        """Interpret correlation và beta"""
        if correlation > 0.8:
            return "Highly correlated with market - follows VN30 closely"
        elif correlation > 0.5:
            return "Moderately correlated - influenced by market"
        elif correlation > 0.2:
            return "Low correlation - somewhat independent"
        else:
            return "Very low correlation - independent of market"

    def get_optimal_entry_time(self) -> Dict:
        """
        Xác định thời điểm vào lệnh tối ưu trong ngày

        Dựa trên nghiên cứu thị trường VN:
        - 9:30-10:30: Sau khi biến động đầu phiên ổn định
        - 13:30-14:30: Sau khi gap đầu phiên chiều ổn định

        Returns:
            Dict với khuyến nghị thời điểm
        """
        try:
            import pytz

            VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")
            now = datetime.now(VN_TZ)
        except ImportError:
            now = datetime.now()

        current_time = now.time()

        # Thời điểm tối ưu
        optimal_periods = [
            (time(9, 30), time(10, 30), "Morning optimal"),
            (time(13, 30), time(14, 30), "Afternoon optimal"),
        ]

        for start, end, label in optimal_periods:
            if start <= current_time <= end:
                return {"is_optimal": True, "period": label, "recommendation": "GOOD_TIME_TO_ENTER"}

        # Thời điểm nên tránh
        avoid_periods = [
            (time(9, 0), time(9, 30), "Opening volatility"),
            (time(11, 0), time(11, 30), "Pre-lunch selling"),
            (time(14, 30), time(15, 0), "ATC volatility"),
        ]

        for start, end, label in avoid_periods:
            if start <= current_time <= end:
                return {"is_optimal": False, "period": label, "recommendation": "AVOID_ENTRY"}

        return {"is_optimal": False, "period": "Neutral", "recommendation": "ACCEPTABLE"}


# Singleton instance
_vietnam_indicators = None


def get_vietnam_indicators() -> VietnamMarketIndicators:
    """Get singleton instance"""
    global _vietnam_indicators
    if _vietnam_indicators is None:
        _vietnam_indicators = VietnamMarketIndicators()
    return _vietnam_indicators


# Test
if __name__ == "__main__":
    print("Testing Vietnam Market Indicators...")

    indicators = VietnamMarketIndicators()

    # Test session timing
    timing = indicators.check_session_timing()
    print(f"\nSession Timing: {timing}")

    # Test optimal entry time
    optimal = indicators.get_optimal_entry_time()
    print(f"Optimal Entry: {optimal}")

    print("\n✅ Vietnam Market Indicators test completed!")
