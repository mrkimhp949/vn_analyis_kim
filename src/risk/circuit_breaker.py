"""
Circuit Breaker - Giới hạn trades và loss per day
Bảo vệ khỏi lỗi logic hoặc market anomaly
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Dict, Tuple


@dataclass
class DailyStats:
    """Stats trong ngày"""

    date: str
    trades_count: int
    total_loss: float
    total_profit: float
    net_pnl: float
    last_updated: str


class CircuitBreaker:
    """
    Circuit Breaker để bảo vệ khỏi:
    - Trade quá nhiều trong 1 ngày
    - Loss quá nhiều trong 1 ngày
    - Consecutive losses
    - VNINDEX giảm sâu
    """

    def __init__(
        self,
        max_trades_per_day: int = 10,
        max_loss_per_day_pct: float = 0.05,  # 5% vốn
        max_consecutive_losses: int = 5,
        vnindex_drop_threshold: float = -2.5,  # Ngưỡng VNINDEX giảm để ngắt (%)
        total_capital: float = 100_000_000,
        stats_file: str = "circuit_breaker_stats.json",
        max_portfolio_heat: float = 0.70,  # ENHANCEMENT: Max portfolio exposure
        volatility_multiplier: float = 1.5,  # ENHANCEMENT: Adjust thresholds by volatility
    ):
        self.max_trades_per_day = max_trades_per_day
        self.max_loss_per_day_pct = max_loss_per_day_pct
        self.base_max_loss_per_day_pct = max_loss_per_day_pct  # Store original
        self.max_consecutive_losses = max_consecutive_losses
        self.vnindex_drop_threshold = vnindex_drop_threshold / 100.0  # Convert to float
        self.base_vnindex_drop_threshold = self.vnindex_drop_threshold  # Store original
        self.total_capital = total_capital
        self.stats_file = stats_file
        self.max_portfolio_heat = max_portfolio_heat
        self.volatility_multiplier = volatility_multiplier

        self.stats = self._load_stats()
        self._check_new_day()

        # Trạng thái ngắt mạch
        self.tripped = False
        self.tripped_reason = ""

        # OPTIMIZATION: Cache last volatility to avoid redundant recalculations
        self._last_volatility = None

    def _load_stats(self) -> Dict:
        """Load stats từ file"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Đảm bảo các key cần thiết tồn tại
                    data.setdefault("today", self._get_today_stats())
                    data.setdefault("consecutive_losses", 0)
                    data.setdefault("last_trade_date", None)
                    return data
            except Exception:
                pass

        return {
            "today": self._get_today_stats(),
            "consecutive_losses": 0,
            "last_trade_date": None,
        }

    def _save_stats(self):
        """Save stats vào file"""
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

    def _get_today_stats(self) -> Dict:
        """Tạo stats mới cho ngày hôm nay"""
        return asdict(
            DailyStats(
                date=date.today().isoformat(),
                trades_count=0,
                total_loss=0.0,
                total_profit=0.0,
                net_pnl=0.0,
                last_updated=datetime.now().isoformat(),
            )
        )

    def _check_new_day(self):
        """Check xem có phải ngày mới không, reset stats"""
        today = date.today().isoformat()
        if self.stats.get("today", {}).get("date") != today:
            # New day - reset
            self.stats["today"] = self._get_today_stats()
            # Reset trạng thái ngắt mạch mỗi ngày mới
            self.tripped = False
            self.tripped_reason = ""
            # OPTIMIZATION: Reset volatility cache for fresh calculation
            self._last_volatility = None
            self._save_stats()

    def check_and_update(
        self,
        portfolio_pnl_pct: float,
        vnindex_change_pct: float,
        portfolio_heat: float = 0.0,
        market_volatility: float = 0.0,
    ) -> bool:
        """
        ENHANCED: Kiểm tra các điều kiện ngắt mạch với volatility adjustments

        Args:
            portfolio_pnl_pct (float): P&L hiện tại của portfolio trong ngày (dạng float, vd: -0.01 cho -1%).
            vnindex_change_pct (float): % thay đổi của VNINDEX trong ngày.
            portfolio_heat (float): Portfolio exposure (0-1), 1 = fully invested
            market_volatility (float): Market volatility (ATR/Price ratio)

        Returns:
            bool: True nếu ngắt mạch được kích hoạt, False nếu không.
        """
        if self.tripped:
            return True  # Nếu đã ngắt thì không cần check lại

        # Validate input parameters
        if not isinstance(portfolio_pnl_pct, (int, float)):
            raise ValueError(f"portfolio_pnl_pct phải là số, nhận được: {type(portfolio_pnl_pct)}")
        if not isinstance(vnindex_change_pct, (int, float)):
            raise ValueError(
                f"vnindex_change_pct phải là số, nhận được: {type(vnindex_change_pct)}"
            )

        # ENHANCEMENT: Adjust thresholds based on volatility
        self._adjust_thresholds_for_volatility(market_volatility)

        # Check 1: Max loss per day
        if portfolio_pnl_pct < 0 and abs(portfolio_pnl_pct) >= self.max_loss_per_day_pct:
            self.tripped = True
            self.tripped_reason = (
                f"Lỗ trong ngày ({portfolio_pnl_pct:.2%}) "
                f"vượt ngưỡng cho phép ({self.max_loss_per_day_pct:.2%})."
            )
            self._save_stats()
            return True

        # Check 2: VNINDEX giảm sâu
        if vnindex_change_pct < self.vnindex_drop_threshold:
            self.tripped = True
            self.tripped_reason = (
                f"VNINDEX giảm sâu ({vnindex_change_pct:.2%}) "
                f"vượt ngưỡng ({self.vnindex_drop_threshold:.2%})."
            )
            self._save_stats()
            return True

        # Check 3: Max trades per day
        if self.stats["today"]["trades_count"] >= self.max_trades_per_day:
            self.tripped = True
            self.tripped_reason = (
                f"Số lệnh trong ngày ({self.stats['today']['trades_count']}) đạt giới hạn."
            )
            self._save_stats()
            return True

        # Check 4: Consecutive losses
        if self.stats["consecutive_losses"] >= self.max_consecutive_losses:
            self.tripped = True
            self.tripped_reason = (
                f"Số lệnh thua liên tiếp ({self.stats['consecutive_losses']}) đạt giới hạn."
            )
            self._save_stats()
            return True

        # ENHANCEMENT: Check 5: Portfolio heat (overexposure)
        if portfolio_heat > self.max_portfolio_heat:
            self.tripped = True
            self.tripped_reason = (
                f"Portfolio heat quá cao ({portfolio_heat:.1%}) "
                f"vượt ngưỡng ({self.max_portfolio_heat:.1%})."
            )
            self._save_stats()
            return True

        return False

    def _adjust_thresholds_for_volatility(self, market_volatility: float):
        """
        ENHANCEMENT: Adjust circuit breaker thresholds based on market volatility

        Logic:
        - High volatility (>3%): Tighten thresholds (more protective)
        - Normal volatility (1-3%): Use base thresholds
        - Low volatility (<1%): Relax thresholds slightly

        Args:
            market_volatility: Market volatility ratio (e.g., 0.02 = 2%)
        """
        # OPTIMIZATION: Skip recalculation if volatility hasn't changed
        if (
            self._last_volatility is not None
            and abs(market_volatility - self._last_volatility) < 0.0001
        ):
            return  # No change, skip recalculation

        self._last_volatility = market_volatility

        if market_volatility == 0.0:
            # No volatility data - use base thresholds
            self.max_loss_per_day_pct = self.base_max_loss_per_day_pct
            self.vnindex_drop_threshold = self.base_vnindex_drop_threshold
            return

        # Convert to percentage for easier comparison
        vol_pct = market_volatility * 100

        if vol_pct > 3.0:
            # High volatility - tighten limits
            volatility_factor = 0.75  # 25% tighter
            print(
                f"⚠️ High volatility detected ({vol_pct:.1f}%). "
                "Tightening circuit breaker thresholds by 25%"
            )
        elif vol_pct > 2.0:
            # Medium-high volatility - slightly tighter
            volatility_factor = 0.90  # 10% tighter
            print(
                f"📊 Medium-high volatility ({vol_pct:.1f}%). "
                "Tightening circuit breaker thresholds by 10%"
            )
        elif vol_pct < 1.0:
            # Low volatility - can relax slightly
            volatility_factor = 1.10  # 10% looser
            print(
                f"📉 Low volatility ({vol_pct:.1f}%). " "Relaxing circuit breaker thresholds by 10%"
            )
        else:
            # Normal volatility - use base
            volatility_factor = 1.0

        # Apply adjustments
        self.max_loss_per_day_pct = self.base_max_loss_per_day_pct * volatility_factor
        self.vnindex_drop_threshold = self.base_vnindex_drop_threshold * volatility_factor

        print(
            f"🔧 Circuit breaker adjusted: "
            f"max_loss={self.max_loss_per_day_pct:.2%}, "
            f"vnindex_threshold={self.vnindex_drop_threshold:.2%}"
        )

    def is_active(self) -> bool:
        """
        Kiểm tra xem circuit breaker có đang kích hoạt không.

        Returns:
            bool: True nếu circuit breaker đang kích hoạt, False nếu không.
        """
        return self.tripped

    def can_trade(self) -> Tuple[bool, str]:
        """
        DEPRECATED: Use check_and_update instead.
        Check xem có thể vào lệnh mới không.
        """
        if self.tripped:
            return False, self.tripped_reason

        # This part is now mostly redundant as checks are in check_and_update
        today_stats = self.stats["today"]
        if today_stats["trades_count"] >= self.max_trades_per_day:
            return False, f"🚫 Max trades per day reached ({self.max_trades_per_day})"
        if self.stats["consecutive_losses"] >= self.max_consecutive_losses:
            return (
                False,
                f"🚫 Too many consecutive losses ({self.stats['consecutive_losses']})",
            )

        return True, "✅ OK to trade"

    def record_trade(self, pnl: float):
        """
        Record một trade

        Args:
            pnl: Profit/Loss (positive = profit, negative = loss)
        """
        self._check_new_day()

        today_stats = self.stats["today"]

        # Update counts
        today_stats["trades_count"] += 1
        today_stats["last_updated"] = datetime.now().isoformat()

        # Update P&L
        if pnl > 0:
            today_stats["total_profit"] += pnl
            self.stats["consecutive_losses"] = 0  # Reset
        else:
            today_stats["total_loss"] += abs(pnl)
            self.stats["consecutive_losses"] += 1

        today_stats["net_pnl"] = today_stats["total_profit"] - today_stats["total_loss"]

        # Update last trade date
        self.stats["last_trade_date"] = date.today().isoformat()

        self._save_stats()

    def record_pnl(self, portfolio_pnl_pct: float):
        """
        Ghi nhận PnL hiện tại của portfolio ngay lập tức.
        Được gọi sau khi thoát lệnh để cập nhật trạng thái circuit breaker.

        Args:
            portfolio_pnl_pct (float): P&L hiện tại của portfolio (dạng float, vd: -0.01 cho -1%)
        """
        self._check_new_day()

        # Lưu PnL vào stats để tracking
        self.stats["today"]["last_updated"] = datetime.now().isoformat()

        # Kiểm tra ngay xem có cần kích hoạt circuit breaker không
        if portfolio_pnl_pct < 0 and abs(portfolio_pnl_pct) >= self.max_loss_per_day_pct:
            self.tripped = True
            self.tripped_reason = (
                f"Lỗ trong ngày ({portfolio_pnl_pct:.2%}) "
                f"vượt ngưỡng cho phép ({self.max_loss_per_day_pct:.2%})."
            )

        self._save_stats()

    def get_daily_stats(self) -> DailyStats:
        """Lấy stats của ngày hôm nay"""
        self._check_new_day()
        return DailyStats(**self.stats["today"])

    def get_status_message(self) -> str:
        """Lấy status message"""
        self._check_new_day()

        stats = self.get_daily_stats()

        msg = []
        msg.append("🔒 **CIRCUIT BREAKER STATUS**")
        msg.append("=" * 40)
        msg.append(f"📅 Date: {stats.date}")
        msg.append(f"🔄 Trades today: {stats.trades_count}/{self.max_trades_per_day}")
        msg.append(f"📉 Total loss: {stats.total_loss:,.0f} VNĐ")
        msg.append(f"📈 Total profit: {stats.total_profit:,.0f} VNĐ")
        msg.append(f"💰 Net P&L: {stats.net_pnl:+,.0f} VNĐ")
        msg.append(
            f"⚠️ Consecutive losses: {self.stats.get('consecutive_losses', 0)}/{self.max_consecutive_losses}"
        )
        msg.append("")
        msg.append(f"Status: {'TRIPPED - ' + self.tripped_reason if self.tripped else 'OK'}")

        return "\n".join(msg)

    def reset(self):
        """Reset toàn bộ trạng thái (cho testing hoặc manual reset)"""
        self.stats["today"] = self._get_today_stats()
        self.stats["consecutive_losses"] = 0
        self.tripped = False
        self.tripped_reason = ""
        self._save_stats()
        print("Circuit breaker has been reset.")


# Global instance
_circuit_breaker = None


def get_circuit_breaker(total_capital: float = 100_000_000) -> CircuitBreaker:
    """Get singleton instance"""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker(total_capital=total_capital)
    return _circuit_breaker


# Test
if __name__ == "__main__":
    print("Testing Circuit Breaker...")

    breaker = CircuitBreaker(
        max_trades_per_day=5,
        max_loss_per_day_pct=0.05,
        max_consecutive_losses=3,
        total_capital=100_000_000,
    )

    # Test 1: Normal trades
    print("\n1️⃣ Test normal trades:")
    for i in range(3):
        can_trade, reason = breaker.can_trade()
        print(f"Trade {i+1}: {reason}")
        if can_trade:
            # Simulate profit
            breaker.record_trade(1_000_000)

    # Test 2: Consecutive losses
    print("\n2️⃣ Test consecutive losses:")
    for i in range(4):
        can_trade, reason = breaker.can_trade()
        print(f"Loss {i+1}: {reason}")
        if can_trade:
            # Simulate loss
            breaker.record_trade(-500_000)

    # Test 3: Status
    print("\n3️⃣ Status:")
    print(breaker.get_status_message())

    print("\n✅ Test completed!")
