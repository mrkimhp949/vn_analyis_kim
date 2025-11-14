"""
Circuit Breaker - Giới hạn trades và loss per day
Bảo vệ khỏi lỗi logic hoặc market anomaly
"""

import json
import os
from datetime import datetime, date
from typing import Tuple, Dict
from dataclasses import dataclass, asdict


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
    """

    def __init__(
        self,
        max_trades_per_day: int = 10,
        max_loss_per_day_pct: float = 0.05,  # 5% vốn
        max_consecutive_losses: int = 5,
        total_capital: float = 100_000_000,
        stats_file: str = "circuit_breaker_stats.json",
    ):
        self.max_trades_per_day = max_trades_per_day
        self.max_loss_per_day_pct = max_loss_per_day_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.total_capital = total_capital
        self.stats_file = stats_file

        self.stats = self._load_stats()
        self._check_new_day()

    def _load_stats(self) -> Dict:
        """Load stats từ file"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    return json.load(f)
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
        if self.stats["today"]["date"] != today:
            # New day - reset
            self.stats["today"] = self._get_today_stats()
            self._save_stats()

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check xem có thể trade không

        Returns:
            (can_trade, reason)
        """
        self._check_new_day()

        today_stats = self.stats["today"]

        # Check 1: Max trades per day
        if today_stats["trades_count"] >= self.max_trades_per_day:
            return False, f"🚫 Max trades per day reached ({self.max_trades_per_day})"

        # Check 2: Max loss per day
        max_loss = self.total_capital * self.max_loss_per_day_pct
        if today_stats["total_loss"] >= max_loss:
            loss_pct = (today_stats["total_loss"] / self.total_capital) * 100
            return (
                False,
                f"🚫 Max loss per day reached ({loss_pct:.2f}% >= {self.max_loss_per_day_pct*100:.1f}%)",
            )

        # Check 3: Consecutive losses
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

    def get_daily_stats(self) -> DailyStats:
        """Lấy stats của ngày hôm nay"""
        self._check_new_day()
        return DailyStats(**self.stats["today"])

    def get_status_message(self) -> str:
        """Lấy status message"""
        self._check_new_day()

        stats = self.get_daily_stats()
        can_trade, reason = self.can_trade()

        msg = []
        msg.append("🔒 **CIRCUIT BREAKER STATUS**")
        msg.append("=" * 40)
        msg.append(f"📅 Date: {stats.date}")
        msg.append(f"🔄 Trades today: {stats.trades_count}/{self.max_trades_per_day}")
        msg.append(f"📉 Total loss: {stats.total_loss:,.0f} VNĐ")
        msg.append(f"📈 Total profit: {stats.total_profit:,.0f} VNĐ")
        msg.append(f"💰 Net P&L: {stats.net_pnl:+,.0f} VNĐ")
        msg.append(
            f"⚠️ Consecutive losses: {self.stats['consecutive_losses']}/{self.max_consecutive_losses}"
        )
        msg.append("")
        msg.append(f"Status: {reason}")

        return "\n".join(msg)

    def reset_daily_stats(self):
        """Reset stats (for testing or manual reset)"""
        self.stats["today"] = self._get_today_stats()
        self._save_stats()

    def reset_consecutive_losses(self):
        """Reset consecutive losses (after review)"""
        self.stats["consecutive_losses"] = 0
        self._save_stats()


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
