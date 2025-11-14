"""
Portfolio Lock - Prevent race conditions
Đảm bảo không vượt quá max exposure khi scan song song
"""

import threading
from typing import Tuple, Optional
from datetime import datetime


class PortfolioLock:
    """
    Thread-safe portfolio lock để tránh race condition

    Đảm bảo:
    - Không vượt max exposure
    - Không vượt max positions
    - Thread-safe khi add positions
    """

    def __init__(self, max_exposure_pct: float = 0.60, max_positions: int = 10):
        self._lock = threading.Lock()
        self.max_exposure_pct = max_exposure_pct
        self.max_positions = max_positions
        self._pending_positions = {}  # {symbol: value}

    def can_add_position(
        self,
        symbol: str,
        position_value: float,
        total_capital: float,
        current_positions: dict,
    ) -> Tuple[bool, str]:
        """
        Check xem có thể add position không (thread-safe)

        Returns:
            (can_add, reason)
        """
        with self._lock:
            # Check 1: Max positions
            total_positions = len(current_positions) + len(self._pending_positions)
            if total_positions >= self.max_positions:
                return False, f"Max positions reached ({self.max_positions})"

            # Check 2: Already have this symbol
            if symbol in current_positions or symbol in self._pending_positions:
                return False, f"Already have position in {symbol}"

            # Check 3: Max exposure
            current_exposure = sum(
                pos.get("shares", 0) * pos.get("avg_price", 0)
                for pos in current_positions.values()
            )
            pending_exposure = sum(self._pending_positions.values())
            total_exposure = current_exposure + pending_exposure + position_value

            max_exposure = total_capital * self.max_exposure_pct

            if total_exposure > max_exposure:
                exposure_pct = (total_exposure / total_capital) * 100
                return (
                    False,
                    f"Would exceed max exposure ({exposure_pct:.1f}% > {self.max_exposure_pct*100:.1f}%)",
                )

            # OK - Reserve this position
            self._pending_positions[symbol] = position_value
            return True, "OK"

    def confirm_position(self, symbol: str):
        """Confirm position đã được add thành công"""
        with self._lock:
            if symbol in self._pending_positions:
                del self._pending_positions[symbol]

    def cancel_position(self, symbol: str):
        """Cancel position (nếu add failed)"""
        with self._lock:
            if symbol in self._pending_positions:
                del self._pending_positions[symbol]

    def get_pending_exposure(self) -> float:
        """Lấy tổng exposure đang pending"""
        with self._lock:
            return sum(self._pending_positions.values())

    def clear_pending(self):
        """Clear tất cả pending positions (cleanup)"""
        with self._lock:
            self._pending_positions.clear()


# Global instance
_portfolio_lock = None


def get_portfolio_lock() -> PortfolioLock:
    """Get singleton instance"""
    global _portfolio_lock
    if _portfolio_lock is None:
        _portfolio_lock = PortfolioLock()
    return _portfolio_lock


# Test
if __name__ == "__main__":
    print("Testing Portfolio Lock...")

    lock = PortfolioLock(max_exposure_pct=0.60, max_positions=5)

    # Simulate concurrent adds
    current_positions = {}
    total_capital = 100_000_000

    # Try add 6 positions (should fail at 6th)
    for i in range(6):
        symbol = f"STOCK{i+1}"
        position_value = 10_000_000  # 10M each

        can_add, reason = lock.can_add_position(
            symbol, position_value, total_capital, current_positions
        )

        if can_add:
            print(f"✅ Can add {symbol}: {position_value:,} VNĐ")
            lock.confirm_position(symbol)
            current_positions[symbol] = {"shares": 100, "avg_price": 100_000}
        else:
            print(f"❌ Cannot add {symbol}: {reason}")

    print(f"\nPending exposure: {lock.get_pending_exposure():,} VNĐ")
    print("✅ Test completed!")
