"""
Signal Approval Store
Lưu trữ pending signals chờ approval/rejection
"""

import threading
from datetime import datetime
from typing import Dict, Optional

from src.services.position_sizing import EnhancedPositionSize
from src.strategies.entry_logic import EntrySignal


class SignalApprovalStore:
    """
    Thread-safe store để lưu pending signals chờ approval

    Lưu trữ:
    - Entry signal info
    - Position size info
    - Timestamp
    - News sentiment (optional)
    """

    def __init__(self, max_pending: int = 50, expiry_hours: int = 24):
        """
        Args:
            max_pending: Maximum số signals pending
            expiry_hours: Số giờ trước khi signal expire (auto reject)
        """
        self._lock = threading.Lock()
        self._pending_signals: Dict[str, Dict] = {}
        self.max_pending = max_pending
        self.expiry_hours = expiry_hours

    def add_pending_signal(
        self,
        symbol: str,
        entry_signal: EntrySignal,
        position_size: EnhancedPositionSize,
        news_sentiment: Optional[Dict] = None,
    ) -> bool:
        """
        Add signal to pending approval queue

        Returns:
            True if added, False if already exists or queue full
        """
        with self._lock:
            if symbol in self._pending_signals:
                return False

            if len(self._pending_signals) >= self.max_pending:
                # Remove oldest if queue full
                oldest = min(
                    self._pending_signals.items(),
                    key=lambda x: x[1]["timestamp"],
                )
                del self._pending_signals[oldest[0]]

            self._pending_signals[symbol] = {
                "symbol": symbol,
                "entry_signal": entry_signal,
                "position_size": position_size,
                "news_sentiment": news_sentiment,
                "timestamp": datetime.now(),
            }
            return True

    def get_pending_signal(self, symbol: str) -> Optional[Dict]:
        """Get pending signal by symbol"""
        with self._lock:
            signal = self._pending_signals.get(symbol)
            if signal and self._is_expired(signal):
                del self._pending_signals[symbol]
                return None
            return signal

    def remove_pending_signal(self, symbol: str) -> bool:
        """Remove pending signal (after approval/rejection)"""
        with self._lock:
            if symbol in self._pending_signals:
                del self._pending_signals[symbol]
                return True
            return False

    def is_pending(self, symbol: str) -> bool:
        """Check if symbol has pending signal"""
        with self._lock:
            if symbol not in self._pending_signals:
                return False
            signal = self._pending_signals[symbol]
            if self._is_expired(signal):
                del self._pending_signals[symbol]
                return False
            return True

    def get_all_pending(self) -> Dict[str, Dict]:
        """Get all non-expired pending signals"""
        with self._lock:
            expired = [
                symbol
                for symbol, signal in self._pending_signals.items()
                if self._is_expired(signal)
            ]
            for symbol in expired:
                del self._pending_signals[symbol]
            return self._pending_signals.copy()

    def clear_expired(self):
        """Remove all expired signals"""
        with self._lock:
            expired = [
                symbol
                for symbol, signal in self._pending_signals.items()
                if self._is_expired(signal)
            ]
            for symbol in expired:
                del self._pending_signals[symbol]

    def _is_expired(self, signal: Dict) -> bool:
        """Check if signal is expired"""
        age = (datetime.now() - signal["timestamp"]).total_seconds() / 3600
        return age > self.expiry_hours

    def clear_all(self):
        """Clear all pending signals"""
        with self._lock:
            self._pending_signals.clear()


# Global singleton instance
_signal_approval_store = None


def get_signal_approval_store() -> SignalApprovalStore:
    """Get singleton instance"""
    global _signal_approval_store
    if _signal_approval_store is None:
        _signal_approval_store = SignalApprovalStore()
    return _signal_approval_store
