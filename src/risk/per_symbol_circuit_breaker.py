"""
Per-Symbol Circuit Breaker
Prevent repeatedly trading symbols that are losing
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from threading import RLock
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SymbolStats:
    """Stats per symbol"""

    symbol: str
    consecutive_losses: int
    total_trades: int
    total_wins: int
    total_losses: int
    win_rate: float
    last_trade_date: str
    blocked: bool
    blocked_reason: str


class PerSymbolCircuitBreaker:
    """
    Per-symbol circuit breaker to prevent trading bad symbols repeatedly.

    Features:
    - Track consecutive losses per symbol
    - Block symbol after N consecutive losses
    - Track win rate per symbol
    - Block symbol if win rate < threshold after minimum trades
    - Auto-unblock after cooldown period
    - Thread-safe operations
    """

    def __init__(
        self,
        max_consecutive_losses: int = 3,
        min_trades_for_winrate_check: int = 5,
        min_win_rate: float = 0.30,  # 30% minimum win rate
        cooldown_days: int = 7,  # Unblock after 7 days
        stats_file: str = "per_symbol_circuit_breaker.json",
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.min_trades_for_winrate_check = min_trades_for_winrate_check
        self.min_win_rate = min_win_rate
        self.cooldown_days = cooldown_days
        self.stats_file = stats_file

        # Thread safety
        self._lock = RLock()

        # Load stats from file
        self.symbol_stats: Dict[str, SymbolStats] = self._load_stats()

        logger.info(
            f"✅ Per-Symbol Circuit Breaker initialized: "
            f"max_losses={max_consecutive_losses}, "
            f"min_winrate={min_win_rate:.0%}"
        )

    def _load_stats(self) -> Dict[str, SymbolStats]:
        """Load stats from file"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    stats = {}
                    for symbol, stat_dict in data.items():
                        stats[symbol] = SymbolStats(**stat_dict)
                    return stats
            except Exception as e:
                logger.warning(f"⚠️ Error loading per-symbol stats: {e}")

        return {}

    def _save_stats(self):
        """Save stats to file"""
        try:
            data = {symbol: asdict(stat) for symbol, stat in self.symbol_stats.items()}
            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Error saving per-symbol stats: {e}")

    def can_trade(self, symbol: str) -> tuple[bool, str]:
        """
        Check if symbol can be traded

        Returns:
            (can_trade, reason)
        """
        with self._lock:
            # Auto-unblock if cooldown period passed
            self._check_cooldown(symbol)

            if symbol not in self.symbol_stats:
                return True, "✅ OK to trade (new symbol)"

            stats = self.symbol_stats[symbol]

            if stats.blocked:
                return False, stats.blocked_reason

            return True, "✅ OK to trade"

    def record_trade(self, symbol: str, is_win: bool, pnl_percent: float):
        """
        Record a trade for a symbol

        Args:
            symbol: Stock symbol
            is_win: True if trade was profitable
            pnl_percent: P&L percentage (e.g., 5.0 for +5%, -3.0 for -3%)
        """
        with self._lock:
            if symbol not in self.symbol_stats:
                self.symbol_stats[symbol] = SymbolStats(
                    symbol=symbol,
                    consecutive_losses=0,
                    total_trades=0,
                    total_wins=0,
                    total_losses=0,
                    win_rate=0.0,
                    last_trade_date=date.today().isoformat(),
                    blocked=False,
                    blocked_reason="",
                )

            stats = self.symbol_stats[symbol]

            # Update stats
            stats.total_trades += 1
            stats.last_trade_date = date.today().isoformat()

            if is_win:
                stats.total_wins += 1
                stats.consecutive_losses = 0  # Reset
            else:
                stats.total_losses += 1
                stats.consecutive_losses += 1

            # Calculate win rate
            stats.win_rate = (
                stats.total_wins / stats.total_trades if stats.total_trades > 0 else 0.0
            )

            # Check 1: Consecutive losses
            if stats.consecutive_losses >= self.max_consecutive_losses:
                stats.blocked = True
                stats.blocked_reason = f"🚫 Blocked: {stats.consecutive_losses} consecutive losses"
                logger.warning(
                    f"⚠️ Symbol {symbol} BLOCKED: {stats.consecutive_losses} consecutive losses"
                )

            # Check 2: Win rate too low
            if (
                stats.total_trades >= self.min_trades_for_winrate_check
                and stats.win_rate < self.min_win_rate
            ):
                stats.blocked = True
                stats.blocked_reason = (
                    f"🚫 Blocked: Win rate {stats.win_rate:.1%} < {self.min_win_rate:.1%} "
                    f"({stats.total_wins}/{stats.total_trades} trades)"
                )
                logger.warning(f"⚠️ Symbol {symbol} BLOCKED: Low win rate {stats.win_rate:.1%}")

            self._save_stats()

    def _check_cooldown(self, symbol: str):
        """Auto-unblock symbol if cooldown period has passed"""
        if symbol not in self.symbol_stats:
            return

        stats = self.symbol_stats[symbol]

        if not stats.blocked:
            return

        # Check if cooldown period has passed
        last_trade = datetime.fromisoformat(stats.last_trade_date).date()
        days_since_last_trade = (date.today() - last_trade).days

        if days_since_last_trade >= self.cooldown_days:
            logger.info(
                f"✅ Symbol {symbol} UNBLOCKED after {days_since_last_trade} days cooldown. "
                f"Resetting stats."
            )
            # Reset stats
            stats.consecutive_losses = 0
            stats.blocked = False
            stats.blocked_reason = ""
            self._save_stats()

    def get_symbol_stats(self, symbol: str) -> Optional[SymbolStats]:
        """Get stats for a symbol"""
        with self._lock:
            return self.symbol_stats.get(symbol)

    def get_all_blocked_symbols(self) -> list[str]:
        """Get list of all blocked symbols"""
        with self._lock:
            return [symbol for symbol, stats in self.symbol_stats.items() if stats.blocked]

    def unblock_symbol(self, symbol: str):
        """Manually unblock a symbol (for testing/admin)"""
        with self._lock:
            if symbol in self.symbol_stats:
                stats = self.symbol_stats[symbol]
                stats.blocked = False
                stats.blocked_reason = ""
                stats.consecutive_losses = 0
                self._save_stats()
                logger.info(f"✅ Symbol {symbol} manually unblocked")

    def get_status_message(self) -> str:
        """Get status message for all symbols"""
        with self._lock:
            blocked_symbols = self.get_all_blocked_symbols()

            msg = []
            msg.append("🔒 **PER-SYMBOL CIRCUIT BREAKER STATUS**")
            msg.append("=" * 50)
            msg.append(f"Blocked symbols: {len(blocked_symbols)}")

            if blocked_symbols:
                msg.append("\n🚫 **Blocked Symbols:**")
                for symbol in blocked_symbols:
                    stats = self.symbol_stats[symbol]
                    msg.append(f"  • {symbol}: {stats.blocked_reason}")
                    msg.append(
                        f"    Stats: {stats.total_wins}W/{stats.total_losses}L "
                        f"(Win rate: {stats.win_rate:.1%})"
                    )

            # Show top performing symbols
            performing_symbols = [
                (symbol, stats)
                for symbol, stats in self.symbol_stats.items()
                if not stats.blocked and stats.total_trades >= 3
            ]
            performing_symbols.sort(key=lambda x: x[1].win_rate, reverse=True)

            if performing_symbols:
                msg.append("\n✅ **Top Performing Symbols:**")
                for symbol, stats in performing_symbols[:5]:
                    msg.append(
                        f"  • {symbol}: {stats.total_wins}W/{stats.total_losses}L "
                        f"(Win rate: {stats.win_rate:.1%})"
                    )

            return "\n".join(msg)


# Global instance
_per_symbol_circuit_breaker = None


def get_per_symbol_circuit_breaker(
    max_consecutive_losses: int = 3,
    min_win_rate: float = 0.30,
) -> PerSymbolCircuitBreaker:
    """Get singleton instance"""
    global _per_symbol_circuit_breaker
    if _per_symbol_circuit_breaker is None:
        _per_symbol_circuit_breaker = PerSymbolCircuitBreaker(
            max_consecutive_losses=max_consecutive_losses,
            min_win_rate=min_win_rate,
        )
    return _per_symbol_circuit_breaker


# Test
if __name__ == "__main__":
    print("Testing Per-Symbol Circuit Breaker...")

    breaker = PerSymbolCircuitBreaker(
        max_consecutive_losses=3,
        min_trades_for_winrate_check=5,
        min_win_rate=0.30,
    )

    # Test 1: New symbol
    print("\n1️⃣ Test new symbol:")
    can_trade, reason = breaker.can_trade("VNM")
    print(f"  VNM: {reason}")

    # Test 2: Record some losses
    print("\n2️⃣ Test consecutive losses:")
    for i in range(4):
        breaker.record_trade("VNM", is_win=False, pnl_percent=-2.0)
        can_trade, reason = breaker.can_trade("VNM")
        print(f"  After loss {i+1}: {reason}")

    # Test 3: Low win rate
    print("\n3️⃣ Test low win rate:")
    for i in range(5):
        is_win = i < 1  # Only 1 win out of 5
        breaker.record_trade("HPG", is_win=is_win, pnl_percent=2.0 if is_win else -2.0)

    can_trade, reason = breaker.can_trade("HPG")
    print(f"  HPG: {reason}")

    # Test 4: Status
    print("\n4️⃣ Status:")
    print(breaker.get_status_message())

    print("\n✅ Test completed!")
