"""
Signal Performance Tracker
Track and compare ML vs Technical-only signal performance
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SignalPerformance:
    """Performance metrics for a signal type"""

    total_signals: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    sharpe_ratio: float = 0.0


class SignalPerformanceTracker:
    """
    Track performance of ML vs Technical-only signals separately

    Features:
    - Separate tracking for ML and Technical signals
    - Win rate, avg P&L, Sharpe ratio
    - JSON persistence
    - Thread-safe
    """

    def __init__(self, stats_file: str = "signal_performance.json"):
        self.stats_file = Path(stats_file)
        self._lock = RLock()

        # Performance by signal source
        self.ml_performance = SignalPerformance()
        self.technical_performance = SignalPerformance()

        # Load existing stats
        self._load_stats()

        logger.info("✅ Signal Performance Tracker initialized")

    def _load_stats(self):
        """Load stats from file"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "ml" in data:
                    self.ml_performance = SignalPerformance(**data["ml"])
                if "technical" in data:
                    self.technical_performance = SignalPerformance(**data["technical"])

                logger.info(f"📊 Loaded signal performance stats from {self.stats_file}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load signal performance stats: {e}")

    def _save_stats(self):
        """Save stats to file"""
        try:
            data = {
                "ml": asdict(self.ml_performance),
                "technical": asdict(self.technical_performance),
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Could not save signal performance stats: {e}")

    def track_signal(self, is_ml_signal: bool):
        """
        Track when a signal is generated

        Args:
            is_ml_signal: True if ML-based, False if technical-only
        """
        with self._lock:
            if is_ml_signal:
                self.ml_performance.total_signals += 1
            else:
                self.technical_performance.total_signals += 1
            self._save_stats()

    def track_trade(
        self,
        is_ml_signal: bool,
        entry_price: float,
        exit_price: float,
        shares: int,
    ):
        """
        Track trade performance

        Args:
            is_ml_signal: True if ML-based, False if technical-only
            entry_price: Entry price
            exit_price: Exit price
            shares: Number of shares
        """
        with self._lock:
            pnl = (exit_price - entry_price) * shares
            perf = self.ml_performance if is_ml_signal else self.technical_performance

            perf.total_trades += 1

            if pnl > 0:
                perf.winning_trades += 1
                perf.avg_profit = (
                    perf.avg_profit * (perf.winning_trades - 1) + pnl
                ) / perf.winning_trades
            else:
                perf.losing_trades += 1
                perf.avg_loss = (
                    perf.avg_loss * (perf.losing_trades - 1) + abs(pnl)
                ) / perf.losing_trades

            perf.total_pnl += pnl
            perf.win_rate = (
                perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0.0
            )
            perf.avg_pnl = perf.total_pnl / perf.total_trades if perf.total_trades > 0 else 0.0

            # Calculate Sharpe ratio (simplified)
            if perf.avg_loss > 0:
                perf.sharpe_ratio = (perf.avg_profit - perf.avg_loss) / perf.avg_loss
            else:
                perf.sharpe_ratio = 0.0

            self._save_stats()

            # Log significant milestones
            if perf.total_trades in [10, 50, 100, 500]:
                signal_type = "ML" if is_ml_signal else "Technical"
                logger.info(
                    f"📊 {signal_type} Signal Performance ({perf.total_trades} trades):\n"
                    f"   Win Rate: {perf.win_rate:.1%}\n"
                    f"   Avg P&L: {perf.avg_pnl:,.0f} VND\n"
                    f"   Sharpe: {perf.sharpe_ratio:.2f}"
                )

    def get_performance(self, is_ml_signal: bool) -> SignalPerformance:
        """Get performance for signal type"""
        with self._lock:
            return self.ml_performance if is_ml_signal else self.technical_performance

    def get_comparison_report(self) -> str:
        """
        Generate comparison report between ML and Technical signals

        Returns:
            Formatted comparison report
        """
        with self._lock:
            ml = self.ml_performance
            tech = self.technical_performance

            report = []
            report.append("=" * 60)
            report.append("📊 SIGNAL PERFORMANCE COMPARISON")
            report.append("=" * 60)

            # ML Performance
            report.append("\n🤖 ML SIGNALS:")
            report.append(f"  Total Signals: {ml.total_signals}")
            report.append(f"  Total Trades: {ml.total_trades}")
            if ml.total_trades > 0:
                report.append(f"  Win Rate: {ml.win_rate:.1%}")
                report.append(f"  Avg P&L: {ml.avg_pnl:,.0f} VND")
                report.append(f"  Avg Profit: {ml.avg_profit:,.0f} VND")
                report.append(f"  Avg Loss: {ml.avg_loss:,.0f} VND")
                report.append(f"  Sharpe Ratio: {ml.sharpe_ratio:.2f}")
                report.append(f"  Total P&L: {ml.total_pnl:,.0f} VND")

            # Technical Performance
            report.append("\n⚙️ TECHNICAL SIGNALS:")
            report.append(f"  Total Signals: {tech.total_signals}")
            report.append(f"  Total Trades: {tech.total_trades}")
            if tech.total_trades > 0:
                report.append(f"  Win Rate: {tech.win_rate:.1%}")
                report.append(f"  Avg P&L: {tech.avg_pnl:,.0f} VND")
                report.append(f"  Avg Profit: {tech.avg_profit:,.0f} VND")
                report.append(f"  Avg Loss: {tech.avg_loss:,.0f} VND")
                report.append(f"  Sharpe Ratio: {tech.sharpe_ratio:.2f}")
                report.append(f"  Total P&L: {tech.total_pnl:,.0f} VND")

            # Comparison (if both have trades)
            if ml.total_trades > 0 and tech.total_trades > 0:
                report.append("\n🔍 COMPARISON:")
                win_rate_diff = ml.win_rate - tech.win_rate
                report.append(f"  Win Rate Diff: {win_rate_diff:+.1%} (ML vs Tech)")

                pnl_diff = ml.avg_pnl - tech.avg_pnl
                report.append(f"  Avg P&L Diff: {pnl_diff:+,.0f} VND (ML vs Tech)")

                sharpe_diff = ml.sharpe_ratio - tech.sharpe_ratio
                report.append(f"  Sharpe Diff: {sharpe_diff:+.2f} (ML vs Tech)")

                # Recommendation
                if win_rate_diff > 0.05 and pnl_diff > 0:
                    report.append("\n💡 RECOMMENDATION: ML signals performing better")
                elif win_rate_diff < -0.05 and pnl_diff < 0:
                    report.append("\n💡 RECOMMENDATION: Technical signals performing better")
                else:
                    report.append("\n💡 RECOMMENDATION: Performance is similar")

            report.append("=" * 60)

            return "\n".join(report)


# Global instance
_tracker = None


def get_signal_performance_tracker() -> SignalPerformanceTracker:
    """Get singleton instance"""
    global _tracker
    if _tracker is None:
        _tracker = SignalPerformanceTracker()
    return _tracker


# Test
if __name__ == "__main__":
    print("Testing Signal Performance Tracker...\n")

    tracker = SignalPerformanceTracker(stats_file="test_signal_performance.json")

    # Simulate some ML trades
    for i in range(10):
        tracker.track_signal(is_ml_signal=True)
        # Simulate 70% win rate for ML
        if i < 7:
            tracker.track_trade(
                is_ml_signal=True,
                entry_price=100_000,
                exit_price=110_000,
                shares=100,
            )
        else:
            tracker.track_trade(
                is_ml_signal=True,
                entry_price=100_000,
                exit_price=95_000,
                shares=100,
            )

    # Simulate some Technical trades
    for i in range(10):
        tracker.track_signal(is_ml_signal=False)
        # Simulate 60% win rate for Technical
        if i < 6:
            tracker.track_trade(
                is_ml_signal=False,
                entry_price=100_000,
                exit_price=108_000,
                shares=100,
            )
        else:
            tracker.track_trade(
                is_ml_signal=False,
                entry_price=100_000,
                exit_price=96_000,
                shares=100,
            )

    # Print comparison
    print(tracker.get_comparison_report())

    # Cleanup
    import os

    if os.path.exists("test_signal_performance.json"):
        os.remove("test_signal_performance.json")

    print("\n✅ Test completed!")
