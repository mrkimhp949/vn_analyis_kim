"""
Signal Performance Tracker
Track and compare performance of ML vs Technical-only signals
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SignalPerformance:
    """Performance metrics for a signal source"""

    source: str  # 'ml' or 'technical_only'
    total_signals: int
    executed_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_return: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    sharpe_ratio: float
    max_drawdown: float
    total_pnl: float
    last_updated: str


class SignalPerformanceTracker:
    """
    Track and compare ML vs Technical-only signal performance

    Features:
    - Separate tracking for ML and technical-only signals
    - Performance metrics: win rate, avg return, Sharpe, etc.
    - Automatic disable of underperforming source
    - JSON persistence
    """

    def __init__(
        self,
        stats_file: str = "signal_performance.json",
        min_trades_for_evaluation: int = 20,
        min_win_rate_threshold: float = 0.40,  # Disable if win rate < 40%
        auto_disable: bool = True,
    ):
        """
        Args:
            stats_file: Path to persist performance stats
            min_trades_for_evaluation: Minimum trades before evaluating performance
            min_win_rate_threshold: Minimum win rate to keep source enabled
            auto_disable: Auto-disable underperforming source
        """
        self.stats_file = stats_file
        self.min_trades_for_evaluation = min_trades_for_evaluation
        self.min_win_rate_threshold = min_win_rate_threshold
        self.auto_disable = auto_disable

        # Load stats
        self.stats = self._load_stats()

        # Tracking
        self.ml_enabled = self.stats.get("ml_enabled", True)
        self.technical_enabled = self.stats.get("technical_enabled", True)

    def _load_stats(self) -> Dict:
        """Load stats from file"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load signal performance stats: {e}")

        return {
            "ml": {
                "signals": [],
                "trades": [],
            },
            "technical_only": {
                "signals": [],
                "trades": [],
            },
            "ml_enabled": True,
            "technical_enabled": True,
            "last_evaluation": None,
        }

    def _save_stats(self):
        """Save stats to file"""
        try:
            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save signal performance stats: {e}")

    def record_signal(self, symbol: str, signal_source: str, confidence: float, entry_price: float):
        """
        Record a new signal

        Args:
            symbol: Stock symbol
            signal_source: 'ml' or 'technical_only'
            confidence: Signal confidence (0-100)
            entry_price: Entry price
        """
        if signal_source not in ["ml", "technical_only"]:
            logger.warning(f"Invalid signal source: {signal_source}")
            return

        signal_data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "entry_price": entry_price,
        }

        self.stats[signal_source]["signals"].append(signal_data)
        self._save_stats()

        logger.info(
            f"📊 Recorded {signal_source} signal for {symbol} (confidence: {confidence:.1f}%)"
        )

    def record_trade_result(
        self,
        symbol: str,
        signal_source: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        holding_days: int,
    ):
        """
        Record trade result

        Args:
            symbol: Stock symbol
            signal_source: 'ml' or 'technical_only'
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/loss amount
            pnl_pct: Profit/loss percentage
            holding_days: Days held
        """
        if signal_source not in ["ml", "technical_only"]:
            logger.warning(f"Invalid signal source: {signal_source}")
            return

        trade_data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_days": holding_days,
            "is_win": pnl > 0,
        }

        self.stats[signal_source]["trades"].append(trade_data)
        self._save_stats()

        # Evaluate performance after recording
        self._evaluate_performance()

        logger.info(
            f"📊 Recorded {signal_source} trade result for {symbol}: {pnl_pct:+.2f}% in {holding_days} days"
        )

    def get_performance(self, signal_source: str) -> Optional[SignalPerformance]:
        """
        Get performance metrics for a signal source

        Args:
            signal_source: 'ml' or 'technical_only'

        Returns:
            SignalPerformance object or None
        """
        if signal_source not in ["ml", "technical_only"]:
            return None

        trades = self.stats[signal_source].get("trades", [])
        if len(trades) == 0:
            return None

        # Calculate metrics
        total_trades = len(trades)
        wins = [t for t in trades if t["is_win"]]
        losses = [t for t in trades if not t["is_win"]]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        avg_win = sum(t["pnl_pct"] for t in wins) / win_count if win_count > 0 else 0
        avg_loss = sum(t["pnl_pct"] for t in losses) / loss_count if loss_count > 0 else 0
        avg_return = sum(t["pnl_pct"] for t in trades) / total_trades

        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Sharpe ratio (simplified)
        returns = [t["pnl_pct"] for t in trades]
        import numpy as np

        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Max drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in trades:
            cumulative += t["pnl_pct"]
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        total_pnl = sum(t["pnl"] for t in trades)
        total_signals = len(self.stats[signal_source].get("signals", []))

        return SignalPerformance(
            source=signal_source,
            total_signals=total_signals,
            executed_trades=total_trades,
            wins=win_count,
            losses=loss_count,
            win_rate=win_rate,
            avg_return=avg_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_pnl=total_pnl,
            last_updated=datetime.now().isoformat(),
        )

    def get_comparison_report(self) -> str:
        """Generate comparison report between ML and Technical"""
        ml_perf = self.get_performance("ml")
        tech_perf = self.get_performance("technical_only")

        report = []
        report.append("=" * 70)
        report.append("📊 ML vs TECHNICAL-ONLY SIGNAL PERFORMANCE COMPARISON")
        report.append("=" * 70)
        report.append("")

        if ml_perf:
            report.append("🤖 **ML SIGNALS:**")
            report.append(f"   Total Signals: {ml_perf.total_signals}")
            report.append(f"   Executed Trades: {ml_perf.executed_trades}")
            report.append(f"   Win Rate: {ml_perf.win_rate:.1%}")
            report.append(f"   Avg Return: {ml_perf.avg_return:+.2f}%")
            report.append(f"   Avg Win: {ml_perf.avg_win:+.2f}%")
            report.append(f"   Avg Loss: {ml_perf.avg_loss:+.2f}%")
            report.append(f"   Win/Loss Ratio: {ml_perf.win_loss_ratio:.2f}")
            report.append(f"   Sharpe Ratio: {ml_perf.sharpe_ratio:.2f}")
            report.append(f"   Max Drawdown: {ml_perf.max_drawdown:.2f}%")
            report.append(f"   Total P&L: {ml_perf.total_pnl:+,.0f} VND")
            report.append(f"   Status: {'✅ ENABLED' if self.ml_enabled else '❌ DISABLED'}")
        else:
            report.append("🤖 **ML SIGNALS:** No data yet")

        report.append("")

        if tech_perf:
            report.append("📈 **TECHNICAL-ONLY SIGNALS:**")
            report.append(f"   Total Signals: {tech_perf.total_signals}")
            report.append(f"   Executed Trades: {tech_perf.executed_trades}")
            report.append(f"   Win Rate: {tech_perf.win_rate:.1%}")
            report.append(f"   Avg Return: {tech_perf.avg_return:+.2f}%")
            report.append(f"   Avg Win: {tech_perf.avg_win:+.2f}%")
            report.append(f"   Avg Loss: {tech_perf.avg_loss:+.2f}%")
            report.append(f"   Win/Loss Ratio: {tech_perf.win_loss_ratio:.2f}")
            report.append(f"   Sharpe Ratio: {tech_perf.sharpe_ratio:.2f}")
            report.append(f"   Max Drawdown: {tech_perf.max_drawdown:.2f}%")
            report.append(f"   Total P&L: {tech_perf.total_pnl:+,.0f} VND")
            report.append(f"   Status: {'✅ ENABLED' if self.technical_enabled else '❌ DISABLED'}")
        else:
            report.append("📈 **TECHNICAL-ONLY SIGNALS:** No data yet")

        report.append("")
        report.append("=" * 70)

        # Winner
        if ml_perf and tech_perf:
            if ml_perf.win_rate > tech_perf.win_rate:
                report.append("🏆 WINNER: ML Signals (higher win rate)")
            elif tech_perf.win_rate > ml_perf.win_rate:
                report.append("🏆 WINNER: Technical-Only Signals (higher win rate)")
            else:
                report.append("🤝 TIE: Equal win rates")

        return "\n".join(report)

    def _evaluate_performance(self):
        """Evaluate and auto-disable underperforming sources"""
        if not self.auto_disable:
            return

        # Evaluate ML
        ml_perf = self.get_performance("ml")
        if ml_perf and ml_perf.executed_trades >= self.min_trades_for_evaluation:
            if ml_perf.win_rate < self.min_win_rate_threshold:
                self.ml_enabled = False
                self.stats["ml_enabled"] = False
                logger.warning(
                    f"⚠️ AUTO-DISABLED ML signals due to low win rate: {ml_perf.win_rate:.1%} "
                    f"< {self.min_win_rate_threshold:.1%}"
                )

        # Evaluate Technical
        tech_perf = self.get_performance("technical_only")
        if tech_perf and tech_perf.executed_trades >= self.min_trades_for_evaluation:
            if tech_perf.win_rate < self.min_win_rate_threshold:
                self.technical_enabled = False
                self.stats["technical_enabled"] = False
                logger.warning(
                    f"⚠️ AUTO-DISABLED Technical-only signals due to low win rate: {tech_perf.win_rate:.1%} "
                    f"< {self.min_win_rate_threshold:.1%}"
                )

        self.stats["last_evaluation"] = datetime.now().isoformat()
        self._save_stats()

    def is_ml_enabled(self) -> bool:
        """Check if ML signals are enabled"""
        return self.ml_enabled

    def is_technical_enabled(self) -> bool:
        """Check if technical-only signals are enabled"""
        return self.technical_enabled

    def log_comparison_report(self):
        """Generate and log comparison report"""
        report = self.get_comparison_report()
        logger.info(f"\n{report}")
        return report

    def get_recommendation(self) -> str:
        """
        Get recommendation on which signal source to use

        Returns:
            Recommendation string with winner and reasoning
        """
        ml_perf = self.get_performance("ml")
        tech_perf = self.get_performance("technical_only")

        if not ml_perf and not tech_perf:
            return "⚠️ No data yet - continue monitoring both sources"

        if not ml_perf:
            return "📈 Use TECHNICAL-ONLY signals (no ML data yet)"

        if not tech_perf:
            return "🤖 Use ML signals (no technical-only data yet)"

        # Both have data - compare metrics
        ml_score = (
            ml_perf.win_rate * 0.4
            + (ml_perf.win_loss_ratio / 5.0) * 0.3  # Normalize to 0-1 range
            + (ml_perf.sharpe_ratio / 3.0) * 0.3
        )

        tech_score = (
            tech_perf.win_rate * 0.4
            + (tech_perf.win_loss_ratio / 5.0) * 0.3
            + (tech_perf.sharpe_ratio / 3.0) * 0.3
        )

        if ml_score > tech_score * 1.1:  # ML needs to be 10% better
            return (
                f"🤖 Recommend ML signals\n"
                f"   ML: WR={ml_perf.win_rate:.1%}, W/L={ml_perf.win_loss_ratio:.2f}, Sharpe={ml_perf.sharpe_ratio:.2f}\n"
                f"   Tech: WR={tech_perf.win_rate:.1%}, W/L={tech_perf.win_loss_ratio:.2f}, Sharpe={tech_perf.sharpe_ratio:.2f}"
            )
        elif tech_score > ml_score * 1.1:
            return (
                f"📈 Recommend TECHNICAL-ONLY signals\n"
                f"   Tech: WR={tech_perf.win_rate:.1%}, W/L={tech_perf.win_loss_ratio:.2f}, Sharpe={tech_perf.sharpe_ratio:.2f}\n"
                f"   ML: WR={ml_perf.win_rate:.1%}, W/L={ml_perf.win_loss_ratio:.2f}, Sharpe={ml_perf.sharpe_ratio:.2f}"
            )
        else:
            return (
                f"🤝 Both sources performing similarly - use BOTH for diversification\n"
                f"   ML: WR={ml_perf.win_rate:.1%}, W/L={ml_perf.win_loss_ratio:.2f}\n"
                f"   Tech: WR={tech_perf.win_rate:.1%}, W/L={tech_perf.win_loss_ratio:.2f}"
            )

    def reset(self):
        """Reset all stats (for testing)"""
        self.stats = {
            "ml": {"signals": [], "trades": []},
            "technical_only": {"signals": [], "trades": []},
            "ml_enabled": True,
            "technical_enabled": True,
            "last_evaluation": None,
        }
        self.ml_enabled = True
        self.technical_enabled = True
        self._save_stats()
        logger.info("Signal performance tracker reset")


# Singleton
_tracker = None


def get_signal_tracker() -> SignalPerformanceTracker:
    """Get singleton instance"""
    global _tracker
    if _tracker is None:
        _tracker = SignalPerformanceTracker()
    return _tracker
