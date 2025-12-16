# -*- coding: utf-8 -*-
"""
Paper Trading Validation Module

Compares paper trading performance with backtest results to validate
strategy effectiveness and identify discrepancies.

Features:
- Performance comparison (Paper vs Backtest)
- Slippage analysis
- Timing analysis
- Execution quality metrics
- Drift detection

Author: Trading Bot Team
Version: 1.0.0
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Common performance metrics for comparison"""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_return_pct: float = 0.0
    avg_return_per_trade: float = 0.0
    max_drawdown_pct: float = 0.0

    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0

    avg_holding_days: float = 0.0
    total_slippage_pct: float = 0.0
    total_commission_pct: float = 0.0


@dataclass
class ValidationResult:
    """Result of paper trading validation"""

    is_valid: bool
    confidence_score: float  # 0-100

    paper_metrics: PerformanceMetrics
    backtest_metrics: PerformanceMetrics

    # Discrepancy analysis
    return_drift_pct: float  # Difference in returns
    win_rate_drift_pct: float  # Difference in win rate
    slippage_drift_pct: float  # Difference in slippage

    # Alerts
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Detailed comparison
    trade_by_trade: List[Dict] = field(default_factory=list)


@dataclass
class TradeComparison:
    """Compare single trade between paper and backtest"""

    symbol: str
    paper_entry_price: float
    backtest_entry_price: float
    paper_exit_price: float
    backtest_exit_price: float
    paper_pnl_pct: float
    backtest_pnl_pct: float
    pnl_difference_pct: float
    entry_slippage_pct: float
    exit_slippage_pct: float


class PaperTradingValidator:
    """
    Validate paper trading performance against backtesting results.

    This helps identify:
    1. Execution quality issues (higher slippage than expected)
    2. Timing differences (market entry/exit timing)
    3. Strategy drift (divergence from backtest expectations)
    4. Data feed issues (price discrepancies)

    Usage:
        validator = PaperTradingValidator()

        # Load paper trading history
        validator.load_paper_trades("paper_trading.json")

        # Load backtest results
        validator.load_backtest_results("backtest_results/latest.json")

        # Compare
        result = validator.validate()

        if not result.is_valid:
            print("⚠️ Paper trading diverging from backtest!")
            for warning in result.warnings:
                print(f"  - {warning}")
    """

    # Thresholds for validation
    MAX_RETURN_DRIFT_PCT = 10.0  # Max 10% difference in returns
    MAX_WIN_RATE_DRIFT_PCT = 15.0  # Max 15% difference in win rate
    MAX_SLIPPAGE_DRIFT_PCT = 0.5  # Max 0.5% higher slippage
    MIN_CONFIDENCE_SCORE = 60.0  # Minimum confidence to be valid

    def __init__(
        self,
        paper_trades_file: str = "paper_trading.json",
        backtest_results_dir: str = "backtest_results",
    ):
        self.paper_trades_file = paper_trades_file
        self.backtest_results_dir = backtest_results_dir

        self._paper_trades: List[Dict] = []
        self._backtest_trades: List[Dict] = []
        self._paper_account: Dict = {}
        self._backtest_result: Dict = {}

    def load_paper_trades(self, file_path: Optional[str] = None) -> int:
        """
        Load paper trading history.

        Returns:
            Number of trades loaded
        """
        file_path = file_path or self.paper_trades_file

        if not os.path.exists(file_path):
            logger.warning(f"Paper trading file not found: {file_path}")
            return 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self._paper_account = json.load(f)
                self._paper_trades = self._paper_account.get("trades", [])

            logger.info(f"📂 Loaded {len(self._paper_trades)} paper trades")
            return len(self._paper_trades)

        except Exception as e:
            logger.error(f"Failed to load paper trades: {e}")
            return 0

    def load_backtest_results(
        self,
        file_path: Optional[str] = None,
        use_latest: bool = True,
    ) -> int:
        """
        Load backtest results.

        Args:
            file_path: Specific backtest file to load
            use_latest: If True and file_path not specified, use latest

        Returns:
            Number of trades loaded
        """
        if file_path is None and use_latest:
            # Find latest backtest file
            file_path = self._find_latest_backtest()

        if file_path is None or not os.path.exists(file_path):
            logger.warning(f"Backtest file not found: {file_path}")
            return 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self._backtest_result = json.load(f)
                self._backtest_trades = self._backtest_result.get("trades", [])

            logger.info(f"📂 Loaded {len(self._backtest_trades)} backtest trades from {file_path}")
            return len(self._backtest_trades)

        except Exception as e:
            logger.error(f"Failed to load backtest results: {e}")
            return 0

    def _find_latest_backtest(self) -> Optional[str]:
        """Find the latest backtest results file."""
        if not os.path.exists(self.backtest_results_dir):
            return None

        files = [
            f
            for f in os.listdir(self.backtest_results_dir)
            if f.endswith(".json") and f.startswith(("comprehensive", "backtest"))
        ]

        if not files:
            return None

        # Sort by modification time
        files.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.backtest_results_dir, x)),
            reverse=True,
        )

        return os.path.join(self.backtest_results_dir, files[0])

    def validate(self) -> ValidationResult:
        """
        Validate paper trading against backtest.

        Returns:
            ValidationResult with detailed comparison
        """
        # Calculate metrics for both
        paper_metrics = self._calculate_metrics(self._paper_trades, is_paper=True)
        backtest_metrics = self._calculate_metrics(self._backtest_trades, is_paper=False)

        # Calculate drifts
        return_drift = abs(paper_metrics.total_return_pct - backtest_metrics.total_return_pct)
        win_rate_drift = abs(paper_metrics.win_rate - backtest_metrics.win_rate)
        slippage_drift = paper_metrics.total_slippage_pct - backtest_metrics.total_slippage_pct

        # Generate warnings
        warnings = []
        recommendations = []

        if return_drift > self.MAX_RETURN_DRIFT_PCT:
            warnings.append(
                f"Return drift {return_drift:.1f}% exceeds threshold ({self.MAX_RETURN_DRIFT_PCT}%)"
            )
            recommendations.append("Review entry/exit timing and execution quality")

        if win_rate_drift > self.MAX_WIN_RATE_DRIFT_PCT:
            warnings.append(
                f"Win rate drift {win_rate_drift:.1f}% exceeds threshold ({self.MAX_WIN_RATE_DRIFT_PCT}%)"
            )
            recommendations.append("Check signal generation and filtering logic")

        if slippage_drift > self.MAX_SLIPPAGE_DRIFT_PCT:
            warnings.append(f"Slippage {slippage_drift:.2f}% higher than backtest")
            recommendations.append("Consider using limit orders or splitting large orders")

        if paper_metrics.total_trades < 10:
            warnings.append("Insufficient paper trades for reliable validation")
            recommendations.append("Continue paper trading to gather more data")

        # Calculate confidence score
        confidence = self._calculate_confidence(
            return_drift, win_rate_drift, slippage_drift, paper_metrics.total_trades
        )

        # Trade-by-trade comparison (for overlapping symbols)
        trade_by_trade = self._compare_trades()

        is_valid = confidence >= self.MIN_CONFIDENCE_SCORE and len(warnings) <= 2

        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence,
            paper_metrics=paper_metrics,
            backtest_metrics=backtest_metrics,
            return_drift_pct=return_drift,
            win_rate_drift_pct=win_rate_drift,
            slippage_drift_pct=slippage_drift,
            warnings=warnings,
            recommendations=recommendations,
            trade_by_trade=trade_by_trade,
        )

    def _calculate_metrics(
        self,
        trades: List[Dict],
        is_paper: bool = True,
    ) -> PerformanceMetrics:
        """Calculate performance metrics from trade list."""
        if not trades:
            return PerformanceMetrics()

        # Filter completed trades
        if is_paper:
            completed = [t for t in trades if t.get("action", "").startswith("SELL")]
        else:
            completed = [t for t in trades if t.get("exit_price") or t.get("pnl")]

        if not completed:
            return PerformanceMetrics()

        total_trades = len(completed)

        # Calculate P&L for each trade
        pnls = []
        for trade in completed:
            if is_paper:
                # Paper trading: estimate P&L from price
                pnl_pct = trade.get("pnl_percent", 0)
                if pnl_pct == 0 and "price" in trade:
                    # Try to calculate from related buy
                    pnl_pct = 0  # Would need matching logic
                pnls.append(pnl_pct)
            else:
                # Backtest: has direct P&L
                pnl_pct = trade.get("pnl_percent", trade.get("pnl", 0))
                pnls.append(pnl_pct)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        total_return = sum(pnls)
        avg_return = np.mean(pnls) if pnls else 0

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Slippage and commission (estimate)
        slippage_pct = 0.0
        commission_pct = 0.0
        for trade in completed:
            slippage_pct += (
                trade.get("slippage_cost", 0)
                / max(trade.get("price", 1) * trade.get("shares", 1), 1)
                * 100
            )
            commission_pct += (
                trade.get("commission", 0)
                / max(trade.get("price", 1) * trade.get("shares", 1), 1)
                * 100
            )

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_return_pct=total_return,
            avg_return_per_trade=avg_return,
            profit_factor=profit_factor,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            total_slippage_pct=slippage_pct / total_trades if total_trades > 0 else 0,
            total_commission_pct=commission_pct / total_trades if total_trades > 0 else 0,
        )

    def _calculate_confidence(
        self,
        return_drift: float,
        win_rate_drift: float,
        slippage_drift: float,
        num_trades: int,
    ) -> float:
        """Calculate confidence score for validation."""
        score = 100.0

        # Penalize return drift
        score -= min(30, return_drift * 2)

        # Penalize win rate drift
        score -= min(25, win_rate_drift * 1.5)

        # Penalize excess slippage
        score -= min(20, slippage_drift * 20)

        # Penalize low trade count
        if num_trades < 10:
            score -= 20
        elif num_trades < 20:
            score -= 10
        elif num_trades < 30:
            score -= 5

        return max(0, min(100, score))

    def _compare_trades(self) -> List[Dict]:
        """Compare individual trades between paper and backtest."""
        comparisons = []

        # Group paper trades by symbol
        paper_by_symbol = {}
        for trade in self._paper_trades:
            symbol = trade.get("symbol", "")
            if symbol not in paper_by_symbol:
                paper_by_symbol[symbol] = []
            paper_by_symbol[symbol].append(trade)

        # Group backtest trades by symbol
        backtest_by_symbol = {}
        for trade in self._backtest_trades:
            symbol = trade.get("symbol", "")
            if symbol not in backtest_by_symbol:
                backtest_by_symbol[symbol] = []
            backtest_by_symbol[symbol].append(trade)

        # Compare overlapping symbols
        for symbol in set(paper_by_symbol.keys()) & set(backtest_by_symbol.keys()):
            paper_trades = paper_by_symbol[symbol]
            backtest_trades = backtest_by_symbol[symbol]

            # Simple comparison: average prices
            paper_avg_price = np.mean([t.get("price", 0) for t in paper_trades if t.get("price")])
            backtest_avg_entry = np.mean(
                [t.get("entry_price", 0) for t in backtest_trades if t.get("entry_price")]
            )

            if paper_avg_price > 0 and backtest_avg_entry > 0:
                price_diff_pct = (paper_avg_price - backtest_avg_entry) / backtest_avg_entry * 100

                comparisons.append(
                    {
                        "symbol": symbol,
                        "paper_trades": len(paper_trades),
                        "backtest_trades": len(backtest_trades),
                        "paper_avg_price": paper_avg_price,
                        "backtest_avg_price": backtest_avg_entry,
                        "price_difference_pct": round(price_diff_pct, 2),
                    }
                )

        return comparisons

    def generate_report(self, result: Optional[ValidationResult] = None) -> str:
        """Generate human-readable validation report."""
        if result is None:
            result = self.validate()

        lines = [
            "=" * 80,
            "PAPER TRADING VALIDATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"{'Status:':<20} {'✅ VALID' if result.is_valid else '❌ INVALID'}",
            f"{'Confidence Score:':<20} {result.confidence_score:.1f}/100",
            "",
            "PERFORMANCE COMPARISON",
            "-" * 40,
            f"{'Metric':<25} {'Paper':>15} {'Backtest':>15} {'Drift':>10}",
            "-" * 65,
            f"{'Total Trades':<25} {result.paper_metrics.total_trades:>15} {result.backtest_metrics.total_trades:>15}",
            f"{'Win Rate':<25} {result.paper_metrics.win_rate:>14.1f}% {result.backtest_metrics.win_rate:>14.1f}% {result.win_rate_drift_pct:>9.1f}%",
            f"{'Total Return':<25} {result.paper_metrics.total_return_pct:>14.1f}% {result.backtest_metrics.total_return_pct:>14.1f}% {result.return_drift_pct:>9.1f}%",
            f"{'Profit Factor':<25} {result.paper_metrics.profit_factor:>15.2f} {result.backtest_metrics.profit_factor:>15.2f}",
            f"{'Avg Slippage':<25} {result.paper_metrics.total_slippage_pct:>14.2f}% {result.backtest_metrics.total_slippage_pct:>14.2f}% {result.slippage_drift_pct:>9.2f}%",
            "",
        ]

        if result.warnings:
            lines.extend(
                [
                    "⚠️ WARNINGS",
                    "-" * 40,
                ]
            )
            for warning in result.warnings:
                lines.append(f"  • {warning}")
            lines.append("")

        if result.recommendations:
            lines.extend(
                [
                    "💡 RECOMMENDATIONS",
                    "-" * 40,
                ]
            )
            for rec in result.recommendations:
                lines.append(f"  • {rec}")
            lines.append("")

        if result.trade_by_trade:
            lines.extend(
                [
                    "SYMBOL COMPARISON",
                    "-" * 40,
                ]
            )
            for comp in result.trade_by_trade[:10]:  # Top 10
                lines.append(
                    f"  {comp['symbol']}: Paper={comp['paper_trades']} trades, "
                    f"Backtest={comp['backtest_trades']} trades, "
                    f"Price diff={comp['price_difference_pct']:+.2f}%"
                )

        lines.append("=" * 80)

        return "\n".join(lines)

    def save_report(
        self,
        result: Optional[ValidationResult] = None,
        output_file: str = "paper_validation_report.txt",
    ):
        """Save validation report to file."""
        report = self.generate_report(result)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"📄 Validation report saved to {output_file}")


# Singleton instance
_validator_instance: Optional[PaperTradingValidator] = None


def get_paper_trading_validator() -> PaperTradingValidator:
    """Get singleton validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = PaperTradingValidator()
    return _validator_instance


def quick_validate() -> ValidationResult:
    """Quick validation using default files."""
    validator = get_paper_trading_validator()
    validator.load_paper_trades()
    validator.load_backtest_results(use_latest=True)
    return validator.validate()


# CLI for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🔍 Paper Trading Validation")
    print("-" * 40)

    validator = PaperTradingValidator()

    # Load data
    paper_count = validator.load_paper_trades()
    backtest_count = validator.load_backtest_results(use_latest=True)

    print(f"Paper trades: {paper_count}")
    print(f"Backtest trades: {backtest_count}")

    if paper_count > 0 and backtest_count > 0:
        # Validate
        result = validator.validate()

        # Print report
        print("\n" + validator.generate_report(result))

        # Save report
        validator.save_report(result)
    else:
        print("⚠️ Insufficient data for validation")
