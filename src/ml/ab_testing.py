# -*- coding: utf-8 -*-
"""
A/B Testing Framework for ML vs Technical Analysis
Compare performance của ML và Technical signals side-by-side
"""

import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Nguồn signal"""

    ML_MODEL = "ml"
    TECHNICAL_ANALYSIS = "technical"
    ENSEMBLE = "ensemble"  # Combine ML + Technical


@dataclass
class ABTestConfig:
    """Configuration cho A/B test"""

    test_id: str
    name: str
    description: str
    start_date: str
    end_date: Optional[str] = None

    # Traffic allocation (must sum to 1.0)
    ml_allocation: float = 0.50  # 50% ML
    technical_allocation: float = 0.50  # 50% Technical

    # Minimum sample size for statistical significance
    min_sample_size: int = 30

    # Success metrics
    primary_metric: str = "win_rate"  # "win_rate", "avg_profit", "sharpe"
    min_improvement_threshold: float = 0.05  # 5% minimum improvement to declare winner


@dataclass
class ABTestResult:
    """Kết quả cho một variant (ML or Technical)"""

    variant: SignalSource
    total_signals: int
    buy_signals: int
    total_trades: int
    wins: int
    losses: int

    # Performance metrics
    win_rate: float
    avg_profit_pct: float
    total_profit: float
    sharpe_ratio: float
    max_drawdown: float

    # Confidence intervals (95%)
    win_rate_ci_lower: float
    win_rate_ci_upper: float


@dataclass
class ABTestSummary:
    """Tổng kết A/B test"""

    test_id: str
    test_name: str
    start_date: str
    end_date: str
    duration_days: int

    ml_result: ABTestResult
    technical_result: ABTestResult

    # Statistical test results
    winner: Optional[str] = None  # "ml", "technical", or None (no significant difference)
    p_value: float = 1.0  # P-value from statistical test
    is_significant: bool = False  # True if p < 0.05
    improvement_pct: float = 0.0  # % improvement of winner over loser

    recommendation: str = ""


class ABTestingFramework:
    """
    A/B Testing framework để compare ML vs Technical signals

    FEATURES:
    - Random allocation của symbols/trades to variants
    - Performance tracking per variant
    - Statistical significance testing (t-test, chi-square)
    - Winner selection with confidence intervals
    - Automated recommendations
    """

    def __init__(
        self,
        results_dir: str = "ab_tests",
        active_test_file: str = "ab_tests/active_test.json",
        results_log_file: str = "ab_tests/results_log.jsonl",
    ):
        """
        Args:
            results_dir: Directory for A/B test results
            active_test_file: JSON file for active test config
            results_log_file: JSONL file for test results logging
        """
        self.results_dir = Path(results_dir)
        self.active_test_file = Path(active_test_file)
        self.results_log_file = Path(results_log_file)

        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load active test
        self.active_test = self._load_active_test()

        # Performance trackers
        self.ml_tracker = PerformanceTracker()
        self.technical_tracker = PerformanceTracker()

    def start_test(self, config: ABTestConfig):
        """
        Start a new A/B test

        Args:
            config: ABTestConfig object
        """
        # Validate config
        if config.ml_allocation + config.technical_allocation != 1.0:
            raise ValueError("Allocations must sum to 1.0")

        # Reset trackers
        self.ml_tracker = PerformanceTracker()
        self.technical_tracker = PerformanceTracker()

        # Save active test
        self.active_test = asdict(config)
        self._save_active_test()

        logger.info(
            f"🧪 A/B Test started: {config.name}\n"
            f"   ML allocation: {config.ml_allocation:.0%}\n"
            f"   Technical allocation: {config.technical_allocation:.0%}\n"
            f"   Primary metric: {config.primary_metric}"
        )

    def allocate_signal(self, symbol: str) -> SignalSource:
        """
        Allocate symbol to a variant (ML or Technical)

        Uses deterministic hash-based allocation for consistency
        (same symbol always gets same variant)

        Args:
            symbol: Stock symbol

        Returns:
            SignalSource (ML or TECHNICAL)
        """
        if not self.active_test:
            # No active test - default to ML
            return SignalSource.ML_MODEL

        # Hash-based allocation (deterministic)
        hash_value = hash(symbol + self.active_test["test_id"])
        random.seed(hash_value)
        rand = random.random()

        ml_allocation = self.active_test["ml_allocation"]

        if rand < ml_allocation:
            return SignalSource.ML_MODEL
        else:
            return SignalSource.TECHNICAL_ANALYSIS

    def track_signal(
        self,
        variant: SignalSource,
        symbol: str,
        signal: str,  # "BUY", "SELL", "HOLD"
        confidence: float,
    ):
        """
        Track a generated signal

        Args:
            variant: Which variant generated this signal
            symbol: Stock symbol
            signal: Signal type
            confidence: Signal confidence
        """
        tracker = self._get_tracker(variant)
        tracker.track_signal(symbol, signal, confidence)

    def track_trade_result(
        self,
        variant: SignalSource,
        symbol: str,
        is_win: bool,
        profit_pct: float,
        holding_days: int,
    ):
        """
        Track trade result

        Args:
            variant: Which variant generated the signal
            symbol: Stock symbol
            is_win: True if profitable trade
            profit_pct: Profit/loss %
            holding_days: Days held
        """
        tracker = self._get_tracker(variant)
        tracker.track_trade(symbol, is_win, profit_pct, holding_days)

    def get_summary(self) -> Optional[ABTestSummary]:
        """
        Get current A/B test summary

        Returns:
            ABTestSummary or None if no active test
        """
        if not self.active_test:
            logger.warning("No active A/B test")
            return None

        # Calculate results for each variant
        ml_result = self._calculate_result(SignalSource.ML_MODEL, self.ml_tracker)

        technical_result = self._calculate_result(
            SignalSource.TECHNICAL_ANALYSIS, self.technical_tracker
        )

        # Check minimum sample size
        min_samples = self.active_test["min_sample_size"]
        if ml_result.total_trades < min_samples or technical_result.total_trades < min_samples:
            logger.info(
                f"ℹ️ Insufficient samples for significance test "
                f"(ML: {ml_result.total_trades}, Technical: {technical_result.total_trades}, "
                f"min: {min_samples})"
            )
            insufficient = True
        else:
            insufficient = False

        # Statistical significance test
        if not insufficient:
            winner, p_value, is_significant, improvement = self._statistical_test(
                ml_result, technical_result, self.active_test["primary_metric"]
            )
        else:
            winner = None
            p_value = 1.0
            is_significant = False
            improvement = 0.0

        # Generate recommendation
        recommendation = self._generate_recommendation(
            ml_result, technical_result, winner, is_significant, improvement
        )

        # Create summary
        start_date = self.active_test["start_date"]
        end_date = datetime.now().isoformat()
        duration = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days

        summary = ABTestSummary(
            test_id=self.active_test["test_id"],
            test_name=self.active_test["name"],
            start_date=start_date,
            end_date=end_date,
            duration_days=duration,
            ml_result=ml_result,
            technical_result=technical_result,
            winner=winner,
            p_value=p_value,
            is_significant=is_significant,
            improvement_pct=improvement,
            recommendation=recommendation,
        )

        return summary

    def end_test(self) -> Optional[ABTestSummary]:
        """
        End current A/B test and save results

        Returns:
            Final ABTestSummary
        """
        if not self.active_test:
            logger.warning("No active A/B test to end")
            return None

        # Get final summary
        summary = self.get_summary()

        # Log results
        self._log_test_results(summary)

        # Clear active test
        self.active_test = None
        if self.active_test_file.exists():
            self.active_test_file.unlink()

        logger.info(f"🏁 A/B Test ended: {summary.test_name}")
        logger.info(f"   Winner: {summary.winner or 'No significant difference'}")
        logger.info(f"   P-value: {summary.p_value:.4f}")
        logger.info(f"   Improvement: {summary.improvement_pct:+.1f}%")
        logger.info(f"\n{summary.recommendation}")

        return summary

    def _get_tracker(self, variant: SignalSource) -> "PerformanceTracker":
        """Get tracker for variant"""
        if variant == SignalSource.ML_MODEL:
            return self.ml_tracker
        else:
            return self.technical_tracker

    def _calculate_result(
        self, variant: SignalSource, tracker: "PerformanceTracker"
    ) -> ABTestResult:
        """Calculate result for a variant"""

        metrics = tracker.get_metrics()

        # Calculate confidence interval for win rate
        if metrics["total_trades"] > 0:
            win_rate_ci = self._wilson_confidence_interval(metrics["wins"], metrics["total_trades"])
        else:
            win_rate_ci = (0.0, 0.0)

        return ABTestResult(
            variant=variant,
            total_signals=metrics["total_signals"],
            buy_signals=metrics["buy_signals"],
            total_trades=metrics["total_trades"],
            wins=metrics["wins"],
            losses=metrics["losses"],
            win_rate=metrics["win_rate"],
            avg_profit_pct=metrics["avg_profit_pct"],
            total_profit=metrics["total_profit"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate_ci_lower=win_rate_ci[0],
            win_rate_ci_upper=win_rate_ci[1],
        )

    def _statistical_test(
        self, ml_result: ABTestResult, technical_result: ABTestResult, primary_metric: str
    ) -> tuple[Optional[str], float, bool, float]:
        """
        Perform statistical significance test

        Returns:
            (winner, p_value, is_significant, improvement_pct)
        """
        # Extract primary metric values
        ml_value = getattr(ml_result, primary_metric)
        tech_value = getattr(technical_result, primary_metric)

        # Determine winner by metric value
        if ml_value > tech_value:
            winner_candidate = "ml"
            improvement = ((ml_value - tech_value) / tech_value * 100) if tech_value > 0 else 0
        elif tech_value > ml_value:
            winner_candidate = "technical"
            improvement = ((tech_value - ml_value) / ml_value * 100) if ml_value > 0 else 0
        else:
            return None, 1.0, False, 0.0

        # Statistical test based on metric
        if primary_metric == "win_rate":
            # Chi-square test for win rate
            p_value = self._chi_square_test(ml_result, technical_result)
        else:
            # T-test for continuous metrics (avg_profit, sharpe)
            p_value = self._t_test(ml_result, technical_result, primary_metric)

        # Significant if p < 0.05
        is_significant = p_value < 0.05

        # Only declare winner if significant AND meets minimum improvement threshold
        min_improvement = self.active_test["min_improvement_threshold"] * 100
        if is_significant and improvement >= min_improvement:
            winner = winner_candidate
        else:
            winner = None
            improvement = 0.0

        return winner, p_value, is_significant, improvement

    def _chi_square_test(self, ml_result: ABTestResult, technical_result: ABTestResult) -> float:
        """Chi-square test for win rates"""
        try:
            # Contingency table
            observed = np.array(
                [
                    [ml_result.wins, ml_result.losses],
                    [technical_result.wins, technical_result.losses],
                ]
            )

            # Chi-square test
            chi2, p_value, _, _ = stats.chi2_contingency(observed)

            return p_value

        except Exception as e:
            logger.error(f"Error in chi-square test: {e}")
            return 1.0  # No significance

    def _t_test(
        self, ml_result: ABTestResult, technical_result: ABTestResult, metric: str
    ) -> float:
        """T-test for continuous metrics"""
        # Note: This is simplified - ideally we'd have individual trade data
        # For now, use approximation based on summary statistics

        ml_value = getattr(ml_result, metric)
        tech_value = getattr(technical_result, metric)

        # Use two-sample t-test approximation
        # Assumes normal distribution (reasonable for large samples)

        # Estimate standard errors
        ml_se = ml_value / np.sqrt(ml_result.total_trades) if ml_result.total_trades > 0 else 0
        tech_se = (
            tech_value / np.sqrt(technical_result.total_trades)
            if technical_result.total_trades > 0
            else 0
        )

        # Pooled standard error
        pooled_se = np.sqrt(ml_se**2 + tech_se**2)

        if pooled_se > 0:
            # T-statistic
            t_stat = (ml_value - tech_value) / pooled_se

            # Degrees of freedom (approximate)
            df = ml_result.total_trades + technical_result.total_trades - 2

            # P-value (two-tailed)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            p_value = 1.0

        return p_value

    def _wilson_confidence_interval(
        self, successes: int, total: int, confidence: float = 0.95
    ) -> tuple[float, float]:
        """
        Wilson score confidence interval for win rate

        More accurate than normal approximation for small samples
        """
        if total == 0:
            return (0.0, 0.0)

        p = successes / total
        z = stats.norm.ppf((1 + confidence) / 2)

        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return (lower, upper)

    def _generate_recommendation(
        self,
        ml_result: ABTestResult,
        technical_result: ABTestResult,
        winner: Optional[str],
        is_significant: bool,
        improvement: float,
    ) -> str:
        """Generate actionable recommendation"""

        if winner == "ml":
            return (
                f"✅ RECOMMENDATION: Use ML Model\n"
                f"   ML outperforms Technical by {improvement:.1f}%\n"
                f"   Statistical significance: p < 0.05\n"
                f"   ML win rate: {ml_result.win_rate:.1%} (95% CI: [{ml_result.win_rate_ci_lower:.1%}, {ml_result.win_rate_ci_upper:.1%}])\n"
                f"   Technical win rate: {technical_result.win_rate:.1%} (95% CI: [{technical_result.win_rate_ci_lower:.1%}, {technical_result.win_rate_ci_upper:.1%}])"
            )

        elif winner == "technical":
            return (
                f"✅ RECOMMENDATION: Use Technical Analysis\n"
                f"   Technical outperforms ML by {improvement:.1f}%\n"
                f"   Statistical significance: p < 0.05\n"
                f"   Technical win rate: {technical_result.win_rate:.1%} (95% CI: [{technical_result.win_rate_ci_lower:.1%}, {technical_result.win_rate_ci_upper:.1%}])\n"
                f"   ML win rate: {ml_result.win_rate:.1%} (95% CI: [{ml_result.win_rate_ci_lower:.1%}, {ml_result.win_rate_ci_upper:.1%}])"
            )

        else:
            return (
                f"⚖️ RECOMMENDATION: No clear winner\n"
                f"   No statistically significant difference detected\n"
                f"   ML win rate: {ml_result.win_rate:.1%}\n"
                f"   Technical win rate: {technical_result.win_rate:.1%}\n"
                f"   Consider: (1) Continue test for more data, (2) Use ensemble approach"
            )

    def _load_active_test(self) -> Optional[Dict]:
        """Load active test config"""
        if not self.active_test_file.exists():
            return None

        try:
            with open(self.active_test_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading active test: {e}")
            return None

    def _save_active_test(self):
        """Save active test config"""
        try:
            with open(self.active_test_file, "w", encoding="utf-8") as f:
                json.dump(self.active_test, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving active test: {e}")

    def _log_test_results(self, summary: ABTestSummary):
        """Log test results to file"""
        try:
            log_entry = asdict(summary)

            with open(self.results_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.error(f"Error logging test results: {e}")


class PerformanceTracker:
    """Track performance metrics for a variant"""

    def __init__(self):
        self.signals = []  # List of signals
        self.trades = []  # List of trades

    def track_signal(self, symbol: str, signal: str, confidence: float):
        """Track a signal"""
        self.signals.append(
            {
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def track_trade(self, symbol: str, is_win: bool, profit_pct: float, holding_days: int):
        """Track a trade result"""
        self.trades.append(
            {
                "symbol": symbol,
                "is_win": is_win,
                "profit_pct": profit_pct,
                "holding_days": holding_days,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_metrics(self) -> Dict:
        """Calculate metrics"""
        total_signals = len(self.signals)
        buy_signals = sum(1 for s in self.signals if s["signal"] == "BUY")

        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t["is_win"])
        losses = total_trades - wins

        win_rate = wins / total_trades if total_trades > 0 else 0.0

        profits = [t["profit_pct"] for t in self.trades]
        avg_profit = np.mean(profits) if profits else 0.0
        total_profit = sum(profits) if profits else 0.0

        # Sharpe ratio (simplified)
        sharpe = (np.mean(profits) / np.std(profits)) if len(profits) > 1 else 0.0

        # Max drawdown (simplified)
        cumulative = np.cumsum(profits) if profits else [0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0

        return {
            "total_signals": total_signals,
            "buy_signals": buy_signals,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_profit_pct": avg_profit,
            "total_profit": total_profit,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }


# Singleton instance
_ab_framework = None


def get_ab_testing_framework() -> ABTestingFramework:
    """Get singleton instance"""
    global _ab_framework
    if _ab_framework is None:
        _ab_framework = ABTestingFramework()
    return _ab_framework
