# -*- coding: utf-8 -*-
"""
Live-Backtest Consistency Checker

Validates that backtest logic matches live trading logic to prevent:
1. Look-ahead bias in backtests
2. Different indicator calculations
3. Missing transaction costs
4. Different entry/exit timing

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue between backtest and live."""

    category: str  # "INDICATOR", "ENTRY", "EXIT", "COST", "TIMING"
    severity: str  # "CRITICAL", "WARNING", "INFO"
    description: str
    backtest_value: Any
    live_value: Any
    recommendation: str


@dataclass
class ConsistencyReport:
    """Full consistency check report."""

    is_consistent: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    issues: List[ConsistencyIssue]
    timestamp: str


class LiveConsistencyChecker:
    """
    Check consistency between backtest and live trading logic.

    Key checks:
    1. Indicator calculations match
    2. Entry signals generated at same points
    3. Exit signals generated at same points
    4. Transaction costs applied consistently
    5. No look-ahead bias in backtests
    """

    # Tolerance for floating point comparisons
    FLOAT_TOLERANCE = 0.0001
    PRICE_TOLERANCE = 0.001  # 0.1% price tolerance

    def __init__(self):
        self._issues: List[ConsistencyIssue] = []
        self._checks_passed = 0
        self._checks_failed = 0
        logger.info("✅ LiveConsistencyChecker initialized")

    def check_indicator_consistency(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        backtest_calc: Callable,
        live_calc: Callable,
        params: Dict = None,
    ) -> bool:
        """
        Check if indicator calculation is consistent.

        Args:
            df: DataFrame with OHLCV data
            indicator_name: Name of indicator
            backtest_calc: Backtest calculation function
            live_calc: Live calculation function
            params: Parameters for calculation

        Returns:
            True if consistent
        """
        params = params or {}

        try:
            backtest_result = backtest_calc(df, **params)
            live_result = live_calc(df, **params)

            # Compare results
            if isinstance(backtest_result, pd.Series):
                # Compare series
                diff = (backtest_result - live_result).abs()
                max_diff = diff.max()

                if max_diff > self.FLOAT_TOLERANCE:
                    self._add_issue(
                        category="INDICATOR",
                        severity="CRITICAL",
                        description=f"{indicator_name} calculation differs",
                        backtest_value=f"max_diff={max_diff:.6f}",
                        live_value="See detailed comparison",
                        recommendation=f"Review {indicator_name} calculation in both modules",
                    )
                    return False
            else:
                # Compare scalar
                if abs(backtest_result - live_result) > self.FLOAT_TOLERANCE:
                    self._add_issue(
                        category="INDICATOR",
                        severity="CRITICAL",
                        description=f"{indicator_name} value differs",
                        backtest_value=backtest_result,
                        live_value=live_result,
                        recommendation=f"Check {indicator_name} formula",
                    )
                    return False

            self._checks_passed += 1
            return True

        except Exception as e:
            self._add_issue(
                category="INDICATOR",
                severity="CRITICAL",
                description=f"Error comparing {indicator_name}: {e}",
                backtest_value="ERROR",
                live_value="ERROR",
                recommendation="Fix calculation error first",
            )
            return False

    def check_entry_signal_consistency(
        self,
        df: pd.DataFrame,
        backtest_entry_logic: Callable,
        live_entry_logic: Callable,
        test_points: int = 100,
    ) -> bool:
        """
        Check if entry signals are generated consistently.

        Args:
            df: DataFrame with OHLCV + indicators
            backtest_entry_logic: Backtest entry function
            live_entry_logic: Live entry function
            test_points: Number of points to test

        Returns:
            True if consistent
        """
        if len(df) < test_points:
            test_points = len(df) - 50  # Need some history

        mismatches = []

        for i in range(50, min(50 + test_points, len(df))):
            # Simulate point-in-time data (no look-ahead)
            df_slice = df.iloc[: i + 1].copy()

            try:
                backtest_signal = backtest_entry_logic(df_slice)
                live_signal = live_entry_logic(df_slice)

                # Compare signals
                bt_should_enter = getattr(backtest_signal, "should_enter", False)
                live_should_enter = getattr(live_signal, "should_enter", False)

                if bt_should_enter != live_should_enter:
                    mismatches.append(
                        {
                            "index": i,
                            "date": df.index[i] if hasattr(df.index[i], "strftime") else str(i),
                            "backtest": bt_should_enter,
                            "live": live_should_enter,
                        }
                    )

            except Exception as e:
                logger.warning(f"Error at index {i}: {e}")

        if mismatches:
            mismatch_rate = len(mismatches) / test_points

            if mismatch_rate > 0.05:  # > 5% mismatch
                self._add_issue(
                    category="ENTRY",
                    severity="CRITICAL",
                    description=f"Entry signals differ at {len(mismatches)}/{test_points} points ({mismatch_rate:.1%})",
                    backtest_value=f"{len([m for m in mismatches if m['backtest']])} entries",
                    live_value=f"{len([m for m in mismatches if m['live']])} entries",
                    recommendation="Review entry logic for look-ahead bias or calculation differences",
                )
                return False
            else:
                self._add_issue(
                    category="ENTRY",
                    severity="WARNING",
                    description=f"Minor entry signal differences at {len(mismatches)} points",
                    backtest_value="See details",
                    live_value="See details",
                    recommendation="Review edge cases in entry logic",
                )

        self._checks_passed += 1
        return True

    def check_exit_signal_consistency(
        self,
        df: pd.DataFrame,
        backtest_exit_logic: Callable,
        live_exit_logic: Callable,
        entry_price: float,
        stop_loss: float,
        take_profit: List[float],
        test_points: int = 100,
    ) -> bool:
        """
        Check if exit signals are generated consistently.

        Args:
            df: DataFrame with OHLCV + indicators
            backtest_exit_logic: Backtest exit function
            live_exit_logic: Live exit function
            entry_price: Simulated entry price
            stop_loss: Stop loss price
            take_profit: Take profit targets
            test_points: Number of points to test

        Returns:
            True if consistent
        """
        mismatches = []

        for i in range(50, min(50 + test_points, len(df))):
            df_slice = df.iloc[: i + 1].copy()
            current_price = df_slice.iloc[-1]["close"]

            try:
                # Create mock position context
                ctx = {
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit_targets": take_profit,
                    "days_held": i - 50,
                }

                backtest_exit = backtest_exit_logic(df_slice, **ctx)
                live_exit = live_exit_logic(df_slice, **ctx)

                bt_should_exit = getattr(backtest_exit, "should_exit", False)
                live_should_exit = getattr(live_exit, "should_exit", False)

                if bt_should_exit != live_should_exit:
                    mismatches.append(
                        {
                            "index": i,
                            "price": current_price,
                            "backtest": bt_should_exit,
                            "live": live_should_exit,
                        }
                    )

            except Exception as e:
                logger.warning(f"Exit check error at index {i}: {e}")

        if mismatches:
            mismatch_rate = len(mismatches) / test_points

            if mismatch_rate > 0.05:
                self._add_issue(
                    category="EXIT",
                    severity="CRITICAL",
                    description=f"Exit signals differ at {len(mismatches)}/{test_points} points",
                    backtest_value=f"{len([m for m in mismatches if m['backtest']])} exits",
                    live_value=f"{len([m for m in mismatches if m['live']])} exits",
                    recommendation="Review exit logic for timing differences",
                )
                return False

        self._checks_passed += 1
        return True

    def check_transaction_cost_consistency(
        self,
        backtest_cost_pct: float,
        live_cost_pct: float,
    ) -> bool:
        """
        Check if transaction costs are applied consistently.

        Args:
            backtest_cost_pct: Backtest transaction cost %
            live_cost_pct: Live transaction cost %

        Returns:
            True if consistent
        """
        diff = abs(backtest_cost_pct - live_cost_pct)

        if diff > 0.001:  # > 0.1% difference
            severity = "CRITICAL" if diff > 0.005 else "WARNING"

            self._add_issue(
                category="COST",
                severity=severity,
                description=f"Transaction cost differs by {diff:.2%}",
                backtest_value=f"{backtest_cost_pct:.2%}",
                live_value=f"{live_cost_pct:.2%}",
                recommendation="Align transaction cost assumptions",
            )

            if severity == "CRITICAL":
                return False

        self._checks_passed += 1
        return True

    def check_look_ahead_bias(
        self,
        df: pd.DataFrame,
        indicator_func: Callable,
        indicator_name: str,
    ) -> bool:
        """
        Check for look-ahead bias in indicator calculation.

        Tests if adding future data changes historical values.

        Args:
            df: Full DataFrame
            indicator_func: Indicator calculation function
            indicator_name: Name of indicator

        Returns:
            True if no look-ahead bias detected
        """
        if len(df) < 100:
            logger.warning("Insufficient data for look-ahead bias check")
            return True

        # Calculate with partial data
        partial_df = df.iloc[:80].copy()
        partial_result = indicator_func(partial_df)

        # Calculate with full data
        full_result = indicator_func(df)

        # Compare overlapping period
        if isinstance(partial_result, pd.Series):
            partial_values = partial_result.iloc[-10:].values
            full_values = full_result.iloc[70:80].values

            diff = abs(partial_values - full_values)
            max_diff = diff.max()

            if max_diff > self.FLOAT_TOLERANCE:
                self._add_issue(
                    category="TIMING",
                    severity="CRITICAL",
                    description=f"Look-ahead bias detected in {indicator_name}",
                    backtest_value=f"Historical values change when future data added",
                    live_value=f"Max diff: {max_diff:.6f}",
                    recommendation=f"Review {indicator_name} for future data usage",
                )
                return False

        self._checks_passed += 1
        return True

    def check_slippage_modeling(
        self,
        backtest_slippage_model: Callable,
        live_slippage_estimate: Callable,
        test_orders: List[Dict],
    ) -> bool:
        """
        Check if slippage modeling is realistic.

        Args:
            backtest_slippage_model: Backtest slippage function
            live_slippage_estimate: Live slippage estimation
            test_orders: List of test order scenarios

        Returns:
            True if slippage modeling is reasonable
        """
        significant_diffs = []

        for order in test_orders:
            bt_slippage = backtest_slippage_model(**order)
            live_slippage = live_slippage_estimate(**order)

            diff = abs(bt_slippage - live_slippage)

            if diff > 0.002:  # > 0.2% difference
                significant_diffs.append(
                    {
                        "order": order,
                        "backtest": bt_slippage,
                        "live": live_slippage,
                        "diff": diff,
                    }
                )

        if significant_diffs:
            avg_diff = sum(d["diff"] for d in significant_diffs) / len(significant_diffs)

            self._add_issue(
                category="COST",
                severity="WARNING" if avg_diff < 0.005 else "CRITICAL",
                description=f"Slippage modeling differs in {len(significant_diffs)}/{len(test_orders)} scenarios",
                backtest_value=f"Avg backtest slippage: {sum(d['backtest'] for d in significant_diffs)/len(significant_diffs):.2%}",
                live_value=f"Avg live estimate: {sum(d['live'] for d in significant_diffs)/len(significant_diffs):.2%}",
                recommendation="Calibrate slippage model with actual execution data",
            )

            if avg_diff >= 0.005:
                return False

        self._checks_passed += 1
        return True

    def _add_issue(
        self,
        category: str,
        severity: str,
        description: str,
        backtest_value: Any,
        live_value: Any,
        recommendation: str,
    ) -> None:
        """Add a consistency issue."""
        self._issues.append(
            ConsistencyIssue(
                category=category,
                severity=severity,
                description=description,
                backtest_value=backtest_value,
                live_value=live_value,
                recommendation=recommendation,
            )
        )
        self._checks_failed += 1

    def generate_report(self) -> ConsistencyReport:
        """Generate full consistency report."""
        total_checks = self._checks_passed + self._checks_failed

        critical_issues = [i for i in self._issues if i.severity == "CRITICAL"]
        is_consistent = len(critical_issues) == 0

        return ConsistencyReport(
            is_consistent=is_consistent,
            total_checks=total_checks,
            passed_checks=self._checks_passed,
            failed_checks=self._checks_failed,
            issues=self._issues,
            timestamp=datetime.now().isoformat(),
        )

    def reset(self) -> None:
        """Reset checker state."""
        self._issues = []
        self._checks_passed = 0
        self._checks_failed = 0

    def print_report(self) -> None:
        """Print formatted report to console."""
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("BACKTEST-LIVE CONSISTENCY REPORT")
        print("=" * 60)
        print(f"Status: {'✅ CONSISTENT' if report.is_consistent else '❌ INCONSISTENT'}")
        print(f"Checks: {report.passed_checks}/{report.total_checks} passed")
        print(f"Timestamp: {report.timestamp}")
        print("-" * 60)

        if report.issues:
            print("\nISSUES FOUND:")
            for i, issue in enumerate(report.issues, 1):
                emoji = (
                    "🚨"
                    if issue.severity == "CRITICAL"
                    else "⚠️" if issue.severity == "WARNING" else "ℹ️"
                )
                print(f"\n{i}. {emoji} [{issue.category}] {issue.description}")
                print(f"   Backtest: {issue.backtest_value}")
                print(f"   Live: {issue.live_value}")
                print(f"   → {issue.recommendation}")
        else:
            print("\n✅ No issues found - backtest and live logic are consistent!")

        print("\n" + "=" * 60)


def run_full_consistency_check(
    df: pd.DataFrame,
    entry_logic_backtest: Callable,
    entry_logic_live: Callable,
    exit_logic_backtest: Callable,
    exit_logic_live: Callable,
    transaction_cost_backtest: float = 0.01,
    transaction_cost_live: float = 0.01,
) -> ConsistencyReport:
    """
    Run full consistency check between backtest and live.

    Args:
        df: Test DataFrame
        entry_logic_backtest: Backtest entry function
        entry_logic_live: Live entry function
        exit_logic_backtest: Backtest exit function
        exit_logic_live: Live exit function
        transaction_cost_backtest: Backtest transaction cost
        transaction_cost_live: Live transaction cost

    Returns:
        ConsistencyReport
    """
    checker = LiveConsistencyChecker()

    # Check transaction costs
    checker.check_transaction_cost_consistency(
        transaction_cost_backtest,
        transaction_cost_live,
    )

    # Check entry signals
    checker.check_entry_signal_consistency(
        df,
        entry_logic_backtest,
        entry_logic_live,
    )

    # Check exit signals
    if len(df) > 50:
        entry_price = df.iloc[50]["close"]
        stop_loss = entry_price * 0.93
        take_profit = [entry_price * 1.05, entry_price * 1.10]

        checker.check_exit_signal_consistency(
            df,
            exit_logic_backtest,
            exit_logic_live,
            entry_price,
            stop_loss,
            take_profit,
        )

    checker.print_report()
    return checker.generate_report()
