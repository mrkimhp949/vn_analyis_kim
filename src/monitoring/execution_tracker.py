# -*- coding: utf-8 -*-
"""
Execution Cost Tracker - Track actual slippage and transaction costs

Monitors real execution vs expected to:
1. Validate slippage assumptions
2. Adjust future cost estimates
3. Identify problematic symbols/times
4. Generate execution quality reports

Author: Trading Bot Team
Version: 1.0.0
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default file path
DEFAULT_EXECUTION_STATS_FILE = "execution_stats.json"


@dataclass
class ExecutionRecord:
    """Single execution record."""

    symbol: str
    order_type: str  # "MARKET" or "LIMIT"
    side: str  # "BUY" or "SELL"
    expected_price: float
    executed_price: float
    shares: int
    order_value: float
    expected_slippage_pct: float
    actual_slippage_pct: float
    expected_commission: float
    actual_commission: float
    execution_time_ms: int  # Time to fill in milliseconds
    timestamp: str
    session: str  # "ATO", "CONTINUOUS", "ATC"
    avg_daily_volume: int
    order_pct_of_adv: float  # Order as % of average daily volume

    @property
    def slippage_deviation(self) -> float:
        """Deviation from expected slippage."""
        return self.actual_slippage_pct - self.expected_slippage_pct

    @property
    def total_cost_pct(self) -> float:
        """Total execution cost as percentage."""
        return self.actual_slippage_pct + (self.actual_commission / self.order_value * 100)


@dataclass
class SymbolExecutionStats:
    """Aggregated execution stats for a symbol."""

    symbol: str
    total_executions: int = 0
    avg_slippage_pct: float = 0.0
    max_slippage_pct: float = 0.0
    min_slippage_pct: float = 0.0
    avg_slippage_deviation: float = 0.0  # Actual vs Expected
    avg_execution_time_ms: int = 0
    buy_executions: int = 0
    sell_executions: int = 0
    market_order_count: int = 0
    limit_order_count: int = 0
    last_updated: str = ""


@dataclass
class DailyExecutionStats:
    """Daily aggregated execution stats."""

    date: str
    total_executions: int = 0
    total_volume: int = 0
    total_value: float = 0.0
    avg_slippage_pct: float = 0.0
    total_slippage_cost: float = 0.0
    total_commission_cost: float = 0.0
    worst_slippage_symbol: str = ""
    worst_slippage_pct: float = 0.0
    best_slippage_symbol: str = ""
    best_slippage_pct: float = 0.0


class ExecutionCostTracker:
    """
    Track and analyze actual execution costs vs estimates.

    Features:
    - Record every execution with expected vs actual costs
    - Calculate rolling averages by symbol, time, order type
    - Identify systematic under/over-estimation
    - Generate calibration recommendations
    """

    # Thresholds for alerts
    SLIPPAGE_WARNING_THRESHOLD = 0.005  # 0.5% deviation from expected
    SLIPPAGE_CRITICAL_THRESHOLD = 0.01  # 1.0% deviation

    def __init__(
        self,
        stats_file: str = DEFAULT_EXECUTION_STATS_FILE,
        max_records_per_symbol: int = 500,
        calibration_lookback_days: int = 30,
    ):
        self.stats_file = stats_file
        self.max_records_per_symbol = max_records_per_symbol
        self.calibration_lookback_days = calibration_lookback_days

        self._lock = RLock()
        self._records: Dict[str, List[ExecutionRecord]] = {}
        self._symbol_stats: Dict[str, SymbolExecutionStats] = {}
        self._daily_stats: Dict[str, DailyExecutionStats] = {}

        # Calibration factors (learned from actual executions)
        self._slippage_calibration: Dict[str, float] = {}

        self._load_stats()
        logger.info("✅ ExecutionCostTracker initialized")

    def record_execution(
        self,
        symbol: str,
        order_type: str,
        side: str,
        expected_price: float,
        executed_price: float,
        shares: int,
        expected_slippage_pct: float,
        expected_commission: float,
        actual_commission: float,
        execution_time_ms: int = 0,
        session: str = "CONTINUOUS",
        avg_daily_volume: int = 0,
    ) -> ExecutionRecord:
        """
        Record an execution and update statistics.

        Args:
            symbol: Stock symbol
            order_type: "MARKET" or "LIMIT"
            side: "BUY" or "SELL"
            expected_price: Price we expected to execute at
            executed_price: Actual execution price
            shares: Number of shares
            expected_slippage_pct: Expected slippage percentage
            expected_commission: Expected commission in VND
            actual_commission: Actual commission charged
            execution_time_ms: Time to fill order
            session: Trading session (ATO, CONTINUOUS, ATC)
            avg_daily_volume: Average daily volume for the symbol

        Returns:
            ExecutionRecord with all details
        """
        with self._lock:
            # Calculate actual slippage
            if side.upper() == "BUY":
                actual_slippage_pct = (executed_price - expected_price) / expected_price
            else:
                actual_slippage_pct = (expected_price - executed_price) / expected_price

            order_value = executed_price * shares
            order_pct_of_adv = shares / avg_daily_volume if avg_daily_volume > 0 else 0

            record = ExecutionRecord(
                symbol=symbol.upper(),
                order_type=order_type.upper(),
                side=side.upper(),
                expected_price=expected_price,
                executed_price=executed_price,
                shares=shares,
                order_value=order_value,
                expected_slippage_pct=expected_slippage_pct,
                actual_slippage_pct=actual_slippage_pct,
                expected_commission=expected_commission,
                actual_commission=actual_commission,
                execution_time_ms=execution_time_ms,
                timestamp=datetime.now().isoformat(),
                session=session.upper(),
                avg_daily_volume=avg_daily_volume,
                order_pct_of_adv=order_pct_of_adv,
            )

            # Store record
            if symbol not in self._records:
                self._records[symbol] = []
            self._records[symbol].append(record)

            # Trim old records
            if len(self._records[symbol]) > self.max_records_per_symbol:
                self._records[symbol] = self._records[symbol][-self.max_records_per_symbol :]

            # Update statistics
            self._update_symbol_stats(symbol)
            self._update_daily_stats(record)

            # Check for alerts
            self._check_slippage_alert(record)

            # Save periodically
            self._save_stats()

            return record

    def _update_symbol_stats(self, symbol: str) -> None:
        """Update aggregated stats for a symbol."""
        records = self._records.get(symbol, [])
        if not records:
            return

        slippages = [r.actual_slippage_pct for r in records]
        deviations = [r.slippage_deviation for r in records]
        exec_times = [r.execution_time_ms for r in records]

        self._symbol_stats[symbol] = SymbolExecutionStats(
            symbol=symbol,
            total_executions=len(records),
            avg_slippage_pct=sum(slippages) / len(slippages),
            max_slippage_pct=max(slippages),
            min_slippage_pct=min(slippages),
            avg_slippage_deviation=sum(deviations) / len(deviations),
            avg_execution_time_ms=int(sum(exec_times) / len(exec_times)),
            buy_executions=sum(1 for r in records if r.side == "BUY"),
            sell_executions=sum(1 for r in records if r.side == "SELL"),
            market_order_count=sum(1 for r in records if r.order_type == "MARKET"),
            limit_order_count=sum(1 for r in records if r.order_type == "LIMIT"),
            last_updated=datetime.now().isoformat(),
        )

    def _update_daily_stats(self, record: ExecutionRecord) -> None:
        """Update daily aggregated stats."""
        date_str = record.timestamp[:10]  # YYYY-MM-DD

        if date_str not in self._daily_stats:
            self._daily_stats[date_str] = DailyExecutionStats(date=date_str)

        stats = self._daily_stats[date_str]
        stats.total_executions += 1
        stats.total_volume += record.shares
        stats.total_value += record.order_value

        # Update running average
        n = stats.total_executions
        stats.avg_slippage_pct = (stats.avg_slippage_pct * (n - 1) + record.actual_slippage_pct) / n

        stats.total_slippage_cost += record.order_value * record.actual_slippage_pct
        stats.total_commission_cost += record.actual_commission

        # Track worst/best
        if record.actual_slippage_pct > stats.worst_slippage_pct:
            stats.worst_slippage_pct = record.actual_slippage_pct
            stats.worst_slippage_symbol = record.symbol
        if record.actual_slippage_pct < stats.best_slippage_pct or stats.best_slippage_symbol == "":
            stats.best_slippage_pct = record.actual_slippage_pct
            stats.best_slippage_symbol = record.symbol

    def _check_slippage_alert(self, record: ExecutionRecord) -> None:
        """Check if slippage deviation warrants an alert."""
        deviation = abs(record.slippage_deviation)

        if deviation >= self.SLIPPAGE_CRITICAL_THRESHOLD:
            logger.warning(
                f"🚨 CRITICAL SLIPPAGE: {record.symbol} "
                f"expected {record.expected_slippage_pct:.2%}, "
                f"actual {record.actual_slippage_pct:.2%} "
                f"(deviation: {record.slippage_deviation:+.2%})"
            )
        elif deviation >= self.SLIPPAGE_WARNING_THRESHOLD:
            logger.info(
                f"⚠️ Slippage warning: {record.symbol} "
                f"deviation {record.slippage_deviation:+.2%}"
            )

    def get_calibrated_slippage(
        self,
        symbol: str,
        order_type: str = "MARKET",
        fallback: float = 0.004,
    ) -> float:
        """
        Get calibrated slippage estimate based on historical executions.

        Args:
            symbol: Stock symbol
            order_type: "MARKET" or "LIMIT"
            fallback: Fallback slippage if no data

        Returns:
            Calibrated slippage percentage
        """
        stats = self._symbol_stats.get(symbol.upper())

        if not stats or stats.total_executions < 5:
            # Not enough data, use fallback
            return fallback

        # Use historical average with safety margin
        safety_margin = 1.2  # 20% buffer
        calibrated = stats.avg_slippage_pct * safety_margin

        # Cap at reasonable bounds
        calibrated = max(0.001, min(0.03, calibrated))

        return calibrated

    def get_execution_quality_score(self, symbol: str) -> Tuple[float, str]:
        """
        Get execution quality score for a symbol.

        Returns:
            Tuple of (score 0-100, description)
        """
        stats = self._symbol_stats.get(symbol.upper())

        if not stats or stats.total_executions < 3:
            return 50.0, "Insufficient data"

        # Score based on:
        # 1. Average slippage (lower is better)
        # 2. Slippage consistency (lower deviation is better)
        # 3. Execution time (faster is better)

        slippage_score = max(0, 100 - stats.avg_slippage_pct * 5000)  # 2% slippage = 0 score
        consistency_score = max(0, 100 - abs(stats.avg_slippage_deviation) * 10000)
        time_score = max(0, 100 - stats.avg_execution_time_ms / 100)  # 10s = 0 score

        total_score = slippage_score * 0.5 + consistency_score * 0.3 + time_score * 0.2

        if total_score >= 80:
            desc = "Excellent execution quality"
        elif total_score >= 60:
            desc = "Good execution quality"
        elif total_score >= 40:
            desc = "Average execution quality"
        else:
            desc = "Poor execution quality - consider limit orders"

        return total_score, desc

    def get_calibration_recommendations(self) -> Dict[str, Dict]:
        """
        Generate recommendations for adjusting slippage estimates.

        Returns:
            Dict with symbol -> recommendation
        """
        recommendations = {}

        for symbol, stats in self._symbol_stats.items():
            if stats.total_executions < 10:
                continue

            deviation = stats.avg_slippage_deviation

            if abs(deviation) > 0.002:  # 0.2% systematic deviation
                if deviation > 0:
                    action = "INCREASE"
                    reason = f"Actual slippage {deviation:+.2%} higher than expected"
                else:
                    action = "DECREASE"
                    reason = f"Actual slippage {deviation:+.2%} lower than expected"

                recommendations[symbol] = {
                    "action": action,
                    "current_avg_slippage": stats.avg_slippage_pct,
                    "deviation": deviation,
                    "recommended_adjustment": deviation * 1.1,  # 10% buffer
                    "reason": reason,
                    "sample_size": stats.total_executions,
                }

        return recommendations

    def get_daily_report(self, date_str: Optional[str] = None) -> Dict:
        """Get daily execution report."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        stats = self._daily_stats.get(date_str)

        if not stats:
            return {"date": date_str, "message": "No executions recorded"}

        return {
            "date": date_str,
            "total_executions": stats.total_executions,
            "total_volume": stats.total_volume,
            "total_value": stats.total_value,
            "avg_slippage_pct": stats.avg_slippage_pct,
            "total_slippage_cost": stats.total_slippage_cost,
            "total_commission_cost": stats.total_commission_cost,
            "total_execution_cost": stats.total_slippage_cost + stats.total_commission_cost,
            "worst_execution": {
                "symbol": stats.worst_slippage_symbol,
                "slippage": stats.worst_slippage_pct,
            },
            "best_execution": {
                "symbol": stats.best_slippage_symbol,
                "slippage": stats.best_slippage_pct,
            },
        }

    def _load_stats(self) -> None:
        """Load stats from file."""
        if not os.path.exists(self.stats_file):
            return

        try:
            with open(self.stats_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load symbol stats
            for symbol, stats_dict in data.get("symbol_stats", {}).items():
                self._symbol_stats[symbol] = SymbolExecutionStats(**stats_dict)

            # Load daily stats
            for date_str, stats_dict in data.get("daily_stats", {}).items():
                self._daily_stats[date_str] = DailyExecutionStats(**stats_dict)

            # Load calibration
            self._slippage_calibration = data.get("calibration", {})

            logger.info(f"📊 Loaded execution stats: {len(self._symbol_stats)} symbols")

        except Exception as e:
            logger.warning(f"Failed to load execution stats: {e}")

    def _save_stats(self) -> None:
        """Save stats to file."""
        try:
            data = {
                "symbol_stats": {s: asdict(stats) for s, stats in self._symbol_stats.items()},
                "daily_stats": {d: asdict(stats) for d, stats in self._daily_stats.items()},
                "calibration": self._slippage_calibration,
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Failed to save execution stats: {e}")


# Singleton instance
_tracker_instance: Optional[ExecutionCostTracker] = None


def get_execution_tracker() -> ExecutionCostTracker:
    """Get singleton ExecutionCostTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ExecutionCostTracker()
    return _tracker_instance
