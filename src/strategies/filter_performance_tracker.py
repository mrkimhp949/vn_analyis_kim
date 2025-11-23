"""
Filter Performance Tracker
Tracks the performance of individual entry filters to optimize entry logic

Features:
1. Track which filters contribute to winning vs losing trades
2. Calculate filter effectiveness scores
3. Dynamic filter weighting based on historical performance
4. Filter importance ranking
5. Adaptive filter thresholds
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FilterPerformance:
    """Performance metrics for a single filter"""

    filter_name: str
    total_signals: int
    winning_signals: int
    losing_signals: int
    win_rate: float
    avg_profit_when_present: float
    avg_profit_when_absent: float
    effectiveness_score: float  # 0-100
    current_weight: float  # Dynamic weight (0.5-1.5)
    recommendation: str  # "STRONG", "KEEP", "WEAK", "DISABLE"


class FilterPerformanceTracker:
    """
    Track and optimize filter performance over time

    Each trade is tagged with which filters were present/absent
    Performance is calculated to identify high-value filters
    """

    def __init__(self, storage_path: str = "filter_performance.json"):
        self.storage_path = Path(storage_path)

        # Filter performance data
        # {filter_name: {
        #     "present": {"wins": int, "losses": int, "total_pnl": float},
        #     "absent": {"wins": int, "losses": int, "total_pnl": float}
        # }}
        self.filter_stats = defaultdict(
            lambda: {
                "present": {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0},
                "absent": {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0},
            }
        )

        # Load historical data
        self._load_stats()

        logger.info(
            f"✅ Filter Performance Tracker initialized with {len(self.filter_stats)} filters tracked"
        )

    def track_trade_outcome(
        self,
        symbol: str,
        filters_present: List[str],
        all_filters: List[str],
        pnl_percent: float,
        trade_metadata: Optional[Dict] = None,
    ):
        """
        Record trade outcome for filter performance analysis

        Args:
            symbol: Stock symbol
            filters_present: List of filters that were TRUE for this entry
            all_filters: List of all available filters
            pnl_percent: Trade P&L percentage
            trade_metadata: Additional trade info
        """
        is_win = pnl_percent > 0

        # Update stats for each filter
        for filter_name in all_filters:
            was_present = filter_name in filters_present

            if was_present:
                # Filter was present for this trade
                self.filter_stats[filter_name]["present"]["count"] += 1
                self.filter_stats[filter_name]["present"]["total_pnl"] += pnl_percent
                if is_win:
                    self.filter_stats[filter_name]["present"]["wins"] += 1
                else:
                    self.filter_stats[filter_name]["present"]["losses"] += 1
            else:
                # Filter was absent for this trade
                self.filter_stats[filter_name]["absent"]["count"] += 1
                self.filter_stats[filter_name]["absent"]["total_pnl"] += pnl_percent
                if is_win:
                    self.filter_stats[filter_name]["absent"]["wins"] += 1
                else:
                    self.filter_stats[filter_name]["absent"]["losses"] += 1

        # Save stats
        self._save_stats()

        logger.debug(
            f"📊 Tracked trade outcome for {symbol}: "
            f"P&L={pnl_percent:+.1f}%, filters={len(filters_present)}/{len(all_filters)}"
        )

    def get_filter_performance(self, filter_name: str) -> Optional[FilterPerformance]:
        """Get performance metrics for a specific filter"""
        if filter_name not in self.filter_stats:
            return None

        stats = self.filter_stats[filter_name]

        # Calculate metrics when filter was PRESENT
        present = stats["present"]
        present_count = present["count"]
        present_wins = present["wins"]
        present_losses = present["losses"]
        present_total_pnl = present["total_pnl"]

        # Calculate metrics when filter was ABSENT
        absent = stats["absent"]
        absent_count = absent["count"]
        absent_total_pnl = absent["total_pnl"]

        if present_count == 0:
            return None

        # Win rate when present
        win_rate = present_wins / present_count if present_count > 0 else 0

        # Average profit when present vs absent
        avg_profit_present = present_total_pnl / present_count if present_count > 0 else 0
        avg_profit_absent = absent_total_pnl / absent_count if absent_count > 0 else 0

        # Effectiveness score (0-100)
        # Higher score = filter contributes to better outcomes
        effectiveness = self._calculate_effectiveness_score(
            win_rate, avg_profit_present, avg_profit_absent, present_count
        )

        # Current weight based on effectiveness
        weight = self._calculate_filter_weight(effectiveness)

        # Recommendation
        recommendation = self._get_filter_recommendation(effectiveness, present_count)

        return FilterPerformance(
            filter_name=filter_name,
            total_signals=present_count,
            winning_signals=present_wins,
            losing_signals=present_losses,
            win_rate=win_rate,
            avg_profit_when_present=avg_profit_present,
            avg_profit_when_absent=avg_profit_absent,
            effectiveness_score=effectiveness,
            current_weight=weight,
            recommendation=recommendation,
        )

    def get_all_filter_performance(self) -> List[FilterPerformance]:
        """Get performance metrics for all tracked filters"""
        performances = []

        for filter_name in self.filter_stats.keys():
            perf = self.get_filter_performance(filter_name)
            if perf and perf.total_signals >= 5:  # Min 5 signals to be meaningful
                performances.append(perf)

        # Sort by effectiveness score
        performances.sort(key=lambda x: x.effectiveness_score, reverse=True)

        return performances

    def get_filter_weights(self) -> Dict[str, float]:
        """
        Get dynamic weights for all filters

        Use this to adjust filter penalties/bonuses in entry logic

        Returns:
            Dict mapping filter_name to weight (0.5-1.5)
        """
        weights = {}

        for filter_name in self.filter_stats.keys():
            perf = self.get_filter_performance(filter_name)
            if perf and perf.total_signals >= 10:  # Min 10 signals for reliable weight
                weights[filter_name] = perf.current_weight
            else:
                weights[filter_name] = 1.0  # Neutral weight if insufficient data

        return weights

    def should_disable_filter(self, filter_name: str, min_trades: int = 20) -> Tuple[bool, str]:
        """
        Determine if a filter should be disabled based on performance

        Args:
            filter_name: Name of filter to check
            min_trades: Minimum trades before making recommendation

        Returns:
            Tuple of (should_disable, reason)
        """
        perf = self.get_filter_performance(filter_name)

        if not perf:
            return False, "No data"

        if perf.total_signals < min_trades:
            return False, f"Insufficient data ({perf.total_signals} < {min_trades})"

        # Disable if effectiveness is very low
        if perf.effectiveness_score < 20:
            return (
                True,
                f"Very low effectiveness ({perf.effectiveness_score:.0f}/100) - "
                f"filter may be counterproductive",
            )

        # Disable if filter presence correlates with losses
        if perf.win_rate < 0.35 and perf.avg_profit_when_present < -2.0:
            return (
                True,
                f"Low win rate ({perf.win_rate:.1%}) and negative avg P&L "
                f"({perf.avg_profit_when_present:.1f}%)",
            )

        return False, "Performance acceptable"

    def _calculate_effectiveness_score(
        self,
        win_rate: float,
        avg_profit_present: float,
        avg_profit_absent: float,
        sample_size: int,
    ) -> float:
        """
        Calculate filter effectiveness score (0-100)

        Factors:
        1. Win rate when filter present (40% weight)
        2. Profit differential (present vs absent) (40% weight)
        3. Sample size confidence (20% weight)
        """
        # Component 1: Win rate score (0-40)
        win_rate_score = min(win_rate * 100, 100) * 0.4

        # Component 2: Profit differential score (0-40)
        # Positive if avg_profit_present > avg_profit_absent
        profit_diff = avg_profit_present - avg_profit_absent

        if profit_diff > 5.0:
            # Huge improvement
            profit_score = 40
        elif profit_diff > 2.0:
            # Good improvement
            profit_score = 30
        elif profit_diff > 0:
            # Slight improvement
            profit_score = 20
        elif profit_diff > -2.0:
            # Slight negative
            profit_score = 10
        else:
            # Significantly negative
            profit_score = 0

        # Component 3: Confidence score based on sample size (0-20)
        if sample_size >= 50:
            confidence_score = 20
        elif sample_size >= 30:
            confidence_score = 15
        elif sample_size >= 10:
            confidence_score = 10
        else:
            confidence_score = 5

        total_score = win_rate_score + profit_score + confidence_score

        return min(total_score, 100)

    def _calculate_filter_weight(self, effectiveness_score: float) -> float:
        """
        Calculate dynamic weight for filter based on effectiveness

        High effectiveness = higher weight (increase penalty/bonus)
        Low effectiveness = lower weight (reduce penalty/bonus)

        Returns:
            Weight between 0.5 (weak filter) and 1.5 (strong filter)
        """
        if effectiveness_score >= 80:
            return 1.5  # Very effective - amplify this filter
        elif effectiveness_score >= 60:
            return 1.2  # Good - increase weight
        elif effectiveness_score >= 40:
            return 1.0  # Average - keep neutral
        elif effectiveness_score >= 25:
            return 0.75  # Below average - reduce weight
        else:
            return 0.5  # Poor - significantly reduce weight

    def _get_filter_recommendation(self, effectiveness_score: float, sample_size: int) -> str:
        """Get recommendation for filter usage"""
        if sample_size < 10:
            return "INSUFFICIENT_DATA"
        elif effectiveness_score >= 70:
            return "STRONG"
        elif effectiveness_score >= 50:
            return "KEEP"
        elif effectiveness_score >= 30:
            return "WEAK"
        else:
            return "CONSIDER_DISABLE"

    def _save_stats(self):
        """Save filter statistics to disk"""
        try:
            # Convert defaultdict to regular dict for JSON serialization
            data = {
                "last_updated": datetime.now().isoformat(),
                "filters": dict(self.filter_stats),
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"⚠️ Failed to save filter stats: {e}")

    def _load_stats(self):
        """Load filter statistics from disk"""
        try:
            if not self.storage_path.exists():
                logger.info("No existing filter stats found - starting fresh")
                return

            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Load filter data
            if "filters" in data:
                for filter_name, stats in data["filters"].items():
                    self.filter_stats[filter_name] = stats

            logger.info(
                f"✅ Loaded filter stats from {self.storage_path} "
                f"(last updated: {data.get('last_updated', 'unknown')})"
            )

        except Exception as e:
            logger.warning(f"⚠️ Failed to load filter stats: {e}")

    def format_performance_report(self) -> str:
        """Generate comprehensive filter performance report"""
        performances = self.get_all_filter_performance()

        if not performances:
            return "No filter performance data available yet."

        lines = []
        lines.append("📊 *FILTER PERFORMANCE ANALYSIS*")
        lines.append("=" * 60)

        lines.append(f"\n📈 *Total Filters Tracked:* {len(performances)}")

        # Strong filters
        strong_filters = [p for p in performances if p.recommendation == "STRONG"]
        if strong_filters:
            lines.append(f"\n✅ *STRONG FILTERS ({len(strong_filters)}):*")
            for perf in strong_filters[:5]:  # Top 5
                lines.append(
                    f"• {perf.filter_name}: {perf.effectiveness_score:.0f}/100 "
                    f"(WR: {perf.win_rate:.1%}, Avg P&L: {perf.avg_profit_when_present:+.1f}%) "
                    f"[Weight: {perf.current_weight:.2f}x]"
                )

        # Weak filters
        weak_filters = [
            p for p in performances if p.recommendation in ["WEAK", "CONSIDER_DISABLE"]
        ]
        if weak_filters:
            lines.append(f"\n⚠️ *WEAK FILTERS ({len(weak_filters)}):*")
            for perf in weak_filters:
                lines.append(
                    f"• {perf.filter_name}: {perf.effectiveness_score:.0f}/100 "
                    f"(WR: {perf.win_rate:.1%}, Avg P&L: {perf.avg_profit_when_present:+.1f}%) "
                    f"[{perf.recommendation}]"
                )

        # Detailed breakdown
        lines.append(f"\n📋 *ALL FILTERS (Top 10):*")
        for i, perf in enumerate(performances[:10], 1):
            lines.append(
                f"{i}. {perf.filter_name}:\n"
                f"   Score: {perf.effectiveness_score:.0f}/100 | "
                f"WR: {perf.win_rate:.1%} ({perf.winning_signals}W/{perf.losing_signals}L) | "
                f"Avg P&L: {perf.avg_profit_when_present:+.1f}%\n"
                f"   Weight: {perf.current_weight:.2f}x | "
                f"Status: {perf.recommendation}"
            )

        return "\n".join(lines)


# Singleton
_tracker = None


def get_filter_performance_tracker() -> FilterPerformanceTracker:
    """Get filter performance tracker singleton"""
    global _tracker
    if _tracker is None:
        _tracker = FilterPerformanceTracker()
    return _tracker


# Test
if __name__ == "__main__":
    print("Testing Filter Performance Tracker...")

    tracker = FilterPerformanceTracker(storage_path="test_filter_performance.json")

    # Simulate some trades
    all_filters = [
        "trend_alignment",
        "support_resistance",
        "volume",
        "rsi",
        "macd",
        "market_regime",
    ]

    # Winning trade with good filters
    tracker.track_trade_outcome(
        symbol="VCB",
        filters_present=["trend_alignment", "support_resistance", "volume", "rsi"],
        all_filters=all_filters,
        pnl_percent=5.5,
    )

    # Losing trade without key filters
    tracker.track_trade_outcome(
        symbol="VNM",
        filters_present=["market_regime"],
        all_filters=all_filters,
        pnl_percent=-3.2,
    )

    # Generate report
    print("\n" + tracker.format_performance_report())

    print("\n✅ Filter Performance Tracker test completed!")
