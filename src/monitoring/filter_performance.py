"""
Filter Performance Tracker
Tracks effectiveness of each entry filter to identify redundant filters
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
from threading import RLock

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Statistics for a single filter"""

    filter_name: str
    total_checks: int = 0
    blocked_count: int = 0  # How many times this filter blocked entry
    warning_count: int = 0  # How many times this filter added warning
    passed_count: int = 0  # How many times this filter passed

    # Performance metrics (from actual trades)
    blocked_that_won: int = 0  # False negatives (missed opportunities)
    blocked_that_lost: int = 0  # True negatives (avoided losses)
    passed_that_won: int = 0  # True positives (good entries)
    passed_that_lost: int = 0  # False positives (bad entries)

    # Effectiveness metrics
    block_rate: float = 0.0  # % of checks that blocked
    false_negative_rate: float = 0.0  # % of blocks that would have won
    precision: float = 0.0  # % of passes that won
    effectiveness_score: float = 0.0  # Overall effectiveness (0-100)

    last_updated: str = ""


class FilterPerformanceTracker:
    """
    Track filter performance to identify:
    - Which filters are most effective
    - Which filters are redundant
    - Which filters have high false negative rate
    """

    def __init__(self, stats_file: str = "filter_performance.json"):
        self.stats_file = stats_file
        self.filters: Dict[str, FilterStats] = {}
        self._lock = RLock()
        self._load_stats()

        # Known filters to track
        self.filter_names = [
            "market_regime",
            "price_limits",
            "trend_alignment",
            "support_resistance",
            "volume_confirmation",
            "liquidity",
            "volatility",
            "rsi_check",
            "portfolio_correlation",
        ]

        # Initialize filters if not exist
        for filter_name in self.filter_names:
            if filter_name not in self.filters:
                self.filters[filter_name] = FilterStats(filter_name=filter_name)

    def _load_stats(self):
        """Load filter stats from file"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r") as f:
                    data = json.load(f)
                    self.filters = {name: FilterStats(**stats) for name, stats in data.items()}
                logger.info(f"✅ Loaded filter stats from {self.stats_file}")
            except Exception as e:
                logger.error(f"Error loading filter stats: {e}")
                self.filters = {}
        else:
            self.filters = {}

    def _save_stats(self):
        """Save filter stats to file"""
        try:
            with open(self.stats_file, "w") as f:
                data = {name: asdict(stats) for name, stats in self.filters.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving filter stats: {e}")

    def record_filter_check(
        self,
        filter_name: str,
        result: str,  # 'BLOCKED', 'WARNING', 'PASSED'
        symbol: str,
    ):
        """
        Record a filter check result

        Args:
            filter_name: Name of the filter
            result: 'BLOCKED', 'WARNING', or 'PASSED'
            symbol: Stock symbol being checked
        """
        with self._lock:
            if filter_name not in self.filters:
                self.filters[filter_name] = FilterStats(filter_name=filter_name)

            stats = self.filters[filter_name]
            stats.total_checks += 1

            if result == "BLOCKED":
                stats.blocked_count += 1
            elif result == "WARNING":
                stats.warning_count += 1
            elif result == "PASSED":
                stats.passed_count += 1

            stats.last_updated = datetime.now().isoformat()

            # Update block rate
            if stats.total_checks > 0:
                stats.block_rate = stats.blocked_count / stats.total_checks

            self._save_stats()

            logger.debug(
                f"Filter '{filter_name}': {result} for {symbol} "
                f"(block_rate: {stats.block_rate:.1%})"
            )

    def record_trade_outcome(
        self,
        symbol: str,
        filters_that_passed: List[str],
        filters_that_blocked: List[str],
        won: bool,
    ):
        """
        Record trade outcome to update filter effectiveness

        Args:
            symbol: Stock symbol
            filters_that_passed: Filters that passed for this trade
            filters_that_blocked: Filters that would have blocked (from backtest)
            won: Whether the trade was profitable
        """
        with self._lock:
            # Update filters that passed
            for filter_name in filters_that_passed:
                if filter_name not in self.filters:
                    continue

                stats = self.filters[filter_name]
                if won:
                    stats.passed_that_won += 1
                else:
                    stats.passed_that_lost += 1

                # Update precision
                total_passed_with_outcome = stats.passed_that_won + stats.passed_that_lost
                if total_passed_with_outcome > 0:
                    stats.precision = stats.passed_that_won / total_passed_with_outcome

            # Update filters that blocked (would have missed this opportunity)
            for filter_name in filters_that_blocked:
                if filter_name not in self.filters:
                    continue

                stats = self.filters[filter_name]
                if won:
                    stats.blocked_that_won += 1  # False negative
                else:
                    stats.blocked_that_lost += 1  # True negative

                # Update false negative rate
                total_blocked_with_outcome = stats.blocked_that_won + stats.blocked_that_lost
                if total_blocked_with_outcome > 0:
                    stats.false_negative_rate = stats.blocked_that_won / total_blocked_with_outcome

            # Recalculate effectiveness scores
            self._update_effectiveness_scores()
            self._save_stats()

    def _update_effectiveness_scores(self):
        """
        Calculate effectiveness score for each filter

        Effectiveness = precision * (1 - false_negative_rate) * 100
        - High precision (passes lead to wins)
        - Low false negative rate (blocks don't miss opportunities)
        """
        for stats in self.filters.values():
            if stats.total_checks < 10:
                # Need minimum samples
                stats.effectiveness_score = 50.0  # Neutral
                continue

            # Calculate effectiveness
            precision = stats.precision if stats.precision > 0 else 0.5
            fnr_penalty = 1 - stats.false_negative_rate

            # Weight precision higher (70%) than FNR (30%)
            stats.effectiveness_score = (precision * 0.7 + fnr_penalty * 0.3) * 100

    def get_filter_stats(self, filter_name: str) -> Optional[FilterStats]:
        """Get stats for a specific filter"""
        with self._lock:
            return self.filters.get(filter_name)

    def get_all_stats(self) -> Dict[str, FilterStats]:
        """Get all filter stats"""
        with self._lock:
            return dict(self.filters)

    def get_report(self) -> str:
        """
        Generate a report on filter performance

        Returns:
            Formatted report string
        """
        with self._lock:
            report = "📊 FILTER PERFORMANCE REPORT\n"
            report += "=" * 80 + "\n\n"

            # Sort by effectiveness score
            sorted_filters = sorted(
                self.filters.values(),
                key=lambda x: x.effectiveness_score,
                reverse=True,
            )

            for stats in sorted_filters:
                if stats.total_checks == 0:
                    continue

                report += f"Filter: {stats.filter_name}\n"
                report += f"  Total Checks: {stats.total_checks}\n"
                report += f"  Block Rate: {stats.block_rate:.1%}\n"
                report += (
                    f"  Precision: {stats.precision:.1%} "
                    f"({stats.passed_that_won}W/{stats.passed_that_lost}L)\n"
                )
                report += (
                    f"  False Negative Rate: {stats.false_negative_rate:.1%} "
                    f"({stats.blocked_that_won} missed wins)\n"
                )
                report += f"  Effectiveness Score: {stats.effectiveness_score:.1f}/100\n"

                # Recommendation
                if stats.effectiveness_score >= 70:
                    recommendation = "✅ KEEP - Effective filter"
                elif stats.effectiveness_score >= 50:
                    recommendation = "⚠️ REVIEW - Moderate effectiveness"
                elif stats.false_negative_rate > 0.5:
                    recommendation = "🔴 RELAX - Too many false negatives"
                else:
                    recommendation = "🟡 OPTIMIZE - Low effectiveness"

                report += f"  Recommendation: {recommendation}\n\n"

            return report

    def get_redundant_filters(self, correlation_threshold: float = 0.9) -> List[str]:
        """
        Identify potentially redundant filters

        Two filters are redundant if they:
        1. Have similar block rates
        2. Block similar symbols

        Returns:
            List of potentially redundant filter names
        """
        # TODO: Implement correlation analysis between filters
        # For now, return filters with very low effectiveness
        redundant = []

        with self._lock:
            for stats in self.filters.values():
                if stats.total_checks >= 50:  # Need sufficient samples
                    if stats.effectiveness_score < 40:
                        redundant.append(stats.filter_name)

        return redundant

    def get_dashboard_data(self) -> Dict:
        """
        Get comprehensive dashboard data for filter analytics.

        Returns:
            Dict with dashboard metrics and visualizable data
        """
        with self._lock:
            dashboard = {
                "summary": self._get_summary_metrics(),
                "filters": self._get_filter_metrics(),
                "recommendations": self._get_recommendations(),
                "trends": self._get_trend_data(),
                "generated_at": datetime.now().isoformat(),
            }
            return dashboard

    def _get_summary_metrics(self) -> Dict:
        """Get high-level summary metrics."""
        total_checks = sum(f.total_checks for f in self.filters.values())
        total_blocks = sum(f.blocked_count for f in self.filters.values())
        total_warnings = sum(f.warning_count for f in self.filters.values())
        total_passes = sum(f.passed_count for f in self.filters.values())

        # Win/loss from passed trades
        total_wins = sum(f.passed_that_won for f in self.filters.values())
        total_losses = sum(f.passed_that_lost for f in self.filters.values())

        # Missed opportunities (false negatives)
        total_missed = sum(f.blocked_that_won for f in self.filters.values())
        total_avoided = sum(f.blocked_that_lost for f in self.filters.values())

        return {
            "total_checks": total_checks,
            "total_blocks": total_blocks,
            "total_warnings": total_warnings,
            "total_passes": total_passes,
            "overall_block_rate": total_blocks / total_checks if total_checks > 0 else 0,
            "overall_win_rate": (
                total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
            ),
            "missed_opportunities": total_missed,
            "avoided_losses": total_avoided,
            "active_filters": sum(1 for f in self.filters.values() if f.total_checks > 0),
        }

    def _get_filter_metrics(self) -> List[Dict]:
        """Get detailed metrics for each filter."""
        metrics = []
        for stats in sorted(
            self.filters.values(), key=lambda x: x.effectiveness_score, reverse=True
        ):
            if stats.total_checks == 0:
                continue

            metrics.append(
                {
                    "name": stats.filter_name,
                    "total_checks": stats.total_checks,
                    "block_rate": round(stats.block_rate * 100, 1),
                    "precision": round(stats.precision * 100, 1),
                    "false_negative_rate": round(stats.false_negative_rate * 100, 1),
                    "effectiveness_score": round(stats.effectiveness_score, 1),
                    "passed_wins": stats.passed_that_won,
                    "passed_losses": stats.passed_that_lost,
                    "blocked_wins": stats.blocked_that_won,
                    "blocked_losses": stats.blocked_that_lost,
                    "status": self._get_filter_status(stats),
                }
            )
        return metrics

    def _get_filter_status(self, stats: FilterStats) -> str:
        """Determine filter health status."""
        if stats.total_checks < 10:
            return "INSUFFICIENT_DATA"
        if stats.effectiveness_score >= 70:
            return "HEALTHY"
        if stats.effectiveness_score >= 50:
            return "MODERATE"
        if stats.false_negative_rate > 0.5:
            return "TOO_STRICT"
        return "NEEDS_OPTIMIZATION"

    def _get_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []

        for stats in self.filters.values():
            if stats.total_checks < 10:
                continue

            # Too strict filter (high false negative rate)
            if stats.false_negative_rate > 0.4:
                recommendations.append(
                    {
                        "filter": stats.filter_name,
                        "type": "RELAX",
                        "priority": "HIGH",
                        "reason": f"False negative rate {stats.false_negative_rate:.1%} - missing profitable trades",
                        "suggestion": "Consider relaxing threshold or converting to soft filter (warning only)",
                    }
                )

            # Low precision filter
            elif stats.precision < 0.4 and (stats.passed_that_won + stats.passed_that_lost) >= 10:
                recommendations.append(
                    {
                        "filter": stats.filter_name,
                        "type": "TIGHTEN",
                        "priority": "MEDIUM",
                        "reason": f"Precision only {stats.precision:.1%} - passing too many losing trades",
                        "suggestion": "Consider tightening threshold or adding additional conditions",
                    }
                )

            # Very low block rate (might be redundant)
            elif stats.block_rate < 0.05 and stats.total_checks >= 50:
                recommendations.append(
                    {
                        "filter": stats.filter_name,
                        "type": "REVIEW",
                        "priority": "LOW",
                        "reason": f"Block rate only {stats.block_rate:.1%} - filter rarely triggers",
                        "suggestion": "Review if filter is still relevant or needs threshold adjustment",
                    }
                )

            # Very high block rate
            elif stats.block_rate > 0.7:
                recommendations.append(
                    {
                        "filter": stats.filter_name,
                        "type": "REVIEW",
                        "priority": "MEDIUM",
                        "reason": f"Block rate {stats.block_rate:.1%} - filter blocks most signals",
                        "suggestion": "Verify threshold is appropriate for current market conditions",
                    }
                )

        # Sort by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return recommendations

    def _get_trend_data(self) -> Dict:
        """Get trend data for visualization (placeholder for time-series)."""
        # TODO: Implement time-series tracking for trend analysis
        return {
            "note": "Time-series tracking not yet implemented",
            "top_performers": [
                f.filter_name
                for f in sorted(
                    self.filters.values(), key=lambda x: x.effectiveness_score, reverse=True
                )[:3]
                if f.total_checks >= 10
            ],
            "needs_attention": [
                f.filter_name
                for f in self.filters.values()
                if f.total_checks >= 10 and f.effectiveness_score < 50
            ],
        }

    def print_dashboard(self) -> None:
        """Print a formatted dashboard to console."""
        dashboard = self.get_dashboard_data()

        print("\n" + "=" * 80)
        print("📊 FILTER EFFECTIVENESS DASHBOARD")
        print("=" * 80)

        # Summary
        summary = dashboard["summary"]
        print(f"\n📈 SUMMARY")
        print(f"   Total Checks: {summary['total_checks']:,}")
        print(f"   Overall Block Rate: {summary['overall_block_rate']:.1%}")
        print(f"   Overall Win Rate: {summary['overall_win_rate']:.1%}")
        print(f"   Missed Opportunities: {summary['missed_opportunities']}")
        print(f"   Avoided Losses: {summary['avoided_losses']}")
        print(f"   Active Filters: {summary['active_filters']}")

        # Filter Performance Table
        print(f"\n📋 FILTER PERFORMANCE")
        print("-" * 80)
        print(
            f"{'Filter':<25} {'Checks':>8} {'Block%':>8} {'Prec%':>8} {'FNR%':>8} {'Score':>8} {'Status':<15}"
        )
        print("-" * 80)

        for f in dashboard["filters"]:
            status_icon = {
                "HEALTHY": "✅",
                "MODERATE": "⚠️",
                "TOO_STRICT": "🔴",
                "NEEDS_OPTIMIZATION": "🟡",
                "INSUFFICIENT_DATA": "⬜",
            }.get(f["status"], "❓")

            print(
                f"{f['name']:<25} {f['total_checks']:>8} {f['block_rate']:>7.1f}% "
                f"{f['precision']:>7.1f}% {f['false_negative_rate']:>7.1f}% "
                f"{f['effectiveness_score']:>7.1f} {status_icon} {f['status']:<12}"
            )

        # Recommendations
        if dashboard["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS")
            print("-" * 80)
            for rec in dashboard["recommendations"][:5]:  # Top 5
                priority_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(
                    rec["priority"], "⬜"
                )
                print(f"\n{priority_icon} [{rec['priority']}] {rec['filter']}: {rec['type']}")
                print(f"   Reason: {rec['reason']}")
                print(f"   Suggestion: {rec['suggestion']}")

        # Trends
        trends = dashboard["trends"]
        if trends["top_performers"]:
            print(f"\n🏆 Top Performers: {', '.join(trends['top_performers'])}")
        if trends["needs_attention"]:
            print(f"⚠️  Needs Attention: {', '.join(trends['needs_attention'])}")

        print(f"\n📅 Generated: {dashboard['generated_at']}")
        print("=" * 80 + "\n")

    def export_dashboard_json(self, filepath: str = "filter_dashboard.json") -> str:
        """Export dashboard data to JSON file."""
        dashboard = self.get_dashboard_data()
        with open(filepath, "w") as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Dashboard exported to {filepath}")
        return filepath


# Singleton
_filter_tracker = None


def get_filter_performance_tracker() -> FilterPerformanceTracker:
    """Get filter performance tracker singleton"""
    global _filter_tracker
    if _filter_tracker is None:
        _filter_tracker = FilterPerformanceTracker()
    return _filter_tracker


if __name__ == "__main__":
    print("Testing Filter Performance Tracker...")

    tracker = FilterPerformanceTracker()

    # Simulate some filter checks
    tracker.record_filter_check("trend_alignment", "PASSED", "VNM")
    tracker.record_filter_check("volume_confirmation", "BLOCKED", "VIC")
    tracker.record_filter_check("rsi_check", "PASSED", "HPG")

    # Simulate trade outcomes
    tracker.record_trade_outcome(
        symbol="VNM",
        filters_that_passed=["trend_alignment", "rsi_check"],
        filters_that_blocked=[],
        won=True,
    )

    tracker.record_trade_outcome(
        symbol="VIC",
        filters_that_passed=[],
        filters_that_blocked=["volume_confirmation"],
        won=True,  # Volume filter missed this opportunity
    )

    # Print report
    print(tracker.get_report())

    print("\n✅ Filter Performance Tracker test completed!")
