"""
Performance Monitoring & Metrics
Track trading performance and system health
"""

import functools
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class TradeMetric:
    """Single trade metric"""

    symbol: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_percent: float
    hold_days: int
    entry_date: str
    exit_date: str


class PerformanceMonitor:
    """
    Monitor trading performance

    Tracks:
    - Win rate
    - Average profit/loss
    - Sharpe ratio
    - Maximum drawdown
    - Total return
    """

    def __init__(self, db_path="metrics.json"):
        self.db_path = db_path
        self.trades: List[TradeMetric] = []
        self.load()

    def load(self):
        """Load metrics from file"""
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.trades = [TradeMetric(**t) for t in data.get("trades", [])]
        except FileNotFoundError:
            self.trades = []

    def save(self):
        """Save metrics to file"""
        data = {
            "trades": [asdict(t) for t in self.trades],
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def track_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        shares: int,
        entry_date: str,
        exit_date: str,
    ):
        """Track a completed trade"""
        pnl = (exit_price - entry_price) * shares
        pnl_percent = ((exit_price - entry_price) / entry_price) * 100

        # Calculate hold days
        try:
            entry_dt = datetime.fromisoformat(entry_date)
            exit_dt = datetime.fromisoformat(exit_date)
            hold_days = (exit_dt - entry_dt).days
        except Exception:
            hold_days = 0

        trade = TradeMetric(
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares,
            pnl=pnl,
            pnl_percent=pnl_percent,
            hold_days=hold_days,
            entry_date=entry_date,
            exit_date=exit_date,
        )

        self.trades.append(trade)
        self.save()

    def get_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "total_pnl": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
            }

        # Basic stats
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

        avg_profit = (
            sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        )
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0

        total_pnl = sum(t.pnl for t in self.trades)

        # Sharpe ratio (simplified)
        returns = [t.pnl_percent for t in self.trades]
        if len(returns) > 1:
            import numpy as np

            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Max drawdown
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0

        for trade in self.trades:
            cumulative_pnl += trade.pnl
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "total_pnl": total_pnl,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_hold_days": (
                sum(t.hold_days for t in self.trades) / total_trades if total_trades > 0 else 0
            ),
        }

    def get_summary(self) -> str:
        """Get formatted summary"""
        metrics = self.get_metrics()

        summary = []
        summary.append("📊 PERFORMANCE METRICS")
        summary.append("=" * 40)
        summary.append(f"Total Trades: {metrics['total_trades']}")
        summary.append(f"Win Rate: {metrics['win_rate']:.1f}%")
        summary.append(f"Avg Profit: {metrics['avg_profit']:,.0f} VNĐ")
        summary.append(f"Avg Loss: {metrics['avg_loss']:,.0f} VNĐ")
        summary.append(f"Total P&L: {metrics['total_pnl']:+,.0f} VNĐ")
        summary.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        summary.append(f"Max Drawdown: {metrics['max_drawdown']:,.0f} VNĐ")
        summary.append(f"Avg Hold: {metrics['avg_hold_days']:.1f} days")

        return "\n".join(summary)


class SystemMonitor:
    """Monitor system health"""

    def __init__(self):
        self.api_calls = {}
        self.errors = []

    def track_api_call(self, api_name: str, duration: float, success: bool):
        """Track API call"""
        if api_name not in self.api_calls:
            self.api_calls[api_name] = {
                "total": 0,
                "success": 0,
                "failed": 0,
                "total_duration": 0,
            }

        self.api_calls[api_name]["total"] += 1
        if success:
            self.api_calls[api_name]["success"] += 1
        else:
            self.api_calls[api_name]["failed"] += 1
        self.api_calls[api_name]["total_duration"] += duration

    def track_error(self, error_type: str, message: str):
        """Track error"""
        self.errors.append(
            {
                "type": error_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only last 100 errors
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]

    def get_api_stats(self) -> Dict:
        """Get API statistics"""
        stats = {}
        for api_name, data in self.api_calls.items():
            stats[api_name] = {
                "total_calls": data["total"],
                "success_rate": (
                    (data["success"] / data["total"] * 100) if data["total"] > 0 else 0
                ),
                "avg_duration": (
                    (data["total_duration"] / data["total"]) if data["total"] > 0 else 0
                ),
            }
        return stats


# Decorator for monitoring API calls
def monitor_api_call(api_name: str):
    """Decorator to monitor API calls"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_system_monitor()
            start_time = time.time()
            success = False

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception:
                monitor.track_error(api_name, str(e))  # noqa: F821
                raise
            finally:
                duration = time.time() - start_time
                monitor.track_api_call(api_name, duration, success)

        return wrapper

    return decorator


# Singleton instances
_perf_monitor = None
_sys_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get performance monitor singleton"""
    global _perf_monitor
    if _perf_monitor is None:
        _perf_monitor = PerformanceMonitor()
    return _perf_monitor


def get_system_monitor() -> SystemMonitor:
    """Get system monitor singleton"""
    global _sys_monitor
    if _sys_monitor is None:
        _sys_monitor = SystemMonitor()
    return _sys_monitor


# Test
if __name__ == "__main__":
    print("Testing monitoring...")

    monitor = PerformanceMonitor("test_metrics.json")

    # Track some trades
    monitor.track_trade("VCB", 60000, 65000, 100, "2025-11-01", "2025-11-10")
    monitor.track_trade("FPT", 100000, 95000, 50, "2025-11-02", "2025-11-08")
    monitor.track_trade("VNM", 62000, 68000, 80, "2025-11-03", "2025-11-12")

    # Get metrics
    print(monitor.get_summary())

    print("\n✅ Monitoring test completed!")
