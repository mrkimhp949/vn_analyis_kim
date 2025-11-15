# -*- coding: utf-8 -*-
"""
Walk-Forward Testing
Test chiến lược với out-of-sample data để tránh overfitting
"""
import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Kết quả của một walk-forward window"""

    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_samples: int
    test_return: float
    test_sharpe: float
    test_max_drawdown: float
    test_win_rate: float
    num_trades: int


class WalkForwardTester:
    """
    Walk-Forward Testing

    Quy trình:
    1. Chia data thành nhiều windows
    2. Mỗi window: train trên in-sample, test trên out-of-sample
    3. Roll forward và lặp lại
    4. Aggregate kết quả

    Ví dụ:
    - Train: 180 days (6 months)
    - Test: 30 days (1 month)
    - Step: 30 days (roll forward 1 month)
    """

    def __init__(
        self,
        train_period: int = 180,  # days
        test_period: int = 30,  # days
        step: int = 30,  # days
    ):
        self.train_period = train_period
        self.test_period = test_period
        self.step = step

    def run_walk_forward(
        self,
        symbols: List[str],
        strategy_function: callable,
        initial_capital: float = 100_000_000,
    ) -> Dict:
        """
        Chạy walk-forward test

        Args:
            symbols: List các mã để test
            strategy_function: Function(train_data, test_data) -> trades
            initial_capital: Vốn ban đầu

        Returns:
            Dict với kết quả tổng hợp
        """
        print("🔄 Starting Walk-Forward Test...")
        print("   Train: {self.train_period} days")
        print("   Test: {self.test_period} days")
        print("   Step: {self.step} days")
        print(f"   Symbols: {len(symbols)}")

        all_results = []

        for symbol in symbols:
            try:
                symbol_results = self._test_symbol(
                    symbol, strategy_function, initial_capital
                )
                all_results.extend(symbol_results)
            except Exception:
                logger.error(f"Error testing {symbol}")

        if not all_results:
            return {"total_windows": 0, "message": "No results"}

        # Aggregate results
        return self._aggregate_results(all_results)

    def _test_symbol(
        self, symbol: str, strategy_function: callable, initial_capital: float
    ) -> List[WalkForwardResult]:
        """Test một symbol với walk-forward"""
        from src.data.loader import load_data

        # Load full history
        df = load_data(symbol, lookback=500, use_cache=True)

        if df.empty or len(df) < self.train_period + self.test_period:
            logger.warning(f"Insufficient data for {symbol}")
            return []

        results = []
        window_id = 0

        # Calculate windows
        total_days = len(df)
        start_idx = 0

        while start_idx + self.train_period + self.test_period <= total_days:
            # Define train and test periods
            train_start_idx = start_idx
            train_end_idx = start_idx + self.train_period
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + self.test_period

            train_data = df.iloc[train_start_idx:train_end_idx].copy()
            test_data = df.iloc[test_start_idx:test_end_idx].copy()

            # Run strategy
            try:
                trades = strategy_function(train_data, test_data)

                # Calculate metrics
                metrics = self._calculate_metrics(trades, test_data, initial_capital)

                result = WalkForwardResult(
                    window_id=window_id,
                    train_start=train_data["time"].iloc[0].strftime("%Y-%m-%d"),
                    train_end=train_data["time"].iloc[-1].strftime("%Y-%m-%d"),
                    test_start=test_data["time"].iloc[0].strftime("%Y-%m-%d"),
                    test_end=test_data["time"].iloc[-1].strftime("%Y-%m-%d"),
                    train_samples=len(train_data),
                    test_samples=len(test_data),
                    test_return=metrics["return"],
                    test_sharpe=metrics["sharpe"],
                    test_max_drawdown=metrics["max_drawdown"],
                    test_win_rate=metrics["win_rate"],
                    num_trades=len(trades),
                )

                results.append(result)
                window_id += 1

            except Exception:
                logger.error(f"Error in window {window_id} for {symbol}")

            # Move forward
            start_idx += self.step

        return results

    def _calculate_metrics(
        self, trades: List[Dict], test_data: pd.DataFrame, initial_capital: float
    ) -> Dict:
        """Tính metrics cho test period"""
        if not trades:
            return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}

        # Calculate returns
        capital = initial_capital
        equity_curve = [capital]

        for trade in trades:
            pnl = trade.get("pnl", 0)
            capital += pnl
            equity_curve.append(capital)

        # Total return
        total_return = ((capital - initial_capital) / initial_capital) * 100

        # Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = (
            (np.mean(returns) / np.std(returns) * np.sqrt(252))
            if len(returns) > 1 and np.std(returns) > 0
            else 0
        )

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        # Win rate
        winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        win_rate = (winning_trades / len(trades) * 100) if trades else 0

        return {
            "return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd * 100,
            "win_rate": win_rate,
        }

    def _aggregate_results(self, results: List[WalkForwardResult]) -> Dict:
        """Tổng hợp kết quả từ tất cả windows"""
        if not results:
            return {}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([vars(r) for r in results])

        # Overall statistics
        total_windows = len(df)
        profitable_windows = (df["test_return"] > 0).sum()

        avg_return = df["test_return"].mean()
        std_return = df["test_return"].std()

        avg_sharpe = df["test_sharpe"].mean()
        avg_max_dd = df["test_max_drawdown"].mean()
        avg_win_rate = df["test_win_rate"].mean()

        # Consistency metrics
        positive_windows_pct = (
            (profitable_windows / total_windows * 100) if total_windows > 0 else 0
        )

        # Best and worst windows
        best_window = (
            df.loc[df["test_return"].idxmax()].to_dict() if not df.empty else {}
        )
        worst_window = (
            df.loc[df["test_return"].idxmin()].to_dict() if not df.empty else {}
        )

        # Stability score (0-100)
        # Higher is better: consistent returns, low drawdown, high win rate
        stability_score = min(
            100,
            (
                positive_windows_pct * 0.4
                + (100 - avg_max_dd) * 0.3
                + avg_win_rate * 0.3
            ),
        )

        return {
            "total_windows": total_windows,
            "profitable_windows": int(profitable_windows),
            "positive_windows_pct": positive_windows_pct,
            "avg_return": avg_return,
            "std_return": std_return,
            "avg_sharpe": avg_sharpe,
            "avg_max_drawdown": avg_max_dd,
            "avg_win_rate": avg_win_rate,
            "stability_score": stability_score,
            "best_window": best_window,
            "worst_window": worst_window,
            "all_windows": results,
        }

    def format_report(self, results: Dict) -> str:
        """Format báo cáo walk-forward test"""
        if not results or results.get("total_windows", 0) == 0:
            return "📊 Không có kết quả walk-forward test"

        lines = []
        lines.append("📊 **WALK-FORWARD TEST RESULTS**")
        lines.append("=" * 50)
        lines.append("")

        # Overall stats
        lines.append("📈 **TỔNG QUAN**")
        lines.append(f"• Tổng windows: {results['total_windows']}")
        lines.append(
            f"• Windows có lời: {results['profitable_windows']} ({results['positive_windows_pct']:.1f}%)"
        )
        lines.append(
            f"• Return trung bình: {results['avg_return']:.2f}% ± {results['std_return']:.2f}%"
        )
        lines.append(f"• Sharpe ratio TB: {results['avg_sharpe']:.2f}")
        lines.append(f"• Max drawdown TB: {results['avg_max_drawdown']:.2f}%")
        lines.append(f"• Win rate TB: {results['avg_win_rate']:.1f}%")
        lines.append(f"• Stability score: {results['stability_score']:.1f}/100")
        lines.append("")

        # Best window
        if results.get("best_window"):
            best = results["best_window"]
            lines.append("🏆 **BEST WINDOW**")
            lines.append(f"• Period: {best['test_start']} to {best['test_end']}")
            lines.append(f"• Return: {best['test_return']:.2f}%")
            lines.append(f"• Sharpe: {best['test_sharpe']:.2f}")
            lines.append(f"• Trades: {best['num_trades']}")
            lines.append("")

        # Worst window
        if results.get("worst_window"):
            worst = results["worst_window"]
            lines.append("💔 **WORST WINDOW**")
            lines.append(f"• Period: {worst['test_start']} to {worst['test_end']}")
            lines.append(f"• Return: {worst['test_return']:.2f}%")
            lines.append(f"• Sharpe: {worst['test_sharpe']:.2f}")
            lines.append(f"• Trades: {worst['num_trades']}")
            lines.append("")

        # Interpretation
        lines.append("💡 **ĐÁNH GIÁ**")
        if results["stability_score"] >= 70:
            lines.append("✅ Chiến lược ổn định và robust")
        elif results["stability_score"] >= 50:
            lines.append("⚠️ Chiến lược khá ổn định nhưng cần cải thiện")
        else:
            lines.append("❌ Chiến lược không ổn định, có thể bị overfitting")

        if results["positive_windows_pct"] >= 60:
            lines.append("✅ Tỷ lệ windows có lời cao")
        else:
            lines.append("⚠️ Tỷ lệ windows có lời thấp")

        return "\n".join(lines)


# Example strategy function
def example_strategy(train_data: pd.DataFrame, test_data: pd.DataFrame) -> List[Dict]:
    """
    Example strategy function

    Args:
        train_data: Training data (không dùng trong simple strategy)
        test_data: Test data để generate trades

    Returns:
        List of trades với format: {'entry_date', 'exit_date', 'pnl'}
    """
    from src.ml.signals.generator import MLSignalGenerator

    ml_gen = MLSignalGenerator()
    trades = []

    # Simple strategy: buy on BUY signal, sell after 5 days
    position = None

    for i in range(len(test_data)):
        if position is None:
            # Look for entry
            df_slice = test_data.iloc[: i + 1]
            if len(df_slice) < 20:
                continue

            signal = ml_gen.analyze(df_slice)

            if signal["signal"] == "BUY" and signal["confidence"] >= 60:
                position = {
                    "entry_date": test_data.iloc[i]["time"],
                    "entry_price": test_data.iloc[i]["close"],
                    "entry_idx": i,
                }
        else:
            # Check exit (after 5 days or at end)
            hold_days = i - position["entry_idx"]

            if hold_days >= 5 or i == len(test_data) - 1:
                exit_price = test_data.iloc[i]["close"]
                pnl = (exit_price - position["entry_price"]) * 100  # Assume 100 shares

                trades.append(
                    {
                        "entry_date": position["entry_date"],
                        "exit_date": test_data.iloc[i]["time"],
                        "pnl": pnl,
                    }
                )

                position = None

    return trades


# Test
if __name__ == "__main__":
    print("Testing Walk-Forward...")

    tester = WalkForwardTester(train_period=180, test_period=30, step=30)

    test_symbols = ["VCB", "FPT"]

    results = tester.run_walk_forward(
        symbols=test_symbols,
        strategy_function=example_strategy,
        initial_capital=100_000_000,
    )

    report = tester.format_report(results)
    print(report)

    print("\n✅ Test completed!")
