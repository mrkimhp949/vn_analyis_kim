# -*- coding: utf-8 -*-
"""
Backtest Visualizer - Charts and analysis
"""

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting.engine import BacktestResult


class BacktestVisualizer:
    """Create visualizations for backtest results"""

    @staticmethod
    def plot_equity_curve(result: BacktestResult, save_path: Optional[str] = None):
        """Plot equity curve over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        equity_df = result.equity_curve
        dates = pd.to_datetime(equity_df["date"])

        # Plot 1: Equity curve
        ax1.plot(
            dates,
            equity_df["equity"],
            linewidth=2,
            label="Portfolio Value",
            color="#2E86AB",
        )
        ax1.axhline(
            y=result.initial_capital,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Initial Capital",
        )
        ax1.fill_between(
            dates,
            result.initial_capital,
            equity_df["equity"],
            where=equity_df["equity"] >= result.initial_capital,
            alpha=0.3,
            color="green",
            label="Profit",
        )
        ax1.fill_between(
            dates,
            result.initial_capital,
            equity_df["equity"],
            where=equity_df["equity"] < result.initial_capital,
            alpha=0.3,
            color="red",
            label="Loss",
        )

        ax1.set_title("Portfolio Equity Curve", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel("Value (VND)", fontsize=12)
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Drawdown
        rolling_max = equity_df["equity"].expanding().max()
        drawdown = (equity_df["equity"] - rolling_max) / rolling_max * 100

        ax2.fill_between(dates, 0, drawdown, alpha=0.4, color="red", label="Drawdown")
        ax2.plot(dates, drawdown, linewidth=1.5, color="darkred")

        ax2.set_title("Portfolio Drawdown", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Drawdown (%)", fontsize=12)
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved equity curve to {save_path}")

        plt.show()

    @staticmethod
    def plot_trade_analysis(result: BacktestResult, save_path: Optional[str] = None):
        """Plot trade analysis charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        trades = result.trades

        if not trades:
            print("No trades to visualize")
            return

        # Plot 1: PnL Distribution
        pnls = [t.pnl for t in trades]
        colors = ["green" if p > 0 else "red" for p in pnls]

        ax1.bar(range(len(pnls)), pnls, color=colors, alpha=0.6)
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax1.set_title("Trade PnL Distribution", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Trade Number", fontsize=12)
        ax1.set_ylabel("PnL (VND)", fontsize=12)
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Win/Loss Distribution
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]

        win_loss_data = [len(wins), len(losses)]
        labels = [f"Wins\n({len(wins)})", f"Losses\n({len(losses)})"]
        colors_pie = ["#2ecc71", "#e74c3c"]

        ax2.pie(
            win_loss_data,
            labels=labels,
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12},
        )
        ax2.set_title("Win/Loss Ratio", fontsize=14, fontweight="bold")

        # Plot 3: Holding Period Distribution
        holding_days = [t.holding_days for t in trades]

        ax3.hist(holding_days, bins=20, color="#3498db", alpha=0.7, edgecolor="black")
        ax3.axvline(
            x=np.mean(holding_days),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(holding_days):.1f} days",
        )
        ax3.set_title("Holding Period Distribution", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Days Held", fontsize=12)
        ax3.set_ylabel("Number of Trades", fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        # Plot 4: PnL % Distribution
        pnl_pcts = [t.pnl_percent for t in trades]

        ax4.hist(pnl_pcts, bins=30, color="#9b59b6", alpha=0.7, edgecolor="black")
        ax4.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax4.axvline(
            x=np.mean(pnl_pcts),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(pnl_pcts):.2f}%",
        )
        ax4.set_title("PnL % Distribution", fontsize=14, fontweight="bold")
        ax4.set_xlabel("PnL (%)", fontsize=12)
        ax4.set_ylabel("Number of Trades", fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved trade analysis to {save_path}")

        plt.show()

    @staticmethod
    def plot_monthly_returns(result: BacktestResult, save_path: Optional[str] = None):
        """Plot monthly returns heatmap"""
        equity_df = result.equity_curve.copy()
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df["year"] = equity_df["date"].dt.year
        equity_df["month"] = equity_df["date"].dt.month

        # Calculate monthly returns
        monthly_returns = []
        for year in equity_df["year"].unique():
            for month in range(1, 13):
                month_data = equity_df[
                    (equity_df["year"] == year) & (equity_df["month"] == month)
                ]
                if len(month_data) > 0:
                    start_val = month_data["equity"].iloc[0]
                    end_val = month_data["equity"].iloc[-1]
                    monthly_return = ((end_val - start_val) / start_val) * 100
                    monthly_returns.append(
                        {"year": year, "month": month, "return": monthly_return}
                    )

        if not monthly_returns:
            print("Not enough data for monthly returns")
            return

        returns_df = pd.DataFrame(monthly_returns)
        pivot_table = returns_df.pivot(index="month", columns="year", values="return")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(
            pivot_table.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10
        )

        # Set ticks
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )

        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.values[i, j]
                if not np.isnan(value):
                    _text = ax.text(  # noqa: F841
                        j,
                        i,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        color="black" if abs(value) < 5 else "white",
                        fontsize=10,
                    )

        ax.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Month", fontsize=12)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Return (%)", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved monthly returns to {save_path}")

        plt.show()

    @staticmethod
    def create_full_report(
        result: BacktestResult, output_dir: str = "backtest_results"
    ):
        """Create complete visualization report"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📊 Creating backtest visualization report in {output_dir}/")

        BacktestVisualizer.plot_equity_curve(
            result, save_path=f"{output_dir}/equity_curve.png"
        )

        BacktestVisualizer.plot_trade_analysis(
            result, save_path=f"{output_dir}/trade_analysis.png"
        )

        BacktestVisualizer.plot_monthly_returns(
            result, save_path=f"{output_dir}/monthly_returns.png"
        )

        print(f"\n✅ Report complete! Check {output_dir}/ for charts")


if __name__ == "__main__":
    # Example: Visualize existing backtest results
    from backtesting.strategy_runner import run_simple_backtest

    print("Running backtest...")
    result = run_simple_backtest(months_back=6)

    print("\nCreating visualizations...")
    BacktestVisualizer.create_full_report(result)
