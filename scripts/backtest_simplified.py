"""
Backtesting Script for Simplified Entry Logic
Tests the new 8-filter entry logic vs old 14-filter logic
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data.loader import load_data
from src.ml.features.enhanced_v2 import add_ml_features
from src.ml.signals.generator import MLSignalGenerator
from src.strategies.entry_logic_simplified import SimplifiedEntryLogic
from src.config.entry_config import get_entry_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleBacktester:
    """
    Simple backtester for testing entry logic
    """

    def __init__(
        self, initial_capital=100_000_000, commission=0.0015, slippage=0.001, use_simplified=True
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.use_simplified = use_simplified

        # Initialize entry logic
        if use_simplified:
            logger.info("✅ Using SIMPLIFIED Entry Logic (8 filters)")
            self.entry_logic = SimplifiedEntryLogic()
        else:
            logger.info("⚠️ Using ORIGINAL Entry Logic (14 filters)")
            from src.strategies.entry_logic import ImprovedEntryLogic

            self.entry_logic = ImprovedEntryLogic()

        self.ml_generator = MLSignalGenerator()

    def backtest_symbol(self, symbol, start_date=None, end_date=None, lookback=500):
        """
        Run backtest on a single symbol

        Returns:
            dict with backtest results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 BACKTESTING: {symbol}")
        logger.info(f"{'='*70}")

        # Load data
        try:
            df = load_data(symbol, lookback=lookback, use_cache=False)
            if df.empty or len(df) < 200:
                logger.error(f"❌ Insufficient data for {symbol}")
                return None

            # Load index for ML features
            index_df = load_data("VNINDEX", lookback=lookback, is_index=True, use_cache=False)
            df = add_ml_features(df, index_df=index_df)

        except Exception as e:
            logger.error(f"❌ Error loading data for {symbol}: {e}")
            return None

        # Filter date range if specified
        if start_date:
            df = df[df["time"] >= start_date]
        if end_date:
            df = df[df["time"] <= end_date]

        if len(df) < 100:
            logger.error(f"❌ Not enough data after date filtering for {symbol}")
            return None

        logger.info(f"📅 Backtest period: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        logger.info(f"📊 Total bars: {len(df)}")

        # Initialize tracking
        capital = self.initial_capital
        position = None
        trades = []
        portfolio_values = []

        # Walk through data (skip first 200 bars for indicators)
        for i in range(200, len(df)):
            current_date = df.iloc[i]["time"]
            current_price = df.iloc[i]["close"]

            # Get data up to current point (look-back window)
            hist_df = df.iloc[: i + 1].copy()

            # Check if we have a position
            if position is not None:
                # Check exit conditions (simple: stop loss or take profit)
                pnl_pct = (
                    (current_price - position["entry_price"]) / position["entry_price"]
                ) * 100

                exit_reason = None
                if current_price <= position["stop_loss"]:
                    exit_reason = "Stop Loss"
                elif current_price >= position["take_profit"]:
                    exit_reason = "Take Profit"
                elif (current_date - position["entry_date"]).days >= 20:  # Max holding
                    exit_reason = "Max Holding (20 days)"

                if exit_reason:
                    # Close position
                    shares = position["shares"]
                    entry_cost = shares * position["entry_price"]
                    exit_value = shares * current_price
                    gross_pnl = exit_value - entry_cost

                    # Deduct costs
                    entry_commission = entry_cost * self.commission
                    exit_commission = exit_value * self.commission
                    total_commission = entry_commission + exit_commission
                    slippage_cost = entry_cost * self.slippage

                    net_pnl = gross_pnl - total_commission - slippage_cost
                    net_pnl_pct = (net_pnl / entry_cost) * 100

                    # Update capital
                    capital += net_pnl

                    # Record trade
                    trade = {
                        "symbol": symbol,
                        "entry_date": position["entry_date"],
                        "entry_price": position["entry_price"],
                        "exit_date": current_date,
                        "exit_price": current_price,
                        "shares": shares,
                        "stop_loss": position["stop_loss"],
                        "take_profit": position["take_profit"],
                        "holding_days": (current_date - position["entry_date"]).days,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "net_pnl_pct": net_pnl_pct,
                        "commission": total_commission,
                        "slippage_cost": slippage_cost,
                        "exit_reason": exit_reason,
                        "confidence": position["confidence"],
                    }
                    trades.append(trade)

                    logger.info(
                        f"🔴 EXIT: {symbol} @ {current_price:,.0f} | "
                        f"P&L: {net_pnl:+,.0f} ({net_pnl_pct:+.1f}%) | "
                        f"Reason: {exit_reason}"
                    )

                    position = None

            # If no position, check for entry signal
            if position is None:
                try:
                    # Generate ML signal
                    ml_signal = self.ml_generator.analyze(hist_df, index_df=index_df)

                    # Check entry logic
                    entry_signal = self.entry_logic.analyze_entry(
                        df=hist_df, ml_signal=ml_signal, symbol=symbol
                    )

                    if entry_signal.should_enter and entry_signal.signal_type == "BUY":
                        # Calculate position size (simple: 20% of capital)
                        position_value = capital * 0.20 * entry_signal.position_size_multiplier
                        shares = int(position_value / current_price)

                        if shares > 0:
                            # Open position
                            position = {
                                "symbol": symbol,
                                "entry_date": current_date,
                                "entry_price": current_price,
                                "shares": shares,
                                "stop_loss": entry_signal.stop_loss,
                                "take_profit": (
                                    entry_signal.take_profit_targets[1]
                                    if len(entry_signal.take_profit_targets) > 1
                                    else current_price * 1.1
                                ),
                                "confidence": entry_signal.confidence,
                                "strength": entry_signal.strength.name,
                            }

                            logger.info(
                                f"🟢 ENTRY: {symbol} @ {current_price:,.0f} | "
                                f"Shares: {shares:,} | "
                                f"Confidence: {entry_signal.confidence}% | "
                                f"Strength: {entry_signal.strength.name} | "
                                f"SL: {entry_signal.stop_loss:,.0f} | "
                                f"TP: {position['take_profit']:,.0f}"
                            )

                except Exception as e:
                    logger.warning(f"⚠️ Error analyzing entry for {symbol}: {e}")
                    continue

            # Track portfolio value
            current_value = capital
            if position:
                current_value += position["shares"] * current_price

            portfolio_values.append(
                {"date": current_date, "value": current_value, "cash": capital, "symbol": symbol}
            )

        # Close any open position at end
        if position:
            current_price = df.iloc[-1]["close"]
            shares = position["shares"]
            entry_cost = shares * position["entry_price"]
            exit_value = shares * current_price
            net_pnl = exit_value - entry_cost - (entry_cost * self.commission * 2)
            capital += net_pnl

            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "exit_date": df.iloc[-1]["time"],
                    "exit_price": current_price,
                    "shares": shares,
                    "net_pnl": net_pnl,
                    "net_pnl_pct": (net_pnl / entry_cost) * 100,
                    "exit_reason": "End of backtest",
                    "confidence": position["confidence"],
                }
            )

        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        portfolio_df = pd.DataFrame(portfolio_values)

        if len(trades_df) == 0:
            logger.warning(f"⚠️ No trades for {symbol}")
            return {
                "symbol": symbol,
                "total_trades": 0,
                "win_rate": 0,
                "total_return": 0,
                "sharpe_ratio": 0,
            }

        winning_trades = trades_df[trades_df["net_pnl"] > 0]
        losing_trades = trades_df[trades_df["net_pnl"] <= 0]

        total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
        avg_win = winning_trades["net_pnl"].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades["net_pnl"].mean() if len(losing_trades) > 0 else 0
        win_rate = (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0

        # Calculate max drawdown
        portfolio_df["peak"] = portfolio_df["value"].cummax()
        portfolio_df["drawdown"] = (
            (portfolio_df["value"] - portfolio_df["peak"]) / portfolio_df["peak"]
        ) * 100
        max_drawdown = portfolio_df["drawdown"].min()

        # Sharpe ratio (simplified)
        returns = trades_df["net_pnl_pct"].values
        sharpe_ratio = (
            (np.mean(returns) / np.std(returns)) * np.sqrt(252 / 20) if len(returns) > 1 else 0
        )  # Annualized

        results = {
            "symbol": symbol,
            "initial_capital": self.initial_capital,
            "final_capital": capital,
            "total_return": total_return,
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_confidence": trades_df["confidence"].mean(),
            "trades": trades_df,
            "portfolio_values": portfolio_df,
        }

        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 BACKTEST RESULTS: {symbol}")
        logger.info(f"{'='*70}")
        logger.info(f"💰 Initial Capital: {self.initial_capital:,.0f} VNĐ")
        logger.info(f"💰 Final Capital:   {capital:,.0f} VNĐ")
        logger.info(f"📈 Total Return:    {total_return:+.2f}%")
        logger.info(f"📊 Total Trades:    {len(trades_df)}")
        logger.info(f"✅ Winning Trades:  {len(winning_trades)} ({win_rate:.1f}%)")
        logger.info(f"❌ Losing Trades:   {len(losing_trades)}")
        logger.info(f"💚 Avg Win:         {avg_win:+,.0f} VNĐ")
        logger.info(f"💔 Avg Loss:        {avg_loss:+,.0f} VNĐ")
        logger.info(f"📉 Max Drawdown:    {max_drawdown:.2f}%")
        logger.info(f"📊 Sharpe Ratio:    {sharpe_ratio:.2f}")
        logger.info(f"🎯 Avg Confidence:  {trades_df['confidence'].mean():.1f}%")
        logger.info(f"{'='*70}\n")

        return results


def main():
    """Run backtest on multiple symbols"""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 BACKTESTING SIMPLIFIED ENTRY LOGIC")
    logger.info("=" * 70 + "\n")

    # Test symbols
    symbols = ["VNM", "HPG", "VCB", "FPT", "VIC"]

    # Backtest period (last 2 years)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    logger.info(f"📅 Backtest Period: {start_date} to {end_date}")
    logger.info(f"📊 Test Symbols: {', '.join(symbols)}\n")

    # Run backtest
    backtester = SimpleBacktester(use_simplified=True)

    all_results = []
    for symbol in symbols:
        try:
            result = backtester.backtest_symbol(symbol, start_date=start_date, end_date=end_date)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"❌ Error backtesting {symbol}: {e}")
            continue

    # Aggregate results
    if len(all_results) > 0:
        logger.info("\n" + "=" * 70)
        logger.info("📊 AGGREGATE RESULTS")
        logger.info("=" * 70)

        total_trades = sum(r["total_trades"] for r in all_results)
        avg_return = np.mean([r["total_return"] for r in all_results])
        avg_win_rate = np.mean([r["win_rate"] for r in all_results])
        avg_sharpe = np.mean([r["sharpe_ratio"] for r in all_results])
        avg_drawdown = np.mean([r["max_drawdown"] for r in all_results])

        logger.info(f"📊 Symbols Tested:   {len(all_results)}")
        logger.info(f"📊 Total Trades:     {total_trades}")
        logger.info(f"📈 Avg Return:       {avg_return:+.2f}%")
        logger.info(f"✅ Avg Win Rate:     {avg_win_rate:.1f}%")
        logger.info(f"📊 Avg Sharpe:       {avg_sharpe:.2f}")
        logger.info(f"📉 Avg Max Drawdown: {avg_drawdown:.2f}%")
        logger.info("=" * 70 + "\n")

        # Save results
        results_df = pd.DataFrame(
            [
                {
                    "symbol": r["symbol"],
                    "total_return": r["total_return"],
                    "total_trades": r["total_trades"],
                    "win_rate": r["win_rate"],
                    "sharpe_ratio": r["sharpe_ratio"],
                    "max_drawdown": r["max_drawdown"],
                    "avg_confidence": r["avg_confidence"],
                }
                for r in all_results
            ]
        )

        output_file = "backtest_results/simplified_entry_logic_results.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"✅ Results saved to: {output_file}")

    else:
        logger.error("❌ No successful backtest results")


if __name__ == "__main__":
    main()
