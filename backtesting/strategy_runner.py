# -*- coding: utf-8 -*-
"""
Strategy Runner - Run backtests with actual trading logic
"""

import logging
from datetime import datetime, timedelta
from typing import List

from src.data.loader import load_data
from src.strategies.entry_logic import ImprovedEntryLogic
from src.strategies.exit_logic import ImprovedExitStrategy

from backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


class StrategyRunner:
    """
    Run backtests using actual entry/exit logic

    This integrates:
    - ImprovedEntryLogic for entry signals
    - ImprovedExitStrategy for exit decisions
    - BacktestEngine for simulation
    """

    def __init__(self, config: BacktestConfig = None):
        self.engine = BacktestEngine(config)
        self.entry_logic = ImprovedEntryLogic()
        self.exit_logic = ImprovedExitStrategy()

        logger.info("Strategy runner initialized")

    def _load_symbol_data(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> dict:
        """Load historical data for all symbols"""
        data_cache = {}
        for symbol in symbols:
            try:
                days_diff = (end_date - start_date).days
                lookback = int(days_diff * 1.5)  # Add 50% buffer

                df = load_data(symbol=symbol, lookback=lookback, use_cache=True)
                if df is not None and len(df) > 0:
                    df = df[df["time"] <= end_date].copy()
                    data_cache[symbol] = df
                    logger.info(f"Loaded {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data for {symbol}")
            except Exception:
                logger.error(f"Error loading {symbol}")
        return data_cache

    def _get_trading_dates(
        self, data_cache: dict, start_date: datetime, end_date: datetime
    ) -> List:
        """Extract and sort unique trading dates from data"""
        all_dates = set()
        for df in data_cache.values():
            all_dates.update(df["time"].dt.date)
        trading_dates = sorted(list(all_dates))
        return [d for d in trading_dates if start_date.date() <= d <= end_date.date()]

    def _check_exits_for_date(self, current_date, data_cache: dict, current_prices: dict):
        """Check exit conditions for all open positions"""
        for symbol in list(self.engine.positions.keys()):
            if symbol not in data_cache:
                continue

            df = data_cache[symbol]
            df_up_to_date = df[df["time"].dt.date <= current_date].copy()

            if len(df_up_to_date) < 50:
                continue

            current_price = df_up_to_date["close"].iloc[-1]
            current_prices[symbol] = current_price

            trade = self.engine.positions[symbol]

            exit_decision = self.exit_logic.check_exit(
                symbol=symbol,
                entry_price=trade.entry_price,
                current_price=current_price,
                stop_loss=trade.stop_loss,
                take_profit_targets=[trade.take_profit],
                entry_date=trade.entry_date,
                df=df_up_to_date,
            )

            if exit_decision.should_exit:
                self.engine.close_position(
                    symbol=symbol,
                    date=datetime.combine(current_date, datetime.min.time()),
                    exit_price=current_price,
                    reason=(
                        exit_decision.exit_reason.value if exit_decision.exit_reason else "Unknown"
                    ),
                )

    def _check_entries_for_date(
        self,
        current_date,
        symbols: List[str],
        data_cache: dict,
        current_prices: dict,
        use_ml_signals: bool,
    ):
        """Check entry conditions for symbols without positions"""
        for symbol in symbols:
            if symbol not in data_cache:
                continue

            if symbol in self.engine.positions:
                continue

            df = data_cache[symbol]
            df_up_to_date = df[df["time"].dt.date <= current_date].copy()

            if len(df_up_to_date) < 200:
                continue

            current_price = df_up_to_date["close"].iloc[-1]
            current_prices[symbol] = current_price

            ml_signal = None
            if use_ml_signals:
                ml_signal = {
                    "signal": "HOLD",
                    "confidence": 50,
                    "reason": "Backtest",
                }

            try:
                entry_signal = self.entry_logic.analyze_entry(df_up_to_date, ml_signal)

                if entry_signal.signal_type == "BUY":
                    self.engine.open_position(
                        symbol=symbol,
                        date=datetime.combine(current_date, datetime.min.time()),
                        entry_price=entry_signal.entry_price,
                        stop_loss=entry_signal.stop_loss,
                        take_profit=(
                            entry_signal.take_profit_targets[1]
                            if len(entry_signal.take_profit_targets) > 1
                            else entry_signal.entry_price * 1.15
                        ),
                        reason=(
                            f"{entry_signal.strength.value}: " f"{', '.join(entry_signal.reasons)}"
                        ),
                    )
            except Exception:
                logger.error(f"Error analyzing {symbol}")

    def _close_remaining_positions(self, data_cache: dict, end_date: datetime):
        """Close any remaining open positions at backtest end"""
        for symbol in list(self.engine.positions.keys()):
            if symbol in data_cache:
                df = data_cache[symbol]
                final_price = df[df["time"].dt.date <= end_date.date()]["close"].iloc[-1]
                self.engine.close_position(
                    symbol=symbol,
                    date=end_date,
                    exit_price=final_price,
                    reason="End of backtest",
                )

    def run_backtest(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        use_ml_signals: bool = False,
    ) -> BacktestResult:
        """
        Run backtest on multiple symbols

        Args:
            symbols: List of stock symbols
            start_date: Backtest start date
            end_date: Backtest end date
            use_ml_signals: Whether to use ML predictions

        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Starting backtest: {start_date.date()} → {end_date.date()}")
        logger.info(f"Symbols: {', '.join(symbols)}")

        # Step 1: Load historical data
        data_cache = self._load_symbol_data(symbols, start_date, end_date)
        if not data_cache:
            raise ValueError("No data loaded for any symbol")

        # Step 2: Get trading dates
        trading_dates = self._get_trading_dates(data_cache, start_date, end_date)
        logger.info(f"Simulating {len(trading_dates)} trading days...")

        # Step 3: Simulate each trading day
        for current_date in trading_dates:
            current_prices = {}

            # Check exits first
            self._check_exits_for_date(current_date, data_cache, current_prices)

            # Check entries
            self._check_entries_for_date(
                current_date, symbols, data_cache, current_prices, use_ml_signals
            )

            # Update equity curve
            self.engine.update_equity(
                date=datetime.combine(current_date, datetime.min.time()),
                current_prices=current_prices,
            )

        # Step 4: Close remaining positions
        self._close_remaining_positions(data_cache, end_date)

        # Calculate and return results
        results = self.engine.calculate_results()
        logger.info("Backtest complete!")

        return results


def run_simple_backtest(symbols: List[str] = None, months_back: int = 12) -> BacktestResult:
    """
    Quick helper function to run a simple backtest

    Args:
        symbols: List of symbols (defaults to common VN stocks)
        months_back: How many months of history to test

    Returns:
        BacktestResult
    """
    if symbols is None:
        symbols = ["VCB", "HPG", "VHM", "VNM", "VIC", "GAS", "MSN", "MWG", "TCB", "BID"]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)

    config = BacktestConfig(
        initial_capital=100_000_000,  # 100M VND
        commission_rate=0.0015,
        position_size_pct=0.20,
        max_positions=5,
    )

    runner = StrategyRunner(config)
    results = runner.run_backtest(
        symbols=symbols, start_date=start_date, end_date=end_date, use_ml_signals=False
    )

    runner.engine.print_results(results)

    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Running 12-month backtest on VN30 stocks...")
    results = run_simple_backtest(months_back=12)

    print(f"\n✅ Backtest complete! Final return: {results.total_return_pct:+.2f}%")
