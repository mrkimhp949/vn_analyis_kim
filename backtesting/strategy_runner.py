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

    def __init__(self, config: BacktestConfig = None, entry_logic: ImprovedEntryLogic = None):
        self.engine = BacktestEngine(config)
        # Use provided entry_logic or create one with relaxed parameters for backtesting
        if entry_logic is None:
            # Relaxed parameters for backtesting to allow more trades
            self.entry_logic = ImprovedEntryLogic(
                min_confidence=40,  # Lower from 60 to allow more signals
                min_risk_reward=1.5,  # Lower from 2.0 to be less strict
                support_distance_percent=5.0,  # More flexible
                require_trend_alignment=False,  # Allow trades in sideways markets
                require_volume_confirmation=False,  # Don't require volume confirmation
                min_liquidity_value=2_000_000_000,  # Lower from 5B to 2B VND
                min_avg_volume=50_000,  # Lower from 150k to 50k
            )
        else:
            self.entry_logic = entry_logic
        self.exit_logic = ImprovedExitStrategy()

        # Statistics for debugging
        self.stats = {
            "signals_checked": 0,
            "signals_rejected": 0,
            "signals_passed": 0,
            "rejection_reasons": {},
        }

        logger.info(
            f"Strategy runner initialized "
            f"(min_confidence={self.entry_logic.min_confidence}%, "
            f"trend_req={self.entry_logic.require_trend_alignment}, "
            f"volume_req={self.entry_logic.require_volume_confirmation})"
        )

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
                # Create a neutral market regime for backtesting
                # This prevents filters from being too strict
                market_regime = {
                    "regime": "SIDEWAYS",
                    "tradeable": True,
                    "confidence": 50,
                }

                self.stats["signals_checked"] += 1
                entry_signal = self.entry_logic.analyze_entry(
                    df_up_to_date, ml_signal, market_regime=market_regime, symbol=symbol
                )

                if entry_signal.signal_type == "BUY":
                    self.stats["signals_passed"] += 1
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
                else:
                    # Track rejection reasons
                    self.stats["signals_rejected"] += 1
                    signal_type = entry_signal.signal_type
                    self.stats["rejection_reasons"][signal_type] = (
                        self.stats["rejection_reasons"].get(signal_type, 0) + 1
                    )
                    # Log why entry was rejected for debugging
                    logger.debug(
                        f"[{current_date}] {symbol}: Signal={signal_type}, "
                        f"reasons={getattr(entry_signal, 'reasons', [])}"
                    )
            except Exception as e:
                self.stats["signals_rejected"] += 1
                self.stats["rejection_reasons"]["ERROR"] = (
                    self.stats["rejection_reasons"].get("ERROR", 0) + 1
                )
                logger.error(f"Error analyzing {symbol} on {current_date}: {e}", exc_info=True)

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

        # Reset statistics
        self.stats = {
            "signals_checked": 0,
            "signals_rejected": 0,
            "signals_passed": 0,
            "rejection_reasons": {},
        }

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

        # Log statistics
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Signals checked: {self.stats['signals_checked']}")
        logger.info(f"Signals passed: {self.stats['signals_passed']}")
        logger.info(f"Signals rejected: {self.stats['signals_rejected']}")
        if self.stats["signals_checked"] > 0:
            pass_rate = (self.stats["signals_passed"] / self.stats["signals_checked"]) * 100
            logger.info(f"Pass rate: {pass_rate:.1f}%")
        if self.stats["rejection_reasons"]:
            logger.info("Rejection reasons:")
            for reason, count in sorted(
                self.stats["rejection_reasons"].items(), key=lambda x: x[1], reverse=True
            ):
                logger.info(f"  {reason}: {count}")
        logger.info("=" * 60 + "\n")

        if self.stats["signals_passed"] == 0 and self.stats["signals_checked"] > 0:
            logger.warning(
                "⚠️ WARNING: No trades executed! Consider relaxing entry criteria:\n"
                "  - Lower min_confidence (currently {})\n"
                "  - Lower min_risk_reward (currently {})\n"
                "  - Set require_trend_alignment=False\n"
                "  - Set require_volume_confirmation=False".format(
                    self.entry_logic.min_confidence, self.entry_logic.min_risk_reward
                )
            )

        logger.info("Backtest complete!")

        return results


def run_simple_backtest(
    symbols: List[str] = None,
    months_back: int = 12,
    max_symbols: int = 100,
    min_volume: int = 100_000,
    use_validated_only: bool = True,
) -> BacktestResult:
    """
    Quick helper function to run a simple backtest

    Args:
        symbols: List of symbols (defaults to loading validated tickers from List.csv)
        months_back: How many months of history to test
        max_symbols: Maximum number of symbols to test (default: 100, None = all available)
                     ⚠️ Warning: Backtesting ALL ~1600 symbols can take HOURS!
        min_volume: Minimum average volume for filtering (default: 100k)
        use_validated_only: Only use validated tickers with data available (default: True)
                           This filters out delisted stocks and low liquidity stocks

    Returns:
        BacktestResult

    Note:
        Backtesting ALL symbols is NOT recommended because:
        - Takes too long (hours/days)
        - Many stocks have no data or are delisted
        - Low liquidity stocks are not tradeable
        - Strategy may not work on all stocks

        Recommended: Use max_symbols=50-200 with use_validated_only=True
    """
    if symbols is None:
        # Try to load from List.csv with intelligent filtering
        try:
            if use_validated_only:
                # Use validated tickers (filters out delisted, low volume stocks)
                from src.data.ticker_loader import get_ticker_loader

                loader = get_ticker_loader()
                validated_symbols = loader.get_validated_tickers(
                    force_validate=False,  # Use cache for speed
                    min_volume=min_volume,
                    max_tickers=max_symbols,  # Limit during validation
                )

                if validated_symbols:
                    symbols = validated_symbols
                    logger.info(
                        f"✅ Loaded {len(symbols)} validated symbols "
                        f"(min_volume={min_volume:,}, filtered from {len(loader.all_tickers)} total)"
                    )
                else:
                    # Fallback: try getting all and limit manually
                    from src.config.legacy_config import get_tickers

                    all_symbols = get_tickers()
                    if all_symbols:
                        symbols = all_symbols[:max_symbols] if max_symbols else all_symbols
                        logger.warning(
                            f"⚠️ No validated symbols found, using {len(symbols)} symbols without validation"
                        )
                    else:
                        raise ValueError("No symbols available")
            else:
                # Load all without validation (faster but may include invalid stocks)
                from src.config.legacy_config import get_tickers

                all_symbols = get_tickers()
                if all_symbols:
                    symbols = all_symbols
                    logger.info(f"Loaded {len(symbols)} symbols from List.csv (no validation)")

                    # Warn if too many symbols
                    if len(symbols) > 200:
                        logger.warning(
                            f"⚠️ WARNING: Backtesting {len(symbols)} symbols will take a LONG time! "
                            f"Consider using max_symbols=100 or use_validated_only=True"
                        )

                    # Limit if max_symbols specified
                    if max_symbols and len(symbols) > max_symbols:
                        symbols = symbols[:max_symbols]
                        logger.info(f"Limited to {max_symbols} symbols for backtest")
                else:
                    raise ValueError("No symbols available")

        except Exception as e:
            # Fallback to VN30 stocks if error
            symbols = ["VCB", "HPG", "VHM", "VNM", "VIC", "GAS", "MSN", "MWG", "TCB", "BID"]
            logger.warning(f"Error loading symbols: {e}, using default VN30 stocks")

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

    import argparse

    parser = argparse.ArgumentParser(description="Run backtest on multiple symbols")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: load from List.csv)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of months of history to test (default: 12)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=100,
        help="Maximum number of symbols to test (default: 100, None = all available) ⚠️ Testing all ~1600 symbols takes HOURS!",
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=100_000,
        help="Minimum average volume for filtering (default: 100000)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation (faster but may include invalid/delisted stocks)",
    )

    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        print(f"Running backtest on {len(symbols)} specified symbols...")
    else:
        print("Running backtest on symbols from List.csv...")
        if args.max_symbols:
            print(f"Limited to first {args.max_symbols} symbols")

    results = run_simple_backtest(
        symbols=symbols,
        months_back=args.months,
        max_symbols=args.max_symbols,
        min_volume=args.min_volume,
        use_validated_only=not args.no_validation,
    )

    print(f"\n✅ Backtest complete! Final return: {results.total_return_pct:+.2f}%")
