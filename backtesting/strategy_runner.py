# -*- coding: utf-8 -*-
"""
Strategy Runner - Run backtests with actual trading logic

IMPROVEMENTS V2:
- Market regime detection integration
- Real ML signal integration (not placeholder)
- Partial exit support
- Position size multiplier from entry signals
- Circuit breaker integration
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.data.loader import load_data
from src.strategies.entry_logic import ImprovedEntryLogic
from src.strategies.exit_logic import ImprovedExitStrategy

from backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)

# Optional imports for ML and market regime
try:
    from src.market.regime_detector import MarketRegimeDetector, detect_regime

    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False
    logger.warning("Market regime detector not available")

try:
    from src.ml.models.predictor import MLPredictor

    ML_PREDICTOR_AVAILABLE = True
except ImportError:
    ML_PREDICTOR_AVAILABLE = False
    logger.warning("ML predictor not available")


class StrategyRunner:
    """
    Run backtests using actual entry/exit logic

    This integrates:
    - ImprovedEntryLogic for entry signals
    - ImprovedExitStrategy for exit decisions
    - BacktestEngine for simulation
    - MarketRegimeDetector for market regime analysis
    - MLPredictor for ML-based signals
    - Partial exit support
    """

    def __init__(self, config: BacktestConfig = None, use_ml: bool = True, use_regime: bool = True):
        self.engine = BacktestEngine(config)
        self.entry_logic = ImprovedEntryLogic()
        self.exit_logic = ImprovedExitStrategy()

        # Market regime detector
        self.regime_detector = None
        self.use_regime = use_regime and REGIME_DETECTOR_AVAILABLE
        if self.use_regime:
            try:
                self.regime_detector = MarketRegimeDetector()
                logger.info("✅ Market regime detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize regime detector: {e}")
                self.use_regime = False

        # ML predictor
        self.ml_predictor = None
        self.use_ml = use_ml and ML_PREDICTOR_AVAILABLE
        if self.use_ml:
            try:
                self.ml_predictor = MLPredictor()
                if self.ml_predictor.load_models():
                    logger.info("✅ ML predictor initialized and models loaded")
                else:
                    logger.warning("ML models not available, disabling ML signals")
                    self.use_ml = False
            except Exception as e:
                logger.warning(f"Failed to initialize ML predictor: {e}")
                self.use_ml = False

        # Cache for market regime (updated daily)
        self._current_regime: Optional[Dict] = None
        self._regime_date: Optional[datetime] = None

        # VNINDEX data cache for regime detection
        self._vnindex_data = None

        logger.info("Strategy runner initialized")
        logger.info(f"  ML signals: {'enabled' if self.use_ml else 'disabled'}")
        logger.info(f"  Market regime: {'enabled' if self.use_regime else 'disabled'}")

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

    def _load_vnindex_data(self, start_date: datetime, end_date: datetime):
        """Load VNINDEX data for market regime detection"""
        if not self.use_regime:
            return

        try:
            days_diff = (end_date - start_date).days
            lookback = int(days_diff * 1.5) + 200  # Extra buffer for regime calculation

            # Try to load VNINDEX
            df = load_data(symbol="VNINDEX", lookback=lookback, use_cache=True, data_type="index")
            if df is not None and len(df) >= 200:
                self._vnindex_data = df
                logger.info(f"✅ Loaded {len(df)} bars of VNINDEX for regime detection")
            else:
                logger.warning("VNINDEX data insufficient for regime detection")
                self._vnindex_data = None
        except Exception as e:
            logger.warning(f"Failed to load VNINDEX data: {e}")
            self._vnindex_data = None

    def _update_market_regime(self, current_date) -> Optional[Dict]:
        """Update market regime for current date"""
        if not self.use_regime or self.regime_detector is None:
            return None

        # Only update once per day
        if self._regime_date is not None and self._regime_date == current_date:
            return self._current_regime

        if self._vnindex_data is None:
            return None

        try:
            # Filter VNINDEX data up to current date
            df_up_to_date = self._vnindex_data[
                self._vnindex_data["time"].dt.date <= current_date
            ].copy()

            if len(df_up_to_date) < 200:
                return None

            # Detect regime
            regime = self.regime_detector.detect(df_up_to_date)

            self._current_regime = {
                "regime": regime.regime,
                "confidence": regime.confidence,
                "tradeable": regime.tradeable,
                "components": regime.components,
            }
            self._regime_date = current_date

            logger.debug(
                f"📊 Market regime: {regime.regime} "
                f"(confidence: {regime.confidence:.1f}%, tradeable: {regime.tradeable})"
            )

            return self._current_regime

        except Exception as e:
            logger.warning(f"Failed to detect market regime: {e}")
            return None

    def _get_ml_signal(self, df, symbol: str) -> Optional[Dict]:
        """Get ML signal for a symbol"""
        if not self.use_ml or self.ml_predictor is None:
            return None

        try:
            # Generate features from dataframe
            from src.ml.features.technical import add_ml_features, get_feature_columns

            # Add ML features to dataframe
            df_with_features = add_ml_features(df.copy())
            if df_with_features is None or len(df_with_features) == 0:
                return None

            # Get feature columns
            feature_cols = get_feature_columns()

            # Check if all feature columns exist
            missing_cols = [col for col in feature_cols if col not in df_with_features.columns]
            if missing_cols:
                logger.debug(f"Missing feature columns for {symbol}: {missing_cols[:5]}...")
                return None

            # Get latest features
            latest_features = df_with_features[feature_cols].iloc[-1:].values

            # Predict
            prediction = self.ml_predictor.predict(latest_features)
            if prediction is None or len(prediction) == 0:
                return None

            prob = prediction[0]

            # Convert probability to signal
            if prob >= 0.6:
                signal = "BUY"
                confidence = int(prob * 100)
            elif prob <= 0.4:
                signal = "SELL"
                confidence = int((1 - prob) * 100)
            else:
                signal = "HOLD"
                confidence = 50

            return {
                "signal": signal,
                "confidence": confidence,
                "probability": prob,
                "reason": f"ML prediction: {prob:.2f}",
            }

        except Exception as e:
            logger.debug(f"ML signal generation failed for {symbol}: {e}")
            return None

    def _get_trading_dates(
        self, data_cache: dict, start_date: datetime, end_date: datetime
    ) -> List:
        """Extract and sort unique trading dates from data"""
        all_dates = set()
        for df in data_cache.values():
            all_dates.update(df["time"].dt.date)
        trading_dates = sorted(list(all_dates))
        return [d for d in trading_dates if start_date.date() <= d <= end_date.date()]

    def _check_exits_for_date(
        self,
        current_date,
        data_cache: dict,
        current_prices: dict,
        market_regime: Optional[Dict] = None,
    ):
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

            # Get ML signal for exit decision
            ml_signal = self._get_ml_signal(df_up_to_date, symbol) if self.use_ml else None

            # Get partial exits from trade
            partial_exits = (
                [pe["price"] for pe in trade.partial_exits] if trade.partial_exits else []
            )

            exit_decision = self.exit_logic.check_exit(
                symbol=symbol,
                entry_price=trade.entry_price,
                current_price=current_price,
                stop_loss=trade.stop_loss,
                take_profit_targets=[trade.take_profit],
                entry_date=trade.entry_date,
                df=df_up_to_date,
                ml_signal=ml_signal,
                market_regime=market_regime,
                partial_exits=partial_exits,
            )

            if exit_decision.should_exit:
                exit_date = datetime.combine(current_date, datetime.min.time())
                exit_reason = (
                    exit_decision.exit_reason.value if exit_decision.exit_reason else "Unknown"
                )

                # Handle partial vs full exit
                if exit_decision.exit_type.startswith("PARTIAL"):
                    # Extract percentage from exit_type (e.g., "PARTIAL_50%" -> 0.5)
                    try:
                        pct_str = exit_decision.exit_type.split("_")[1].replace("%", "")
                        exit_pct = float(pct_str) / 100.0
                    except (IndexError, ValueError):
                        exit_pct = 0.5  # Default to 50%

                    self.engine.partial_close_position(
                        symbol=symbol,
                        date=exit_date,
                        exit_price=current_price,
                        exit_percent=exit_pct,
                        reason=exit_reason,
                    )
                else:
                    # Full exit
                    self.engine.close_position(
                        symbol=symbol,
                        date=exit_date,
                        exit_price=current_price,
                        reason=exit_reason,
                    )

    def _check_entries_for_date(
        self,
        current_date,
        symbols: List[str],
        data_cache: dict,
        current_prices: dict,
        use_ml_signals: bool,
        market_regime: Optional[Dict] = None,
    ):
        """Check entry conditions for symbols without positions"""
        # Skip entries if market regime is not tradeable
        if market_regime and not market_regime.get("tradeable", True):
            logger.debug(
                f"Skipping entries: Market regime not tradeable ({market_regime.get('regime')})"
            )
            return

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

            # Get real ML signal instead of placeholder
            ml_signal = None
            if use_ml_signals and self.use_ml:
                ml_signal = self._get_ml_signal(df_up_to_date, symbol)

            try:
                entry_signal = self.entry_logic.analyze_entry(
                    df_up_to_date,
                    ml_signal,
                    market_regime=market_regime,  # Pass market regime to entry logic
                )

                if entry_signal.signal_type == "BUY":
                    # Get position size multiplier from entry signal
                    position_multiplier = getattr(entry_signal, "position_size_multiplier", 1.0)

                    # Adjust multiplier based on market regime
                    if market_regime:
                        regime = market_regime.get("regime", "SIDEWAYS")
                        if regime == "BULL":
                            position_multiplier = min(
                                position_multiplier * 1.2, 1.5
                            )  # Increase in bull
                        elif regime == "BEAR":
                            position_multiplier = max(
                                position_multiplier * 0.7, 0.3
                            )  # Decrease in bear

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
                            f"{entry_signal.strength.value}: {', '.join(entry_signal.reasons)}"
                        ),
                        position_size_multiplier=position_multiplier,
                    )
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

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

        # Step 1.5: Load VNINDEX for market regime detection
        if self.use_regime:
            self._load_vnindex_data(start_date, end_date)

        # Step 2: Get trading dates
        trading_dates = self._get_trading_dates(data_cache, start_date, end_date)
        logger.info(f"Simulating {len(trading_dates)} trading days...")

        # Step 3: Simulate each trading day
        for current_date in trading_dates:
            current_prices = {}

            # Update market regime for current date
            market_regime = self._update_market_regime(current_date)

            # Check if circuit breaker triggered - emergency close all
            if self.engine.circuit_breaker_triggered:
                logger.warning(f"Circuit breaker active on {current_date}")
                # Get current prices for emergency close
                for symbol in list(self.engine.positions.keys()):
                    if symbol in data_cache:
                        df = data_cache[symbol]
                        df_up_to_date = df[df["time"].dt.date <= current_date]
                        if len(df_up_to_date) > 0:
                            current_prices[symbol] = df_up_to_date["close"].iloc[-1]

                self.engine.emergency_close_all(
                    datetime.combine(current_date, datetime.min.time()),
                    current_prices,
                    "Daily loss circuit breaker",
                )
                continue  # Skip to next day

            # Check exits first (with market regime)
            self._check_exits_for_date(current_date, data_cache, current_prices, market_regime)

            # Check entries (with market regime and ML signals)
            self._check_entries_for_date(
                current_date, symbols, data_cache, current_prices, use_ml_signals, market_regime
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


def run_simple_backtest(
    symbols: List[str] = None,
    months_back: int = 12,
    use_ml: bool = True,
    use_regime: bool = True,
) -> BacktestResult:
    """
    Quick helper function to run a simple backtest

    Args:
        symbols: List of symbols (defaults to common VN stocks)
        months_back: How many months of history to test
        use_ml: Whether to use ML predictions
        use_regime: Whether to use market regime detection

    Returns:
        BacktestResult
    """
    if symbols is None:
        symbols = ["VCB", "HPG", "VHM", "VNM", "VIC", "GAS", "MSN", "MWG", "TCB", "BID"]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)

    # Use centralized transaction costs from constants
    config = BacktestConfig(
        initial_capital=100_000_000,  # 100M VND
        # commission_rate and slippage now use defaults from constants.py
        position_size_pct=0.20,
        max_positions=5,
    )

    runner = StrategyRunner(config, use_ml=use_ml, use_regime=use_regime)
    results = runner.run_backtest(
        symbols=symbols, start_date=start_date, end_date=end_date, use_ml_signals=use_ml
    )

    runner.engine.print_results(results)

    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Running 12-month backtest on VN30 stocks...")
    results = run_simple_backtest(months_back=12)

    print(f"\n✅ Backtest complete! Final return: {results.total_return_pct:+.2f}%")
