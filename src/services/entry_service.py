"""
Entry Signal Service
Handles entry signal generation and validation
"""

import asyncio
import logging
from typing import Dict, List, Optional

import pandas as pd
from src.data.loader import load_data
from src.config.exceptions import DataQualityError
from src.strategies.entry_logic import ImprovedEntryLogic
from src.ml.signals.enhanced import EnhancedMLSignalGenerator
from src.portfolio.lock import get_portfolio_lock
from src.strategies.position_sizing import EnhancedPositionSizer

from src.utils.validation import DataValidator
from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

logger = logging.getLogger(__name__)


class EntrySignalService:
    """
    Service for entry signal operations

    Responsibilities:
    - Scan tickers for entry signals
    - Validate entry conditions
    - Calculate position sizes
    - Filter and rank signals
    """

    def __init__(self):
        self.ml_generator = EnhancedMLSignalGenerator()
        self.entry_logic = ImprovedEntryLogic()
        self.position_sizer = EnhancedPositionSizer()
        self.portfolio_lock = get_portfolio_lock()

        logger.info("✅ Entry Signal Service initialized")

    async def scan_for_entries(
        self,
        tickers: List[str],
        existing_symbols: set,
        market_regime: Dict,
        vnindex_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict]:
        """
        Scan tickers for entry signals

        Args:
            tickers: List of ticker symbols to scan
            existing_symbols: Set of symbols already in portfolio
            market_regime: Market regime information
            vnindex_df: VNINDEX DataFrame for correlation

        Returns:
            List of entry signals
        """
        signals = []

        # Scan in parallel
        tasks = [
            self._scan_single_ticker(
                symbol, existing_symbols, market_regime, vnindex_df
            )
            for symbol in tickers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect valid signals
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Scan error: {result}")
                continue

            if result and result.get("signal"):
                signals.append(result)

        logger.info(f"📊 Found {len(signals)} entry signals from {len(tickers)} tickers")

        return signals

    async def _scan_single_ticker(
        self,
        symbol: str,
        existing_symbols: set,
        market_regime: Dict,
        vnindex_df: Optional[pd.DataFrame],
    ) -> Optional[Dict]:
        """Scan a single ticker for entry signal"""
        try:
            # Skip if already in portfolio or pending
            if symbol in existing_symbols or self.portfolio_lock.is_pending(symbol):
                return None

            # Load data
            df = load_data(symbol, lookback=200)

            # Validate data
            try:
                DataValidator.validate_dataframe(df, min_rows=50)
            except DataQualityError:
                logger.debug(f"[{symbol}] Data validation failed")
                return None

            # ML analysis với error handling
            ml_signal = None
            try:
                ml_signal = self.ml_generator.analyze(df, vnindex_df)
            except Exception:
                logger.warning(f"⚠️ Lỗi ML analysis cho {symbol}")
                # Tiếp tục với ml_signal = None

            # Entry logic
            entry_signal = self.entry_logic.analyze_entry(
                df=df, ml_signal=ml_signal, market_regime=market_regime
            )

            # Check if should enter
            if not entry_signal.should_enter:
                return None

            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                symbol=symbol,
                entry_price=entry_signal.entry_price,
                stop_loss=entry_signal.stop_loss,
                take_profit=entry_signal.take_profit_targets[0],
                confidence=entry_signal.confidence,
                signal_strength=entry_signal.strength.name,
                market_regime=market_regime,
            )

            # Check if position size valid
            if position_size.shares == 0:
                logger.debug(f"[{symbol}] Position size = 0, skipping")
                return None

            # Mark as pending
            self.portfolio_lock.add_pending(symbol)

            return {
                "symbol": symbol,
                "signal": entry_signal,
                "position_size": position_size,
                "ml_signal": ml_signal,
            }

        except Exception:
            logger.error(f"[{symbol}] Error scanning", exc_info=True)
            return None

    def filter_and_rank_signals(
        self, signals: List[Dict], max_signals: int = 5
    ) -> List[Dict]:
        """
        Filter and rank signals by quality

        Args:
            signals: List of entry signals
            max_signals: Maximum signals to return

        Returns:
            Filtered and ranked signals
        """
        if not signals:
            return []

        # Sort by confidence * strength
        def signal_score(sig):
            confidence = sig["signal"].confidence
            strength = sig["signal"].strength.value
            return confidence * strength

        sorted_signals = sorted(signals, key=signal_score, reverse=True)

        # Take top N
        top_signals = sorted_signals[:max_signals]

        logger.info(f"📊 Filtered {len(signals)} signals to top {len(top_signals)}")

        return top_signals


# Singleton
_entry_service = None


def get_entry_service() -> EntrySignalService:
    """Get entry service singleton"""
    global _entry_service
    if _entry_service is None:
        _entry_service = EntrySignalService()
    return _entry_service
