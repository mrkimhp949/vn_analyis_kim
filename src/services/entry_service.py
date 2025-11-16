"""
Entry Signal Service
Handles entry signal generation and validation
"""

import asyncio
import logging
from typing import Dict, List, Optional

import pandas as pd

from src.config.exceptions import DataQualityError
from src.config.trading_config import get_config
from src.data.loader import load_data
from src.ml.signals.enhanced import EnhancedMLSignalGenerator
from src.portfolio.lock import get_portfolio_lock
from src.strategies.entry_logic import ImprovedEntryLogic
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
        # Align entry threshold with centralized config
        cfg = get_config(validate=False)
        # ENHANCED: Allow more flexible entry requirements
        self.entry_logic = ImprovedEntryLogic(
            min_confidence=cfg.trading.min_confidence,
            min_risk_reward=cfg.trading.min_risk_reward,
            require_trend_alignment=True,  # Still require but can adjust
            require_volume_confirmation=False,  # Make volume optional for more signals
        )
        self.position_sizer = EnhancedPositionSizer()
        self.portfolio_lock = get_portfolio_lock()

        logger.info(
            f"✅ Entry Signal Service initialized "
            f"(min_conf={cfg.trading.min_confidence}%, "
            f"R:R>={cfg.trading.min_risk_reward}, "
            f"volume_req=False)"
        )

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
            self._scan_single_ticker(symbol, existing_symbols, market_regime, vnindex_df)
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
            # Skip if already in portfolio or pending (align with tests)
            if symbol in existing_symbols:
                return None

            # Skip only if pending (being processed)
            if self.portfolio_lock.is_pending(symbol):
                logger.debug(f"[{symbol}] Đang pending, skip")
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
                if ml_signal is None:
                    logger.debug(f"[{symbol}] ML analysis returned None")
            except Exception as e:
                logger.warning(f"⚠️ Lỗi ML analysis cho {symbol}: {type(e).__name__}: {str(e)}")
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
                logger.debug(
                    f"[{symbol}] Position size = 0, skipping. "
                    f"Reason: {', '.join(position_size.warnings) if position_size.warnings else 'Unknown'}"
                )
                return None

            # Validate entry price and stop loss
            if entry_signal.entry_price <= 0 or entry_signal.stop_loss <= 0:
                logger.warning(
                    f"[{symbol}] Invalid prices: entry={entry_signal.entry_price:.0f}, "
                    f"stop_loss={entry_signal.stop_loss:.0f}"
                )
                return None

            # Mark as pending (pass position value for exposure tracking)
            position_value = position_size.value if position_size else 0.0
            self.portfolio_lock.add_pending(symbol, position_value)

            return {
                "symbol": symbol,
                "signal": entry_signal,
                "position_size": position_size,
                "ml_signal": ml_signal,
            }

        except Exception:
            logger.error(f"[{symbol}] Error scanning", exc_info=True)
            return None

    def filter_and_rank_signals(self, signals: List[Dict], max_signals: int = 5) -> List[Dict]:
        """
        Filter and rank signals by quality using multiple factors

        Ranking factors:
        1. Confidence score (0-100)
        2. Signal strength (1-5)
        3. Risk/reward ratio (from position size)
        4. Number of warnings (negative factor)
        5. Number of positive reasons (positive factor)

        Args:
            signals: List of entry signals
            max_signals: Maximum signals to return

        Returns:
            Filtered and ranked signals
        """
        if not signals:
            return []

        def signal_score(sig):
            """Calculate composite signal score using multiple factors"""
            entry_signal = sig["signal"]
            position_size = sig.get("position_size", None)

            def to_float(val, default=0.0):
                try:
                    if isinstance(val, (int, float)):
                        return float(val)
                    # Avoid coercing unittest.mock objects
                    import numbers

                    if isinstance(val, numbers.Number):
                        return float(val)
                    return default
                except Exception:
                    return default

            # Base score: confidence * strength (weighted 50%)
            confidence = to_float(getattr(entry_signal, "confidence", 0), 0.0)
            strength = getattr(entry_signal, "strength", None)
            strength_value = to_float(
                getattr(strength, "value", 0) if strength is not None else 0, 0.0
            )
            base_score = (confidence / 100.0) * (strength_value / 5.0) * 0.5

            # Risk/reward bonus (weighted 25%)
            # Calculate R:R from entry signal (entry price, stop loss, take profit)
            rr_score = 0.0
            raw_entry = getattr(entry_signal, "entry_price", 0)
            entry_price = to_float(raw_entry, 0.0)
            raw_sl = getattr(entry_signal, "stop_loss", 0)
            stop_loss = to_float(raw_sl, 0.0)
            take_profits = getattr(entry_signal, "take_profit_targets", None)
            if (
                take_profits
                and isinstance(take_profits, (list, tuple))
                and entry_price > 0
                and stop_loss > 0
            ):
                risk = abs(entry_price - stop_loss)
                if risk > 0:
                    # Use TP2 (index 1) as target, or TP1 if TP2 not available
                    tp_raw = take_profits[1] if len(take_profits) > 1 else take_profits[0]
                    tp_target = to_float(tp_raw, 0.0)
                    reward = abs(tp_target - entry_price)
                    rr_ratio = reward / risk if risk > 0 else 0
                    # Normalize R:R to 0-0.25 score (2:1 = 0.1, 5:1 = 0.25)
                    rr_score = min(rr_ratio / 5.0, 1.0) * 0.25

            # Position size quality bonus (weighted 10%)
            # Larger positions with valid risk indicate stronger conviction
            position_bonus = 0.0
            if position_size:
                shares_ps = getattr(position_size, "shares", 0)
                risk_pct_ps = getattr(position_size, "risk_percent", 0)
                position_percent = to_float(getattr(position_size, "position_percent", 0), 0.0)
                try:
                    shares_val = int(shares_ps) if isinstance(shares_ps, (int,)) else 0
                except Exception:
                    shares_val = 0
                risk_pct_val = to_float(risk_pct_ps, 0.0)
                if shares_val > 0 and risk_pct_val > 0:
                    # Score based on position size being meaningful but not excessive
                    if 0.05 <= position_percent <= 0.15:  # 5-15% range
                        position_bonus = 0.10
                    elif position_percent > 0.15:
                        position_bonus = 0.05  # Slightly penalize oversized positions

            # Reasons bonus (weighted 10%)
            reasons = getattr(entry_signal, "reasons", []) or []
            if not isinstance(reasons, (list, tuple)):
                reasons = []
            reasons_bonus = min(len(reasons) / 8.0, 1.0) * 0.10

            # Warnings penalty (weighted 5%)
            warnings = getattr(entry_signal, "warnings", []) or []
            if not isinstance(warnings, (list, tuple)):
                warnings = []
            warnings_penalty = max(0, 1.0 - (len(warnings) / 5.0)) * 0.05

            # Combine scores
            total_score = base_score + rr_score + position_bonus + reasons_bonus + warnings_penalty

            return total_score

        # Sort by composite score
        sorted_signals = sorted(signals, key=signal_score, reverse=True)

        # Take top N
        top_signals = sorted_signals[:max_signals]

        # Log ranking details
        if top_signals:
            logger.info(
                f"📊 Ranked {len(signals)} signals to top {len(top_signals)}. "
                f"Top signal: {top_signals[0]['symbol']} "
                f"(score: {signal_score(top_signals[0]):.3f}, "
                f"confidence: {top_signals[0]['signal'].confidence}%, "
                f"strength: {top_signals[0]['signal'].strength.name})"
            )

        return top_signals


# Singleton
_entry_service = None


def get_entry_service() -> EntrySignalService:
    """Get entry service singleton"""
    global _entry_service
    if _entry_service is None:
        _entry_service = EntrySignalService()
    return _entry_service
