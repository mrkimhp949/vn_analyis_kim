"""
Entry Signal Service
Handles entry signal generation and validation

IMPROVEMENTS V3:
- Per-symbol circuit breaker integration
- Vietnam market session boundary check
- Price floor/ceiling validation
- Position size vs volume validation
- Refactored for better maintainability (10/10)
- Separated concerns: constants, scoring, notifications
- Improved type hints and documentation
"""

from __future__ import annotations

import asyncio
import logging
import numbers
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.config.exceptions import DataQualityError
from src.config.trading_config import get_config
from src.data.loader import load_data
from src.ml.signals.enhanced import EnhancedMLSignalGenerator
from src.portfolio.lock import get_portfolio_lock
from src.strategies.entry_logic import ImprovedEntryLogic
from src.strategies.position_sizing import EnhancedPositionSizer
from src.utils.validation import DataValidator
from utils.dataframe_utils import safe_get_latest

if TYPE_CHECKING:
    from src.notifications.telegram import TelegramNotificationService
    from src.strategies.entry_logic import EntrySignal
    from src.strategies.position_sizing import PositionSize

logger = logging.getLogger(__name__)


# =============================================================================
# OPTIONAL DEPENDENCIES (Graceful degradation)
# =============================================================================

# Import per-symbol circuit breaker
try:
    from src.risk.per_symbol_circuit_breaker import get_per_symbol_circuit_breaker

    PER_SYMBOL_CB_AVAILABLE = True
except ImportError:
    PER_SYMBOL_CB_AVAILABLE = False

# Import Vietnam market validator
try:
    from src.utils.vietnam_market import get_vietnam_market_validator
    from src.market.schedule import is_near_session_boundary

    VN_MARKET_VALIDATOR_AVAILABLE = True
except ImportError:
    VN_MARKET_VALIDATOR_AVAILABLE = False

# Import T+2 Settlement Tracker
try:
    from src.portfolio.settlement import get_settlement_tracker

    T2_SETTLEMENT_AVAILABLE = True
except ImportError:
    T2_SETTLEMENT_AVAILABLE = False


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EntryServiceConfig:
    """Configuration constants for EntrySignalService."""

    # Data validation
    MIN_DATA_ROWS: int = 50
    LOOKBACK_PERIOD: int = 200

    # Signal filtering
    DEFAULT_MAX_SIGNALS: int = 5

    # Scoring weights (must sum to 1.0)
    WEIGHT_BASE_SCORE: float = 0.50
    WEIGHT_RR_SCORE: float = 0.25
    WEIGHT_POSITION_BONUS: float = 0.10
    WEIGHT_REASONS_BONUS: float = 0.10
    WEIGHT_WARNINGS_PENALTY: float = 0.05

    # Position size scoring thresholds
    OPTIMAL_POSITION_MIN: float = 0.05  # 5%
    OPTIMAL_POSITION_MAX: float = 0.15  # 15%

    # R:R normalization
    MAX_RR_FOR_SCORING: float = 5.0

    # Reasons/warnings normalization
    MAX_REASONS_FOR_SCORING: int = 8
    MAX_WARNINGS_FOR_SCORING: int = 5


# Global config instance
SERVICE_CONFIG = EntryServiceConfig()


# =============================================================================
# DATA CLASSES FOR TYPE SAFETY
# =============================================================================


@dataclass
class ScanResult:
    """Result of scanning a single ticker."""

    symbol: str
    signal: Optional[EntrySignal] = None
    position_size: Optional[PositionSize] = None
    ml_signal: Optional[Dict[str, Any]] = None
    skip_reason: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if scan result has a valid entry signal."""
        return self.signal is not None and self.position_size is not None


@dataclass
class T2CashCheckResult:
    """Result of T+2 cash availability check."""

    sufficient: bool
    available: float
    required: float
    pending_settlements: float = 0.0
    buffer: float = 0.0
    warning: Optional[str] = None


@dataclass
class NoSignalSummary:
    """Summary of symbols without entry signals."""

    symbols: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)

    def add(self, symbol: str, reason: str) -> None:
        """Add a symbol with its skip reason."""
        self.symbols.append(symbol)
        self.reasons[symbol] = reason

    @property
    def count(self) -> int:
        """Get count of symbols without signals."""
        return len(self.symbols)

    def get_reason_counts(self) -> Dict[str, int]:
        """Group and count reasons."""
        reason_counts: Dict[str, int] = {}
        for reason in self.reasons.values():
            # Clean up reason for grouping
            clean_reason = reason.split("(")[0].strip() if "(" in reason else reason
            clean_reason = (
                clean_reason.split(":")[0].strip() if ":" in clean_reason else clean_reason
            )
            reason_counts[clean_reason] = reason_counts.get(clean_reason, 0) + 1
        return reason_counts


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def safe_to_float(val: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.

    Handles various types including Mock objects from tests.

    Args:
        val: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, numbers.Number):
            return float(val)
        return default
    except (TypeError, ValueError):
        return default


def safe_get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Safely get attribute from object.

    Args:
        obj: Object to get attribute from
        attr: Attribute name
        default: Default value if attribute doesn't exist

    Returns:
        Attribute value or default
    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


# =============================================================================
# SIGNAL SCORING CALCULATOR
# =============================================================================


class SignalScorer:
    """
    Calculator for signal quality scores.

    Separates scoring logic from main service for better testability
    and maintainability.
    """

    def __init__(self, config: EntryServiceConfig = SERVICE_CONFIG):
        self.config = config

    def calculate_score(self, signal_data: Dict[str, Any]) -> float:
        """
        Calculate composite signal score using multiple factors.

        Scoring breakdown:
        - Base score (50%): confidence * strength
        - R:R score (25%): risk/reward ratio
        - Position bonus (10%): position size quality
        - Reasons bonus (10%): number of positive reasons
        - Warnings penalty (5%): fewer warnings = better

        Args:
            signal_data: Dict containing 'signal' and optionally 'position_size'

        Returns:
            Composite score between 0.0 and 1.0
        """
        entry_signal = signal_data.get("signal")
        position_size = signal_data.get("position_size")

        if entry_signal is None:
            return 0.0

        base_score = self._calculate_base_score(entry_signal)
        rr_score = self._calculate_rr_score(entry_signal)
        position_bonus = self._calculate_position_bonus(position_size)
        reasons_bonus = self._calculate_reasons_bonus(entry_signal)
        warnings_penalty = self._calculate_warnings_penalty(entry_signal)

        return base_score + rr_score + position_bonus + reasons_bonus + warnings_penalty

    def _calculate_base_score(self, entry_signal: Any) -> float:
        """Calculate base score from confidence and strength."""
        confidence = safe_to_float(safe_get_attr(entry_signal, "confidence", 0), 0.0)
        strength = safe_get_attr(entry_signal, "strength")
        strength_value = safe_to_float(safe_get_attr(strength, "value", 0) if strength else 0, 0.0)

        # Normalize: confidence (0-100) -> 0-1, strength (1-5) -> 0-1
        normalized_confidence = confidence / 100.0
        normalized_strength = strength_value / 5.0

        return normalized_confidence * normalized_strength * self.config.WEIGHT_BASE_SCORE

    def _calculate_rr_score(self, entry_signal: Any) -> float:
        """Calculate risk/reward ratio score."""
        entry_price = safe_to_float(safe_get_attr(entry_signal, "entry_price", 0), 0.0)
        stop_loss = safe_to_float(safe_get_attr(entry_signal, "stop_loss", 0), 0.0)
        take_profits = safe_get_attr(entry_signal, "take_profit_targets")

        if not take_profits or not isinstance(take_profits, (list, tuple)):
            return 0.0

        if entry_price <= 0 or stop_loss <= 0:
            return 0.0

        risk = abs(entry_price - stop_loss)
        if risk <= 0:
            return 0.0

        # Use TP2 if available, otherwise TP1
        tp_index = 1 if len(take_profits) > 1 else 0
        tp_target = safe_to_float(take_profits[tp_index], 0.0)
        reward = abs(tp_target - entry_price)

        rr_ratio = reward / risk
        # Normalize: R:R of 5:1 or higher = max score
        normalized_rr = min(rr_ratio / self.config.MAX_RR_FOR_SCORING, 1.0)

        return normalized_rr * self.config.WEIGHT_RR_SCORE

    def _calculate_position_bonus(self, position_size: Any) -> float:
        """Calculate position size quality bonus."""
        if position_size is None:
            return 0.0

        shares = safe_get_attr(position_size, "shares", 0)
        risk_pct = safe_to_float(safe_get_attr(position_size, "risk_percent", 0), 0.0)
        position_percent = safe_to_float(safe_get_attr(position_size, "position_percent", 0), 0.0)

        try:
            shares_val = int(shares) if isinstance(shares, int) else 0
        except (TypeError, ValueError):
            shares_val = 0

        if shares_val <= 0 or risk_pct <= 0:
            return 0.0

        # Optimal position size range: 5-15%
        if self.config.OPTIMAL_POSITION_MIN <= position_percent <= self.config.OPTIMAL_POSITION_MAX:
            return self.config.WEIGHT_POSITION_BONUS
        elif position_percent > self.config.OPTIMAL_POSITION_MAX:
            # Slightly penalize oversized positions
            return self.config.WEIGHT_POSITION_BONUS * 0.5

        return 0.0

    def _calculate_reasons_bonus(self, entry_signal: Any) -> float:
        """Calculate bonus based on number of positive reasons."""
        reasons = safe_get_attr(entry_signal, "reasons", []) or []
        if not isinstance(reasons, (list, tuple)):
            reasons = []

        # Normalize: 8+ reasons = max bonus
        normalized = min(len(reasons) / self.config.MAX_REASONS_FOR_SCORING, 1.0)
        return normalized * self.config.WEIGHT_REASONS_BONUS

    def _calculate_warnings_penalty(self, entry_signal: Any) -> float:
        """Calculate penalty based on warnings (fewer = better)."""
        warnings = safe_get_attr(entry_signal, "warnings", []) or []
        if not isinstance(warnings, (list, tuple)):
            warnings = []

        # Normalize: 0 warnings = full bonus, 5+ warnings = no bonus
        penalty_factor = max(0, 1.0 - (len(warnings) / self.config.MAX_WARNINGS_FOR_SCORING))
        return penalty_factor * self.config.WEIGHT_WARNINGS_PENALTY


# =============================================================================
# NOTIFICATION HELPER
# =============================================================================


class NoSignalNotifier:
    """
    Helper class for sending no-signal notifications.

    Separates notification logic from main service.
    """

    @staticmethod
    async def send_summary(
        notification_service: TelegramNotificationService,
        total_scanned: int,
        no_signal_summary: NoSignalSummary,
    ) -> None:
        """
        Send summary notification when no signals found.

        Args:
            notification_service: Telegram notification service
            total_scanned: Total number of tickers scanned
            no_signal_summary: Summary of symbols without signals
        """
        if notification_service is None or no_signal_summary.count == 0:
            return

        try:
            message = NoSignalNotifier._build_message(total_scanned, no_signal_summary)
            await notification_service.send_message(message)
        except Exception as e:
            logger.error(f"Error sending no-signal notification: {e}")

    @staticmethod
    def _build_message(total_scanned: int, summary: NoSignalSummary) -> str:
        """Build notification message."""
        lines = [
            "🔍 *TỔNG HỢP KHÔNG TÌM THẤY TÍN HIỆU MUA*",
            f"📊 Đã quét: {total_scanned} mã cổ phiếu",
            f"📉 Không tìm thấy tín hiệu: {summary.count} mã",
            "",
            "*CHI TIẾT THEO NGUYÊN NHÂN:*",
        ]

        # Add reason counts
        reason_counts = summary.get_reason_counts()
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / summary.count) * 100
            lines.append(f"• {reason}: {count} mã ({percentage:.1f}%)")

        # Add top 3 reasons with examples
        top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_reasons:
            lines.append("")
            lines.append("*NGUYÊN NHÂN CHÍNH:*")
            for reason, count in top_reasons:
                examples = [s for s, r in summary.reasons.items() if reason in r][:2]
                examples_str = ", ".join(examples) if examples else "N/A"
                lines.append(f"• {reason}: {count} mã (VD: {examples_str})")

        # Add timestamp
        lines.append(f"\n⏰ {datetime.now().strftime('%H:%M %d/%m/%Y')}")

        return "\n".join(lines)


# =============================================================================
# MAIN SERVICE CLASS
# =============================================================================


class EntrySignalService:
    """
    Service for entry signal operations.

    Responsibilities:
    - Scan tickers for entry signals
    - Validate entry conditions
    - Calculate position sizes
    - Filter and rank signals

    Attributes:
        ml_generator: ML signal generator
        entry_logic: Entry logic analyzer
        position_sizer: Position size calculator
        portfolio_lock: Portfolio lock manager
        scorer: Signal quality scorer
        config: Service configuration
    """

    def __init__(self, config: EntryServiceConfig = SERVICE_CONFIG):
        """
        Initialize EntrySignalService.

        Args:
            config: Service configuration (uses default if not provided)
        """
        self.config = config
        self.ml_generator = EnhancedMLSignalGenerator()
        self.scorer = SignalScorer(config)

        # Load trading config
        cfg = get_config(validate=False)

        # Initialize entry logic with config values
        self.entry_logic = ImprovedEntryLogic(
            min_confidence=cfg.trading.min_confidence,
            min_risk_reward=cfg.trading.min_risk_reward,
            require_trend_alignment=True,
            require_volume_confirmation=False,  # More flexible for signals
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
        market_regime: Dict[str, Any],
        vnindex_df: Optional[pd.DataFrame] = None,
        notification_service: Optional[TelegramNotificationService] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scan tickers for entry signals.

        Args:
            tickers: List of ticker symbols to scan
            existing_symbols: Set of symbols already in portfolio
            market_regime: Market regime information
            vnindex_df: VNINDEX DataFrame for correlation
            notification_service: Notification service for sending alerts

        Returns:
            List of entry signals with structure:
            {
                'symbol': str,
                'signal': EntrySignal,
                'position_size': PositionSize,
                'ml_signal': Optional[Dict]
            }
        """
        signals: List[Dict[str, Any]] = []
        no_signal_summary = NoSignalSummary()

        # Scan in parallel
        tasks = [
            self._scan_single_ticker(symbol, existing_symbols, market_regime, vnindex_df)
            for symbol in tickers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for symbol, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.error(f"Scan error for {symbol}: {result}")
                no_signal_summary.add(symbol, f"Lỗi: {str(result)}")
                continue

            if result and result.get("signal"):
                signals.append(result)
            else:
                reason = self._extract_skip_reason(result)
                no_signal_summary.add(symbol, reason)

        # Log results
        if signals:
            logger.info(f"📊 Found {len(signals)} entry signals from {len(tickers)} tickers")

        # Send notification if no signals found
        if notification_service and not signals and no_signal_summary.count > 0:
            await NoSignalNotifier.send_summary(
                notification_service, len(tickers), no_signal_summary
            )

        return signals

    def _extract_skip_reason(self, result: Optional[Dict[str, Any]]) -> str:
        """Extract skip reason from scan result."""
        if result is None:
            return "Không rõ lý do"

        if hasattr(result, "warnings") and result.warnings:
            return ", ".join(result.warnings)

        return result.get("skip_reason", "Không rõ lý do")

    async def _scan_single_ticker(
        self,
        symbol: str,
        existing_symbols: set,
        market_regime: Dict[str, Any],
        vnindex_df: Optional[pd.DataFrame],
    ) -> Optional[Dict[str, Any]]:
        """
        Scan a single ticker for entry signal.

        Args:
            symbol: Ticker symbol
            existing_symbols: Symbols already in portfolio
            market_regime: Market regime info
            vnindex_df: VNINDEX data for correlation

        Returns:
            Signal dict if valid entry found, None otherwise
        """
        try:
            # Pre-validation checks
            skip_reason = self._pre_validate_symbol(symbol, existing_symbols)
            if skip_reason:
                return None

            # Load and validate data
            df = self._load_and_validate_data(symbol)
            if df is None:
                return None

            # ML analysis
            ml_signal = self._run_ml_analysis(symbol, df, vnindex_df)

            # Entry logic analysis
            entry_signal = self.entry_logic.analyze_entry(
                df=df, ml_signal=ml_signal, market_regime=market_regime, symbol=symbol
            )

            if not entry_signal.should_enter:
                if entry_signal.warnings:
                    logger.debug(f"[{symbol}] No entry: {', '.join(entry_signal.warnings)}")
                return None

            # Vietnam market validations
            if not self._validate_vietnam_market(symbol, df, entry_signal):
                return None

            # Calculate position size
            position_size = self._calculate_position(symbol, entry_signal, market_regime)
            if position_size is None or position_size.shares == 0:
                return None

            # T+2 cash check
            if not self._check_t2_cash(symbol, position_size.value):
                return None

            # Volume constraint check
            position_size = self._apply_volume_constraint(symbol, df, position_size, entry_signal)
            if position_size is None:
                return None

            # Final validation
            if entry_signal.entry_price <= 0 or entry_signal.stop_loss <= 0:
                logger.warning(
                    f"[{symbol}] Invalid prices: entry={entry_signal.entry_price:.0f}, "
                    f"stop_loss={entry_signal.stop_loss:.0f}"
                )
                return None

            # Mark as pending (atomic operation)
            self.portfolio_lock.add_pending(symbol, position_size.value)

            return {
                "symbol": symbol,
                "signal": entry_signal,
                "position_size": position_size,
                "ml_signal": ml_signal,
            }

        except Exception:
            logger.error(f"[{symbol}] Error scanning", exc_info=True)
            return None

    def _pre_validate_symbol(self, symbol: str, existing_symbols: set) -> Optional[str]:
        """
        Pre-validate symbol before scanning.

        Returns skip reason if should skip, None if OK to proceed.
        """
        # Skip if already in portfolio
        if symbol in existing_symbols:
            return "Already in portfolio"

        # Skip if pending
        if self.portfolio_lock.is_pending(symbol):
            logger.debug(f"[{symbol}] Đang pending, skip")
            return "Pending"

        # Per-symbol circuit breaker check
        if PER_SYMBOL_CB_AVAILABLE:
            per_symbol_cb = get_per_symbol_circuit_breaker()
            can_trade, reason = per_symbol_cb.can_trade(symbol)
            if not can_trade:
                logger.info(f"[{symbol}] Blocked by per-symbol circuit breaker: {reason}")
                return f"Circuit breaker: {reason}"

        # Vietnam market session boundary check
        if VN_MARKET_VALIDATOR_AVAILABLE:
            is_near_boundary, boundary_type = is_near_session_boundary()
            if is_near_boundary:
                logger.debug(f"[{symbol}] Skipping near session boundary ({boundary_type})")
                return f"Near session boundary: {boundary_type}"

        return None

    def _load_and_validate_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load and validate data for symbol."""
        df = load_data(symbol, lookback=self.config.LOOKBACK_PERIOD)

        try:
            DataValidator.validate_dataframe(df, min_rows=self.config.MIN_DATA_ROWS)
            return df
        except DataQualityError:
            logger.debug(f"[{symbol}] Data validation failed")
            return None

    def _run_ml_analysis(
        self, symbol: str, df: pd.DataFrame, vnindex_df: Optional[pd.DataFrame]
    ) -> Optional[Dict[str, Any]]:
        """Run ML analysis with error handling."""
        try:
            ml_signal = self.ml_generator.analyze(df, vnindex_df)
            if ml_signal is None:
                logger.debug(f"[{symbol}] ML analysis returned None")
            return ml_signal
        except Exception as e:
            logger.warning(f"⚠️ ML analysis error for {symbol}: {type(e).__name__}: {e}")
            return None

    def _validate_vietnam_market(self, symbol: str, df: pd.DataFrame, entry_signal: Any) -> bool:
        """Validate Vietnam market specific rules."""
        if not VN_MARKET_VALIDATOR_AVAILABLE or len(df) < 2:
            return True

        validator = get_vietnam_market_validator()
        current_price = safe_get_latest(df, "close", 0)
        reference_price = df["close"].iloc[-2]

        is_safe, warning = validator.check_price_floor_ceiling(
            current_price, reference_price, symbol
        )
        if not is_safe:
            logger.info(f"[{symbol}] Skipping: {warning}")
            return False

        return True

    def _calculate_position(
        self, symbol: str, entry_signal: Any, market_regime: Dict[str, Any]
    ) -> Optional[Any]:
        """Calculate position size."""
        position_size = self.position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=entry_signal.entry_price,
            stop_loss=entry_signal.stop_loss,
            take_profit=entry_signal.take_profit_targets[0],
            confidence=entry_signal.confidence,
            signal_strength=entry_signal.strength.name,
            market_regime=market_regime,
        )

        if position_size.shares == 0:
            warnings_str = (
                ", ".join(position_size.warnings) if position_size.warnings else "Unknown"
            )
            logger.debug(f"[{symbol}] Position size = 0: {warnings_str}")
            return None

        return position_size

    def _check_t2_cash(self, symbol: str, position_value: float) -> bool:
        """Check T+2 cash availability."""
        if not (T2_SETTLEMENT_AVAILABLE and VN_MARKET_VALIDATOR_AVAILABLE):
            return True

        try:
            result = self._check_t2_cash_availability(symbol, position_value)
            if not result["sufficient"]:
                logger.info(
                    f"[{symbol}] Insufficient cash after T+2: "
                    f"Required={result['required']:,.0f}, Available={result['available']:,.0f}"
                )
                return False
            if result.get("warning"):
                logger.warning(f"[{symbol}] T+2 Warning: {result['warning']}")
            return True
        except Exception as e:
            logger.warning(f"[{symbol}] T+2 check failed: {e}, proceeding anyway")
            return True

    def _apply_volume_constraint(
        self, symbol: str, df: pd.DataFrame, position_size: Any, entry_signal: Any
    ) -> Optional[Any]:
        """Apply volume constraint to position size."""
        if not VN_MARKET_VALIDATOR_AVAILABLE or "volume" not in df.columns:
            return position_size

        validator = get_vietnam_market_validator()
        avg_volume = df["volume"].tail(20).mean()

        is_safe, warning = validator.validate_position_size_vs_volume(
            position_size.shares, avg_volume, symbol
        )

        if is_safe:
            return position_size

        # Reduce position size instead of skipping
        max_shares = int(avg_volume * validator.max_position_pct_of_volume)
        from src.config.constants import VIETNAM_LOT_SIZE

        max_shares = (max_shares // VIETNAM_LOT_SIZE) * VIETNAM_LOT_SIZE

        if max_shares > 0:
            logger.warning(
                f"[{symbol}] Reducing position from {position_size.shares} "
                f"to {max_shares} shares due to volume constraint"
            )
            position_size.shares = max_shares
            position_size.value = max_shares * entry_signal.entry_price
            position_size.warnings.append(warning)
            return position_size

        logger.info(f"[{symbol}] Skipping: {warning}")
        return None

    def filter_and_rank_signals(
        self, signals: List[Dict[str, Any]], max_signals: int = None
    ) -> List[Dict[str, Any]]:
        """
        Filter and rank signals by quality.

        Args:
            signals: List of entry signals
            max_signals: Maximum signals to return (default: 5)

        Returns:
            Filtered and ranked signals (best first)
        """
        if not signals:
            return []

        if max_signals is None:
            max_signals = self.config.DEFAULT_MAX_SIGNALS

        # Sort by composite score (descending)
        sorted_signals = sorted(
            signals, key=lambda sig: self.scorer.calculate_score(sig), reverse=True
        )

        top_signals = sorted_signals[:max_signals]

        # Log ranking details
        if top_signals:
            top = top_signals[0]
            score = self.scorer.calculate_score(top)
            logger.info(
                f"📊 Ranked {len(signals)} signals to top {len(top_signals)}. "
                f"Top: {top['symbol']} (score={score:.3f}, "
                f"conf={top['signal'].confidence}%, "
                f"strength={top['signal'].strength.name})"
            )

        return top_signals

    def _check_t2_cash_availability(self, symbol: str, position_value: float) -> Dict[str, Any]:
        """
        Check T+2 cash availability before entry.

        Vietnam market uses T+2 settlement:
        - Day T: Trade executed
        - Day T+2: Cash settlement

        Must ensure sufficient cash for:
        1. New trade value
        2. Pending T+2 settlement obligations
        3. Buffer for safety (10%)

        Args:
            symbol: Stock symbol
            position_value: Value of new position

        Returns:
            Dict with sufficient, available, required, warning
        """
        try:
            config = get_config(validate=False)
            settlement_tracker = get_settlement_tracker()
            vn_validator = get_vietnam_market_validator()

            total_capital = config.trading.total_capital

            # Get current portfolio value
            used_capital = self._get_used_capital()

            # Get pending settlements
            settlement_summary = settlement_tracker.get_settlement_summary()
            pending_stock_value = settlement_summary.get("pending_stock_value", 0)

            # Calculate available cash
            gross_available = total_capital - used_capital - pending_stock_value

            # Calculate T+2 requirement
            total_t2_required, buffer = vn_validator.calculate_t2_cash_requirement(
                pending_settlements={},
                new_trade_value=position_value,
            )

            total_required = total_t2_required + buffer
            is_sufficient = gross_available >= total_required

            # Warning if close to limit
            warning = None
            if is_sufficient and gross_available < total_required * 1.2:
                warning = (
                    f"Cash utilization high: {gross_available:,.0f} available, "
                    f"{total_required:,.0f} required (including T+2 buffer)"
                )

            logger.debug(
                f"[{symbol}] T+2 Check: Available={gross_available:,.0f}, "
                f"Required={total_required:,.0f}, Sufficient={is_sufficient}"
            )

            return {
                "sufficient": is_sufficient,
                "available": gross_available,
                "required": total_required,
                "pending_settlements": pending_stock_value,
                "buffer": buffer,
                "warning": warning,
            }

        except Exception as e:
            logger.warning(f"[{symbol}] T+2 cash check error: {e}")
            return {
                "sufficient": True,
                "available": 0,
                "required": position_value,
                "warning": f"T+2 check failed: {e}",
            }

    def _get_used_capital(self) -> float:
        """
        Get currently used capital from portfolio.

        Uses lazy import to avoid circular dependencies.
        """
        try:
            # Lazy import to avoid circular dependency
            from src.portfolio.manager import PortfolioManager

            pm = PortfolioManager()
            positions = pm.get_positions()
            return sum(pos.get("shares", 0) * pos.get("avg_price", 0) for pos in positions.values())
        except Exception:
            return 0.0


# =============================================================================
# SINGLETON PATTERN
# =============================================================================

_entry_service: Optional[EntrySignalService] = None


def get_entry_service() -> EntrySignalService:
    """
    Get entry service singleton.

    Returns:
        EntrySignalService instance
    """
    global _entry_service
    if _entry_service is None:
        _entry_service = EntrySignalService()
    return _entry_service


def reset_entry_service() -> None:
    """
    Reset entry service singleton.

    Useful for testing or reconfiguration.
    """
    global _entry_service
    _entry_service = None
