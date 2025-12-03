"""
Exit Management Service
Handles exit signal checking and execution
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from src.config.exceptions import DataQualityError
from src.data.loader import load_data
from src.ml.signals.enhanced import EnhancedMLSignalGenerator
from src.portfolio.manager import get_portfolio_manager
from src.portfolio.paper_trading import get_paper_account
from src.strategies.exit_logic import ImprovedExitStrategy
from src.utils.validation import DataValidator
from utils.dataframe_utils import safe_get_latest

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from src.services.risk_service import RiskManagementService

logger = logging.getLogger(__name__)

# Constants
DEFAULT_LOOKBACK_DAYS = 200
MIN_DATA_ROWS = 20


class ExitManagementService:
    """
    Service for exit management operations

    Responsibilities:
    - Check active positions for exit signals
    - Execute exits (full or partial)
    - Track exit performance
    - Update position tracking
    """

    def __init__(self):
        self.exit_strategy = ImprovedExitStrategy()
        self.ml_generator = EnhancedMLSignalGenerator()
        self.portfolio_manager = get_portfolio_manager()
        self.paper_account = get_paper_account()
        self._risk_service: Optional["RiskManagementService"] = None

        logger.info("✅ Exit Management Service initialized")

    def _get_risk_service(self) -> "RiskManagementService":
        """Lazy load risk service to avoid circular import"""
        if self._risk_service is None:
            from src.services.risk_service import get_risk_service

            self._risk_service = get_risk_service()
        return self._risk_service

    async def check_all_positions(
        self, market_regime: Dict, vnindex_df: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """
        Check all active positions for exit signals

        Args:
            market_regime: Market regime information
            vnindex_df: VNINDEX DataFrame

        Returns:
            List of exit decisions
        """
        positions = self.portfolio_manager.get_positions()

        if not positions:
            logger.info("📊 No active positions to check")
            return []

        logger.info(f"📊 Checking {len(positions)} active positions for exits")

        # Check positions in parallel
        tasks = [
            self._check_single_position(symbol, pos_data, market_regime, vnindex_df)
            for symbol, pos_data in positions.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect exit decisions
        exits = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ Exit check error: {result}")
                continue

            if result and result.get("should_exit"):
                exits.append(result)

        logger.info(f"📊 Found {len(exits)} exit signals")

        return exits

    async def _check_single_position(
        self,
        symbol: str,
        pos_data: Dict,
        market_regime: Dict,
        vnindex_df: Optional[pd.DataFrame],
    ) -> Optional[Dict]:
        """Check a single position for exit signal"""
        try:
            # Load data
            df = load_data(symbol, lookback=DEFAULT_LOOKBACK_DAYS)

            # Validate data
            try:
                DataValidator.validate_dataframe(df, min_rows=MIN_DATA_ROWS)
            except DataQualityError:
                logger.warning(f"⚠️ [{symbol}] Data validation failed")
                return None

            # Get current price safely
            current_price = safe_get_latest(df, "close", 0)

            if current_price <= 0:
                logger.warning(f"⚠️ [{symbol}] Invalid current price: {current_price}")
                return None

            # ML signal với error handling
            ml_signal = None
            try:
                ml_signal = self.ml_generator.analyze(df, vnindex_df)
                if ml_signal is None:
                    logger.debug(f"📊 [{symbol}] ML analysis returned None")
            except Exception as e:
                logger.warning(f"⚠️ [{symbol}] ML analysis error: {type(e).__name__}: {e}")

            # Check exit
            exit_decision = self.exit_strategy.check_exit(
                symbol=symbol,
                entry_price=pos_data["avg_price"],
                current_price=current_price,
                stop_loss=pos_data.get("stop_loss"),
                take_profit_targets=pos_data.get("take_profit_targets", []),
                entry_date=datetime.fromisoformat(pos_data["entry_date"]),
                df=df,
                ml_signal=ml_signal,
                market_regime=market_regime,
                partial_exits=pos_data.get("partial_exits", []),
            )

            if exit_decision.should_exit:
                return {
                    "symbol": symbol,
                    "should_exit": True,
                    "decision": exit_decision,
                    "position": pos_data,
                    "current_price": current_price,
                }

            return None

        except Exception:
            logger.error(f"❌ [{symbol}] Error checking exit", exc_info=True)
            return None

    def _calculate_pnl(
        self, entry_price: float, current_price: float, shares: int
    ) -> Tuple[float, float]:
        """
        Calculate P&L for a position

        Args:
            entry_price: Entry price per share
            current_price: Current price per share
            shares: Number of shares

        Returns:
            Tuple of (absolute_pnl, pnl_percentage)
        """
        pnl = (current_price - entry_price) * shares
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        return pnl, pnl_pct

    async def execute_exit(
        self, symbol: str, exit_decision: Dict, current_price: float, max_retries: int = 3
    ) -> bool:
        """
        Execute an exit (full or partial) with retry mechanism

        Args:
            symbol: Stock symbol
            exit_decision: Exit decision from check
            current_price: Current price
            max_retries: Maximum retry attempts

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                result = await self._execute_exit_internal(symbol, exit_decision, current_price)
                if result:
                    return True

                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ [{symbol}] Exit attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

            except Exception as e:
                logger.error(f"❌ [{symbol}] Exit attempt {attempt + 1} error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        logger.error(f"❌ [{symbol}] All {max_retries} exit attempts failed")
        return False

    async def _execute_exit_internal(
        self, symbol: str, exit_decision: Dict, current_price: float
    ) -> bool:
        """Internal exit execution logic"""
        decision = exit_decision["decision"]
        pos_data = exit_decision["position"]

        # Validate entry price
        if pos_data["avg_price"] <= 0:
            logger.error(
                f"❌ [{symbol}] Invalid entry price: {pos_data['avg_price']}. "
                "Cannot calculate P&L."
            )
            return False

        # Calculate P&L
        pnl, pnl_pct = self._calculate_pnl(
            entry_price=pos_data["avg_price"],
            current_price=current_price,
            shares=pos_data["shares"],
        )

        # Execute paper trade
        success, message, _ = self.paper_account.execute_sell(
            symbol=symbol,
            price=current_price,
            exit_type=decision.exit_type,
            reason=decision.exit_reason.value,
        )

        if success:
            logger.info(f"✅ [{symbol}] Exit executed: {message}")

            # Record P&L for circuit breaker tracking
            risk_service = self._get_risk_service()
            risk_service.record_trade(pnl_pct)
            logger.debug(f"📊 [{symbol}] Recorded exit P&L: {pnl_pct:.2f}%")

            # Clear position tracking if full exit
            if decision.exit_type == "FULL":
                self.exit_strategy.clear_position_tracking(symbol)

            return True
        else:
            logger.error(f"❌ [{symbol}] Exit failed: {message}")
            return False


# Thread-safe singleton
_exit_service: Optional[ExitManagementService] = None
_lock = threading.Lock()


def get_exit_service() -> ExitManagementService:
    """Get exit service singleton (thread-safe)"""
    global _exit_service
    if _exit_service is None:
        with _lock:
            if _exit_service is None:
                _exit_service = ExitManagementService()
    return _exit_service
