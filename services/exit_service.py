"""
Exit Management Service
Handles exit signal checking and execution
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from src.data.loader import load_data
from src.config.exceptions import DataQualityError
from src.strategies.exit_logic import ImprovedExitStrategy
from src.ml.signals.enhanced import EnhancedMLSignalGenerator
from src.portfolio.paper_trading import get_paper_account
from src.portfolio.manager import get_portfolio_manager

from utils.validation import DataValidator

logger = logging.getLogger(__name__)


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

        logger.info("✅ Exit Management Service initialized")

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
                logger.error(f"Exit check error: {result}")
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
            df = load_data(symbol, lookback=200)

            # Validate data
            try:
                DataValidator.validate_dataframe(df, min_rows=20)
            except DataQualityError:
                logger.warning(f"[{symbol}] Data validation failed")
                return None

            # Get current price
            current_price = df.iloc[-1]["close"]

            # ML signal
            ml_signal = self.ml_generator.analyze(df, vnindex_df)

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
                    "decision": exit_decision,
                    "position": pos_data,
                    "current_price": current_price,
                }

            return None

        except Exception:
            logger.error(f"[{symbol}] Error checking exit", exc_info=True)
            return None

    async def execute_exit(self, symbol: str, exit_decision: Dict, current_price: float) -> bool:
        """
        Execute an exit (full or partial)

        Args:
            symbol: Stock symbol
            exit_decision: Exit decision from check
            current_price: Current price

        Returns:
            True if successful, False otherwise
        """
        try:
            decision = exit_decision["decision"]
            pos_data = exit_decision["position"]

            # Calculate P&L for circuit breaker
            (current_price - pos_data["avg_price"]) * pos_data["shares"]

            # Execute paper trade
            success, message, _ = self.paper_account.execute_sell(
                symbol=symbol,
                price=current_price,
                exit_type=decision.exit_type,
                reason=decision.exit_reason.value,
            )

            if success:
                logger.info(f"✅ Exit executed: {symbol} - {message}")

                # Clear position tracking if full exit
                if decision.exit_type == "FULL":
                    self.exit_strategy.clear_position_tracking(symbol)

                return True
            else:
                logger.error(f"❌ Exit failed: {symbol} - {message}")
                return False

        except Exception:
            logger.error(f"❌ Error executing exit for {symbol}", exc_info=True)
            return False


# Singleton
_exit_service = None


def get_exit_service() -> ExitManagementService:
    """Get exit service singleton"""
    global _exit_service
    if _exit_service is None:
        _exit_service = ExitManagementService()
    return _exit_service
