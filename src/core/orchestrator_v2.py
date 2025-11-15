"""
Trading Orchestrator V2 - Refactored with Services
Simplified orchestrator using service-oriented architecture
"""

import logging
from typing import Dict, Optional
import pandas as pd
from telegram import Bot

from services import (
    RiskManagementService,
    EntrySignalService,
    ExitManagementService,
    NotificationService,
    get_risk_service,
    get_entry_service,
    get_exit_service,
    get_notification_service,
)
from portfolio_manager import get_portfolio_manager
from ticker_loader import get_ticker_loader
from market_regime_proxy import ProxyMarketRegimeAnalyzer
from data_loader import load_data
from config import LOOKBACK, MAX_SCAN_UNIVERSE

logger = logging.getLogger(__name__)


class TradingOrchestratorV2:
    """
    Simplified Trading Orchestrator using services
    
    Responsibilities:
    - Coordinate services
    - Manage scan workflow
    - Handle errors gracefully
    """
    
    def __init__(
        self,
        bot_instance: Bot,
        chat_id: str,
        vnindex_df: Optional[pd.DataFrame] = None
    ):
        self.bot = bot_instance
        self.chat_id = chat_id
        self.vnindex_df = vnindex_df
        
        # Initialize services
        self.risk_service = get_risk_service()
        self.entry_service = get_entry_service()
        self.exit_service = get_exit_service()
        self.notification_service = get_notification_service(bot_instance, chat_id)
        
        # Other dependencies
        self.portfolio_manager = get_portfolio_manager()
        self.ticker_loader = get_ticker_loader()
        self.market_analyzer = ProxyMarketRegimeAnalyzer()
        
        logger.info("✅ Trading Orchestrator V2 initialized")
    
    async def run_scan(self, market_regime: Dict) -> None:
        """
        Main scan workflow
        
        Args:
            market_regime: Market regime information
        """
        try:
            # Step 1: Pre-flight checks
            if not await self._preflight_checks(market_regime):
                return
            
            # Step 2: Get scan universe
            tickers = self._get_scan_universe()
            existing_symbols = set(self.portfolio_manager.get_positions().keys())
            
            # Step 3: Send scan start notification
            await self.notification_service.send_scan_start(
                ticker_count=len(tickers),
                market_regime=market_regime
            )
            
            # Step 4: Check exits first
            exit_count = await self._check_and_execute_exits(market_regime)
            
            # Step 5: Scan for new entries
            signal_count = await self._scan_and_execute_entries(
                tickers, existing_symbols, market_regime
            )
            
            # Step 6: Send summary
            await self._send_scan_summary(
                signal_count, exit_count, market_regime
            )
            
            logger.info("✅ Scan completed successfully")
        
        except Exception as e:
            logger.error(f"❌ Error in scan workflow: {e}", exc_info=True)
            await self.notification_service.send_risk_alert(
                alert_type="SCAN_ERROR",
                message=f"Scan failed: {str(e)}"
            )
    
    async def _preflight_checks(self, market_regime: Dict) -> bool:
        """
        Pre-flight checks before scanning
        
        Returns:
            True if checks pass, False otherwise
        """
        # Check 1: Load VNINDEX if not provided
        if self.vnindex_df is None or self.vnindex_df.empty:
            try:
                self.vnindex_df = load_data("VNINDEX", lookback=LOOKBACK, is_index=True)
            except Exception as e:
                logger.warning(f"Could not load VNINDEX: {e}")
        
        # Check 2: Circuit breaker
        vnindex_change = 0.0
        if self.vnindex_df is not None and not self.vnindex_df.empty:
            vnindex_change = self.vnindex_df['close'].pct_change().iloc[-1]
        
        portfolio_pnl = self.portfolio_manager.get_daily_pnl_pct()
        
        tripped = await self.risk_service.check_and_update_circuit_breaker(
            portfolio_pnl_pct=portfolio_pnl,
            vnindex_change_pct=vnindex_change
        )
        
        if tripped:
            reason = self.risk_service.circuit_breaker.tripped_reason
            logger.critical(f"🚨 Circuit breaker tripped: {reason}")
            
            await self.notification_service.send_risk_alert(
                alert_type="CIRCUIT_BREAKER",
                message=reason
            )
            
            # Still check exits, but no new entries
            await self._check_and_execute_exits(market_regime)
            return False
        
        # Check 3: Can trade?
        can_trade, reason = await self.risk_service.can_trade()
        if not can_trade:
            logger.warning(f"Trading not allowed: {reason}")
            await self.notification_service.send_risk_alert(
                alert_type="TRADING_BLOCKED",
                message=reason
            )
            return False
        
        return True
    
    def _get_scan_universe(self) -> list:
        """Get list of tickers to scan"""
        try:
            return self.ticker_loader.get_validated_tickers(
                force_validate=False,
                min_volume=100_000,
                max_tickers=MAX_SCAN_UNIVERSE
            )
        except Exception as e:
            logger.error(f"Error getting tickers: {e}")
            from config import TICKERS
            return TICKERS[:MAX_SCAN_UNIVERSE]
    
    async def _check_and_execute_exits(self, market_regime: Dict) -> int:
        """
        Check and execute exits
        
        Returns:
            Number of exits executed
        """
        try:
            # Check all positions
            exits = await self.exit_service.check_all_positions(
                market_regime=market_regime,
                vnindex_df=self.vnindex_df
            )
            
            if not exits:
                return 0
            
            # Execute exits
            executed_count = 0
            for exit_data in exits:
                # Send notification
                await self.notification_service.send_exit_signal(exit_data)
                
                # Execute
                success = await self.exit_service.execute_exit(
                    symbol=exit_data['symbol'],
                    exit_decision=exit_data,
                    current_price=exit_data['current_price']
                )
                
                if success:
                    executed_count += 1
                    
                    # Record for circuit breaker
                    pos_data = exit_data['position']
                    pnl = (
                        exit_data['current_price'] - pos_data['avg_price']
                    ) * pos_data['shares']
                    self.risk_service.record_trade(pnl)
            
            logger.info(f"✅ Executed {executed_count}/{len(exits)} exits")
            return executed_count
        
        except Exception as e:
            logger.error(f"Error checking exits: {e}", exc_info=True)
            return 0
    
    async def _scan_and_execute_entries(
        self,
        tickers: list,
        existing_symbols: set,
        market_regime: Dict
    ) -> int:
        """
        Scan for and execute entry signals
        
        Returns:
            Number of signals found
        """
        try:
            # Scan for signals
            signals = await self.entry_service.scan_for_entries(
                tickers=tickers,
                existing_symbols=existing_symbols,
                market_regime=market_regime,
                vnindex_df=self.vnindex_df
            )
            
            if not signals:
                return 0
            
            # Filter and rank
            top_signals = self.entry_service.filter_and_rank_signals(
                signals, max_signals=5
            )
            
            # Send notifications
            for signal_data in top_signals:
                await self.notification_service.send_entry_signal(signal_data)
            
            logger.info(f"✅ Found {len(top_signals)} entry signals")
            return len(top_signals)
        
        except Exception as e:
            logger.error(f"Error scanning entries: {e}", exc_info=True)
            return 0
    
    async def _send_scan_summary(
        self,
        signal_count: int,
        exit_count: int,
        market_regime: Dict
    ) -> None:
        """Send scan summary notification"""
        try:
            portfolio_summary = self.portfolio_manager.get_detailed_analysis()
            
            await self.notification_service.send_scan_summary(
                signal_count=signal_count,
                exit_count=exit_count,
                market_regime=market_regime,
                portfolio_summary=portfolio_summary
            )
        except Exception as e:
            logger.error(f"Error sending summary: {e}")


# For backward compatibility
def create_orchestrator_v2(
    bot_instance: Bot,
    chat_id: str,
    vnindex_df: Optional[pd.DataFrame] = None
) -> TradingOrchestratorV2:
    """Create orchestrator V2 instance"""
    return TradingOrchestratorV2(bot_instance, chat_id, vnindex_df)
