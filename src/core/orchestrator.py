# -*- coding: utf-8 -*-
"""
Trading Orchestrator - Core coordination module for the trading bot.

This module orchestrates the entire trading workflow including:
- Market scanning for entry opportunities
- Position management and exit monitoring
- ML signal generation with circuit breaker protection
- Risk management integration
- Telegram notifications

Architecture:
    - Supports both legacy mode (bot_instance, chat_id) and modern DI mode
    - Uses atomic operations for position management to prevent race conditions
    - Implements ML circuit breaker for automatic fallback to technical analysis

Author: Trading Bot Team
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from telegram import Bot

from src.config.exceptions import DataLoadError
from src.config.trading_config import get_config
from src.data.loader import load_data
from src.data.ticker_loader import get_ticker_loader
from src.market.regime_proxy import ProxyMarketRegimeAnalyzer
from src.ml.monitor import get_ml_model_monitor
from src.ml.signals.enhanced import EnhancedMLSignalGenerator
from src.monitoring.signal_performance import get_signal_performance_tracker
from src.portfolio.lock import get_portfolio_lock
from src.portfolio.manager import get_portfolio_manager
from src.portfolio.paper_trading import get_paper_account
from src.portfolio.risk_manager import get_portfolio_risk_manager
from src.risk.circuit_breaker import get_circuit_breaker
from src.strategies.manager import get_strategy_manager

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


class MarketRegime(TypedDict, total=False):
    """Type definition for market regime data."""

    regime: str
    confidence: int
    tradeable: bool
    volatility: Optional[float]
    trend: Optional[str]


class PendingExitData(TypedDict):
    """Type definition for pending exit data."""

    pos_data: Dict[str, Any]
    exit_decision: Any
    current_price: float
    timestamp: str


class ScanResult(TypedDict, total=False):
    """Type definition for scan result."""

    symbol: str
    signal: bool
    skipped_buy: bool
    warnings: List[str]
    is_watchlist: bool


class MLFailureReason(Enum):
    """Categorized ML failure reasons for monitoring."""

    DATA_QUALITY = "data_quality"
    VNINDEX_LOAD_FAIL = "vnindex_load_fail"
    MODEL_ERROR = "model_error"
    FEATURE_ERROR = "feature_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass
class MLCircuitBreakerConfig:
    """Configuration for ML circuit breaker behavior."""

    # Thresholds
    failure_threshold: float = 0.05  # Disable ML at 5% failure rate
    recovery_threshold: float = 0.02  # Re-enable at 2% failure rate
    min_samples: int = 50  # Minimum attempts before activating

    # Strict mode settings (when circuit breaker active)
    strict_confidence_threshold: int = 70  # Higher confidence required
    strict_size_multiplier: float = 0.5  # Reduce position size by 50%

    @classmethod
    def from_env(cls) -> "MLCircuitBreakerConfig":
        """Load config from environment variables."""
        return cls(
            failure_threshold=float(os.getenv("ML_CB_FAILURE_THRESHOLD", 0.05)),
            recovery_threshold=float(os.getenv("ML_CB_RECOVERY_THRESHOLD", 0.02)),
            min_samples=int(os.getenv("ML_CB_MIN_SAMPLES", 50)),
            strict_confidence_threshold=int(os.getenv("ML_CB_STRICT_CONFIDENCE", 70)),
            strict_size_multiplier=float(os.getenv("ML_CB_STRICT_SIZE_MULT", 0.5)),
        )


@dataclass
class VNIndexCacheConfig:
    """Configuration for VNINDEX caching."""

    ttl_seconds: int = 3600  # Cache for 1 hour
    min_rows: int = 50  # Minimum rows required

    @classmethod
    def from_env(cls) -> "VNIndexCacheConfig":
        """Load config from environment variables."""
        return cls(
            ttl_seconds=int(os.getenv("VNINDEX_CACHE_TTL", 3600)),
            min_rows=int(os.getenv("VNINDEX_MIN_ROWS", 50)),
        )


@dataclass
class MLAnalysisConfig:
    """Configuration for ML analysis behavior."""

    max_retries: int = 2
    retry_delay_base: float = 0.5  # Base delay in seconds
    timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "MLAnalysisConfig":
        """Load config from environment variables."""
        return cls(
            max_retries=int(os.getenv("ML_MAX_RETRIES", 2)),
            retry_delay_base=float(os.getenv("ML_RETRY_DELAY", 0.5)),
            timeout_seconds=float(os.getenv("ML_TIMEOUT", 10.0)),
        )


@dataclass
class MLMetrics:
    """Metrics tracking for ML analysis performance."""

    failure_count: int = 0
    success_count: int = 0
    failures_by_error: Dict[str, int] = field(default_factory=dict)
    failures_by_symbol: Dict[str, int] = field(default_factory=dict)
    failure_reasons: Dict[str, int] = field(
        default_factory=lambda: {reason.value: 0 for reason in MLFailureReason}
    )

    @property
    def total_attempts(self) -> int:
        """Total ML analysis attempts."""
        return self.failure_count + self.success_count

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.failure_count / self.total_attempts

    def record_success(self) -> None:
        """Record a successful ML analysis."""
        self.success_count += 1

    def record_failure(
        self,
        symbol: str,
        error_type: str,
        error_msg: str,
    ) -> None:
        """Record an ML analysis failure with categorization."""
        self.failure_count += 1

        # Track by error type
        self.failures_by_error[error_type] = self.failures_by_error.get(error_type, 0) + 1

        # Track by symbol
        self.failures_by_symbol[symbol] = self.failures_by_symbol.get(symbol, 0) + 1

        # Categorize failure reason
        reason = self._categorize_failure(error_type, error_msg)
        self.failure_reasons[reason.value] += 1

    def _categorize_failure(self, error_type: str, error_msg: str) -> MLFailureReason:
        """Categorize failure for root cause analysis."""
        error_msg_lower = error_msg.lower()
        error_type_lower = error_type.lower()

        if "timeout" in error_type_lower or "timeout" in error_msg_lower:
            return MLFailureReason.TIMEOUT
        if "data_quality" in error_type_lower or any(
            kw in error_msg_lower for kw in ["insufficient", "missing", "invalid"]
        ):
            return MLFailureReason.DATA_QUALITY
        if "vnindex" in error_msg_lower:
            return MLFailureReason.VNINDEX_LOAD_FAIL
        if any(kw in error_msg_lower for kw in ["model", "prediction"]):
            return MLFailureReason.MODEL_ERROR
        if "feature" in error_msg_lower:
            return MLFailureReason.FEATURE_ERROR
        return MLFailureReason.UNKNOWN

    def get_summary(self) -> str:
        """Get formatted summary of ML metrics."""
        top_errors = sorted(self.failures_by_error.items(), key=lambda x: x[1], reverse=True)[:3]
        top_symbols = sorted(self.failures_by_symbol.items(), key=lambda x: x[1], reverse=True)[:3]

        return (
            f"📊 ML Metrics Summary:\n"
            f"   Total: {self.total_attempts} (✅{self.success_count} / ❌{self.failure_count})\n"
            f"   Failure rate: {self.failure_rate:.1%}\n"
            f"   Top errors: {', '.join(f'{err}({cnt})' for err, cnt in top_errors)}\n"
            f"   Top failing symbols: {', '.join(f'{sym}({cnt})' for sym, cnt in top_symbols)}"
        )


# =============================================================================
# HELPER CLASSES
# =============================================================================


class TelegramMessageFormatter:
    """Helper class for formatting Telegram messages with proper escaping."""

    @staticmethod
    def escape_html(text: Any) -> str:
        """Escape HTML special characters for Telegram."""
        if text is None:
            return ""
        result = str(text)
        return result.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def escape_markdown(text: Any) -> str:
        """Escape Markdown special characters for Telegram."""
        if text is None:
            return ""
        result = str(text)
        special_chars = [
            "_",
            "*",
            "[",
            "]",
            "(",
            ")",
            "~",
            "`",
            ">",
            "#",
            "+",
            "-",
            "=",
            "|",
            "{",
            "}",
            ".",
            "!",
        ]
        for char in special_chars:
            result = result.replace(char, f"\\{char}")
        return result

    @classmethod
    def format_entry_recommendation(
        cls,
        symbol: str,
        entry_signal: Any,
        position: Any,
        market_regime: MarketRegime,
        news_context: Optional[Dict] = None,
    ) -> str:
        """Format entry recommendation message in HTML format."""
        safe_symbol = cls.escape_html(symbol)

        msg = f"🎯 <b>TÍN HIỆU VÀO LỆNH - {safe_symbol}</b>\n\n"

        if market_regime:
            regime = cls.escape_html(str(market_regime.get("regime", "N/A")))
            msg += f"📊 <b>Market:</b> {regime} ({market_regime.get('confidence', 0)}%)\n\n"

        msg += f"💪 <b>Signal:</b> {entry_signal.strength.name}\n"
        msg += f"🎲 <b>Confidence:</b> {entry_signal.confidence}%\n"
        msg += f"📈 <b>Shares:</b> {position.shares:,} ({position.shares // 100} lô)\n"
        msg += f"💰 <b>Value:</b> {position.value:,.0f} VNĐ ({position.position_percent:.1f}%)\n\n"

        if position.recommended_entries:
            msg += "💵 <b>GIÁ VÀO (DCA):</b>\n"
            for entry in position.recommended_entries[:2]:
                msg += f"  L{entry['level']}: {entry['price']:,.0f} - "
                msg += f"{entry['shares']:,} CP ({entry['percent']}%)\n"
            msg += "\n"

        msg += f"🛑 <b>Stop Loss:</b> {entry_signal.stop_loss:,.0f} VNĐ "
        sl_pct = (
            (entry_signal.stop_loss - entry_signal.entry_price) / entry_signal.entry_price * 100
        )
        msg += f"({sl_pct:+.1f}%)\n\n"

        msg += "🎯 <b>Take Profit:</b>\n"
        for i, tp in enumerate(entry_signal.take_profit_targets[:2], 1):
            tp_pct = ((tp - entry_signal.entry_price) / entry_signal.entry_price) * 100
            msg += f"  TP{i}: {tp:,.0f} (+{tp_pct:.1f}%)\n"

        if entry_signal.reasons:
            msg += "\n✅ <b>Lý do:</b>\n"
            for reason in entry_signal.reasons[:2]:
                safe_reason = cls.escape_html(str(reason))
                msg += f"• {safe_reason}\n"

        if entry_signal.warnings:
            safe_warning = cls.escape_html(str(entry_signal.warnings[0]))
            msg += f"\n⚠️ <b>Cảnh báo:</b> {safe_warning}\n"

        msg += f"\n💸 <b>Risk:</b> {position.max_loss:,.0f} VNĐ ({position.risk_percent:.2f}%)"

        if news_context and news_context.get("articles"):
            sentiment_label = cls.escape_html(str(news_context.get("sentiment_label", "N/A")))
            msg += f"\n\n📰 <b>News Sentiment:</b> {sentiment_label} ({news_context.get('sentiment_score', 0):+.2f})\n"
            for article in news_context.get("top_headlines", [])[:2]:
                published = article.get("published_at", "")[:16].replace("T", " ")
                title = cls.escape_html(str(article.get("title", "")))
                source = cls.escape_html(str(article.get("source", "")))
                msg += f"  • {title} ({source}, {published})\n"
                if article.get("url"):
                    msg += f"    {article['url']}\n"

        return msg

    @classmethod
    def format_exit_recommendation(
        cls,
        symbol: str,
        pos_data: Dict[str, Any],
        exit_decision: Any,
        current_price: float,
    ) -> str:
        """Format exit recommendation message in HTML format."""
        entry_price = pos_data.get("avg_price", 0)
        shares = pos_data.get("shares", 0)
        pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        pnl_amount = (current_price - entry_price) * shares

        pnl_emoji = "🟢" if pnl_percent >= 0 else "🔴"

        exit_reason = exit_decision.exit_reason.value if exit_decision.exit_reason else "Unknown"
        safe_reason = cls.escape_html(exit_reason)
        safe_message = cls.escape_html(exit_decision.message)
        safe_symbol = cls.escape_html(symbol)

        msg = f"📊 <b>KHUYẾN NGHỊ THOÁT LỆNH - {safe_symbol}</b>\n\n"
        msg += "⚠️ <b>Cần xác nhận của bạn để thực hiện</b>\n\n"

        msg += "📈 <b>Thông tin vị thế:</b>\n"
        msg += f"  • Số lượng: {shares:,} CP\n"
        msg += f"  • Giá vào: {entry_price:,.0f} VNĐ\n"
        msg += f"  • Giá hiện tại: {current_price:,.0f} VNĐ\n"
        msg += f"  • {pnl_emoji} P&amp;L: {pnl_percent:+.2f}% ({pnl_amount:+,.0f} VNĐ)\n\n"

        msg += f"🎯 <b>Lý do thoát:</b> {safe_reason}\n"
        msg += f"📝 <b>Chi tiết:</b> {safe_message}\n"
        msg += f"⚡ <b>Mức độ khẩn cấp:</b> {exit_decision.urgency}/5\n\n"

        msg += "💡 <b>Hành động:</b>\n"
        msg += f"  • Gửi <code>/sell {safe_symbol}</code> để xác nhận bán\n"
        msg += f"  • Gửi <code>/hold {safe_symbol}</code> để giữ lại\n"
        msg += f"  • Gửi <code>/sell {safe_symbol} 50%</code> để bán một phần\n\n"

        msg += f"⏰ Khuyến nghị lúc: {datetime.now().strftime('%H:%M %d/%m/%Y')}"

        return msg

    @classmethod
    def format_buy_signal(
        cls,
        symbol: str,
        entry_signal: Any,
        position_size_info: Any,
    ) -> str:
        """Format buy signal notification message."""
        # Calculate R:R ratio
        try:
            if (
                entry_signal.take_profit_targets
                and entry_signal.entry_price != entry_signal.stop_loss
            ):
                risk_reward_ratio = (
                    entry_signal.take_profit_targets[0] - entry_signal.entry_price
                ) / (entry_signal.entry_price - entry_signal.stop_loss)
            else:
                risk_reward_ratio = 0
        except (ZeroDivisionError, IndexError):
            risk_reward_ratio = 0

        tp1_target = entry_signal.take_profit_targets[0] if entry_signal.take_profit_targets else 0

        confidence_emoji = (
            "🟢"
            if entry_signal.confidence >= 70
            else "🟡" if entry_signal.confidence >= 50 else "🔴"
        )

        return (
            "**🚀 TÍN HIỆU MUA MỚI 🚀**\n\n"
            f"**Mã:** `{symbol}`\n"
            f"**Độ tin cậy:** `{entry_signal.confidence}%` {confidence_emoji}\n"
            f"**Giá vào:** `{entry_signal.entry_price:,.0f}`\n"
            f"**Mục tiêu 1:** `{tp1_target:,.0f}`\n"
            f"**Dừng lỗ:** `{entry_signal.stop_loss:,.0f}`\n"
            f"**R:R (TP1):** `{risk_reward_ratio:.2f}`\n\n"
            f"**Lý do:** {', '.join(entry_signal.reasons)}\n\n"
            "**--- Quản lý vốn ---**\n"
            f"**Số CP mua:** `{position_size_info.shares}`\n"
            f"**Giá trị lệnh:** `{position_size_info.value:,.0f} VNĐ`\n"
            f"**Rủi ro lệnh:** `{position_size_info.risk_amount:,.0f} VNĐ` "
            f"({position_size_info.risk_percent:.2%})\n\n"
        )


class VNIndexCache:
    """Cache manager for VNINDEX data with TTL support."""

    def __init__(self, config: VNIndexCacheConfig, lookback: int = 200):
        self._config = config
        self._lookback = lookback
        self._cached_df: Optional[pd.DataFrame] = None
        self._cache_timestamp: Optional[float] = None
        self._logger = logging.getLogger(__name__)

    def get(self) -> Optional[pd.DataFrame]:
        """
        Get cached VNINDEX data or load fresh if cache expired.

        Returns:
            VNINDEX DataFrame or None if load fails
        """
        current_time = time.time()

        # Check if cache is still valid
        if self._is_cache_valid(current_time):
            cache_age = current_time - self._cache_timestamp
            self._logger.debug(f"✅ Using cached VNINDEX (age: {cache_age:.0f}s)")
            return self._cached_df

        # Load fresh VNINDEX
        return self._load_fresh()

    def _is_cache_valid(self, current_time: float) -> bool:
        """Check if cache is still valid."""
        if self._cached_df is None or self._cache_timestamp is None:
            return False
        cache_age = current_time - self._cache_timestamp
        return cache_age < self._config.ttl_seconds

    def _load_fresh(self) -> Optional[pd.DataFrame]:
        """Load fresh VNINDEX data."""
        try:
            self._logger.info("📊 Loading VNINDEX for ML analysis...")
            index_df = load_data("VNINDEX", lookback=self._lookback, is_index=True)

            if self._validate_data(index_df):
                self._cached_df = index_df
                self._cache_timestamp = time.time()
                self._logger.info(f"✅ VNINDEX loaded successfully ({len(index_df)} rows)")
                return index_df

            self._logger.warning("⚠️ VNINDEX data is empty or insufficient")
            return self._get_stale_fallback()

        except Exception as e:
            self._logger.error(f"❌ Failed to load VNINDEX: {e}")
            return self._get_stale_fallback()

    def _validate_data(self, df: Optional[pd.DataFrame]) -> bool:
        """Validate VNINDEX data."""
        return df is not None and not df.empty and len(df) >= self._config.min_rows

    def _get_stale_fallback(self) -> Optional[pd.DataFrame]:
        """Return stale cache as fallback if available."""
        if self._cached_df is not None:
            self._logger.warning("⚠️ Using stale VNINDEX cache as fallback")
            return self._cached_df
        return None

    def invalidate(self) -> None:
        """Invalidate the cache."""
        self._cached_df = None
        self._cache_timestamp = None


# =============================================================================
# MAIN ORCHESTRATOR CLASS
# =============================================================================


class TradingOrchestrator:
    """
    Core orchestrator for the trading bot.

    Manages the entire trading workflow including:
    - Market scanning for entry opportunities
    - Position management and exit monitoring
    - ML signal generation with circuit breaker protection
    - Risk management integration
    - Telegram notifications

    Supports both legacy mode (bot_instance, chat_id) and modern DI mode.

    Example:
        # Legacy mode
        orchestrator = TradingOrchestrator(bot, chat_id)

        # Modern DI mode (via factory)
        orchestrator = create_orchestrator()

        # Run scan
        await orchestrator.run_scan(market_regime)
    """

    def __init__(
        self,
        bot_instance: Optional[Bot] = None,
        chat_id: Optional[str] = None,
        vnindex_df: Optional[pd.DataFrame] = None,
        # Dependency injection parameters
        config: Optional[Any] = None,
        data_loader: Optional[Any] = None,
        ml_generator: Optional[Any] = None,
        strategy_manager: Optional[Any] = None,
        portfolio_manager: Optional[Any] = None,
        risk_service: Optional[Any] = None,
        entry_service: Optional[Any] = None,
        exit_service: Optional[Any] = None,
        notification_service: Optional[Any] = None,
        circuit_breaker: Optional[Any] = None,
        paper_account: Optional[Any] = None,
    ):
        """
        Initialize TradingOrchestrator.

        Args:
            bot_instance: Telegram bot instance (legacy mode)
            chat_id: Telegram chat ID (legacy mode)
            vnindex_df: Pre-loaded VNINDEX data (optional)
            config: Trading configuration
            data_loader: Data loader service
            ml_generator: ML signal generator
            strategy_manager: Strategy manager
            portfolio_manager: Portfolio manager
            risk_service: Risk management service
            entry_service: Entry logic service
            exit_service: Exit logic service
            notification_service: Notification service
            circuit_breaker: Circuit breaker instance
            paper_account: Paper trading account
        """
        self._logger = logging.getLogger(__name__)

        # Telegram setup
        self.bot = bot_instance
        self.chat_id = chat_id
        self.vnindex_df = vnindex_df

        # Load configurations
        self._trading_config = config or get_config(validate=False)
        self._ml_cb_config = MLCircuitBreakerConfig.from_env()
        self._vnindex_cache_config = VNIndexCacheConfig.from_env()
        self._ml_analysis_config = MLAnalysisConfig.from_env()

        # Get lookback from config
        self._lookback = self._trading_config.data.lookback

        # Initialize core services
        self._init_services(
            data_loader=data_loader,
            ml_generator=ml_generator,
            strategy_manager=strategy_manager,
            portfolio_manager=portfolio_manager,
            circuit_breaker=circuit_breaker,
            paper_account=paper_account,
        )

        # Initialize optional services
        self.risk_service = risk_service
        self.entry_service = entry_service
        self.exit_service = exit_service
        self.notification_service = notification_service

        # Strategy components (initialized via _setup_strategies)
        self.entry_logic: Optional[Any] = None
        self.position_sizer: Optional[Any] = None
        self.exit_strategy: Optional[Any] = None

        # Initialize ML tracking
        self._ml_metrics = MLMetrics()
        self._ml_enabled = True
        self._ml_circuit_breaker_active = False

        # Initialize VNINDEX cache
        self._vnindex_cache = VNIndexCache(self._vnindex_cache_config, self._lookback)

        # Initialize pending exits tracking
        self._pending_exits: Dict[str, PendingExitData] = {}
        self._pending_exits_ttl = 3600  # 1 hour

        # Message formatter
        self._formatter = TelegramMessageFormatter()

    def _init_services(
        self,
        data_loader: Optional[Any],
        ml_generator: Optional[Any],
        strategy_manager: Optional[Any],
        portfolio_manager: Optional[Any],
        circuit_breaker: Optional[Any],
        paper_account: Optional[Any],
    ) -> None:
        """Initialize core services with dependency injection or defaults."""
        self.data_loader = data_loader
        self.portfolio_manager = portfolio_manager or get_portfolio_manager()
        self.portfolio_risk_manager = get_portfolio_risk_manager(total_capital=100_000_000)
        self.portfolio_lock = get_portfolio_lock()
        self.market_analyzer = ProxyMarketRegimeAnalyzer()
        self.ticker_loader = get_ticker_loader()
        self.ml_generator = ml_generator or EnhancedMLSignalGenerator()
        self.paper_account = paper_account or get_paper_account()
        self.ml_monitor = get_ml_model_monitor()
        self.strategy_manager = strategy_manager or get_strategy_manager()
        self.circuit_breaker = circuit_breaker or get_circuit_breaker()
        self.signal_tracker = get_signal_performance_tracker()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def run_scan(self, market_regime: MarketRegime) -> None:
        """
        Run comprehensive scan for trading opportunities.

        This is the main entry point for the scanning workflow:
        1. Validate inputs and check circuit breaker
        2. Setup strategies based on market regime
        3. Check active positions for exit signals
        4. Scan for new entry opportunities
        5. Send summary report

        Args:
            market_regime: Current market regime data
        """
        # Validate and normalize market regime
        market_regime = self._validate_market_regime(market_regime)

        # Setup strategies based on market regime
        self._setup_strategies(market_regime)

        # Get current state
        active_positions = self.portfolio_manager.get_positions()
        existing_symbols = set(active_positions.keys())
        self._sync_position_sizer(active_positions)

        # Get scan universe
        current_tickers = self._get_scan_universe()
        self._logger.info(f"🔍 Quét {len(current_tickers)} mã...")

        # Send scan start notification
        await self._send_scan_start_message(current_tickers, market_regime)

        # Check active positions for exits
        await self._check_active_positions(market_regime)

        # Scan for new entries
        self._logger.info(f"\n🔍 Quét {len(current_tickers)} mã để tìm cơ hội mua mới")
        self.portfolio_lock.clear_pending()

        signal_count, watchlist_candidates = await self._scan_for_new_entries(
            current_tickers, existing_symbols, market_regime
        )

        # Send summary report
        await self._send_summary_report(signal_count, watchlist_candidates, market_regime)

        # Cleanup stale pending exits
        self._cleanup_stale_pending_exits()

    async def confirm_exit(self, symbol: str, percent: float = 100.0) -> bool:
        """
        Confirm and execute exit after user confirmation.

        Args:
            symbol: Stock symbol
            percent: Percentage to sell (default 100%)

        Returns:
            True if successful
        """
        if symbol not in self._pending_exits:
            await self._send_message(f"❌ Không tìm thấy khuyến nghị thoát lệnh cho {symbol}")
            return False

        pending = self._pending_exits[symbol]

        # Update exit type if partial
        if percent < 100:
            pending["exit_decision"].exit_type = f"PARTIAL_{int(percent)}%"

        # Execute the exit
        await self._execute_exit(
            symbol,
            pending["pos_data"],
            pending["exit_decision"],
            pending["current_price"],
        )

        # Remove from pending
        del self._pending_exits[symbol]
        return True

    async def cancel_exit(self, symbol: str) -> bool:
        """
        Cancel exit recommendation (user chose to hold).

        Args:
            symbol: Stock symbol

        Returns:
            True if successful
        """
        if symbol not in self._pending_exits:
            await self._send_message(f"❌ Không tìm thấy khuyến nghị thoát lệnh cho {symbol}")
            return False

        del self._pending_exits[symbol]

        await self._send_message(
            f"✅ Đã hủy khuyến nghị thoát lệnh cho {symbol}. Tiếp tục giữ vị thế.",
            parse_mode="Markdown",
        )

        self._logger.info(f"🔄 User chọn HOLD cho {symbol}, hủy khuyến nghị thoát lệnh")
        return True

    def get_pending_exits(self) -> Dict[str, PendingExitData]:
        """Get list of pending exit recommendations."""
        return self._pending_exits.copy()

    def get_ml_metrics(self) -> MLMetrics:
        """Get ML analysis metrics."""
        return self._ml_metrics

    def is_ml_circuit_breaker_active(self) -> bool:
        """Check if ML circuit breaker is active."""
        return self._ml_circuit_breaker_active

    # =========================================================================
    # MARKET REGIME & STRATEGY SETUP
    # =========================================================================

    def _validate_market_regime(self, market_regime: Optional[MarketRegime]) -> MarketRegime:
        """Validate and normalize market regime data."""
        if not market_regime or not isinstance(market_regime, dict):
            self._logger.error("❌ Invalid market_regime provided to run_scan")
            market_regime = {}

        # Ensure required keys with defaults
        return {
            "regime": market_regime.get("regime", "SIDEWAYS"),
            "confidence": market_regime.get("confidence", 50),
            "tradeable": market_regime.get("tradeable", True),
            "volatility": market_regime.get("volatility"),
            "trend": market_regime.get("trend"),
        }

    def _setup_strategies(self, market_regime: MarketRegime) -> None:
        """Setup and adjust strategies based on market regime."""
        strategies = self.strategy_manager.get_strategies()
        self.entry_logic = strategies["entry_logic"]
        self.position_sizer = strategies["position_sizer"]
        self.exit_strategy = strategies["exit_strategy"]

        self.strategy_manager.apply_market_adjustments(market_regime)

        self._logger.info(
            f"🔧 Đã thiết lập và điều chỉnh chiến lược cho chế độ: {market_regime.get('regime', 'Sideways')}"
        )

    # =========================================================================
    # SCAN UNIVERSE & POSITION SYNC
    # =========================================================================

    def _get_scan_universe(self) -> List[str]:
        """Get list of tickers to scan."""
        try:
            return self.ticker_loader.get_validated_tickers(
                force_validate=True,
                min_volume=self._trading_config.data.min_volume,
                max_tickers=1000,
            )
        except Exception:
            self._logger.error("Lỗi khi lấy danh sách ticker", exc_info=True)
            # Fallback to legacy config
            from src.config.legacy_config import TICKERS, MAX_SCAN_UNIVERSE

            return TICKERS[:MAX_SCAN_UNIVERSE]

    def _sync_position_sizer(self, active_positions: Dict[str, Any]) -> None:
        """Sync position sizer with active positions."""
        if not self.position_sizer:
            return

        self.position_sizer.current_positions = {}
        for symbol, pos in active_positions.items():
            if pos.get("shares", 0) > 0:
                entry_price = pos.get("avg_price", 0)
                self.position_sizer.current_positions[symbol] = {
                    "shares": pos.get("shares", 0),
                    "entry_price": entry_price,
                    "current_price": pos.get("metadata", {}).get("last_price", entry_price),
                    "unrealized_pnl": 0,
                }

    # =========================================================================
    # POSITION CHECKING & EXIT LOGIC
    # =========================================================================

    async def _check_active_positions(self, market_regime: MarketRegime) -> None:
        """Check and process active positions for exit signals."""
        positions = self.portfolio_manager.get_positions()
        if not positions:
            return

        self._logger.info(f"\n📊 Kiểm tra {len(positions)} vị thế đang nắm giữ...")

        tasks = [
            self._check_single_position(symbol, pos_data, market_regime)
            for symbol, pos_data in positions.items()
        ]
        await asyncio.gather(*tasks)

    async def _check_single_position(
        self,
        symbol: str,
        pos_data: Dict[str, Any],
        market_regime: MarketRegime,
    ) -> None:
        """Check a single position for exit signals."""
        try:
            df = load_data(symbol, lookback=self._lookback)
            if df.empty:
                return

            current_price = df.iloc[-1]["close"]

            # Update current price in portfolio
            try:
                self.portfolio_manager.update_position_price(symbol, float(current_price))
            except Exception:
                self._logger.debug(f"Không thể cập nhật last_price cho {symbol}")

            # ML analysis for exit decision
            ml_signal = await self._get_ml_signal(df, symbol, context="exit_check")

            # Check exit conditions
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

            if exit_decision and exit_decision.should_exit:
                await self._handle_exit_decision(symbol, pos_data, exit_decision, current_price)

        except Exception:
            self._logger.error(f"Lỗi khi kiểm tra vị thế {symbol}", exc_info=True)

    async def _handle_exit_decision(
        self,
        symbol: str,
        pos_data: Dict[str, Any],
        exit_decision: Any,
        current_price: float,
    ) -> None:
        """Handle exit decision based on configuration."""
        auto_sell = os.getenv("AUTO_SELL", "false").lower() == "true"
        auto_sell_stop_loss = os.getenv("AUTO_SELL_STOP_LOSS", "true").lower() == "true"

        from src.strategies.exit_logic import ExitReason

        is_stop_loss = exit_decision.exit_reason == ExitReason.STOP_LOSS
        is_emergency = exit_decision.exit_reason in [
            ExitReason.MARKET_CRASH,
            ExitReason.EMERGENCY_EXIT,
        ]

        should_auto_sell = auto_sell or (is_stop_loss and auto_sell_stop_loss) or is_emergency

        if should_auto_sell:
            await self._execute_exit(symbol, pos_data, exit_decision, current_price)
        else:
            await self._send_exit_recommendation(symbol, pos_data, exit_decision, current_price)

    async def _execute_exit(
        self,
        symbol: str,
        pos_data: Dict[str, Any],
        exit_decision: Any,
        current_price: float,
    ) -> None:
        """Execute exit order."""
        try:
            # Send exit notification
            msg = self.exit_strategy.format_exit_message(symbol, exit_decision, use_html=True)
            await self._send_message(msg, parse_mode="HTML")

            # Record trade result
            pnl = (current_price - pos_data["avg_price"]) * pos_data["shares"]
            self.circuit_breaker.record_trade(pnl)

            # Get exit reason
            exit_reason_str = (
                exit_decision.exit_reason.value
                if exit_decision.exit_reason
                else exit_decision.message
            )

            # Execute sell
            success, sell_msg, _ = self.paper_account.execute_sell(
                symbol=symbol,
                price=current_price,
                exit_type=exit_decision.exit_type,
                reason=exit_reason_str,
            )

            if success:
                self._logger.info(f"✅ Giao dịch bán được thực thi: {sell_msg}")
                await self._post_exit_cleanup(symbol)
            else:
                self._logger.error(f"❌ Lỗi thực thi lệnh bán cho {symbol}: {sell_msg}")
                await self._send_message(f"❌ Lỗi bán {symbol}: {sell_msg}")

        except Exception:
            self._logger.error(f"Lỗi khi thực hiện thoát lệnh {symbol}", exc_info=True)

    async def _post_exit_cleanup(self, symbol: str) -> None:
        """Cleanup after successful exit."""
        # Check if position still exists
        updated_positions = self.portfolio_manager.get_positions()
        position_still_exists = (
            symbol in updated_positions and updated_positions[symbol].get("shares", 0) > 0
        )

        if not position_still_exists:
            self.exit_strategy.clear_position_tracking(symbol)
            self._logger.debug(f"🧹 Cleared tracking for {symbol} (position fully closed)")

        # Record PnL and check circuit breaker
        current_pnl = self.portfolio_manager.get_daily_pnl_pct()
        self.circuit_breaker.record_pnl(current_pnl)

        if self.circuit_breaker.is_active():
            self._logger.warning(
                f"⚠️ Circuit breaker đã kích hoạt sau khi thoát {symbol}. PnL: {current_pnl:.2%}"
            )
            await self._send_message(
                "🚨 *CIRCUIT BREAKER KÍCH HOẠT*\n\n"
                f"Sau khi thoát {symbol}\n"
                f"PnL hiện tại: {current_pnl:.2%}\n"
                f"Lý do: {self.circuit_breaker.tripped_reason}",
                parse_mode="Markdown",
            )

    async def _send_exit_recommendation(
        self,
        symbol: str,
        pos_data: Dict[str, Any],
        exit_decision: Any,
        current_price: float,
    ) -> None:
        """Send exit recommendation and wait for user confirmation."""
        try:
            msg = self._formatter.format_exit_recommendation(
                symbol, pos_data, exit_decision, current_price
            )
            await self._send_message(msg, parse_mode="HTML")

            self._logger.info(
                f"📤 Đã gửi khuyến nghị THOÁT LỆNH cho {symbol}, chờ xác nhận từ user"
            )

            # Store pending exit
            self._pending_exits[symbol] = {
                "pos_data": pos_data,
                "exit_decision": exit_decision,
                "current_price": current_price,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception:
            self._logger.error(f"Lỗi khi gửi khuyến nghị thoát lệnh {symbol}", exc_info=True)

    # =========================================================================
    # ENTRY SCANNING
    # =========================================================================

    async def _scan_for_new_entries(
        self,
        current_tickers: List[str],
        existing_symbols: set,
        market_regime: MarketRegime,
    ) -> tuple[int, List[Dict]]:
        """Scan tickers in parallel for new entry signals."""
        signal_count = 0
        watchlist_candidates: List[Dict] = []
        no_signal_symbols: List[str] = []
        no_signal_reasons: Dict[str, List[str]] = {}
        results_lock = asyncio.Lock()

        async def scan_ticker(symbol: str) -> None:
            nonlocal signal_count
            try:
                if self.portfolio_lock.is_pending(symbol):
                    return

                entry_result = await self._process_ticker_for_entry(symbol, market_regime)

                if entry_result:
                    async with results_lock:
                        if entry_result.get("signal"):
                            signal_count += 1
                        elif entry_result.get("warnings"):
                            no_signal_symbols.append(entry_result["symbol"])
                            no_signal_reasons[entry_result["symbol"]] = entry_result["warnings"]
                        elif entry_result.get("is_watchlist"):
                            watchlist_candidates.append(entry_result)

            except Exception as e:
                self._logger.error(f"Lỗi nghiêm trọng khi quét mã {symbol}: {e}", exc_info=True)
                async with results_lock:
                    no_signal_symbols.append(symbol)
                    no_signal_reasons[symbol] = [f"Lỗi khi quét: {str(e)}"]

        tasks = [scan_ticker(symbol) for symbol in current_tickers]
        await asyncio.gather(*tasks)

        # Send summary if no signals found
        if signal_count == 0 and no_signal_symbols:
            await self._send_no_signal_summary(
                current_tickers, no_signal_symbols, no_signal_reasons
            )

        return signal_count, watchlist_candidates

    async def _process_ticker_for_entry(
        self,
        symbol: str,
        market_regime: MarketRegime,
    ) -> Optional[ScanResult]:
        """
        Process a single ticker for entry signal.

        Returns:
            ScanResult with signal info, warnings, or watchlist status
        """
        try:
            # Load data
            df = load_data(symbol, lookback=self._lookback)
            if df.empty or len(df) < 50:
                return {
                    "symbol": symbol,
                    "warnings": ["Không đủ dữ liệu lịch sử"],
                    "is_watchlist": False,
                }

            # Get ML signal
            ml_signal = await self._get_ml_signal(df, symbol, context="entry_scan")

            # Check ML circuit breaker strict mode
            ml_cb_strict_mode = self._ml_circuit_breaker_active
            if ml_cb_strict_mode:
                self._logger.warning(
                    f"⚠️ [{symbol}] ML circuit breaker active - applying strict risk controls"
                )

            # Validate entry logic
            if not self.entry_logic:
                self._logger.error("❌ Entry logic not initialized")
                return {
                    "symbol": symbol,
                    "warnings": ["Lỗi: Entry logic chưa được khởi tạo"],
                    "is_watchlist": False,
                }

            # Analyze entry
            entry_signal = self.entry_logic.analyze_entry(
                df=df,
                ml_signal=ml_signal,
                market_regime=market_regime,
                symbol=symbol,
            )

            # Validate entry signal
            if not entry_signal or not entry_signal.should_enter:
                warnings = getattr(entry_signal, "warnings", ["Không rõ lý do"])
                return {"symbol": symbol, "warnings": warnings, "is_watchlist": False}

            # Apply strict mode confidence check
            if ml_cb_strict_mode:
                threshold = self._ml_cb_config.strict_confidence_threshold
                if entry_signal.confidence < threshold:
                    self._logger.warning(
                        f"⚠️ [{symbol}] ML CB strict mode: Confidence {entry_signal.confidence}% < {threshold}%"
                    )
                    return {
                        "symbol": symbol,
                        "warnings": [
                            f"ML circuit breaker active - need confidence ≥{threshold}% (got {entry_signal.confidence}%)"
                        ],
                        "is_watchlist": False,
                    }

            # Track signal source
            is_ml_signal = getattr(entry_signal, "telemetry", {}).get("signal_source") == "ml"
            self.signal_tracker.track_signal(is_ml_signal=is_ml_signal)

            # Process entry if signal is valid
            return await self._process_valid_entry(
                symbol, df, entry_signal, market_regime, ml_cb_strict_mode, is_ml_signal
            )

        except DataLoadError:
            return {"symbol": symbol, "warnings": ["Lỗi tải dữ liệu"], "is_watchlist": False}
        except Exception as e:
            self._logger.error(f"[{symbol}] Lỗi không xác định: {e}", exc_info=True)
            return {
                "symbol": symbol,
                "warnings": [f"Lỗi không xác định: {str(e)}"],
                "is_watchlist": False,
            }

    async def _process_valid_entry(
        self,
        symbol: str,
        df: pd.DataFrame,
        entry_signal: Any,
        market_regime: MarketRegime,
        ml_cb_strict_mode: bool,
        is_ml_signal: bool,
    ) -> Optional[ScanResult]:
        """Process a valid entry signal."""
        # Get current positions
        current_positions = self.portfolio_manager.get_positions()
        active_positions = {
            sym: pos for sym, pos in current_positions.items() if pos.get("shares", 0) > 0
        }

        # Check if already has position
        if symbol in active_positions:
            self._logger.info(f"ℹ️ [{symbol}] Đã có position active, chỉ gửi notification")

        # Calculate position size
        position_size_info = self._calculate_position_size(
            symbol, entry_signal, market_regime, ml_cb_strict_mode
        )

        if not position_size_info or position_size_info.shares <= 0:
            return None

        # If already has position, just send notification
        if symbol in active_positions:
            await self._send_buy_notification(symbol, entry_signal, position_size_info)
            return {"signal": True, "skipped_buy": True}

        # Execute buy for new position
        return await self._execute_buy(
            symbol, entry_signal, position_size_info, active_positions, is_ml_signal
        )

    def _calculate_position_size(
        self,
        symbol: str,
        entry_signal: Any,
        market_regime: MarketRegime,
        ml_cb_strict_mode: bool,
    ) -> Optional[Any]:
        """Calculate position size with optional strict mode reduction."""
        take_profit_price = (
            entry_signal.take_profit_targets[0]
            if entry_signal.take_profit_targets
            else entry_signal.entry_price * 1.1
        )

        position_size_info = self.position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=entry_signal.entry_price,
            stop_loss=entry_signal.stop_loss,
            take_profit=take_profit_price,
            confidence=entry_signal.confidence,
            signal_strength=entry_signal.strength.name,
            market_regime=market_regime,
        )

        # Apply strict mode size reduction
        if ml_cb_strict_mode and position_size_info:
            multiplier = self._ml_cb_config.strict_size_multiplier
            original_shares = position_size_info.shares
            original_value = position_size_info.value

            position_size_info.shares = max(1, int(position_size_info.shares * multiplier))
            position_size_info.value = position_size_info.shares * entry_signal.entry_price

            self._logger.info(
                f"📊 [{symbol}] ML CB strict mode: Position size reduced from "
                f"{original_shares} → {position_size_info.shares} shares"
            )

        return position_size_info

    async def _execute_buy(
        self,
        symbol: str,
        entry_signal: Any,
        position_size_info: Any,
        active_positions: Dict[str, Any],
        is_ml_signal: bool,
    ) -> Optional[ScanResult]:
        """Execute buy order with atomic position management."""
        total_capital = self._trading_config.trading.total_capital

        with self.portfolio_lock.atomic_position_add(
            symbol=symbol,
            position_value=position_size_info.value,
            total_capital=total_capital,
            current_positions=active_positions,
        ) as (can_add, reason):
            if not can_add:
                self._logger.info(f"⚠️ [{symbol}] Cannot add position: {reason}")
                return None

            take_profit = (
                entry_signal.take_profit_targets[0] if entry_signal.take_profit_targets else None
            )

            trade_metadata = {
                "signal_source": "ml" if is_ml_signal else "technical",
                "confidence": entry_signal.confidence,
                "signal_reason": ", ".join(entry_signal.reasons),
            }

            success, message, trade = self.paper_account.execute_buy(
                symbol=symbol,
                shares=position_size_info.shares,
                price=entry_signal.entry_price,
                signal_confidence=entry_signal.confidence,
                signal_reason=", ".join(entry_signal.reasons),
                stop_loss=entry_signal.stop_loss,
                take_profit=take_profit,
                is_limit_order=getattr(entry_signal, "is_limit_order", False),
                limit_price=getattr(entry_signal, "limit_price", None),
                metadata=trade_metadata,
            )

            if not success:
                self._logger.error(f"❌ Paper trade failed for {symbol}: {message}")
                raise Exception(f"Paper trade failed: {message}")

            self._logger.info(f"✅ Paper trade successful: {message}")

            # Send notification (non-blocking)
            try:
                await self._send_buy_notification(symbol, entry_signal, position_size_info)
            except Exception as e:
                self._logger.error(f"⚠️ Failed to send buy notification for {symbol}: {e}")

            return {"signal": True}

    # =========================================================================
    # ML SIGNAL GENERATION
    # =========================================================================

    async def _get_ml_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        context: str = "unknown",
    ) -> Optional[Any]:
        """
        Get ML signal with retry logic and circuit breaker protection.

        Args:
            df: Price data DataFrame
            symbol: Stock symbol
            context: Context for logging (e.g., "entry_scan", "exit_check")

        Returns:
            ML signal or None if failed/disabled
        """
        if not self._should_use_ml():
            return None

        config = self._ml_analysis_config
        error_type = "unknown"
        error_msg = ""

        for attempt in range(config.max_retries + 1):
            try:
                # Validate data
                if df is None or len(df) < 50:
                    raise ValueError(f"Insufficient data: {len(df) if df is not None else 0} rows")

                if "close" not in df.columns or "volume" not in df.columns:
                    raise ValueError(f"Missing required columns: {list(df.columns)}")

                # Get cached VNINDEX
                cached_vnindex = self._vnindex_cache.get()

                # Run ML analysis with timeout
                ml_signal = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.ml_generator.analyze,
                        df,
                        index_df=cached_vnindex,
                        symbol=symbol,
                    ),
                    timeout=config.timeout_seconds,
                )

                if ml_signal is not None:
                    self._ml_metrics.record_success()
                    if attempt > 0:
                        self._logger.info(
                            f"✅ ML analysis succeeded on retry {attempt} for {symbol}"
                        )
                    return ml_signal

            except asyncio.TimeoutError:
                error_type = "timeout"
                error_msg = f"ML analysis timed out after {config.timeout_seconds}s"
                if attempt < config.max_retries:
                    self._logger.warning(
                        f"⏰ ML timeout for {symbol}, retrying ({attempt + 1}/{config.max_retries})..."
                    )
                    await asyncio.sleep(config.retry_delay_base * (2**attempt))
                    continue

            except ValueError as e:
                error_type = "data_quality"
                error_msg = str(e)
                break  # Don't retry data quality errors

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                if attempt < config.max_retries and "VNINDEX" not in str(e):
                    self._logger.warning(
                        f"⚠️ ML error for {symbol}, retrying ({attempt + 1}/{config.max_retries}): {e}"
                    )
                    await asyncio.sleep(config.retry_delay_base * (2**attempt))
                    continue
                break

        # Track failure
        self._logger.error(
            f"❌ ML analysis failed ({context}) for {symbol} after {attempt} retries: {error_type}: {error_msg}"
        )
        self._logger.debug(f"ML error traceback for {symbol}:\n{traceback.format_exc()}")

        self._ml_metrics.record_failure(symbol, error_type, error_msg)
        await self._check_ml_circuit_breaker()

        return None

    def _should_use_ml(self) -> bool:
        """Check if ML analysis should be used."""
        if not self._ml_enabled:
            return False
        return os.getenv("USE_ML_ANALYSIS", "true").lower() == "true"

    async def _check_ml_circuit_breaker(self) -> None:
        """Check and update ML circuit breaker status."""
        metrics = self._ml_metrics
        config = self._ml_cb_config

        if metrics.total_attempts < config.min_samples:
            return

        failure_rate = metrics.failure_rate

        # Check if should TRIP circuit breaker
        if not self._ml_circuit_breaker_active and failure_rate >= config.failure_threshold:
            self._ml_circuit_breaker_active = True
            self._ml_enabled = False

            alert_msg = (
                f"🚨 ML CIRCUIT BREAKER ACTIVATED 🚨\n\n"
                f"Failure rate: {failure_rate:.1%} (threshold: {config.failure_threshold:.1%})\n"
                f"Total failures: {metrics.failure_count}/{metrics.total_attempts}\n\n"
                f"🔧 Switching to TECHNICAL ANALYSIS only\n"
                f"ML will auto-recover when failure rate drops below {config.recovery_threshold:.1%}"
            )

            self._logger.critical(alert_msg)
            await self._send_message(alert_msg, parse_mode="Markdown")

        # Check if should RECOVER
        elif self._ml_circuit_breaker_active and failure_rate <= config.recovery_threshold:
            self._ml_circuit_breaker_active = False
            self._ml_enabled = True

            recovery_msg = (
                f"✅ ML CIRCUIT BREAKER RECOVERED\n\n"
                f"Failure rate improved: {failure_rate:.1%}\n"
                f"🤖 ML analysis RE-ENABLED"
            )

            self._logger.info(recovery_msg)
            await self._send_message(recovery_msg, parse_mode="Markdown")

    # =========================================================================
    # NEWS ANALYSIS
    # =========================================================================

    def adjust_signal_with_news(self, entry_signal: Any, news_context: Optional[Dict]) -> Any:
        """Adjust entry signal based on news analysis."""
        if not news_context or not news_context.get("articles"):
            return entry_signal

        news_sentiment = news_context.get("sentiment_score", 0.0)
        has_litigation = any(
            "litigation" in article.get("topics", [])
            for article in news_context.get("articles", [])
        )
        has_dividend = any(
            "dividend" in article.get("topics", []) for article in news_context.get("articles", [])
        )

        # Apply sentiment adjustments
        if news_sentiment >= 0.8:
            entry_signal.confidence = min(100, entry_signal.confidence + 15)
            entry_signal.reasons.append(f"📰 Tin tức RẤT tích cực ({news_sentiment:+.2f})")
        elif news_sentiment >= 0.5:
            entry_signal.confidence = min(100, entry_signal.confidence + 10)
            entry_signal.reasons.append(f"📰 Tin tức tích cực ({news_sentiment:+.2f})")
        elif news_sentiment <= -0.8 or has_litigation:
            entry_signal.should_enter = False
            entry_signal.warnings.append(
                f"📰 Tin tức RẤT tiêu cực hoặc kiện tụng ({news_sentiment:+.2f})"
            )
        elif news_sentiment <= -0.5:
            entry_signal.confidence = max(0, entry_signal.confidence - 15)
            entry_signal.warnings.append(f"📰 Tin tức tiêu cực ({news_sentiment:+.2f})")
        else:
            entry_signal.reasons.append(f"📰 Tin tức trung lập ({news_sentiment:+.2f})")

        if has_dividend and news_sentiment > 0:
            entry_signal.confidence = min(100, entry_signal.confidence + 5)
            entry_signal.reasons.append("💰 Tin cổ tức")

        # Check confidence threshold
        if (
            entry_signal.should_enter
            and self.entry_logic
            and entry_signal.confidence < self.entry_logic.min_confidence
        ):
            entry_signal.should_enter = False
            entry_signal.warnings.append(
                f"Confidence giảm xuống dưới ngưỡng sau khi điều chỉnh tin tức ({entry_signal.confidence}%)"
            )

        return entry_signal

    # =========================================================================
    # NOTIFICATIONS
    # =========================================================================

    async def _send_message(
        self,
        text: str,
        parse_mode: Optional[str] = None,
    ) -> None:
        """Send message via Telegram bot."""
        if not self.bot or not self.chat_id:
            self._logger.warning("⚠️ Telegram bot not configured, skipping message")
            return

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
            )
        except Exception as e:
            self._logger.error(f"Failed to send Telegram message: {e}")

    async def _send_scan_start_message(
        self,
        current_tickers: List[str],
        market_regime: MarketRegime,
    ) -> None:
        """Send scan start notification."""
        try:
            regime_text = market_regime.get("regime", "UNKNOWN")
            await self._send_message(
                f"🔍 Đang quét {len(current_tickers)} mã...\n"
                f"Chế độ thị trường: *{regime_text}* (Confidence: {market_regime.get('confidence', 50)}%)\n"
                f"Số mã tiềm năng: *{len(current_tickers)}*",
                parse_mode="Markdown",
            )
        except Exception:
            self._logger.error("Lỗi gửi Telegram (scan start)", exc_info=True)

    async def _send_buy_notification(
        self,
        symbol: str,
        entry_signal: Any,
        position_size_info: Any,
    ) -> None:
        """Send buy signal notification."""
        message = self._formatter.format_buy_signal(symbol, entry_signal, position_size_info)
        await self._send_message(message, parse_mode="Markdown")

    async def _send_no_signal_summary(
        self,
        all_tickers: List[str],
        no_signal_symbols: List[str],
        no_signal_reasons: Dict[str, List[str]],
    ) -> None:
        """Send summary when no buy signals found."""
        try:
            # Group similar reasons
            reason_counts: Dict[str, int] = {}
            for symbol, reasons in no_signal_reasons.items():
                for reason in reasons:
                    clean_reason = reason.split("(")[0].strip() if "(" in reason else reason
                    clean_reason = (
                        clean_reason.split(":")[0].strip() if ":" in clean_reason else clean_reason
                    )
                    reason_counts[clean_reason] = reason_counts.get(clean_reason, 0) + 1

            # Build summary message
            summary = "🔍 *TỔNG HỢP KHÔNG TÌM THẤY TÍN HIỆU MUA*\n"
            summary += f"📊 Đã quét: {len(all_tickers)} mã\n"
            summary += f"📉 Không tìm thấy tín hiệu: {len(no_signal_symbols)} mã\n\n"

            summary += "*CHI TIẾT THEO NGUYÊN NHÂN:*\n"
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(no_signal_symbols)) * 100
                summary += f"• {reason}: {count} mã ({percentage:.1f}%)\n"

            # Add examples
            top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_reasons:
                summary += "\n*VÍ DỤ:*\n"
                for reason, _ in top_reasons:
                    examples = [
                        s
                        for s, reasons in no_signal_reasons.items()
                        if any(r.startswith(reason) for r in reasons)
                    ][:2]
                    if examples:
                        summary += f"• {reason}: {', '.join(examples)}\n"

            summary += f"\n⏰ {datetime.now().strftime('%H:%M %d/%m/%Y')}"

            await self._send_message(summary, parse_mode="Markdown")
            self._logger.info("✅ Đã gửi thông báo tổng hợp không tìm thấy tín hiệu")

        except Exception as e:
            self._logger.error(f"Lỗi khi gửi thông báo tổng hợp: {e}", exc_info=True)

    async def _send_summary_report(
        self,
        signal_count: int,
        watchlist_candidates: List[Dict],
        market_regime: MarketRegime,
    ) -> None:
        """Send end-of-scan summary report."""
        try:
            portfolio_summary = self.portfolio_manager.get_detailed_analysis()

            summary_msg = "**--- BÁO CÁO QUÉT ---**\n"
            summary_msg += f"Thời gian: {datetime.now().strftime('%H:%M %d-%m-%Y')}\n"
            summary_msg += (
                f"📊 Thị trường: *{market_regime.get('regime', 'N/A')}* "
                f"(Conf: {market_regime.get('confidence', 0)}%)\n"
            )
            summary_msg += f"💡 Tín hiệu mua mới: **{signal_count}**\n"

            if watchlist_candidates:
                summary_msg += f"👀 Watchlist: {len(watchlist_candidates)}\n"

            summary_msg += "\n" + portfolio_summary

            await self._send_message(summary_msg, parse_mode="Markdown")
        except Exception:
            self._logger.error("Lỗi gửi báo cáo tóm tắt", exc_info=True)
            await self._send_message("Lỗi khi tạo báo cáo")

    # =========================================================================
    # RISK ANALYSIS
    # =========================================================================

    async def perform_post_scan_risk_analysis(self) -> None:
        """Perform portfolio risk analysis after scan."""
        active_positions = self.portfolio_manager.get_positions()
        if not self.portfolio_risk_manager or not active_positions:
            return

        try:
            risk_positions = {
                sym: {
                    "shares": pos.get("shares", 0),
                    "avg_price": pos.get("avg_price", 0),
                    "current_price": pos.get("current_price", pos.get("avg_price", 0)),
                    "stop_loss": pos.get("stop_loss", pos.get("avg_price", 0) * 0.93),
                }
                for sym, pos in active_positions.items()
            }

            risk_metrics = self.portfolio_risk_manager.calculate_portfolio_risk(risk_positions)

            if risk_metrics.risk_status in ["HIGH", "CRITICAL"]:
                risk_summary = self.portfolio_risk_manager.get_risk_summary(risk_positions)
                await self._send_message(
                    f"⚠️ *PORTFOLIO RISK ALERT*\n\n{risk_summary}",
                    parse_mode="Markdown",
                )
        except Exception:
            self._logger.error("Lỗi portfolio risk analysis", exc_info=True)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _cleanup_stale_pending_exits(self) -> None:
        """Remove pending exits older than TTL."""
        now = datetime.now()
        stale_symbols = []

        for symbol, data in self._pending_exits.items():
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
                age_seconds = (now - timestamp).total_seconds()
                if age_seconds > self._pending_exits_ttl:
                    stale_symbols.append(symbol)
            except (KeyError, ValueError):
                stale_symbols.append(symbol)

        for symbol in stale_symbols:
            del self._pending_exits[symbol]
            self._logger.debug(f"🧹 Cleaned up stale pending exit for {symbol}")

        if stale_symbols:
            self._logger.info(f"🧹 Cleaned up {len(stale_symbols)} stale pending exits")

    # =========================================================================
    # LEGACY COMPATIBILITY METHODS & PROPERTIES
    # =========================================================================

    # Legacy property accessors for ML tracking (backward compatibility)
    @property
    def _ml_failure_count(self) -> int:
        """Legacy accessor for ML failure count."""
        return self._ml_metrics.failure_count

    @_ml_failure_count.setter
    def _ml_failure_count(self, value: int) -> None:
        """Legacy setter for ML failure count."""
        self._ml_metrics.failure_count = value

    @property
    def _ml_success_count(self) -> int:
        """Legacy accessor for ML success count."""
        return self._ml_metrics.success_count

    @_ml_success_count.setter
    def _ml_success_count(self, value: int) -> None:
        """Legacy setter for ML success count."""
        self._ml_metrics.success_count = value

    @property
    def _ml_failures_by_error(self) -> Dict[str, int]:
        """Legacy accessor for ML failures by error type."""
        return self._ml_metrics.failures_by_error

    @property
    def _ml_failures_by_symbol(self) -> Dict[str, int]:
        """Legacy accessor for ML failures by symbol."""
        return self._ml_metrics.failures_by_symbol

    @property
    def _ml_failure_reasons(self) -> Dict[str, int]:
        """Legacy accessor for ML failure reasons."""
        return self._ml_metrics.failure_reasons

    def _get_cached_vnindex(self) -> Optional[pd.DataFrame]:
        """Legacy method for getting cached VNINDEX (backward compatibility)."""
        return self._vnindex_cache.get()

    @property
    def _cached_vnindex_df(self) -> Optional[pd.DataFrame]:
        """Legacy accessor for cached VNINDEX DataFrame."""
        return self._vnindex_cache._cached_df

    @_cached_vnindex_df.setter
    def _cached_vnindex_df(self, value: Optional[pd.DataFrame]) -> None:
        """Legacy setter for cached VNINDEX DataFrame."""
        self._vnindex_cache._cached_df = value

    @property
    def _vnindex_cache_timestamp(self) -> Optional[float]:
        """Legacy accessor for VNINDEX cache timestamp."""
        return self._vnindex_cache._cache_timestamp

    @_vnindex_cache_timestamp.setter
    def _vnindex_cache_timestamp(self, value: Optional[float]) -> None:
        """Legacy setter for VNINDEX cache timestamp."""
        self._vnindex_cache._cache_timestamp = value

    def _track_ml_failure(self, symbol: str, error_details: Dict[str, Any]) -> None:
        """Legacy method for tracking ML failures (backward compatibility)."""
        self._ml_metrics.record_failure(
            symbol=symbol,
            error_type=error_details.get("error_type", "unknown"),
            error_msg=error_details.get("error_msg", ""),
        )

    def _get_ml_failure_rate(self) -> float:
        """Legacy method for getting ML failure rate (backward compatibility)."""
        return self._ml_metrics.failure_rate

    async def check_active_positions(self, market_regime: MarketRegime) -> None:
        """Legacy method for checking active positions (backward compatibility)."""
        await self._check_active_positions(market_regime)

    async def execute_exit(
        self,
        symbol: str,
        pos_data: Dict[str, Any],
        exit_decision: Any,
        current_price: float,
    ) -> None:
        """Legacy method for executing exit (backward compatibility)."""
        await self._execute_exit(symbol, pos_data, exit_decision, current_price)

    def get_scan_universe(self) -> List[str]:
        """Legacy method for getting scan universe (backward compatibility)."""
        return self._get_scan_universe()

    def sync_position_sizer_with_active_positions(self, active_positions: Dict[str, Any]) -> None:
        """Legacy method for syncing position sizer (backward compatibility)."""
        self._sync_position_sizer(active_positions)

    async def scan_for_new_entries(
        self,
        current_tickers: List[str],
        existing_symbols: set,
        market_regime: MarketRegime,
    ) -> tuple[int, List[Dict]]:
        """Legacy method for scanning new entries (backward compatibility)."""
        return await self._scan_for_new_entries(current_tickers, existing_symbols, market_regime)

    async def send_scan_start_message(
        self,
        current_tickers: List[str],
        market_regime: MarketRegime,
    ) -> None:
        """Legacy method for sending scan start message (backward compatibility)."""
        await self._send_scan_start_message(current_tickers, market_regime)

    async def send_summary_report(
        self,
        signal_count: int,
        watchlist_candidates: List[Dict],
        market_regime: MarketRegime,
    ) -> None:
        """Legacy method for sending summary report (backward compatibility)."""
        await self._send_summary_report(signal_count, watchlist_candidates, market_regime)

    async def send_buy_signal_notification(
        self,
        symbol: str,
        entry_signal: Any,
        position_size_info: Any,
        news_sentiment: Optional[Dict] = None,
    ) -> None:
        """Legacy method for sending buy notification (backward compatibility)."""
        await self._send_buy_notification(symbol, entry_signal, position_size_info)

    async def process_single_ticker_for_entry(
        self,
        symbol: str,
        market_regime: MarketRegime,
    ) -> Optional[ScanResult]:
        """Legacy method for processing ticker entry (backward compatibility)."""
        return await self._process_ticker_for_entry(symbol, market_regime)

    def format_entry_recommendation(
        self,
        symbol: str,
        entry_signal: Any,
        position: Any,
        market_regime: MarketRegime,
        news_context: Optional[Dict] = None,
    ) -> str:
        """Format entry recommendation message (legacy compatibility)."""
        return self._formatter.format_entry_recommendation(
            symbol, entry_signal, position, market_regime, news_context
        )

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters (legacy compatibility)."""
        return self._formatter.escape_html(text)

    def _format_exit_recommendation(
        self,
        symbol: str,
        pos_data: Dict[str, Any],
        exit_decision: Any,
        current_price: float,
        use_html: bool = True,
    ) -> str:
        """Format exit recommendation message (legacy compatibility)."""
        return self._formatter.format_exit_recommendation(
            symbol, pos_data, exit_decision, current_price
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_orchestrator(
    bot_instance: Optional[Bot] = None,
    chat_id: Optional[str] = None,
    **kwargs,
) -> TradingOrchestrator:
    """
    Factory function to create TradingOrchestrator with proper configuration.

    Args:
        bot_instance: Telegram bot instance
        chat_id: Telegram chat ID
        **kwargs: Additional arguments passed to TradingOrchestrator

    Returns:
        Configured TradingOrchestrator instance

    Example:
        # Basic usage
        orchestrator = create_orchestrator(bot, chat_id)

        # With custom services
        orchestrator = create_orchestrator(
            bot, chat_id,
            ml_generator=custom_ml_generator,
            portfolio_manager=custom_portfolio_manager,
        )
    """
    return TradingOrchestrator(
        bot_instance=bot_instance,
        chat_id=chat_id,
        **kwargs,
    )
