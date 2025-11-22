# -*- coding: utf-8 -*-
"""
Orchestrator for the trading bot.
Tách logic điều phối ra khỏi bot_runner_improved.py
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from src.risk.circuit_breaker import get_circuit_breaker
from src.data.loader import load_data
from src.config.exceptions import DataLoadError
from src.market.regime_proxy import ProxyMarketRegimeAnalyzer
from src.ml.monitor import get_ml_model_monitor
from src.ml.signals.generator import MLSignalGenerator
from src.portfolio.paper_trading import get_paper_account
from src.portfolio.lock import get_portfolio_lock
from src.portfolio.manager import get_portfolio_manager
from src.portfolio.risk_manager import get_portfolio_risk_manager
from src.strategies.manager import get_strategy_manager
from telegram import Bot
from src.data.ticker_loader import get_ticker_loader

# Import các thành phần cần thiết từ project
# (Giả định các import này vẫn hoạt động sau khi tách file)
from src.config.legacy_config import MAX_SCAN_UNIVERSE, MIN_VOLUME
from src.monitoring.signal_performance import get_signal_performance_tracker

# Get LOOKBACK safely with fallback
try:
    from src.config.legacy_config import LOOKBACK

    if LOOKBACK is None or not isinstance(LOOKBACK, int):
        LOOKBACK = 200
except Exception as e:
    logging.warning(f"Failed to import LOOKBACK from config: {e}")
    LOOKBACK = 200  # Fallback value

# Ensure LOOKBACK is valid
if not isinstance(LOOKBACK, int) or LOOKBACK < 50:
    logging.warning(f"Invalid LOOKBACK value: {LOOKBACK}, using 200")
    LOOKBACK = 200


class TradingOrchestrator:
    """
    Lớp điều phối chính, quản lý toàn bộ luồng quét và giao dịch.
    """

    def __init__(
        self,
        bot_instance: Optional[Bot] = None,
        chat_id: Optional[str] = None,
        vnindex_df: Optional[pd.DataFrame] = None,
        # Dependency injection parameters (optional)
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
        Initialize TradingOrchestrator with flexible constructor.

        Supports both legacy mode (bot_instance, chat_id) and
        modern dependency injection mode.

        Legacy usage:
            orchestrator = TradingOrchestrator(bot, chat_id)

        Modern usage (via factory):
            orchestrator = create_orchestrator()
        """
        # Legacy telegram bot setup
        self.bot = bot_instance
        self.chat_id = chat_id
        self.vnindex_df = vnindex_df

        # Use injected dependencies or create defaults
        self.config = config
        self.data_loader = data_loader
        self.portfolio_manager = portfolio_manager or get_portfolio_manager()
        self.portfolio_risk_manager = get_portfolio_risk_manager(total_capital=100_000_000)
        self.portfolio_lock = get_portfolio_lock()
        self.market_analyzer = ProxyMarketRegimeAnalyzer()
        self.ticker_loader = get_ticker_loader()
        self.ml_generator = ml_generator or MLSignalGenerator()
        self.paper_account = paper_account or get_paper_account()
        self.ml_monitor = get_ml_model_monitor()
        self.strategy_manager = strategy_manager or get_strategy_manager()
        self.circuit_breaker = circuit_breaker or get_circuit_breaker()
        self.signal_tracker = get_signal_performance_tracker()

        # New injected services (optional)
        self.risk_service = risk_service
        self.entry_service = entry_service
        self.exit_service = exit_service
        self.notification_service = notification_service

        # Các thành phần chiến lược sẽ được lấy từ StrategyManager
        self.entry_logic: Optional[Any] = None
        self.position_sizer: Optional[Any] = None
        self.exit_strategy: Optional[Any] = None

        # ENHANCEMENT: ML failure tracking for monitoring and debugging
        self._ml_failure_count = 0
        self._ml_success_count = 0
        self._ml_failures_by_error = {}  # {error_type: count}
        self._ml_failures_by_symbol = {}  # {symbol: count}

        # ML CIRCUIT BREAKER: Auto-disable ML when failure rate too high
        self._ml_enabled = True  # Can be disabled by circuit breaker
        self._ml_circuit_breaker_threshold = 0.30  # Disable at 30% failure rate
        self._ml_circuit_breaker_min_samples = 20  # Need 20 attempts before activating
        self._ml_recovery_threshold = 0.10  # Re-enable at 10% failure rate
        self._ml_circuit_breaker_active = False

    def _setup_strategies(self, market_regime: Dict):
        """Lấy và gán các chiến lược từ StrategyManager và điều chỉnh theo thị trường."""
        # Lấy các đối tượng chiến lược gốc
        strategies = self.strategy_manager.get_strategies()
        self.entry_logic = strategies["entry_logic"]
        self.position_sizer = strategies["position_sizer"]
        self.exit_strategy = strategies["exit_strategy"]

        # Yêu cầu StrategyManager tự điều chỉnh dựa trên trạng thái thị trường
        self.strategy_manager.apply_market_adjustments(market_regime)

        logging.info(
            f"🔧 Đã thiết lập và điều chỉnh chiến lược cho chế độ: {market_regime.get('regime', 'Sideways')}"
        )

    def adjust_signal_with_news(self, entry_signal, news_context):
        """
        Điều chỉnh tín hiệu entry dựa trên phân tích tin tức.
        """
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

        # ENHANCED IMPACT
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

    def format_entry_recommendation(
        self, symbol, entry_signal, position, market_regime, news_context=None
    ):
        """Format entry recommendation message"""

        msg = f"🎯 *TÍN HIỆU VÀO LỆNH - {symbol}*\n\n"

        if market_regime:
            msg += f"📊 *Market:* {market_regime['regime']} ({market_regime.get('confidence', 0)}%)\n\n"

        msg += f"💪 *Signal:* {entry_signal.strength.name}\n"
        msg += f"🎲 *Confidence:* {entry_signal.confidence}%\n"
        msg += f"📈 *Shares:* {position.shares:,} ({position.shares//100} lô)\n"
        msg += f"💰 *Value:* {position.value:,.0f} VNĐ ({position.position_percent:.1f}%)\n\n"

        if position.recommended_entries:
            msg += "💵 *GIÁ VÀO (DCA):*\n"
            for entry in position.recommended_entries[:2]:
                msg += f"  L{entry['level']}: {entry['price']:,.0f} - "
                msg += f"{entry['shares']:,} CP ({entry['percent']}%)\n"
            msg += "\n"

        msg += f"🛑 *Stop Loss:* {entry_signal.stop_loss:,.0f} VNĐ "
        sl_pct = (
            (entry_signal.stop_loss - entry_signal.entry_price) / entry_signal.entry_price * 100
        )
        msg += f"({sl_pct:+.1f}%)\n\n"

        msg += "🎯 *Take Profit:*\n"
        for i, tp in enumerate(entry_signal.take_profit_targets[:2], 1):
            tp_pct = ((tp - entry_signal.entry_price) / entry_signal.entry_price) * 100
            msg += f"  TP{i}: {tp:,.0f} (+{tp_pct:.1f}%)\n"

        if entry_signal.reasons:
            msg += "\n✅ *Lý do:*\n"
            for reason in entry_signal.reasons[:2]:
                msg += f"• {reason}\n"

        if entry_signal.warnings:
            msg += f"\n⚠️ *Cảnh báo:* {entry_signal.warnings[0]}\n"

        msg += f"\n💸 *Risk:* {position.max_loss:,.0f} VNĐ ({position.risk_percent:.2f}%)"

        if news_context and news_context.get("articles"):
            msg += f"\n\n📰 *News Sentiment:* {news_context['sentiment_label']} ({news_context['sentiment_score']:+.2f})\n"
            for article in news_context.get("top_headlines", [])[:2]:
                published = article.get("published_at", "")[:16].replace("T", " ")
                msg += f"  • {article['title']} ({article['source']}, {published})\n"
                if article.get("url"):
                    msg += f"    {article['url']}\n"

        return msg

    async def run_scan(self, market_regime: Dict):
        """
        Chạy quá trình quét toàn diện để tìm kiếm cơ hội và quản lý vị thế.
        """
        # VALIDATION: Validate inputs
        if not market_regime or not isinstance(market_regime, dict):
            logging.error("❌ Invalid market_regime provided to run_scan")
            market_regime = {"regime": "UNKNOWN", "confidence": 0, "tradeable": False}

        # Ensure required keys exist
        market_regime.setdefault("regime", "SIDEWAYS")
        market_regime.setdefault("confidence", 50)
        market_regime.setdefault("tradeable", True)

        # 0. KIỂM TRA NGẮT MẠCH (CIRCUIT BREAKER)
        vnindex_change = 0.0
        try:
            if (
                self.vnindex_df is not None
                and not self.vnindex_df.empty
                and len(self.vnindex_df) > 1
            ):
                vnindex_change = self.vnindex_df["close"].pct_change().iloc[-1]
                if pd.isna(vnindex_change):
                    vnindex_change = 0.0
            else:
                logging.warning("⚠️ VNINDEX data unavailable for circuit breaker check")
        except Exception as e:
            logging.error(f"❌ Error calculating VNINDEX change: {e}")
            vnindex_change = 0.0

        # Lấy PNL thực tế từ portfolio manager nếu có
        current_pnl_pct = 0.0
        try:
            current_pnl_pct = self.portfolio_manager.get_daily_pnl_pct()
        except Exception as e:
            logging.error(f"❌ Error getting daily PnL: {e}")
            current_pnl_pct = 0.0

        if self.circuit_breaker.check_and_update(
            portfolio_pnl_pct=current_pnl_pct, vnindex_change_pct=vnindex_change
        ):
            reason = self.circuit_breaker.tripped_reason
            logging.critical(f"🚨 NGẮT MẠCH ĐANG KÍCH HOẠT: {reason}. Dừng mọi lệnh mua mới.")
            await self.bot.send_message(
                self.chat_id,
                f"🚨 *NGẮT MẠCH TỰ ĐỘNG*\n\nLý do: {reason}\n\nTạm dừng tất cả các lệnh mua mới.",
                parse_mode="Markdown",
            )
            # Vẫn cho phép kiểm tra thoát lệnh
            await self.check_active_positions(market_regime)
            return

        # 1. THIẾT LẬP CHIẾN LƯỢC DỰA TRÊN THỊ TRƯỜNG
        self._setup_strategies(market_regime)

        # 2. LẤY DANH SÁCH MÃ VÀ VỊ THẾ HIỆN TẠI
        active_positions = self.portfolio_manager.get_positions()
        existing_symbols = set(active_positions.keys())
        self.sync_position_sizer_with_active_positions(active_positions)

        current_tickers = self.get_scan_universe()
        logging.info(f"🔍 Quét {len(current_tickers)} mã...")

        # 3. Gửi thông báo bắt đầu quét
        await self.send_scan_start_message(current_tickers, market_regime)

        # 4. KIỂM TRA THOÁT LỆNH TRƯỚC
        await self.check_active_positions(market_regime)

        # 5. QUÉT TÌM LỆNH MUA MỚI (SONG SONG)
        logging.info(f"\n🔍 Quét {len(current_tickers)} mã để tìm cơ hội mua mới")
        self.portfolio_lock.clear_pending()

        signal_count, watchlist_candidates = await self.scan_for_new_entries(
            current_tickers, existing_symbols, market_regime
        )

        # 6. GỬI BÁO CÁO TÓM TẮT
        await self.send_summary_report(signal_count, watchlist_candidates, market_regime)

    def get_scan_universe(self) -> List[str]:
        """Lấy danh sách các mã cổ phiếu cần quét."""
        try:
            return self.ticker_loader.get_validated_tickers(
                force_validate=True,
                min_volume=MIN_VOLUME,
                max_tickers=1000,  # Force validate để áp dụng thiết lập mới
            )
        except Exception:
            logging.error("Lỗi khi lấy danh sách ticker", exc_info=True)
            # Fallback to a default list if loader fails
            from src.config.legacy_config import TICKERS

            return TICKERS[:MAX_SCAN_UNIVERSE]

    def sync_position_sizer_with_active_positions(self, active_positions: Dict):
        """Đồng bộ position_sizer với các vị thế đang hoạt động."""
        if not self.position_sizer:
            return
        self.position_sizer.current_positions = {}
        for symbol, pos in active_positions.items():
            if pos.get("shares", 0) > 0:
                # Positions in DB use 'avg_price' as entry price
                entry_price = pos.get("avg_price", 0)
                self.position_sizer.current_positions[symbol] = {
                    "shares": pos.get("shares", 0),
                    "entry_price": entry_price,
                    # Prefer last known price from metadata if available
                    "current_price": pos.get("metadata", {}).get("last_price", entry_price),
                    "unrealized_pnl": 0,
                }

    async def send_scan_start_message(self, current_tickers, market_regime):
        """Gửi thông báo bắt đầu quét qua Telegram."""
        try:
            regime_text = market_regime.get("regime", "UNKNOWN")
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"🔍 Đang quét {len(current_tickers)} mã...\n"
                f"Chế độ thị trường: *{regime_text}* (Confidence: {market_regime.get('confidence', 50)}%)\n"
                f"Số mã tiềm năng: *{len(current_tickers)}*",
                parse_mode="Markdown",
            )
        except Exception:
            logging.error("Lỗi gửi Telegram (scan start)", exc_info=True)

    async def check_active_positions(self, market_regime: Dict):
        """Kiểm tra và xử lý các vị thế đang nắm giữ."""
        positions = self.portfolio_manager.get_positions()
        if not positions:
            return

        logging.info(f"\n📊 Kiểm tra {len(positions)} vị thế đang nắm giữ...")

        tasks = [
            self._check_single_position(symbol, pos_data, market_regime)
            for symbol, pos_data in positions.items()
        ]
        await asyncio.gather(*tasks)

    async def _check_single_position(
        self,
        symbol: str,
        pos_data: Dict,
        market_regime: Dict,
    ):
        """Logic kiểm tra cho một vị thế cụ thể."""
        try:
            df = load_data(symbol, lookback=LOOKBACK)
            if df.empty:
                return

            current_price = df.iloc[-1]["close"]

            # Cập nhật giá hiện tại vào metadata để portfolio phản ánh P&L theo thời gian thực
            try:
                self.portfolio_manager.update_position_price(symbol, float(current_price))
            except Exception:
                logging.debug(f"Không thể cập nhật last_price cho {symbol}")

            # ML analysis với enhanced error handling
            ml_signal = None
            if self._should_use_ml():
                try:
                    ml_signal = self.ml_generator.analyze(
                        df, index_df=self.vnindex_df, symbol=symbol
                    )
                    # Track successful ML analysis
                    if ml_signal is not None:
                        self._ml_success_count += 1
                except Exception as e:
                    # ENHANCEMENT: Detailed error logging with diagnostic info
                    import traceback

                    error_details = {
                        "symbol": symbol,
                        "error_type": type(e).__name__,
                        "error_msg": str(e),
                        "context": "exit_check",
                    }
                    logging.error(
                        f"❌ ML analysis failed (exit check) for {symbol}: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    logging.debug(f"ML error traceback for {symbol}:\n{traceback.format_exc()}")

                    # Track ML failure for monitoring
                    self._track_ml_failure(symbol, error_details)

                    # Check circuit breaker after tracking failure
                    self._check_ml_circuit_breaker()

                    # Tiếp tục với ml_signal = None

            exit_decision = self.exit_strategy.check_exit(
                symbol=symbol,
                # Positions stored in DB expose 'avg_price' as the effective entry price
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
                await self.execute_exit(symbol, pos_data, exit_decision, current_price)

        except Exception:
            logging.error(f"Lỗi khi kiểm tra vị thế {symbol}", exc_info=True)

    async def execute_exit(self, symbol, pos_data, exit_decision, current_price):
        """Thực hiện thoát lệnh bán dựa trên quyết định thoát."""
        try:
            # Gửi thông báo thoát lệnh
            msg = self.exit_strategy.format_exit_message(symbol, exit_decision)
            await self.bot.send_message(self.chat_id, msg, parse_mode="Markdown")

            # Ghi nhận kết quả giao dịch vào Circuit Breaker
            # Use avg_price from DB as entry price
            pnl = (current_price - pos_data["avg_price"]) * pos_data["shares"]
            self.circuit_breaker.record_trade(pnl)

            # Get exit reason safely (can be None)
            exit_reason_str = (
                exit_decision.exit_reason.value
                if exit_decision.exit_reason
                else exit_decision.message
            )

            success, sell_msg, _ = self.paper_account.execute_sell(
                symbol=symbol,
                price=current_price,
                exit_type=exit_decision.exit_type,
                reason=exit_reason_str,
            )
            if success:
                logging.info(f"✅ Giao dịch bán được thực thi: {sell_msg}")

                # CRITICAL: Clear tracking if position no longer exists or shares = 0
                # This handles:
                # 1. FULL exits (obvious)
                # 2. Multiple partial exits that reduce shares to 0 (memory leak fix)
                # 3. Any edge cases where position is closed but exit_type != "FULL"
                updated_positions = self.portfolio_manager.get_positions()
                position_still_exists = (
                    symbol in updated_positions and updated_positions[symbol].get("shares", 0) > 0
                )

                if not position_still_exists:
                    self.exit_strategy.clear_position_tracking(symbol)
                    logging.debug(f"🧹 Cleared tracking for {symbol} (position fully closed)")

                # GHI NHẬN PNL NGAY LẬP TỨC sau khi thoát lệnh
                current_pnl = self.portfolio_manager.get_daily_pnl_pct()
                self.circuit_breaker.record_pnl(current_pnl)

                # Kiểm tra xem circuit breaker có kích hoạt không
                if self.circuit_breaker.is_active():
                    logging.warning(
                        f"⚠️ Circuit breaker đã kích hoạt sau khi thoát {symbol}. PnL: {current_pnl:.2%}"
                    )
                    await self.bot.send_message(
                        self.chat_id,
                        "🚨 *CIRCUIT BREAKER KÍCH HOẠT*\n\n"
                        f"Sau khi thoát {symbol}\n"
                        f"PnL hiện tại: {current_pnl:.2%}\n"
                        f"Lý do: {self.circuit_breaker.tripped_reason}",
                        parse_mode="Markdown",
                    )
            else:
                logging.error(f"❌ Lỗi thực thi lệnh bán cho {symbol}: {sell_msg}")
                await self.bot.send_message(self.chat_id, f"❌ Lỗi bán {symbol}: {sell_msg}")

        except Exception:
            logging.error(f"Lỗi khi thực hiện thoát lệnh {symbol}", exc_info=True)

    async def scan_for_new_entries(self, current_tickers, existing_symbols, market_regime):
        """Quét song song để tìm các tín hiệu vào lệnh mới."""
        signal_count = 0
        watchlist_candidates = []
        no_signal_symbols = []
        no_signal_reasons = {}
        results_lock = asyncio.Lock()

        async def _scan_ticker(symbol: str):
            nonlocal signal_count
            try:
                # Skip only if pending (being processed)
                if self.portfolio_lock.is_pending(symbol):
                    return

                entry_result = await self.process_single_ticker_for_entry(symbol, market_regime)

                if entry_result:
                    if entry_result.get("signal"):
                        async with results_lock:
                            signal_count += 1
                    elif entry_result.get("warnings"):
                        async with results_lock:
                            no_signal_symbols.append(entry_result["symbol"])
                            no_signal_reasons[entry_result["symbol"]] = entry_result["warnings"]
                    elif entry_result.get("is_watchlist"):
                        async with results_lock:
                            watchlist_candidates.append(entry_result)

            except Exception as e:
                logging.error(f"Lỗi nghiêm trọng khi quét mã {symbol}: {str(e)}", exc_info=True)
                async with results_lock:
                    no_signal_symbols.append(symbol)
                    no_signal_reasons[symbol] = [f"Lỗi khi quét: {str(e)}"]

        tasks = [_scan_ticker(symbol) for symbol in current_tickers]
        await asyncio.gather(*tasks)

        # Gửi thông báo tổng hợp nếu không có tín hiệu nào
        if signal_count == 0 and no_signal_symbols:
            await self._send_no_signal_summary(
                current_tickers, no_signal_symbols, no_signal_reasons
            )

        return signal_count, watchlist_candidates

    async def _send_no_signal_summary(self, all_tickers, no_signal_symbols, no_signal_reasons):
        """Gửi thông báo tổng hợp khi không tìm thấy tín hiệu mua nào."""
        try:
            # Nhóm các lý do tương tự lại với nhau
            reason_counts = {}
            for symbol, reasons in no_signal_reasons.items():
                for reason in reasons:
                    # Làm sạch lý do để nhóm các lý do tương tự
                    clean_reason = reason.split("(")[0].strip() if "(" in reason else reason
                    clean_reason = (
                        clean_reason.split(":")[0].strip() if ":" in clean_reason else clean_reason
                    )
                    reason_counts[clean_reason] = reason_counts.get(clean_reason, 0) + 1

            # Tạo thông báo tổng hợp
            summary = "🔍 *TỔNG HỢP KHÔNG TÌM THẤY TÍN HIỆU MUA*\n"
            summary += f"📊 Đã quét: {len(all_tickers)} mã\n"
            summary += f"📉 Không tìm thấy tín hiệu: {len(no_signal_symbols)} mã\n\n"

            # Thêm chi tiết theo nguyên nhân
            summary += "*CHI TIẾT THEO NGUYÊN NHÂN:*\n"
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(no_signal_symbols)) * 100
                summary += f"• {reason}: {count} mã ({percentage:.1f}%)\n"

            # Thêm ví dụ cho các lý do phổ biến
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

            # Gửi thông báo qua Telegram bot
            if self.bot and self.chat_id:
                await self.bot.send_message(self.chat_id, summary, parse_mode="Markdown")
                logging.info("✅ Đã gửi thông báo tổng hợp không tìm thấy tín hiệu")
            else:
                logging.warning("⚠️ Không thể gửi thông báo: bot hoặc chat_id không khả dụng")

        except Exception as e:
            logging.error(f"Lỗi khi gửi thông báo tổng hợp không tín hiệu: {str(e)}", exc_info=True)

    async def process_single_ticker_for_entry(self, symbol: str, market_regime: dict):
        """
        Xử lý logic để tìm tín hiệu vào lệnh cho một mã cổ phiếu.
        Bao gồm: Lấy dữ liệu, phân tích ML, kiểm tra entry logic, phân tích tin tức,
                 tính toán position size và gửi thông báo.
        Returns:
            - dict: Chứa 'signal': True nếu có tín hiệu, hoặc 'warnings' nếu không có tín hiệu
        """
        try:
            # Lấy dữ liệu
            df = load_data(symbol, lookback=LOOKBACK)
            if df.empty or len(df) < 50:  # Cần đủ dữ liệu để phân tích
                return {
                    "symbol": symbol,
                    "warnings": ["Không đủ dữ liệu lịch sử"],
                    "is_watchlist": False,
                }

            # Phân tích ML với enhanced error handling
            ml_signal = None
            if self._should_use_ml():
                try:
                    ml_signal = self.ml_generator.analyze(
                        df, index_df=self.vnindex_df, symbol=symbol
                    )
                    # Track successful ML analysis
                    if ml_signal is not None:
                        self._ml_success_count += 1
                except Exception as e:
                    # ENHANCEMENT: Detailed error logging with diagnostic info
                    import traceback

                    error_details = {
                        "symbol": symbol,
                        "error_type": type(e).__name__,
                        "error_msg": str(e),
                        "df_shape": df.shape if df is not None else None,
                        "df_columns": list(df.columns) if df is not None else None,
                    }
                    logging.error(
                        f"❌ ML analysis failed for {symbol}: {type(e).__name__}: {str(e)}\n"
                        f"   Details: df_shape={error_details['df_shape']}, "
                        f"df_columns={len(error_details['df_columns']) if error_details['df_columns'] else 0}"
                    )
                    logging.debug(f"ML error traceback for {symbol}:\n{traceback.format_exc()}")

                    # Track ML failure for monitoring
                    self._track_ml_failure(symbol, error_details)

                    # Check circuit breaker after tracking failure
                    self._check_ml_circuit_breaker()

                    # Tiếp tục với ml_signal = None, entry_logic sẽ xử lý technical fallback

            # 1. Entry Logic with validation
            if not self.entry_logic:
                logging.error("❌ Entry logic not initialized")
                return {
                    "symbol": symbol,
                    "warnings": ["Lỗi: Entry logic chưa được khởi tạo"],
                    "is_watchlist": False,
                }

            entry_signal = self.entry_logic.analyze_entry(
                df=df,
                ml_signal=ml_signal,
                market_regime=market_regime,
            )

            # Validate entry signal
            if not entry_signal or not entry_signal.should_enter:
                warnings = getattr(entry_signal, "warnings", ["Không rõ lý do"])
                return {"symbol": symbol, "warnings": warnings, "is_watchlist": False}

            # Track signal generation (ML vs Technical)
            is_ml_signal = getattr(entry_signal, "telemetry", {}).get("signal_source") == "ml"
            self.signal_tracker.track_signal(is_ml_signal=is_ml_signal)

            # 2. News Analysis (nếu có tín hiệu)
            news_sentiment = {"score": 0.5, "comment": "Neutral"}
            if entry_signal.should_enter:
                # news_sentiment = self.news_analyzer.analyze(symbol)
                pass  # Tạm thời bỏ qua

            # 3. Position Sizing
            if entry_signal.should_enter:
                # Get current positions (read once)
                current_positions = self.portfolio_manager.get_positions()

                # Filter out positions with invalid shares
                active_positions = {
                    sym: pos for sym, pos in current_positions.items() if pos.get("shares", 0) > 0
                }

                # Check if symbol already has an active position
                if symbol in active_positions:
                    logging.info(
                        f"ℹ️ [{symbol}] Đã có position active trong portfolio, "
                        f"vẫn gửi notification nhưng không mua thêm."
                    )
                    # Skip to notification section below (symbol_has_position will be True)

                # Get config for capital calculation
                from src.config.trading_config import get_config

                config = get_config(validate=False)
                max_positions = config.trading.max_positions

                # Get take_profit from entry signal (use first target if available)
                take_profit_price = (
                    entry_signal.take_profit_targets[0]
                    if entry_signal.take_profit_targets
                    else entry_signal.entry_price * 1.1  # Default 10% gain
                )

                # Calculate position size using EnhancedPositionSizer
                position_size_info = self.position_sizer.calculate_position_size(
                    symbol=symbol,
                    entry_price=entry_signal.entry_price,
                    stop_loss=entry_signal.stop_loss,  # Fixed: was stop_loss_price
                    take_profit=take_profit_price,
                    confidence=entry_signal.confidence,
                    signal_strength=entry_signal.strength.name,
                    market_regime=market_regime,
                    # Optional: portfolio_risk, win_rate, avg_win_loss_ratio
                )

                # 4. Paper Trade & Notification
                # If symbol already has position, skip buying but still send notification
                if symbol in active_positions:
                    # Đã có position - chỉ gửi notification, không mua thêm
                    logging.info(
                        f"📢 [{symbol}] Gửi notification tín hiệu mua "
                        f"(đã có position, không mua thêm)"
                    )

                    # Gửi thông báo Telegram
                    await self.send_buy_signal_notification(
                        symbol, entry_signal, position_size_info, news_sentiment
                    )
                    return {"signal": True, "skipped_buy": True}

                # Normal flow: Execute buy for new positions
                # CRITICAL FIX: Use atomic context manager to prevent race conditions
                if position_size_info and position_size_info.shares > 0:
                    total_capital = config.trading.total_capital

                    # ATOMIC POSITION ADD: Automatically handles reserve → confirm/cancel
                    with self.portfolio_lock.atomic_position_add(
                        symbol=symbol,
                        position_value=position_size_info.value,
                        total_capital=total_capital,
                        current_positions=active_positions,
                    ) as (can_add, reason):
                        if not can_add:
                            logging.info(f"⚠️ [{symbol}] Cannot add position: {reason}")
                            return None

                        # Thực hiện paper trade (BUY)
                        take_profit = (
                            entry_signal.take_profit_targets[0]
                            if entry_signal.take_profit_targets
                            else None
                        )

                        # Prepare metadata with signal source for performance tracking
                        trade_metadata = {
                            "signal_source": "ml" if is_ml_signal else "technical",
                            "confidence": entry_signal.confidence,
                            "signal_reason": ", ".join(entry_signal.reasons),
                        }

                        # ENHANCEMENT: Pass limit order info if applicable
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
                            logging.error(f"❌ Paper trade failed for {symbol}: {message}")
                            # Raise exception to auto-cancel reservation via context manager
                            raise Exception(f"Paper trade failed: {message}")

                        # Success - context manager will auto-confirm
                        logging.info(f"✅ Paper trade successful: {message}")

                        # Gửi thông báo Telegram
                        await self.send_buy_signal_notification(
                            symbol, entry_signal, position_size_info, news_sentiment
                        )
                        return {"signal": True}

            # 5. Watchlist Logic
            # ... (logic để thêm vào watchlist nếu không phải tín hiệu mua)
            return None

        except DataLoadError:
            return {"symbol": symbol, "warnings": ["Lỗi tải dữ liệu"], "is_watchlist": False}
        except Exception as e:
            error_msg = f"Lỗi không xác định: {str(e)}"
            logging.error(f"[{symbol}] {error_msg}", exc_info=True)
            return {"symbol": symbol, "warnings": [error_msg], "is_watchlist": False}

    async def send_buy_signal_notification(
        self, symbol, entry_signal, position_size_info, news_sentiment
    ):
        """Gửi thông báo tín hiệu mua qua Telegram."""
        # Tính toán R:R cho mục tiêu đầu tiên
        try:
            # Đảm bảo take_profit_targets không rỗng và stop_loss khác entry_price
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

        # Format confidence với emoji dựa trên mức độ
        confidence_emoji = (
            "🟢"
            if entry_signal.confidence >= 70
            else "🟡" if entry_signal.confidence >= 50 else "🔴"
        )

        message = (
            "**🚀 TÍN HIỆU MUA MỚI 🚀**\n\n"
            f"**Mã:** `{symbol}`\n"
            f"**Độ tin cậy:** `{entry_signal.confidence}%` {confidence_emoji}\n"
            f"**Giá vào:** `{entry_signal.entry_price:,.0f}`\n"
            f"**Mục tiêu 1:** `{tp1_target:,.0f}`\n"
            f"**Dừng lỗ:** `{entry_signal.stop_loss:,.0f}`\n"
            f"**R:R (TP1):** `{risk_reward_ratio:.2f}`\n\n"
            f"**Lý do:** {', '.join(entry_signal.reasons)}\n\n"
            "**--- Quản lý vốn ---**\n"
            f"**Số CP mua:** `{position_size_info.shares}`\n"  # Fixed: was position_size_info['shares_to_buy']
            f"**Giá trị lệnh:** `{position_size_info.value:,.0f} VNĐ`\n"  # Fixed: was position_size_info['trade_value']
            f"**Rủi ro lệnh:** `{position_size_info.risk_amount:,.0f} VNĐ` "  # Fixed: was position_size_info['risk_per_trade']
            f"({position_size_info.risk_percent:.2%})\n\n"  # Fixed: was position_size_info['risk_pct_of_capital']
            # "**--- Tin tức ---**\n"
            # f"**Sentiment:** {news_sentiment['comment']} ({news_sentiment['score']:.2f})\n"
        )
        await self.bot.send_message(self.chat_id, message, parse_mode="Markdown")

    async def perform_post_scan_risk_analysis(self):
        """Thực hiện phân tích rủi ro sau khi quét."""
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
                await self.bot.send_message(
                    self.chat_id,
                    f"⚠️ *PORTFOLIO RISK ALERT*\n\n{risk_summary}",
                    parse_mode="Markdown",
                )
        except Exception:
            logging.error("Lỗi portfolio risk analysis", exc_info=True)

    async def send_summary_report(self, signal_count, watchlist_candidates, market_regime: Dict):
        """Gửi báo cáo tóm tắt cuối phiên quét."""
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

            await self.bot.send_message(self.chat_id, summary_msg, parse_mode="Markdown")
        except Exception:
            logging.error("Lỗi gửi báo cáo tóm tắt", exc_info=True)
            await self.bot.send_message(self.chat_id, "Lỗi khi tạo báo cáo")

    def _track_ml_failure(self, symbol: str, error_details: dict):
        """
        Track ML analysis failures for monitoring and debugging

        Args:
            symbol: Stock symbol that failed
            error_details: Dict with error information
        """
        self._ml_failure_count += 1

        # Track by error type
        error_type = error_details.get("error_type", "Unknown")
        self._ml_failures_by_error[error_type] = self._ml_failures_by_error.get(error_type, 0) + 1

        # Track by symbol
        self._ml_failures_by_symbol[symbol] = self._ml_failures_by_symbol.get(symbol, 0) + 1

        # Log summary periodically (every 10 failures)
        if self._ml_failure_count % 10 == 0:
            failure_rate = self._get_ml_failure_rate()
            top_errors = sorted(
                self._ml_failures_by_error.items(), key=lambda x: x[1], reverse=True
            )[:3]
            top_symbols = sorted(
                self._ml_failures_by_symbol.items(), key=lambda x: x[1], reverse=True
            )[:3]

            logging.warning(
                f"📊 ML Failure Summary:\n"
                f"   Total failures: {self._ml_failure_count}\n"
                f"   Total successes: {self._ml_success_count}\n"
                f"   Failure rate: {failure_rate:.1%}\n"
                f"   Top errors: {', '.join(f'{err}({cnt})' for err, cnt in top_errors)}\n"
                f"   Top failing symbols: {', '.join(f'{sym}({cnt})' for sym, cnt in top_symbols)}"
            )

    def _get_ml_failure_rate(self) -> float:
        """Calculate ML failure rate for monitoring"""
        total = self._ml_failure_count + self._ml_success_count
        return self._ml_failure_count / total if total > 0 else 0.0

    def _check_ml_circuit_breaker(self):
        """
        Check and update ML circuit breaker status.

        Logic:
        1. If failure rate > threshold AND min_samples met → DISABLE ML
        2. If failure rate < recovery threshold AND circuit active → RE-ENABLE ML
        3. Send alerts on status change
        """
        total_attempts = self._ml_failure_count + self._ml_success_count

        # Need minimum samples before activating circuit breaker
        if total_attempts < self._ml_circuit_breaker_min_samples:
            return

        failure_rate = self._get_ml_failure_rate()

        # Check if should TRIP circuit breaker (disable ML)
        if not self._ml_circuit_breaker_active and failure_rate >= self._ml_circuit_breaker_threshold:
            self._ml_circuit_breaker_active = True
            self._ml_enabled = False

            alert_msg = (
                f"🚨 ML CIRCUIT BREAKER ACTIVATED 🚨\n\n"
                f"Failure rate: {failure_rate:.1%} (threshold: {self._ml_circuit_breaker_threshold:.1%})\n"
                f"Total failures: {self._ml_failure_count}/{total_attempts}\n\n"
                f"🔧 Switching to TECHNICAL ANALYSIS only\n"
                f"ML will auto-recover when failure rate drops below {self._ml_recovery_threshold:.1%}"
            )

            logging.critical(alert_msg)

            # Send alert via Telegram if available
            if self.bot and self.chat_id:
                import asyncio

                try:
                    asyncio.create_task(self.bot.send_message(self.chat_id, alert_msg, parse_mode="Markdown"))
                except Exception as e:
                    logging.error(f"Failed to send ML circuit breaker alert: {e}")

        # Check if should RECOVER (re-enable ML)
        elif self._ml_circuit_breaker_active and failure_rate <= self._ml_recovery_threshold:
            self._ml_circuit_breaker_active = False
            self._ml_enabled = True

            recovery_msg = (
                f"✅ ML CIRCUIT BREAKER RECOVERED\n\n"
                f"Failure rate improved: {failure_rate:.1%} (recovery threshold: {self._ml_recovery_threshold:.1%})\n"
                f"Total failures: {self._ml_failure_count}/{total_attempts}\n\n"
                f"🤖 ML analysis RE-ENABLED"
            )

            logging.info(recovery_msg)

            # Send recovery alert
            if self.bot and self.chat_id:
                import asyncio

                try:
                    asyncio.create_task(self.bot.send_message(self.chat_id, recovery_msg, parse_mode="Markdown"))
                except Exception as e:
                    logging.error(f"Failed to send ML recovery alert: {e}")

    def _should_use_ml(self) -> bool:
        """
        Determine if ML analysis should be used.

        Returns False if:
        1. ML circuit breaker is active (too many failures)
        2. USE_ML_ANALYSIS env var is false
        """
        if not self._ml_enabled:
            return False

        use_ml_env = os.getenv("USE_ML_ANALYSIS", "true").lower() == "true"
        return use_ml_env
