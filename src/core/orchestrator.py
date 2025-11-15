# -*- coding: utf-8 -*-
"""
Orchestrator for the trading bot.
Tách logic điều phối ra khỏi bot_runner_improved.py
"""
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
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
from src.config.legacy_config import LOOKBACK, MAX_SCAN_UNIVERSE


class TradingOrchestrator:
    """
    Lớp điều phối chính, quản lý toàn bộ luồng quét và giao dịch.
    """

    def __init__(
        self, bot_instance: Bot, chat_id: str, vnindex_df: Optional[pd.DataFrame] = None
    ):
        self.bot = bot_instance
        self.chat_id = chat_id
        self.vnindex_df = vnindex_df  # Lưu trữ vnindex_df ngay khi khởi tạo
        self.portfolio_manager = get_portfolio_manager()
        self.portfolio_risk_manager = get_portfolio_risk_manager(
            total_capital=100_000_000
        )
        self.portfolio_lock = get_portfolio_lock()
        self.market_analyzer = ProxyMarketRegimeAnalyzer()
        self.ticker_loader = get_ticker_loader()
        self.ml_generator = MLSignalGenerator()
        self.paper_account = get_paper_account()
        self.ml_monitor = get_ml_model_monitor()
        self.strategy_manager = get_strategy_manager()
        self.circuit_breaker = get_circuit_breaker()

        # Các thành phần chiến lược sẽ được lấy từ StrategyManager
        self.entry_logic: Optional[Any] = None
        self.position_sizer: Optional[Any] = None
        self.exit_strategy: Optional[Any] = None

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
            "dividend" in article.get("topics", [])
            for article in news_context.get("articles", [])
        )

        # ENHANCED IMPACT
        if news_sentiment >= 0.8:
            entry_signal.confidence = min(100, entry_signal.confidence + 15)
            entry_signal.reasons.append(
                f"📰 Tin tức RẤT tích cực ({news_sentiment:+.2f})"
            )
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
            (entry_signal.stop_loss - entry_signal.entry_price)
            / entry_signal.entry_price
            * 100
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

        msg += (
            f"\n💸 *Risk:* {position.max_loss:,.0f} VNĐ ({position.risk_percent:.2f}%)"
        )

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
        # 0. KIỂM TRA NGẮT MẠCH (CIRCUIT BREAKER)
        vnindex_change = (
            self.vnindex_df["close"].pct_change().iloc[-1]
            if self.vnindex_df is not None and not self.vnindex_df.empty
            else 0.0
        )
        # Lấy PNL thực tế từ portfolio manager nếu có
        current_pnl_pct = self.portfolio_manager.get_daily_pnl_pct()

        if self.circuit_breaker.check_and_update(
            portfolio_pnl_pct=current_pnl_pct, vnindex_change_pct=vnindex_change
        ):
            reason = self.circuit_breaker.tripped_reason
            logging.critical(
                f"🚨 NGẮT MẠCH ĐANG KÍCH HOẠT: {reason}. Dừng mọi lệnh mua mới."
            )
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
        await self.send_summary_report(
            signal_count, watchlist_candidates, market_regime
        )

    def get_scan_universe(self) -> List[str]:
        """Lấy danh sách các mã cổ phiếu cần quét."""
        try:
            return self.ticker_loader.get_validated_tickers(
                force_validate=False, min_volume=100_000, max_tickers=MAX_SCAN_UNIVERSE
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
                entry_price = pos.get("entry_price", 0)
                self.position_sizer.current_positions[symbol] = {
                    "shares": pos.get("shares", 0),
                    "entry_price": entry_price,
                    "current_price": pos.get("current_price", entry_price),
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
            df = load_data(symbol, LOOKBACK)
            if df.empty:
                return

            current_price = df.iloc[-1]["close"]

            # ML analysis với error handling
            ml_signal = None
            try:
                ml_signal = self.ml_generator.analyze(df, index_df=self.vnindex_df)
            except Exception:
                logging.warning(f"⚠️ Lỗi ML analysis cho {symbol} (exit check)")
                # Tiếp tục với ml_signal = None

            exit_decision = self.exit_strategy.check_exit(
                symbol=symbol,
                entry_price=pos_data["entry_price"],
                current_price=current_price,
                stop_loss=pos_data.get("stop_loss"),
                take_profit_targets=pos_data.get("take_profit_targets", []),
                entry_date=datetime.fromisoformat(pos_data["entry_date"]),
                df=df,
                ml_signal=ml_signal,
                market_regime=market_regime,
                partial_exits=pos_data.get("partial_exits", []),
            )

            if exit_decision and exit_decision["reason"]:
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
            pnl = (current_price - pos_data["entry_price"]) * pos_data["shares"]
            self.circuit_breaker.record_trade(pnl)

            success, sell_msg, _ = self.paper_account.execute_sell(
                symbol=symbol,
                price=current_price,
                exit_type=exit_decision.exit_type,
                reason=exit_decision.exit_reason.value,
            )
            if success:
                logging.info(f"✅ Giao dịch bán được thực thi: {sell_msg}")
                if exit_decision.exit_type == "FULL":
                    self.exit_strategy.clear_position_tracking(symbol)

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
                await self.bot.send_message(
                    self.chat_id, f"❌ Lỗi bán {symbol}: {sell_msg}"
                )

        except Exception:
            logging.error(f"Lỗi khi thực hiện thoát lệnh {symbol}", exc_info=True)

    async def scan_for_new_entries(
        self, current_tickers, existing_symbols, market_regime
    ):
        """Quét song song để tìm các tín hiệu vào lệnh mới."""
        signal_count = 0
        watchlist_candidates = []
        results_lock = asyncio.Lock()

        async def _scan_ticker(symbol: str):
            nonlocal signal_count
            try:
                if symbol in existing_symbols or self.portfolio_lock.is_pending(symbol):
                    return

                # Logic xử lý được chuyển hết vào process_single_ticker_for_entry
                entry_result = await self.process_single_ticker_for_entry(
                    symbol, market_regime
                )

                if entry_result and entry_result.get("signal"):
                    async with results_lock:
                        signal_count += 1
                elif entry_result and entry_result.get("is_watchlist"):
                    async with results_lock:
                        watchlist_candidates.append(entry_result)

            except Exception:
                logging.error(
                    f"Lỗi nghiêm trọng khi quét mã {symbol}", exc_info=True
                )

        tasks = [_scan_ticker(symbol) for symbol in current_tickers]
        await asyncio.gather(*tasks)
        return signal_count, watchlist_candidates

    async def process_single_ticker_for_entry(self, symbol: str, market_regime: dict):
        """
        Xử lý logic để tìm tín hiệu vào lệnh cho một mã cổ phiếu.
        Bao gồm: Lấy dữ liệu, phân tích ML, kiểm tra entry logic, phân tích tin tức,
                 tính toán position size và gửi thông báo.
        """
        try:
            # Lấy dữ liệu
            df = load_data(symbol, lookback=LOOKBACK)
            if df.empty or len(df) < 50:  # Cần đủ dữ liệu để phân tích
                return None

            # Phân tích ML với error handling
            ml_signal = None
            try:
                ml_signal = self.ml_generator.analyze(symbol, df, self.vnindex_df)
            except Exception:
                logging.warning(f"⚠️ Lỗi ML analysis cho {symbol} (entry scan)")
                # Tiếp tục với ml_signal = None, entry_logic sẽ xử lý

            # 1. Entry Logic
            entry_signal = self.entry_logic.analyze_entry(
                df=df,
                ml_signal=ml_signal,
                market_regime=market_regime,
            )

            # 2. News Analysis (nếu có tín hiệu)
            news_sentiment = {"score": 0.5, "comment": "Neutral"}
            if entry_signal and entry_signal.should_enter:
                # news_sentiment = self.news_analyzer.analyze(symbol)
                pass  # Tạm thời bỏ qua

            # 3. Position Sizing
            if entry_signal and entry_signal.should_enter:
                # Kiểm tra xem có đủ vốn không
                if not self.portfolio_risk_manager.can_open_new_position():
                    logging.warning(
                        f"Bỏ qua tín hiệu {symbol} do đã đạt giới hạn rủi ro/số vị thế."
                    )
                    return None

                position_size_info = self.position_sizer.calculate_position_size(
                    symbol=symbol,
                    entry_price=entry_signal.entry_price,
                    stop_loss_price=entry_signal.stop_loss,
                    risk_appetite=market_regime.get("confidence", 50) / 100.0,
                    # news_sentiment=news_sentiment["score"],
                )

                # 4. Paper Trade & Notification
                if position_size_info and position_size_info["shares_to_buy"] > 0:
                    # Đánh dấu mã này đang chờ xử lý để tránh quét lại
                    self.portfolio_lock.add_pending(symbol)

                    # Thực hiện paper trade
                    self.paper_account.execute_trade(
                        symbol=symbol,
                        action="BUY",
                        shares=position_size_info["shares_to_buy"],
                        price=entry_signal.entry_price,
                        reason=", ".join(entry_signal.reasons),
                    )

                    # Gửi thông báo Telegram
                    await self.send_buy_signal_notification(
                        symbol, entry_signal, position_size_info, news_sentiment
                    )
                    return {"signal": True}

            # 5. Watchlist Logic
            # ... (logic để thêm vào watchlist nếu không phải tín hiệu mua)
            return None
        except DataLoadError:
            # logging.warning(f"[{symbol}] Lỗi tải dữ liệu") # Giảm log nhiễu
            return None
        except Exception:
            logging.error(
                f"[{symbol}] Lỗi không xác định trong process_single_ticker_for_entry"
            )
            import traceback

            logging.error(traceback.format_exc())
            return None

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

        tp1_target = (
            entry_signal.take_profit_targets[0]
            if entry_signal.take_profit_targets
            else 0
        )

        message = (
            "**🚀 TÍN HIỆU MUA MỚI 🚀**\n\n"
            f"**Mã:** `{symbol}`\n"
            f"**Giá vào:** `{entry_signal.entry_price:,.0f}`\n"
            f"**Mục tiêu 1:** `{tp1_target:,.0f}`\n"
            f"**Dừng lỗ:** `{entry_signal.stop_loss:,.0f}`\n"
            f"**R:R (TP1):** `{risk_reward_ratio:.2f}`\n\n"
            f"**Lý do:** {', '.join(entry_signal.reasons)}\n\n"
            "**--- Quản lý vốn ---**\n"
            f"**Số CP mua:** `{position_size_info['shares_to_buy']}`\n"
            f"**Giá trị lệnh:** `{position_size_info['trade_value']:,.0f} VNĐ`\n"
            f"**Rủi ro lệnh:** `{position_size_info['risk_per_trade']:,.0f} VNĐ` "
            f"({position_size_info['risk_pct_of_capital']:.2%})\n\n"
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
            risk_metrics = self.portfolio_risk_manager.calculate_portfolio_risk(
                risk_positions
            )
            if risk_metrics.risk_status in ["HIGH", "CRITICAL"]:
                risk_summary = self.portfolio_risk_manager.get_risk_summary(
                    risk_positions
                )
                await self.bot.send_message(
                    self.chat_id,
                    f"⚠️ *PORTFOLIO RISK ALERT*\n\n{risk_summary}",
                    parse_mode="Markdown",
                )
        except Exception:
            logging.error("Lỗi portfolio risk analysis", exc_info=True)

    async def send_summary_report(
        self, signal_count, watchlist_candidates, market_regime: Dict
    ):
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

            await self.bot.send_message(
                self.chat_id, summary_msg, parse_mode="Markdown"
            )
        except Exception:
            logging.error("Lỗi gửi báo cáo tóm tắt", exc_info=True)
            await self.bot.send_message(self.chat_id, "Lỗi khi tạo báo cáo")
