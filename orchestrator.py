# -*- coding: utf-8 -*-
"""
Orchestrator for the trading bot.
Tách logic điều phối ra khỏi bot_runner_improved.py
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from telegram import Bot

# Import các thành phần cần thiết từ project
# (Giả định các import này vẫn hoạt động sau khi tách file)
from config import LOOKBACK, MAX_SCAN_UNIVERSE, WATCHLIST_SIZE
from data_loader import load_data
from ml_signals import MLSignalGenerator
from improved_entry_logic import ImprovedEntryLogic
from position_sizing_enhanced import EnhancedPositionSizer
from exit_strategy_enhanced import EnhancedExitStrategy
from news_analyzer import analyze_news_trend
from portfolio_manager import get_portfolio_manager
from portfolio_lock import get_portfolio_lock
from portfolio_risk_manager import get_portfolio_risk_manager
from market_regime_proxy import ProxyMarketRegimeAnalyzer
from ticker_loader import get_ticker_loader
from paper_trading import get_paper_account
from ml_model_monitor import get_ml_model_monitor
from exceptions import DataLoadError, DataQualityError


class TradingOrchestrator:
    """
    Lớp điều phối chính, quản lý toàn bộ luồng quét và giao dịch.
    """

    def __init__(self, bot_instance: Bot, chat_id: str):
        self.bot = bot_instance
        self.chat_id = chat_id
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

        # Các thành phần có thể được thay thế bởi StrategyManager
        self.entry_logic: Optional[ImprovedEntryLogic] = None
        self.position_sizer: Optional[EnhancedPositionSizer] = None
        self.exit_strategy: Optional[EnhancedExitStrategy] = None

    def set_strategies(self, entry_logic, position_sizer, exit_strategy):
        """Gán các đối tượng chiến lược từ bên ngoài."""
        self.entry_logic = entry_logic
        self.position_sizer = position_sizer
        self.exit_strategy = exit_strategy

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
            msg += f"💵 *GIÁ VÀO (DCA):*\n"
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

        msg += f"🎯 *Take Profit:*\n"
        for i, tp in enumerate(entry_signal.take_profit_targets[:2], 1):
            tp_pct = ((tp - entry_signal.entry_price) / entry_signal.entry_price) * 100
            msg += f"  TP{i}: {tp:,.0f} (+{tp_pct:.1f}%)\n"

        if entry_signal.reasons:
            msg += f"\n✅ *Lý do:*\n"
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

    async def run_scan(self, market_regime: Dict, vnindex_df: Optional[pd.DataFrame]):
        """
        Chạy quá trình quét toàn diện để tìm kiếm cơ hội và quản lý vị thế.
        """
        if not all([self.entry_logic, self.position_sizer, self.exit_strategy]):
            logging.critical(
                "❌ Các chiến lược (entry, sizing, exit) chưa được thiết lập cho Orchestrator."
            )
            return

        # 1. Lấy danh sách các vị thế và mã cổ phiếu cần quét
        active_positions = self.portfolio_manager.get_positions()
        existing_symbols = set(active_positions.keys())
        self.sync_position_sizer_with_active_positions(active_positions)

        current_tickers = self.get_scan_universe()
        logging.info(f"🔍 Quét {len(current_tickers)} mã...")

        # 2. Gửi thông báo bắt đầu quét
        await self.send_scan_start_message(current_tickers, market_regime)

        # 3. KIỂM TRA THOÁT LỆNH TRƯỚC
        await self.check_active_positions(market_regime, vnindex_df)

        # 4. QUÉT TÌM LỆNH MUA MỚI (SONG SONG)
        logging.info(f"\n🔍 Quét {len(current_tickers)} mã để tìm cơ hội mua mới")
        logging.info(
            f"📊 Đang nắm giữ: {len(existing_symbols)} mã ({', '.join(existing_symbols) if existing_symbols else 'không có'})"
        )

        self.portfolio_lock.clear_pending()

        signal_count, watchlist_candidates = await self.scan_for_new_entries(
            current_tickers, existing_symbols, market_regime, vnindex_df
        )

        # 5. PHÂN TÍCH RỦI RO DANH MỤC SAU KHI QUÉT
        await self.perform_post_scan_risk_analysis()

        # 6. TỔNG KẾT
        await self.send_summary_and_watchlist(
            len(current_tickers), signal_count, market_regime, watchlist_candidates
        )

    def get_scan_universe(self) -> List[str]:
        """Lấy danh sách các mã cổ phiếu cần quét."""
        try:
            return self.ticker_loader.get_validated_tickers(
                force_validate=False, min_volume=100_000, max_tickers=MAX_SCAN_UNIVERSE
            )
        except Exception as e:
            logging.error(f"Lỗi khi lấy danh sách ticker: {e}", exc_info=True)
            # Fallback to a default list if loader fails
            from config import TICKERS

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

    async def send_scan_start_message(self, tickers: List[str], market_regime: Dict):
        """Gửi thông báo bắt đầu quét qua Telegram."""
        try:
            regime_text = market_regime.get("regime", "UNKNOWN")
            msg = f"🔍 Đang quét {len(tickers)} mã...\n📊 Market: {regime_text}"
            await self.bot.send_message(chat_id=self.chat_id, text=msg)
        except Exception as e:
            logging.error(f"Lỗi gửi Telegram (scan start): {e}", exc_info=True)

    async def check_active_positions(
        self, market_regime: Dict, vnindex_df: Optional[pd.DataFrame]
    ):
        """Kiểm tra và xử lý các vị thế đang nắm giữ."""
        positions = self.portfolio_manager.get_positions()
        if not positions:
            return

        logging.info(f"\n📊 Kiểm tra {len(positions)} vị thế đang nắm giữ...")

        tasks = [
            self._check_single_position(symbol, pos_data, market_regime, vnindex_df)
            for symbol, pos_data in positions.items()
        ]
        await asyncio.gather(*tasks)

    async def _check_single_position(
        self,
        symbol: str,
        pos_data: Dict,
        market_regime: Dict,
        vnindex_df: Optional[pd.DataFrame],
    ):
        """Logic kiểm tra cho một vị thế cụ thể."""
        try:
            df = load_data(symbol, LOOKBACK)
            if df.empty:
                return

            current_price = df.iloc[-1]["close"]
            ml_signal = self.ml_generator.analyze(df, index_df=vnindex_df)

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

            if exit_decision.should_exit:
                msg = self.exit_strategy.format_exit_message(symbol, exit_decision)
                await self.bot.send_message(self.chat_id, msg, parse_mode="Markdown")

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
                else:
                    logging.error(f"❌ Lỗi thực thi lệnh bán cho {symbol}: {sell_msg}")
                    await self.bot.send_message(
                        self.chat_id, f"❌ Lỗi bán {symbol}: {sell_msg}"
                    )

            if self.position_sizer:
                self.position_sizer.update_position_price(symbol, current_price)

        except Exception as e:
            logging.error(f"Lỗi check exit {symbol}: {e}", exc_info=True)

    async def scan_for_new_entries(
        self, current_tickers, existing_symbols, market_regime, vnindex_df
    ):
        """Quét song song để tìm các tín hiệu vào lệnh mới."""
        signal_count = 0
        watchlist_candidates = []
        results_lock = asyncio.Lock()

        async def _scan_ticker(symbol: str):
            nonlocal signal_count
            try:
                if symbol in existing_symbols:
                    return

                df = load_data(symbol, LOOKBACK)
                if df.empty or len(df) < 50:
                    return

                # --- Logic xử lý tín hiệu, entry, news, sizing ---
                # (Logic này được giữ nguyên từ bot_runner_improved.py)
                # ...
                # ... (Giả sử logic phức tạp nằm ở đây)
                # ...
                # Cuối cùng, nếu có tín hiệu:
                # async with results_lock:
                #     signal_count += 1
                #     ... (xử lý paper trade, gửi telegram)

                # Để đơn giản, phần logic phức tạp này sẽ được gọi từ một hàm helper
                entry_result = await self.process_single_ticker_for_entry(
                    symbol, df, market_regime, vnindex_df
                )

                if entry_result:
                    if entry_result["type"] == "signal":
                        async with results_lock:
                            # Re-check lock before execution
                            if symbol in existing_symbols:
                                if self.portfolio_lock:
                                    self.portfolio_lock.cancel_position(symbol)
                                return

                        success, paper_msg = self.execute_paper_buy(entry_result)
                        if success:
                            existing_symbols.add(symbol)
                            signal_count += 1
                            logging.info(
                                f"✅ {symbol}: {entry_result['signal'].signal_type} ({entry_result['signal'].confidence}%)"
                            )
                            await self.bot.send_message(
                                self.chat_id,
                                entry_result["message"],
                                parse_mode="Markdown",
                            )
                        else:
                            logging.error(
                                f"❌ Paper trade failed for {symbol}: {paper_msg}"
                            )

                    elif entry_result["type"] == "watchlist":
                        async with results_lock:
                            watchlist_candidates.append(entry_result["data"])

            except (DataLoadError, DataQualityError) as e:
                logging.warning(f"⚠️ {symbol}: {e.message}")
            except Exception as e:
                logging.error(f"Lỗi quét {symbol}: {e}", exc_info=True)

        tasks = [_scan_ticker(symbol) for symbol in current_tickers]
        await asyncio.gather(*tasks)
        return signal_count, watchlist_candidates

    async def process_single_ticker_for_entry(
        self, symbol, df, market_regime, vnindex_df
    ):
        """Xử lý logic đầy đủ cho một mã để tìm tín hiệu vào lệnh."""
        # 1. Get ML Signal & Entry Logic
        ml_signal = self.ml_generator.analyze(df, index_df=vnindex_df)
        if self.ml_monitor:
            self.ml_monitor.record_prediction(
                symbol=symbol,
                predicted_signal=ml_signal.get("signal", "HOLD"),
                predicted_confidence=ml_signal.get(
                    "raw_confidence", ml_signal.get("confidence", 0)
                ),
                model_version="default",
            )

        entry_signal = self.entry_logic.analyze_entry(
            df=df, ml_signal=ml_signal, market_regime=market_regime
        )

        # 2. News Integration
        news_context = analyze_news_trend(symbol) if analyze_news_trend else None
        if news_context:
            entry_signal = self.adjust_signal_with_news(entry_signal, news_context)

        # 3. Check if should enter
        if not entry_signal.should_enter:
            confidence_for_watchlist = max(
                ml_signal.get("confidence", 0), entry_signal.confidence
            )
            if self.entry_logic and confidence_for_watchlist >= max(
                0, self.entry_logic.min_confidence - 5
            ):
                reason = (
                    ", ".join(entry_signal.warnings)
                    if entry_signal.warnings
                    else "Không đạt bộ lọc"
                )
                top_headline = (
                    news_context["top_headlines"][0]["title"]
                    if news_context and news_context.get("top_headlines")
                    else ""
                )
                return {
                    "type": "watchlist",
                    "data": {
                        "symbol": symbol,
                        "confidence": confidence_for_watchlist,
                        "reason": reason,
                        "sentiment": (
                            news_context.get("sentiment_score", 0.0)
                            if news_context
                            else 0.0
                        ),
                        "headline": top_headline,
                    },
                }
            return None

        # 4. Position Sizing
        price = df.iloc[-1]["close"]
        position = self.position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=price,
            stop_loss=entry_signal.stop_loss,
            confidence=entry_signal.confidence,
            signal_strength=entry_signal.strength.name,
            market_regime=market_regime,
        )
        if position.shares == 0:
            return None

        # 5. Risk & Lock Checks
        can_add, lock_reason = self.portfolio_lock.can_add_position(
            symbol=symbol,
            position_value=position.shares * price,
            total_capital=self.position_sizer.total_capital,
            current_positions=self.portfolio_manager.get_positions(),
        )
        if not can_add:
            logging.info(f"🔒 {symbol}: {lock_reason}")
            return None

        # 6. Format message and return result
        msg = self.format_entry_recommendation(
            symbol, entry_signal, position, market_regime, news_context
        )
        return {
            "type": "signal",
            "signal": entry_signal,
            "position": position,
            "message": msg,
            "symbol": symbol,  # Add symbol for execute_paper_buy
        }

    def execute_paper_buy(self, entry_result):
        """Thực thi lệnh mua giấy và cập nhật trạng thái."""
        entry_signal = entry_result["signal"]
        position = entry_result["position"]
        symbol = entry_result["symbol"]

        success, paper_msg, _ = self.paper_account.execute_buy(
            symbol=symbol,
            shares=position.shares,
            price=entry_signal.entry_price,
            signal_confidence=entry_signal.confidence,
            signal_reason=", ".join(entry_signal.reasons[:2]),
            stop_loss=entry_signal.stop_loss,
            take_profit=(
                position.recommended_entries[-1]["price"]
                if position.recommended_entries
                else None
            ),
        )

        if success:
            if self.position_sizer:
                self.position_sizer.add_position(
                    symbol, position.shares, entry_signal.entry_price
                )
            if self.portfolio_lock:
                self.portfolio_lock.confirm_position(symbol)
        else:
            if self.portfolio_lock:
                self.portfolio_lock.cancel_position(symbol)

        return success, paper_msg

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
        except Exception as e:
            logging.error(f"Lỗi portfolio risk analysis: {e}", exc_info=True)

    async def send_summary_and_watchlist(
        self, num_scanned, signal_count, market_regime, watchlist_candidates
    ):
        """Gửi tóm tắt và danh sách theo dõi."""
        regime_text = market_regime.get("regime", "UNKNOWN")
        summary = (
            f"✅ Hoàn thành quét {num_scanned} mã\n"
            f"🎯 Tín hiệu hợp lệ: {signal_count}\n"
            f"📊 Market: {regime_text}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await self.bot.send_message(self.chat_id, text=summary)
        logging.info(summary)

        if signal_count == 0:
            if watchlist_candidates:
                watchlist_candidates.sort(key=lambda x: x["confidence"], reverse=True)
                top_watchlist = watchlist_candidates[:WATCHLIST_SIZE]
                lines = [
                    "• {symbol}: {conf:.0f}% - {reason}{sentiment}{headline}".format(
                        symbol=item["symbol"],
                        conf=item["confidence"],
                        reason=item["reason"],
                        sentiment=(
                            f", sentiment {item.get('sentiment', 0.0):+.2f}"
                            if "sentiment" in item
                            else ""
                        ),
                        headline=(
                            f" | {item['headline']}" if item.get("headline") else ""
                        ),
                    )
                    for item in top_watchlist
                ]
                watchlist_msg = "👀 *WATCHLIST* (chưa đủ điều kiện BUY):\n" + "\n".join(
                    lines
                )
            else:
                watchlist_msg = (
                    "⚠️ Thị trường chưa có mã nào đạt điều kiện BUY. Tiếp tục quan sát."
                )
            await self.bot.send_message(
                self.chat_id, text=watchlist_msg, parse_mode="Markdown"
            )
