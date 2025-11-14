# -*- coding: utf-8 -*-

# Suppress warnings first
import suppress_warnings  # noqa: F401

# ===== LOGGING SETUP =====
import logging
from logging_config import setup_logging

setup_logging()
# =========================

import asyncio
import os
from telegram import Bot
import pandas as pd

# ===== CONFIG IMPORTS =====
try:
    from config import CHAT_ID, TELEGRAM_TOKEN, LOOKBACK

    logging.info("✅ Import config thành công")
except ImportError as e:
    from exceptions import ConfigurationError

    logging.error("❌ Failed to import configuration.", exc_info=True)
    raise ConfigurationError("Failed to import config.", context={"error": str(e)})

# ===== CORE COMPONENTS =====
try:
    from market_regime_proxy import ProxyMarketRegimeAnalyzer as MarketAnalyzer
    from strategy_manager import StrategyManager
    from orchestrator import TradingOrchestrator
    from circuit_breaker import get_circuit_breaker
    from emergency_stop import get_emergency_stop
    from portfolio_manager import get_portfolio_manager
    from data_loader import load_data  # Import load_data here

    logging.info("✅ Import các thành phần cốt lõi thành công")
except ImportError as e:
    logging.critical(
        f"❌ Lỗi nghiêm trọng khi import các thành phần cốt lõi: {e}", exc_info=True
    )
    raise

# ===== INITIALIZE BOT & CORE SERVICES =====
try:
    bot = Bot(token=TELEGRAM_TOKEN)
    logging.info("✅ Telegram bot initialized")
except Exception as e:
    logging.critical(f"❌ Lỗi khởi tạo Telegram bot: {e}", exc_info=True)
    bot = None

# Khởi tạo các dịch vụ chính
market_analyzer = MarketAnalyzer()
strategy_manager = StrategyManager()
portfolio_manager = get_portfolio_manager()
orchestrator = TradingOrchestrator(bot_instance=bot, chat_id=CHAT_ID)

# Gán các chiến lược đã được khởi tạo vào orchestrator
orchestrator.set_strategies(**strategy_manager.get_strategies())


async def run_bot_with_context(bot_instance: Bot, chat_id: str):
    """
    Bot runner chính, đã được tái cấu trúc.
    Chịu trách nhiệm điều phối các bước ở cấp cao nhất.
    """
    if not bot_instance:
        logging.critical("❌ Bot instance không khả dụng, không thể chạy.")
        return

    # ===== 1. KIỂM TRA AN TOÀN =====
    logging.info("🔒 Kiểm tra các hệ thống an toàn...")
    if not check_safety_systems(bot_instance, chat_id):
        return  # Dừng lại nếu kiểm tra an toàn thất bại

    # ===== 2. PHÂN TÍCH THỊ TRƯỜNG =====
    logging.info("📊 Phân tích trạng thái thị trường...")
    market_regime = market_analyzer.analyze_market_regime()
    if not market_regime.get("tradeable", False):
        msg = f"⛔ *THỊ TRƯỜNG KHÔNG PHÙ HỢP*\n\n{market_regime.get('message', 'Không rõ lý do.')}"
        await bot_instance.send_message(chat_id, msg, parse_mode="Markdown")
        logging.warning(
            f"⛔ Thị trường không phù hợp để trade: {market_regime.get('message')}"
        )
        return

    logging.info(f"✅ Thị trường OK: {market_regime.get('message')}")

    # ===== 3. ĐIỀU CHỈNH CHIẾN LƯỢC =====
    strategy_manager.apply_market_adjustments(market_regime)

    # ===== 4. TẢI DỮ LIỆU VN-INDEX (CHO ML) =====
    logging.info("📈 Tải dữ liệu VN-Index cho các tính năng ML...")
    vnindex_df = load_vnindex_data()

    # ===== 5. CHẠY QUÁ TRÌNH QUÉT CHÍNH =====
    await orchestrator.run_scan(market_regime, vnindex_df)

    # ===== 6. PHÂN TÍCH DANH MỤC CUỐI CÙNG =====
    logging.info("\n📊 Phân tích tổng thể danh mục cuối phiên...")
    await check_portfolio_and_recommend(bot_instance, chat_id)


def check_safety_systems(bot_instance: Bot, chat_id: str) -> bool:
    """Kiểm tra Circuit Breaker và Emergency Stop. Trả về True nếu an toàn."""
    # Circuit Breaker
    circuit_breaker = get_circuit_breaker()
    can_trade_cb, cb_reason = circuit_breaker.can_trade()
    if not can_trade_cb:
        msg = f"🚫 *CIRCUIT BREAKER KÍCH HOẠT*\n\n{cb_reason}\n\n{circuit_breaker.get_status_message()}"
        asyncio.run(bot_instance.send_message(chat_id, msg, parse_mode="Markdown"))
        logging.critical(f"🚫 Circuit Breaker: {cb_reason}")
        return False

    # Emergency Stop
    emergency_stop = get_emergency_stop()
    can_trade_es, es_reason = emergency_stop.can_trade()
    if not can_trade_es:
        msg = f"🚨 *DỪNG KHẨN CẤP ĐANG BẬT*\n\n{es_reason}\n\n{emergency_stop.get_status_message()}"
        asyncio.run(bot_instance.send_message(chat_id, msg, parse_mode="Markdown"))
        logging.critical(f"🚨 Emergency Stop: {es_reason}")
        return False

    logging.info("✅ Các hệ thống an toàn đã được kiểm tra.")
    return True


def load_vnindex_data() -> pd.DataFrame | None:
    """Tải dữ liệu VN-Index và xử lý lỗi."""
    try:
        vnindex_df = load_data(
            "VNINDEX", lookback=LOOKBACK, use_cache=True, is_index=True
        )
        if vnindex_df.empty:
            logging.warning(
                "⚠️ Không tải được dữ liệu VN-Index. Phân tích ML có thể bị ảnh hưởng."
            )
            return None
        return vnindex_df
    except Exception as e:
        logging.error(f"Lỗi tải dữ liệu VN-Index: {e}", exc_info=True)
        return None


async def check_portfolio_and_recommend(bot_instance: Bot, chat_id: str):
    """Kiểm tra và gửi báo cáo phân tích danh mục."""
    logging.info("🔍 Gửi báo cáo phân tích danh mục...")
    try:
        analysis_report = portfolio_manager.get_detailed_analysis()
        # Gửi các phần nhỏ nếu tin nhắn quá dài
        if len(analysis_report) > 4000:
            parts = [
                analysis_report[i : i + 4000]
                for i in range(0, len(analysis_report), 4000)
            ]
            for part in parts:
                await bot_instance.send_message(chat_id, part, parse_mode="Markdown")
        else:
            await bot_instance.send_message(
                chat_id, analysis_report, parse_mode="Markdown"
            )
        logging.info("✅ Đã gửi phân tích danh mục.")
    except Exception as e:
        error_msg = f"❌ Lỗi khi gửi phân tích danh mục: {e}"
        logging.error(error_msg, exc_info=True)
        await bot_instance.send_message(chat_id, error_msg)


# ================ RUNNER ====================


async def run_bot():
    """Hàm async chính để chạy bot."""
    await run_bot_with_context(bot, CHAT_ID)


def run_bot_sync():
    """Hàm bao bọc đồng bộ để chạy từ script."""
    try:
        asyncio.run(run_bot())
    except Exception as e:
        logging.critical(
            f"❌ Lỗi nghiêm trọng ở cấp cao nhất khi chạy bot: {e}", exc_info=True
        )


# ================ MAIN ====================

if __name__ == "__main__":
    """Test chạy bot"""
    logging.info("\n" + "=" * 70)
    logging.info("🤖>> BOT RUNNER REFACTORED <<🤖")
    logging.info("=" * 70 + "\n")

    logging.info("Để chạy bot thực tế, sử dụng lệnh:")
    logging.info(
        "  python -c 'from bot_runner_improved import run_bot_sync; run_bot_sync()'"
    )

    # Chạy một lần để test
    run_bot_sync()
