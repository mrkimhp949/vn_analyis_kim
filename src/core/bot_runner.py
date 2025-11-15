# [file name]: bot_runner_improved.py
# [file content begin]
# -*- coding: utf-8 -*-
"""
Bot Runner: Lớp vỏ (wrapper) để khởi chạy TradingOrchestrator.
File này chỉ chịu trách nhiệm khởi tạo và gọi, toàn bộ logic nghiệp vụ
nằm trong TradingOrchestrator.

Version 2.0: Support for both old and new orchestrator with feature flag
"""
import asyncio
import logging
import os
from telegram import Bot
import pandas as pd

# Import các thành phần cần thiết
from src.config.legacy_config import CHAT_ID, TELEGRAM_TOKEN, LOOKBACK
from src.config.exceptions import ConfigurationError
from src.data.loader import load_data
from src.market.regime_proxy import ProxyMarketRegimeAnalyzer
from src.core.orchestrator import TradingOrchestrator

logging.info("✅ Using TradingOrchestrator")

# Khởi tạo các đối tượng toàn cục
try:
    bot = Bot(token=TELEGRAM_TOKEN)
    logging.info("✅ Telegram bot initialized")
except Exception as e:
    logging.critical(f"❌ Lỗi khởi tạo Telegram bot: {e}")
    bot = None

try:
    market_analyzer = ProxyMarketRegimeAnalyzer()
    logging.info("✅ Market analyzer initialized")
except ImportError as e:
    logging.error(f"⚠️ Không có market analyzer: {e}")
    market_analyzer = None


async def run_bot_with_context(bot_instance: Bot, chat_id: str):
    """
    Hàm chính để khởi chạy bot.
    1. Khởi tạo Orchestrator (V1 hoặc V2 based on feature flag).
    2. Lấy trạng thái thị trường.
    3. Chạy quá trình quét của Orchestrator.
    """
    if not bot_instance:
        logging.critical("❌ Bot instance không khả dụng, không thể chạy.")
        return

    logging.info("\n" + "="*50 + "\n🤖 BẮT ĐẦU PHIÊN QUÉT MỚI\n" + "="*50)

    # 1. Tải dữ liệu VNINDEX trước để dùng chung
    vnindex_df = None
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=LOOKBACK)).strftime("%Y-%m-%d")
        vnindex_df = load_data("VNINDEX", start_date=start_date, end_date=end_date, data_type="index")
        if vnindex_df.empty:
            logging.warning("⚠️ Không tải được dữ liệu VNINDEX.")
    except Exception as e:
        logging.error(f"❌ Lỗi khi tải dữ liệu VNINDEX: {e}", exc_info=True)
        # Có thể quyết định dừng nếu dữ liệu VNINDEX là bắt buộc

    # 2. Khởi tạo Orchestrator với context cần thiết
    try:
        orchestrator = TradingOrchestrator(
            bot_instance=bot_instance, 
            chat_id=chat_id,
            vnindex_df=vnindex_df  # Truyền vnindex_df vào
        )
        logging.info("✅ Trading Orchestrator initialized.")
    except Exception as e:
        logging.critical(f"❌ Lỗi khởi tạo TradingOrchestrator: {e}", exc_info=True)
        await bot_instance.send_message(chat_id, f"FATAL: Không thể khởi tạo Orchestrator: {e}")
        return

    # 3. Lấy trạng thái thị trường
    market_regime = {}
    try:
        if market_analyzer:
            # Giờ market_analyzer có thể dùng vnindex_df đã được tải sẵn nếu cần
            market_regime = market_analyzer.analyze_market_regime(vnindex_df=vnindex_df)
            logging.info(f"📊 Trạng thái thị trường: {market_regime.get('regime', 'N/A')} (Confidence: {market_regime.get('confidence', 0)}%)")
    except Exception as e:
        logging.error(f"❌ Lỗi khi phân tích thị trường: {e}", exc_info=True)
        await bot_instance.send_message(chat_id, f"Lỗi phân tích thị trường: {e}")
        # Vẫn tiếp tục với market_regime rỗng, Orchestrator sẽ xử lý

    # 4. Chạy Orchestrator
    try:
        await orchestrator.run_scan(market_regime=market_regime)
    except Exception as e:
        logging.critical(f"❌ Lỗi nghiêm trọng trong quá trình quét của Orchestrator: {e}", exc_info=True)
        await bot_instance.send_message(chat_id, f"Lỗi nghiêm trọng khi đang quét: {e}")

    logging.info("\n" + "="*50 + "\n🏁 KẾT THÚC PHIÊN QUÉT\n" + "="*50)


def run_bot_sync():
    """
    Hàm đồng bộ (sync wrapper) để `main.py` có thể gọi.
    """
    try:
        asyncio.run(run_bot_with_context(bot, CHAT_ID))
    except ConfigurationError as e:
        logging.critical(f"CRITICAL CONFIG ERROR: {e.message}")
        # Maybe send a notification through a different channel if Telegram bot failed
    except Exception as e:
        logging.critical(f"❌ Lỗi không xác định khi chạy bot: {e}", exc_info=True)

# Các hàm cũ như run_sector_analysis, analyze_current_portfolio không còn cần thiết ở đây
# vì chúng đã được tích hợp hoặc sẽ được gọi từ các module khác khi cần.

if __name__ == "__main__":
    """
    Test chạy bot runner.
    Lưu ý: Cần có đầy đủ file config và các biến môi trường.
    """
    print("\n" + "="*70)
    print("🤖 TESTING BOT RUNNER (via ORCHESTRATOR)")
    print("="*70 + "\n")
    
    # Thiết lập logging cơ bản để xem output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not all([CHAT_ID, TELEGRAM_TOKEN]):
        print("❌ Vui lòng cung cấp CHAT_ID và TELEGRAM_TOKEN trong file config hoặc biến môi trường.")
    else:
        print("🚀 Chạy thử bot trong 5 giây...")
        run_bot_sync()
        print("\n✅ Test run hoàn tất. Kiểm tra output log và tin nhắn Telegram.")
# [file content end]