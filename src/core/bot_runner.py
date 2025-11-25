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

from telegram import Bot

from src.config.exceptions import ConfigurationError

# Import các thành phần cần thiết
try:
    from src.config.legacy_config import CHAT_ID, LOOKBACK, TELEGRAM_TOKEN
except (ImportError, ConfigurationError) as e:
    logging.error(f"❌ Lỗi load config: {e}")
    TELEGRAM_TOKEN = None
    CHAT_ID = None
    LOOKBACK = 252

from src.core.orchestrator import TradingOrchestrator
from src.data.loader import load_data
from src.market.regime_proxy import ProxyMarketRegimeAnalyzer

logging.info("✅ Using TradingOrchestrator")

# Khởi tạo các đối tượng toàn cục
try:
    if TELEGRAM_TOKEN:
        bot = Bot(token=TELEGRAM_TOKEN)
        logging.info("✅ Telegram bot initialized")
    else:
        logging.warning("⚠️ TELEGRAM_TOKEN không khả dụng")
        bot = None
except Exception as e:
    logging.critical(f"❌ Lỗi khởi tạo Telegram bot: {e}")
    bot = None

try:
    market_analyzer = ProxyMarketRegimeAnalyzer()
    logging.info("✅ Market analyzer initialized")
except ImportError:
    logging.error("⚠️ Không có market analyzer")
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

    logging.info("\n" + "=" * 50 + "\n🤖 BẮT ĐẦU PHIÊN QUÉT MỚI\n" + "=" * 50)

    # 1. Tải dữ liệu VNINDEX trước để dùng chung
    vnindex_df = None
    try:
        from datetime import datetime, timedelta

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=LOOKBACK)).strftime("%Y-%m-%d")
        vnindex_df = load_data(
            "VNINDEX", start_date=start_date, end_date=end_date, data_type="index"
        )
        if vnindex_df.empty:
            logging.warning("⚠️ Không tải được dữ liệu VNINDEX.")
    except Exception:
        logging.error("❌ Lỗi khi tải dữ liệu VNINDEX", exc_info=True)
        # Có thể quyết định dừng nếu dữ liệu VNINDEX là bắt buộc

    # 2. Khởi tạo Orchestrator với context cần thiết
    try:
        orchestrator = TradingOrchestrator(
            bot_instance=bot_instance,
            chat_id=chat_id,
            vnindex_df=vnindex_df,  # Truyền vnindex_df vào
        )
        logging.info("✅ Trading Orchestrator initialized.")
    except Exception:
        logging.critical("❌ Lỗi khởi tạo TradingOrchestrator", exc_info=True)
        await bot_instance.send_message(chat_id, "FATAL: Không thể khởi tạo Orchestrator")
        return

    # 3. Lấy trạng thái thị trường
    market_regime = {}
    try:
        if market_analyzer:
            # Giờ market_analyzer có thể dùng vnindex_df đã được tải sẵn nếu cần
            market_regime = market_analyzer.analyze_market_regime(vnindex_df=vnindex_df)
            logging.info(
                f"📊 Trạng thái thị trường: {market_regime.get('regime', 'N/A')} "
                f"(Confidence: {market_regime.get('confidence', 0)}%)"
            )
    except Exception:
        logging.error("❌ Lỗi khi phân tích thị trường", exc_info=True)
        await bot_instance.send_message(chat_id, "Lỗi phân tích thị trường")
        # Vẫn tiếp tục với market_regime rỗng, Orchestrator sẽ xử lý

    # 4. Chạy Orchestrator
    try:
        await orchestrator.run_scan(market_regime=market_regime)
    except Exception:
        logging.critical(
            "❌ Lỗi nghiêm trọng trong quá trình quét của Orchestrator",
            exc_info=True,
        )
        await bot_instance.send_message(chat_id, "Lỗi nghiêm trọng khi đang quét")

    logging.info("\n" + "=" * 50 + "\n🏁 KẾT THÚC PHIÊN QUÉT\n" + "=" * 50)


def run_bot_sync():
    """
    Hàm đồng bộ (sync wrapper) để `main.py` có thể gọi.
    """
    if not bot or not CHAT_ID:
        logging.error("❌ Không thể chạy bot: Thiếu TELEGRAM_TOKEN hoặc CHAT_ID")
        logging.error("📝 Vui lòng set các biến môi trường: TELEGRAM_TOKEN, CHAT_ID")
        return

    try:
        asyncio.run(run_bot_with_context(bot, CHAT_ID))
    except ConfigurationError as e:
        # Fix: Use str(e) instead of e.message for compatibility
        logging.critical(f"CRITICAL CONFIG ERROR: {str(e)}")
        # Maybe send a notification through a different channel if Telegram bot failed
    except Exception:
        logging.critical("❌ Lỗi không xác định khi chạy bot", exc_info=True)


# Các hàm cũ như run_sector_analysis, analyze_current_portfolio không còn cần thiết ở đây
# vì chúng đã được tích hợp hoặc sẽ được gọi từ các module khác khi cần.

if __name__ == "__main__":
    """
    Test chạy bot runner.
    Lưu ý: Cần có đầy đủ file config và các biến môi trường.
    """
    print("\n" + "=" * 70)
    print("🤖 TESTING BOT RUNNER (via ORCHESTRATOR)")
    print("=" * 70 + "\n")

    # Thiết lập logging cơ bản để xem output
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not all([CHAT_ID, TELEGRAM_TOKEN]):
        print(
            "❌ Vui lòng cung cấp CHAT_ID và TELEGRAM_TOKEN trong file config hoặc biến môi trường."
        )
    else:
        print("🚀 Chạy thử bot trong 5 giây...")
        run_bot_sync()
        print("\n✅ Test run hoàn tất. Kiểm tra output log và tin nhắn Telegram.")
# [file content end]
