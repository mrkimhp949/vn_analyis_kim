
import asyncio
from typing import Dict, Any, List

from strategies.base_strategy import BaseStrategy
from strategies.ml_strategy import MlStrategy
from strategies.rsi_strategy import RsiStrategy
from data_loader import load_multiple_tickers
from portfolio_manager import get_portfolio_manager
from telegram_notifications import send_telegram_message
import logging

logger = logging.getLogger(__name__)

class LiveBotRunner:
    """
    Chịu trách nhiệm chạy logic giao dịch TRỰC TIẾP.

    Sử dụng một hoặc nhiều chiến lược để tạo tín hiệu và gửi đến portfolio manager.
    """
    def __init__(self, strategies: List[BaseStrategy], config: Dict[str, Any]):
        self.strategies = strategies
        self.config = config
        self.portfolio_manager = get_portfolio_manager()
        self.tickers_to_scan = self.config.get("TICKERS_TO_SCAN", [])
        self.lookback_period = self.config.get("LOOKBACK", "1y")

    async def run_scan(self, **kwargs):
        """
        Thực hiện một chu trình quét tín hiệu hoàn chỉnh.
        """
        logger.info("🚀 Bắt đầu chu trình quét tín hiệu trực tiếp...")

        # 1. Tải dữ liệu cho các mã cần quét
        market_data = await load_multiple_tickers(
            tickers=self.tickers_to_scan,
            period=self.lookback_period
        )
        
        if not market_data:
            logger.warning("Không có dữ liệu thị trường nào được tải. Dừng quét.")
            return

        all_signals = []
        # 2. Chạy từng chiến lược để tạo tín hiệu
        for strategy in self.strategies:
            logger.info(f"--- Đang chạy chiến lược: {strategy.name} ---")
            try:
                # Truyền các tham số bổ sung (ví dụ: vnindex_df) vào generate_signals
                signals = strategy.generate_signals(market_data, **kwargs)
                if signals:
                    all_signals.extend(signals)
                    logger.info(f"✅ Chiến lược {strategy.name} đã tạo ra {len(signals)} tín hiệu.")
            except Exception as e:
                logger.error(f"❌ Lỗi khi chạy chiến lược {strategy.name}: {e}", exc_info=True)

        if not all_signals:
            logger.info("✅ Quét hoàn tất. Không có tín hiệu mới nào được tìm thấy.")
            await send_telegram_message("✅ Quét hoàn tất, không có tín hiệu mới.")
            return

        logger.info(f"🔥 Tổng cộng tìm thấy {len(all_signals)} tín hiệu từ tất cả các chiến lược.")

        # 3. Xử lý các tín hiệu (ví dụ: thêm vào portfolio)
        await self._process_signals(all_signals)

    async def _process_signals(self, signals: List[Dict[str, Any]]):
        """
        Xử lý các tín hiệu đã được tạo ra.
        Ở đây, chúng ta sẽ gửi chúng đến PortfolioManager.
        """
        logger.info(f"💼 Đang xử lý {len(signals)} tín hiệu...")
        for signal in signals:
            try:
                # TODO: Thêm logic quyết định xem có nên thực hiện tín hiệu hay không
                # Ví dụ: kiểm tra portfolio risk, correlation, etc.
                
                # Tính toán kích thước vị thế
                # Cần context của portfolio ở đây
                portfolio_value = self.portfolio_manager.get_portfolio_value()
                portfolio_context = {
                    "total_equity": portfolio_value.get("total_value", 0),
                    "risk_per_trade": self.config.get("RISK_PER_TRADE", 0.02)
                }
                
                # Tìm đúng chiến lược đã tạo ra tín hiệu này để tính position size
                strategy_obj = next((s for s in self.strategies if s.name == signal.get("strategy_name")), None)

                if not strategy_obj:
                    logger.warning(f"Không tìm thấy đối tượng chiến lược cho tín hiệu: {signal}")
                    continue

                shares = strategy_obj.calculate_position_size(signal, portfolio_context)

                if shares > 0:
                    self.portfolio_manager.add_position(
                        symbol=signal["symbol"],
                        shares=shares,
                        entry_price=signal["entry_price"],
                        stop_loss=signal.get("stop_loss"),
                        take_profit=signal.get("take_profit"),
                        metadata={
                            "reason": signal["reason"],
                            "confidence": signal["confidence"],
                            "strategy": signal["strategy_name"]
                        }
                    )
                    
                    # Gửi thông báo
                    message = (
                        f"📈 **TÍN HIỆU MUA MỚI**\n\n"
                        f"**Mã:** `{signal['symbol']}`\n"
                        f"**Chiến lược:** `{signal['strategy_name']}`\n"
                        f"**Giá vào lệnh:** `{signal['entry_price']:,.0f}`\n"
                        f"**Số lượng đề xuất:** `{shares}`\n"
                        f"**Lý do:** _{signal['reason']}_\n"
                        f"**Mức tin cậy:** `{signal['confidence']:.2f}`\n"
                        f"**Dừng lỗ:** `{signal.get('stop_loss', 0):,.0f}`\n"
                        f"**Chốt lời:** `{signal.get('take_profit', 0):,.0f}`"
                    )
                    await send_telegram_message(message)

            except Exception as e:
                logger.error(f"❌ Lỗi khi xử lý tín hiệu cho {signal.get('symbol')}: {e}", exc_info=True)


async def main():
    # --- Đây là ví dụ cách sử dụng ---
    from config import load_config
    
    app_config = load_config()

    # 1. Định nghĩa cấu hình cho các chiến lược
    ml_strategy_config = {
        "name": "ML_Ensemble_v1",
        "confidence_threshold": 0.75,
        "sl_pct": 0.07,
        "tp_pct": 0.15,
    }
    
    rsi_strategy_config = {
        "name": "RSI_Oversold",
        "rsi_period": 14,
        "buy_threshold": 30,
        "sell_threshold": 70,
        "sl_pct": 0.08,
        "tp_pct": 0.20,
    }

    # 2. Khởi tạo các đối tượng chiến lược
    ml_strategy = MlStrategy(ml_strategy_config)
    rsi_strategy = RsiStrategy(rsi_strategy_config)

    # 3. Khởi tạo LiveBotRunner với danh sách các chiến lược
    live_runner = LiveBotRunner(
        strategies=[ml_strategy, rsi_strategy], 
        config=app_config
    )

    # 4. Chạy chu trình quét
    # Trong ứng dụng thực tế, bạn sẽ cần vnindex_df
    from data_loader import load_data
    vnindex_df = await load_data("VNINDEX", period="1y")

    await live_runner.run_scan(vnindex_df=vnindex_df)


if __name__ == "__main__":
    # Cấu hình logging để test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())
