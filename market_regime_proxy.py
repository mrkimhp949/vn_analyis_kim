# market_regime_proxy.py
import sys
import os

# Fix encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        os.environ["PYTHONIOENCODING"] = "utf-8"
    except:
        pass


def safe_print(message):
    """Print an toàn"""
    try:
        print(message)
    except UnicodeEncodeError:
        clean_message = "".join(char for char in message if ord(char) < 128)
        print(clean_message)


import pandas as pd
from data_loader import load_data
from ml_signals import MLSignalGenerator


class ProxyMarketRegimeAnalyzer:
    def __init__(self):
        from config import TICKERS

        # ✅ DÙNG TẤT CẢ MÃ TỪ CONFIG, không hardcode
        self.proxy_stocks = TICKERS
        self.ml_generator = MLSignalGenerator()
        safe_print(
            f"🔧 Khởi tạo Market Regime Analyzer với {len(self.proxy_stocks)} mã từ config"
        )

    def analyze_market_regime(self):
        from config import TICKERS

        safe_print(
            f"📈 PHÂN TÍCH THỊ TRƯỜNG: Bắt đầu phân tích {len(self.proxy_stocks)} mã"
        )

        analyzed = 0
        buy_signals = 0
        total_confidence = 0

        # ✅ PHÂN TÍCH TẤT CẢ MÃ TRONG CONFIG
        for symbol in self.proxy_stocks:
            try:
                safe_print(f"  📊 Phân tích {symbol}...")

                from config import LOOKBACK

                df = load_data(symbol, lookback=LOOKBACK)
                if df.empty or len(df) < 50:
                    safe_print(f"  ⏭️ {symbol}: Không đủ dữ liệu")
                    continue

                ml_signal = self.ml_generator.analyze(df)
                analyzed += 1

                if ml_signal["signal"] == "BUY":
                    buy_signals += 1
                    signal_info = "BUY"
                else:
                    signal_info = ml_signal["signal"]

                total_confidence += ml_signal["confidence"]

                safe_print(f"  {symbol}: {signal_info} ({ml_signal['confidence']}%)")

            except ValueError as e:
                # Bỏ qua mã không hợp lệ (đã bị hủy niêm yết, etc.)
                error_msg = str(e)
                if (
                    "hủy niêm yết" in error_msg
                    or "không tồn tại" in error_msg
                    or "không trả dữ liệu" in error_msg
                ):
                    safe_print(f"  ⏭️ {symbol}: Bỏ qua (không có dữ liệu)")
                else:
                    safe_print(f"  ⚠️ {symbol}: {error_msg[:50]}")
                continue
            except Exception as e:
                safe_print(f"  ⚠️ {symbol}: Lỗi không xác định")
                continue

        safe_print(f"✅ Đã phân tích: {analyzed}/{len(self.proxy_stocks)} mã")

        if analyzed == 0:
            return {
                "regime": "UNKNOWN",
                "tradeable": False,
                "confidence": 0,
                "message": "Khong phan tich duoc thi truong",
            }

        buy_rate = (buy_signals / analyzed) * 100
        avg_confidence = total_confidence / analyzed

        # Xac dinh market regime
        if buy_rate >= 60:
            regime = "BULL"
            tradeable = True
        elif buy_rate >= 40:
            regime = "SIDEWAYS"
            tradeable = True
        else:
            regime = "BEAR"
            tradeable = avg_confidence > 30

        message = f"{regime} market - {buy_rate:.1f}% ma BUY ({analyzed}/{len(self.proxy_stocks)} ma)"

        safe_print(f"KET QUA: {message}")

        return {
            "regime": regime,
            "tradeable": tradeable,
            "confidence": avg_confidence,
            "message": message,
            "analyzed_stocks": analyzed,
            "total_stocks": len(self.proxy_stocks),
            "buy_rate": buy_rate,
        }
