# bot_runner.py
import asyncio
from config import TICKERS, LOOKBACK, CHAT_ID, TELEGRAM_TOKEN
from data_loader import load_data
from indicators import add_indicators
from signals import (
    buy_signal, sell_signal, 
    uptrend_signal, downtrend_signal,
    oversold_signal, overbought_signal
)
from telegram import Bot

# Bot instance cho scheduled job
bot = Bot(token=TELEGRAM_TOKEN)

async def run_bot_with_context(bot_instance, chat_id):
    """Chạy bot với bot instance từ context (dùng cho /run command)"""
    print("🔍 Bắt đầu quét tín hiệu...")
    print(f"📋 Danh sách cổ phiếu: {TICKERS}")
    
    signal_count = 0
    
    for symbol in TICKERS:
        try:
            print(f"📊 Đang phân tích {symbol}...")
            df = load_data(symbol, LOOKBACK)
            print(f"  ↳ Đã tải {len(df)} nến")
            
            df = add_indicators(df)
            
            # Lấy dữ liệu nến cuối
            c = df.iloc[-1]
            
            print(f"  ↳ EMA20: {c['ema20']:.2f} | EMA50: {c['ema50']:.2f} | RSI: {c['rsi']:.2f}")
            
            # Kiểm tra tín hiệu
            buy = buy_signal(df)
            sell = sell_signal(df)
            uptrend = uptrend_signal(df)
            downtrend = downtrend_signal(df)
            oversold = oversold_signal(df)
            overbought = overbought_signal(df)

            msg = None
            
            if buy:
                msg = f"🚀 [{symbol}] BUY MẠNH — EMA20 cắt lên EMA50 (Golden Cross)"
            elif sell:
                msg = f"🔴 [{symbol}] SELL MẠNH — EMA20 cắt xuống EMA50 (Death Cross)"
            elif uptrend:
                msg = f"📈 [{symbol}] XU HƯỚNG TĂNG — EMA20 > EMA50, RSI: {c['rsi']:.1f}"
            elif downtrend:
                msg = f"📉 [{symbol}] XU HƯỚNG GIẢM — EMA20 < EMA50, RSI: {c['rsi']:.1f}"
            elif oversold:
                msg = f"💎 [{symbol}] QUÁ BÁN — RSI: {c['rsi']:.1f} (Cân nhắc mua)"
            elif overbought:
                msg = f"⚠️ [{symbol}] QUÁ MUA — RSI: {c['rsi']:.1f} (Cân nhắc bán)"
            
            if msg:
                await bot_instance.send_message(chat_id=chat_id, text=msg)
                print(f"  ✅ Đã gửi: {msg[:40]}...")
                signal_count += 1
            else:
                print(f"  ↳ ⚪ Không có tín hiệu rõ ràng")

        except Exception as e:
            error_msg = f"❌ Lỗi khi xử lý {symbol}: {e}"
            print(error_msg)
            await bot_instance.send_message(chat_id=chat_id, text=error_msg)
    
    summary = f"✅ Hoàn thành quét {len(TICKERS)} cổ phiếu - Tìm thấy {signal_count} tín hiệu"
    print(summary)
    await bot_instance.send_message(chat_id=chat_id, text=summary)


async def run_bot():
    """Chạy bot với bot instance mặc định (dùng cho scheduled job)"""
    await run_bot_with_context(bot, CHAT_ID)


def run_bot_sync():
    """Wrapper để chạy async function trong sync context"""
    asyncio.run(run_bot())