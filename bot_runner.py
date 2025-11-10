# bot_runner.py
import asyncio
import json
import os
import time
from datetime import datetime
from config import (
    TICKERS, LOOKBACK, CHAT_ID, TELEGRAM_TOKEN,
    KIM_SECTOR, THUY_SECTOR, KIM_TICKERS, THUY_TICKERS, ALL_TICKERS
)
from data_loader import load_data
from ml_signals import MLSignalGenerator
from risk_management import RiskManager
from telegram import Bot
import pandas as pd

# =============== KHỞI TẠO ===============
bot = Bot(token=TELEGRAM_TOKEN)
ml_generator = MLSignalGenerator()
risk_manager = RiskManager(
    total_capital=100_000_000,
    max_position_pct=0.2,
    risk_per_trade_pct=0.02
)

SELECTED_TICKERS_FILE = 'selected_tickers.json'
SIGNAL_CACHE_FILE = 'signals_cache.json'
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)

# ======================================================
# 🎯 PHÂN TÍCH NGÀNH - AUTO SECTOR ANALYSIS
# ======================================================

def run_sector_analysis():
    print("🔍 Bắt đầu phân tích toàn bộ thị trường...")
    print(f"📊 Tổng số: {len(ALL_TICKERS)} mã (Kim: {len(KIM_TICKERS)}, Thủy: {len(THUY_TICKERS)})")

    sector_results = {}
    all_sectors = {**{f"Kim_{k}": v for k, v in KIM_SECTOR.items()},
                   **{f"Thuy_{k}": v for k, v in THUY_SECTOR.items()}}

    for sector_name, tickers in all_sectors.items():
        print(f"\n📊 Đang phân tích: {sector_name} ({len(tickers)} mã)...")
        buy_signals, total_confidence, analyzed = 0, 0, 0

        for symbol in tickers:
            try:
                df = load_data(symbol, LOOKBACK)
                result = ml_generator.analyze(df)

                if result['signal'] == 'BUY':
                    buy_signals += 1
                total_confidence += result['confidence']
                analyzed += 1
                time.sleep(0.3)  # ✅ Thêm delay tránh API spam

            except Exception as e:
                log_error(f"❌ {symbol}: {e}")

        if analyzed > 0:
            avg_confidence = total_confidence / analyzed
            buy_rate = (buy_signals / analyzed) * 100
            sector_results[sector_name] = {
                'total': analyzed,
                'buy_signals': buy_signals,
                'buy_rate': buy_rate,
                'avg_confidence': avg_confidence,
                'score': buy_rate * 0.6 + avg_confidence * 0.4
            }
            print(f"  ✅ BUY: {buy_signals}/{analyzed} ({buy_rate:.1f}%) | Conf: {avg_confidence:.1f}%")

    sorted_sectors = sorted(sector_results.items(), key=lambda x: x[1]['score'], reverse=True)
    print("\n" + "="*70)
    print("📊 XẾP HẠNG NGÀNH")
    print("="*70)
    for i, (sector, data) in enumerate(sorted_sectors[:10], 1):
        print(f"{i}. {sector:20s} | Score: {data['score']:5.1f} | BUY: {data['buy_rate']:5.1f}% | Conf: {data['avg_confidence']:5.1f}%")

    # ✅ Chọn top 3
    top_sectors = [s[0] for s in sorted_sectors[:3]]
    selected_tickers = []
    for sector_name in top_sectors:
        sector_key = sector_name.replace('Kim_', '').replace('Thuy_', '')
        if sector_name.startswith('Kim_'):
            selected_tickers.extend(KIM_SECTOR.get(sector_key, []))
        else:
            selected_tickers.extend(THUY_SECTOR.get(sector_key, []))

    selected_tickers = sorted(list(set(selected_tickers)))

    with open(SELECTED_TICKERS_FILE, 'w') as f:
        json.dump({
            'selected_at': pd.Timestamp.now().isoformat(),
            'top_sectors': top_sectors,
            'tickers': selected_tickers,
            'sector_results': sector_results
        }, f, indent=2)

    print(f"\n✅ ĐÃ CHỌN TOP 3 NGÀNH: {', '.join(top_sectors)}")
    print(f"📋 Tổng {len(selected_tickers)} mã sẽ được quét tuần này\n")

    asyncio.run(send_sector_summary(top_sectors, selected_tickers, sector_results))
    return selected_tickers

async def send_sector_summary(top_sectors, tickers, results):
    msg = "📊 *PHÂN TÍCH THỊ TRƯỜNG TUẦN MỚI*\n\n🏆 *TOP 3 NGÀNH TỐT NHẤT:*\n"
    for i, sector in enumerate(top_sectors, 1):
        data = results[sector]
        msg += f"{i}. {sector}: Score {data['score']:.1f} (BUY {data['buy_rate']:.0f}%)\n"
    msg += f"\n📋 *{len(tickers)} MÃ CHO TUẦN NÀY:*\n{', '.join(tickers[:20])}..."
    try:
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode='Markdown')
    except Exception as e:
        log_error(f"⚠️ Gửi Telegram lỗi: {e}")

def load_selected_tickers():
    if os.path.exists(SELECTED_TICKERS_FILE):
        try:
            with open(SELECTED_TICKERS_FILE, 'r') as f:
                data = json.load(f)
                print(f"✅ Loaded {len(data['tickers'])} mã đã chọn ({data['selected_at']})")
                return data['tickers']
        except Exception as e:
            log_error(f"⚠️ Load selected_tickers lỗi: {e}")
    print(f"⚠️ Chưa có tickers tuần này → dùng mặc định ({len(TICKERS)} mã)")
    return TICKERS

# ======================================================
# ⚙️ BOT RUNNER - CÓ KIỂM SOÁT RỦI RO & CẢNH BÁO TRÙNG LẶP
# ======================================================

async def run_bot_with_context(bot_instance, chat_id):
    current_tickers = load_selected_tickers()
    print(f"🔍 Quét {len(current_tickers)} mã...")
    CONFIDENCE_THRESHOLD = 50
    signal_count = 0
    sent_cache = load_signal_cache()

    await bot_instance.send_message(chat_id=chat_id,
        text=f"🔍 Đang quét {len(current_tickers)} mã cổ phiếu...\n⏳ Vui lòng đợi...")

    for symbol in current_tickers:
        try:
            df = load_data(symbol, LOOKBACK)
            result = ml_generator.analyze(df)
            latest = df.iloc[-1]
            price = latest['close']
            atr = latest['atr']

            # Bỏ qua nếu tín hiệu yếu
            if result['confidence'] < CONFIDENCE_THRESHOLD or result['signal'] == 'HOLD':
                continue

            # Kiểm tra cache để tránh gửi lại
            today = pd.Timestamp.now().date().isoformat()
            cache_key = f"{symbol}_{today}"
            if cache_key in sent_cache:
                print(f"⚪ Đã gửi tín hiệu {symbol} hôm nay, bỏ qua.")
                continue

            # Position sizing + risk adjustment
            position = risk_manager.calculate_position_size(price, atr, result['confidence'], signal=result['signal'])
            adj_factor = risk_manager.adjust_for_portfolio_risk([])
            position['shares'] = int(position['shares'] * adj_factor)
            position['value'] = position['shares'] * price
            position['max_loss'] = position['risk_per_share'] * position['shares']

            limit_prices = risk_manager.suggest_limit_orders(price, atr, result['signal'])
            msg = risk_manager.format_recommendation(symbol, result, position, limit_prices, df)

            await bot_instance.send_message(chat_id, msg, parse_mode='Markdown')
            sent_cache[cache_key] = True
            save_signal_cache(sent_cache)
            print(f"✅ {symbol}: {result['signal']} ({result['confidence']}%)")

            signal_count += 1
            time.sleep(0.5)  # ✅ tránh gọi API liên tục

        except Exception as e:
            log_error(f"❌ Lỗi {symbol}: {e}")

    summary = f"""
✅ Hoàn thành quét {len(current_tickers)} mã
🎯 Tín hiệu hợp lệ: {signal_count}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    await bot_instance.send_message(chat_id, text=summary)
    print(summary)

# ================ TIỆN ÍCH PHỤ =================

def load_signal_cache():
    if os.path.exists(SIGNAL_CACHE_FILE):
        try:
            with open(SIGNAL_CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_signal_cache(data):
    try:
        with open(SIGNAL_CACHE_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        log_error(f"Lỗi ghi signal cache: {e}")

def log_error(msg):
    """Ghi lỗi vào file"""
    with open(os.path.join(LOGS_DIR, 'bot_errors.log'), 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    print(msg)

# ================ RUNNER ====================

async def run_bot():
    await run_bot_with_context(bot, CHAT_ID)

def run_bot_sync():
    asyncio.run(run_bot())