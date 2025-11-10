# bot_runner.py
import asyncio
import json
import os
from config import TICKERS, LOOKBACK, CHAT_ID, TELEGRAM_TOKEN, KIM_SECTOR, THUY_SECTOR, KIM_TICKERS, THUY_TICKERS, ALL_TICKERS
from data_loader import load_data
from ml_signals import MLSignalGenerator
from risk_management import RiskManager
from telegram import Bot
import pandas as pd

bot = Bot(token=TELEGRAM_TOKEN)

# Khởi tạo ML Signal Generator và Risk Manager
ml_generator = MLSignalGenerator()
risk_manager = RiskManager(
    total_capital=100_000_000,
    max_position_pct=0.2,
    risk_per_trade_pct=0.02
)

SELECTED_TICKERS_FILE = 'selected_tickers.json'

# ═══════════════════════════════════════════════════════════
# 🎯 AUTO SECTOR ANALYSIS & SELECTION
# ═══════════════════════════════════════════════════════════

def run_sector_analysis():
    """
    Phân tích toàn bộ thị trường và tự động chọn top sectors
    Chạy vào Thứ 7 hàng tuần
    """
    print("🔍 Bắt đầu phân tích toàn bộ thị trường...")
    print(f"📊 Tổng số: {len(ALL_TICKERS)} mã (Kim: {len(KIM_TICKERS)}, Thủy: {len(THUY_TICKERS)})")
    
    sector_results = {}
    
    # Phân tích từng ngành
    all_sectors = {**{f"Kim_{k}": v for k, v in KIM_SECTOR.items()},
                   **{f"Thuy_{k}": v for k, v in THUY_SECTOR.items()}}
    
    for sector_name, tickers in all_sectors.items():
        print(f"\n📊 Đang phân tích: {sector_name} ({len(tickers)} mã)...")
        
        buy_signals = 0
        total_confidence = 0
        analyzed = 0
        
        for symbol in tickers:
            try:
                df = load_data(symbol, LOOKBACK)
                result = ml_generator.analyze(df)
                
                if result['signal'] == 'BUY':
                    buy_signals += 1
                
                total_confidence += result['confidence']
                analyzed += 1
                
            except Exception as e:
                print(f"  ⚠️ {symbol}: {e}")
        
        if analyzed > 0:
            avg_confidence = total_confidence / analyzed
            buy_rate = (buy_signals / analyzed) * 100
            
            sector_results[sector_name] = {
                'total': analyzed,
                'buy_signals': buy_signals,
                'buy_rate': buy_rate,
                'avg_confidence': avg_confidence,
                'score': (buy_rate * 0.6 + avg_confidence * 0.4)  # Weighted score
            }
            
            print(f"  ✅ BUY: {buy_signals}/{analyzed} ({buy_rate:.1f}%) | Confidence: {avg_confidence:.1f}%")
    
    # Sắp xếp theo score
    sorted_sectors = sorted(sector_results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print("\n" + "="*70)
    print("📊 XẾP HẠNG NGÀNH")
    print("="*70)
    
    for i, (sector, data) in enumerate(sorted_sectors[:10], 1):
        print(f"{i:2d}. {sector:20s} | Score: {data['score']:5.1f} | BUY: {data['buy_rate']:5.1f}% | Conf: {data['avg_confidence']:5.1f}%")
    
    # Tự động chọn top 3 sectors
    top_sectors = [s[0] for s in sorted_sectors[:3]]
    selected_tickers = []
    
    for sector_name in top_sectors:
        sector_key = sector_name.replace('Kim_', '').replace('Thuy_', '')
        
        if sector_name.startswith('Kim_'):
            selected_tickers.extend(KIM_SECTOR.get(sector_key, []))
        else:
            selected_tickers.extend(THUY_SECTOR.get(sector_key, []))
    
    selected_tickers = sorted(list(set(selected_tickers)))
    
    # Lưu vào file
    with open(SELECTED_TICKERS_FILE, 'w') as f:
        json.dump({
            'selected_at': pd.Timestamp.now().isoformat(),
            'top_sectors': top_sectors,
            'tickers': selected_tickers,
            'sector_results': sector_results
        }, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ ĐÃ CHỌN TOP 3 NGÀNH: {', '.join(top_sectors)}")
    print(f"📋 Tổng {len(selected_tickers)} mã sẽ được quét tuần này")
    print("="*70 + "\n")
    
    # Gửi Telegram summary
    asyncio.run(send_sector_summary(top_sectors, selected_tickers, sector_results))
    
    return selected_tickers


async def send_sector_summary(top_sectors, tickers, results):
    """Gửi tổng kết phân tích lên Telegram"""
    msg = f"""
📊 **PHÂN TÍCH THỊ TRƯỜNG TUẦN MỚI**

🏆 **TOP 3 NGÀNH TỐT NHẤT:**
"""
    
    for i, sector in enumerate(top_sectors, 1):
        data = results[sector]
        msg += f"{i}. **{sector}**: Score {data['score']:.1f} (BUY {data['buy_rate']:.0f}%)\n"
    
    msg += f"""
📋 **ĐÃ CHỌN {len(tickers)} MÃ CHO TUẦN NÀY:**
{', '.join(tickers[:20])}{'...' if len(tickers) > 20 else ''}

⏰ {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
💡 Bot sẽ chỉ quét {len(tickers)} mã này từ Thứ 2-6
"""
    
    try:
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode='Markdown')
    except Exception as e:
        print(f"⚠️ Không gửi được Telegram: {e}")


def load_selected_tickers():
    """Load danh sách mã đã chọn"""
    if os.path.exists(SELECTED_TICKERS_FILE):
        try:
            with open(SELECTED_TICKERS_FILE, 'r') as f:
                data = json.load(f)
                print(f"✅ Loaded {len(data['tickers'])} mã đã chọn từ {data['selected_at']}")
                return data['tickers']
        except Exception as e:
            print(f"⚠️ Không load được selected tickers: {e}")
    
    # Fallback: dùng TICKERS mặc định
    print(f"⚠️ Chưa có phân tích tuần này, dùng mặc định {len(TICKERS)} mã")
    return TICKERS


# ═══════════════════════════════════════════════════════════
# 🤖 DAILY BOT RUNNER
# ═══════════════════════════════════════════════════════════

async def run_bot_with_context(bot_instance, chat_id):
    """Chạy bot với ML predictions + Risk Management"""
    
    # Load selected tickers
    current_tickers = load_selected_tickers()
    
    print(f"🔍 Bắt đầu quét {len(current_tickers)} mã đã chọn...")
    
    CONFIDENCE_THRESHOLD = 50
    signal_count = 0
    
    await bot_instance.send_message(
        chat_id=chat_id, 
        text=f"🔍 Đang quét {len(current_tickers)} mã cổ phiếu...\n⏳ Vui lòng đợi..."
    )
    
    for symbol in current_tickers:
        try:
            print(f"📊 Đang phân tích {symbol}...")
            
            # Load data
            df = load_data(symbol, LOOKBACK)
            
            # ML Analysis
            result = ml_generator.analyze(df)
            
            print(f"  ↳ ML: {result['ml_score']:.2f} | {result['signal']} ({result['confidence']}%)")
            
            # Lấy thông tin giá và ATR
            latest = df.iloc[-1]
            current_price = latest['close']
            atr = latest['atr']
            
            msg = None
            
            if result['signal'] in ['BUY', 'SELL'] and result['confidence'] >= CONFIDENCE_THRESHOLD:
                
                # Phân loại tín hiệu
                if result['confidence'] >= 70:
                    signal_strength = "🔥 RẤT MẠNH"
                elif result['confidence'] >= 60:
                    signal_strength = "✅ MẠNH"
                else:
                    signal_strength = "📈 TRUNG BÌNH"
                
                # Position sizing
                position_info = risk_manager.calculate_position_size(
                    current_price=current_price,
                    atr=atr,
                    confidence=result['confidence'],
                    signal=result['signal']
                )
                
                limit_prices = risk_manager.suggest_limit_orders(
                    current_price=current_price,
                    atr=atr,
                    signal=result['signal']
                )
                
                # Điều chỉnh position theo confidence
                original_shares = position_info['shares']
                confidence_multiplier = result['confidence'] / 100
                adjusted_shares = int(original_shares * confidence_multiplier)
                adjusted_shares = max(adjusted_shares // 100 * 100, 100)
                
                position_info['shares'] = adjusted_shares
                position_info['value'] = adjusted_shares * current_price
                position_info['max_loss'] = position_info['risk_per_share'] * adjusted_shares
                position_info['expected_profit_tp2'] = position_info['reward_per_share'] * adjusted_shares
                
                # Format message
                base_msg = risk_manager.format_recommendation(
                    symbol=symbol,
                    result=result,
                    position_info=position_info,
                    limit_prices=limit_prices,
                    df=df
                )
                
                msg = f"""
{'🟢' if result['signal'] == 'BUY' else '🔴'}═══════════════════════════════════════{'🟢' if result['signal'] == 'BUY' else '🔴'}
📊 **ĐÁNH GIÁ:** {signal_strength}
{'🟢' if result['signal'] == 'BUY' else '🔴'}═══════════════════════════════════════{'🟢' if result['signal'] == 'BUY' else '🔴'}

{base_msg}

📌 **LƯU Ý:**
• Position điều chỉnh theo confidence ({result['confidence']}%)
• Gốc: {original_shares:,} CP → Điều chỉnh: {adjusted_shares:,} CP
"""
                signal_count += 1
            
            if msg:
                await bot_instance.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
                print(f"  ✅ Đã gửi tín hiệu")
            else:
                print(f"  ↳ ⚪ Confidence < {CONFIDENCE_THRESHOLD}%")

        except Exception as e:
            error_msg = f"❌ Lỗi {symbol}: {e}"
            print(error_msg)
    
    # Summary
    summary = f"""
✅ **HOÀN THÀNH QUÉT {len(current_tickers)} MÃ**

🎯 Tìm thấy: **{signal_count}** tín hiệu
📊 Threshold: {CONFIDENCE_THRESHOLD}%
⏰ {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    await bot_instance.send_message(chat_id=chat_id, text=summary)
    print(summary)


async def run_bot():
    """Wrapper cho scheduled job"""
    await run_bot_with_context(bot, CHAT_ID)


def run_bot_sync():
    """Sync wrapper"""
    asyncio.run(run_bot())