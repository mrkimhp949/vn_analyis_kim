# -*- coding: utf-8 -*-

import asyncio
import json
import os
import time
from datetime import datetime
from telegram import Bot
import pandas as pd

# ===== CONFIG IMPORTS =====
from config import (
    TICKERS, LOOKBACK, CHAT_ID, TELEGRAM_TOKEN,
    KIM_SECTOR, THUY_SECTOR, ALL_TICKERS
)

# ===== ORIGINAL MODULES =====
from data_loader import load_data
from ml_signals import MLSignalGenerator

# ===== NEW IMPROVED MODULES =====
try:
    # Try proxy version first (recommended)
    from market_regime_proxy import ProxyMarketRegimeAnalyzer as MarketAnalyzer
    print("✅ Using Proxy Market Analyzer (Blue-chip stocks)")
except ImportError:
    try:
        # Fallback to original
        from market_regime import MarketRegimeAnalyzer as MarketAnalyzer
        print("✅ Using Original Market Analyzer (VNINDEX)")
    except ImportError:
        print("⚠️ No market analyzer available - will skip market checks")
        MarketAnalyzer = None

from improved_sector_analysis import EnhancedSectorAnalyzer
from improved_entry_logic import ImprovedEntryLogic
from improved_position_sizing import ConservativePositionSizer
from improved_exit_logic import ImprovedExitStrategy

# =============== KHỞI TẠO ===============
bot = Bot(token=TELEGRAM_TOKEN)
ml_generator = MLSignalGenerator()

# Initialize improved modules
market_analyzer = MarketAnalyzer() if MarketAnalyzer else None

sector_analyzer = EnhancedSectorAnalyzer(
    min_volume=500_000,
    min_price=10_000
)

entry_logic = ImprovedEntryLogic(
    min_confidence=60,
    min_risk_reward=2.0,
    require_trend_alignment=True,
    require_volume_confirmation=False  # Relax for VN market
)

position_sizer = ConservativePositionSizer(
    total_capital=100_000_000,
    max_risk_per_trade=0.02,
    max_position_size=0.10,
    max_total_exposure=0.60,
    min_positions=8
)

exit_strategy = ImprovedExitStrategy()

# Files
SELECTED_TICKERS_FILE = 'selected_tickers.json'
POSITIONS_FILE = 'active_positions.json'
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def check_market_before_trading():
    """
    Check market regime trước khi trade
    
    Returns:
        (can_trade: bool, message: str)
    """
    if not market_analyzer:
        # No analyzer available - assume OK to trade cautiously
        return True, "⚠️ Không có market analyzer - trade thận trọng"
    
    try:
        result = market_analyzer.analyze_market_regime()
        return result['tradeable'], result['message']
    except Exception as e:
        log_error(f"Lỗi check market: {e}")
        # On error, assume OK to trade
        return True, f"⚠️ Lỗi check market - trade thận trọng"


# ======================================================
# PHÂN TÍCH NGÀNH - IMPROVED VERSION
# ======================================================

def run_sector_analysis():
    """Phân tích ngành với enhanced analyzer"""
    print("🔍 Bắt đầu phân tích thị trường (Enhanced)...")
    
    # Combine all sectors
    all_sectors = {
        f"Kim_{k}": v for k, v in KIM_SECTOR.items()
    }
    all_sectors.update({
        f"Thuy_{k}": v for k, v in THUY_SECTOR.items()
    })
    
    # Run analysis
    try:
        result = sector_analyzer.analyze_all_sectors(
            all_sectors,
            lookback=100
        )
        
        # Save results
        with open(SELECTED_TICKERS_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'selected_at': result['analyzed_at'],
                'top_sectors': result['selected_sectors'],
                'tickers': result['selected_tickers'],
                'market_summary': result['market_summary']
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ ĐÃ CHỌN {len(result['selected_tickers'])} MÃ CHO TUẦN NÀY")
        
        # Send Telegram summary
        asyncio.run(send_sector_summary_telegram(result))
        
        return result['selected_tickers']
        
    except Exception as e:
        log_error(f"Lỗi sector analysis: {e}")
        # Fallback to default tickers
        return TICKERS[:10]


async def send_sector_summary_telegram(result):
    """Gửi summary qua Telegram"""
    try:
        msg = "📊 *PHÂN TÍCH THỊ TRƯỜNG TUẦN MỚI*\n\n"
        
        msg += "🏆 *TOP SECTORS:*\n"
        for i, sector in enumerate(result['selected_sectors'][:3], 1):
            score = result['sector_scores'][sector]['total_score']
            msg += f"{i}. {sector}: {score:.1f}/100\n"
        
        msg += f"\n📋 *{len(result['selected_tickers'])} MÃ ĐƯỢC CHỌN*\n"
        
        summary = result['market_summary']
        msg += f"\n📈 *SENTIMENT:* {summary['market_sentiment']}\n"
        msg += f"💯 *Avg Score:* {summary['avg_sector_score']:.1f}/100"
        
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode='Markdown')
    except Exception as e:
        log_error(f"Lỗi gửi Telegram: {e}")


# ======================================================
# BOT RUNNER - IMPROVED VERSION
# ======================================================

async def run_bot_with_context(bot_instance, chat_id):
    """Bot runner với improved logic"""
    
    # ===== CHECK 1: MARKET REGIME =====
    print("📊 Kiểm tra tình trạng thị trường...")
    can_trade, market_message = check_market_before_trading()
    
    if not can_trade:
        msg = f"⛔ *KHÔNG THỂ TRADE*\n\n{market_message}"
        try:
            await bot_instance.send_message(chat_id, msg, parse_mode='Markdown')
        except Exception as e:
            log_error(f"Lỗi gửi Telegram: {e}")
        
        print("⛔ Thị trường không phù hợp để trade")
        return
    
    print(f"✅ Thị trường OK: {market_message}")
    
    # Get market regime info
    try:
        market_regime = market_analyzer.analyze_market_regime() if market_analyzer else None
    except Exception as e:
        log_error(f"Lỗi get market regime: {e}")
        market_regime = None
    
    # ===== CHECK 2: LOAD TICKERS =====
    current_tickers = load_selected_tickers()
    print(f"🔍 Quét {len(current_tickers)} mã...")
    
    signal_count = 0
    
    try:
        regime_text = market_regime['regime'] if market_regime else 'UNKNOWN'
        await bot_instance.send_message(
            chat_id=chat_id,
            text=f"🔍 Đang quét {len(current_tickers)} mã...\n"
                 f"📊 Market: {regime_text}"
        )
    except Exception as e:
        log_error(f"Lỗi gửi Telegram: {e}")
    
    # ===== CHECK 3: CHECK EXITS TRƯỚC =====
    await check_active_positions(bot_instance, chat_id, market_regime)
    
    # ===== CHECK 4: SCAN FOR NEW ENTRIES =====
    for symbol in current_tickers:
        try:
            df = load_data(symbol, LOOKBACK)
            if df.empty:
                continue
            
            # Get ML signal
            ml_signal = ml_generator.analyze(df)
            
            # ===== IMPROVED ENTRY LOGIC =====
            entry_signal = entry_logic.analyze_entry(
                df=df,
                ml_signal=ml_signal,
                market_regime=market_regime
            )
            
            if not entry_signal.should_enter:
                continue
            
            latest = df.iloc[-1]
            price = latest['close']
            
            # ===== IMPROVED POSITION SIZING =====
            position = position_sizer.calculate_position_size(
                symbol=symbol,
                entry_price=price,
                stop_loss=entry_signal.stop_loss,
                confidence=entry_signal.confidence,
                signal_strength=entry_signal.strength.name,
                market_regime=market_regime
            )
            
            if position.shares == 0:
                continue
            
            # ===== FORMAT MESSAGE =====
            msg = format_entry_recommendation(
                symbol, 
                entry_signal, 
                position,
                market_regime
            )
            
            await bot_instance.send_message(chat_id, msg, parse_mode='Markdown')
            
            # Save to pending positions
            save_pending_position(symbol, entry_signal, position)
            
            signal_count += 1
            print(f"✅ {symbol}: {entry_signal.signal_type} ({entry_signal.confidence}%)")
            
            time.sleep(0.5)
            
        except Exception as e:
            log_error(f"Lỗi quét {symbol}: {e}")
    
    # Summary
    regime_text = market_regime['regime'] if market_regime else 'UNKNOWN'
    summary = f"""
✅ Hoàn thành quét {len(current_tickers)} mã
🎯 Tín hiệu hợp lệ: {signal_count}
📊 Market: {regime_text}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    await bot_instance.send_message(chat_id, text=summary)
    print(summary)


async def check_active_positions(bot_instance, chat_id, market_regime):
    """Check các positions đang active"""
    positions = load_active_positions()
    
    if not positions:
        return
    
    print(f"\n📊 Kiểm tra {len(positions)} positions đang active...")
    
    for symbol, pos_data in list(positions.items()):
        try:
            df = load_data(symbol, LOOKBACK)
            if df.empty:
                continue
            
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Get ML signal
            ml_signal = ml_generator.analyze(df)
            
            # Check exit
            exit_decision = exit_strategy.check_exit(
                symbol=symbol,
                entry_price=pos_data['entry_price'],
                current_price=current_price,
                stop_loss=pos_data['stop_loss'],
                take_profit_targets=pos_data['take_profit_targets'],
                entry_date=datetime.fromisoformat(pos_data['entry_date']),
                df=df,
                ml_signal=ml_signal,
                market_regime=market_regime,
                partial_exits=pos_data.get('partial_exits', [])
            )
            
            if exit_decision.should_exit:
                msg = exit_strategy.format_exit_message(symbol, exit_decision)
                await bot_instance.send_message(chat_id, msg, parse_mode='Markdown')
                
                # Update position
                if exit_decision.exit_type == 'FULL':
                    position_sizer.close_position(symbol, current_price)
                    del positions[symbol]
                    print(f"🔴 Đã đóng {symbol}: {exit_decision.exit_reason.value}")
                else:
                    pos_data.setdefault('partial_exits', []).append(current_price)
                    print(f"🟡 Chốt lời 1 phần {symbol}: {exit_decision.exit_type}")
                
                save_active_positions(positions)
            
            # Update price
            position_sizer.update_position_price(symbol, current_price)
            
        except Exception as e:
            log_error(f"Lỗi check exit {symbol}: {e}")


def format_entry_recommendation(symbol, entry_signal, position, market_regime):
    """Format entry recommendation message"""
    
    msg = f"🎯 *TÍN HIỆU VÀO LỆNH - {symbol}*\n\n"
    
    # Market context
    if market_regime:
        msg += f"📊 *Market:* {market_regime['regime']} ({market_regime['confidence']}%)\n\n"
    
    # Signal info
    msg += f"💪 *Signal:* {entry_signal.strength.name}\n"
    msg += f"🎲 *Confidence:* {entry_signal.confidence}%\n"
    msg += f"📈 *Shares:* {position.shares:,} ({position.shares//100} lô)\n"
    msg += f"💰 *Value:* {position.value:,.0f} VNĐ ({position.position_percent:.1f}%)\n\n"
    
    # Entry prices (DCA)
    if position.recommended_entries:
        msg += f"💵 *GIÁ VÀO (DCA):*\n"
        for entry in position.recommended_entries[:2]:  # Show top 2
            msg += f"  L{entry['level']}: {entry['price']:,.0f} - "
            msg += f"{entry['shares']:,} CP ({entry['percent']}%)\n"
        msg += "\n"
    
    # Stop loss
    msg += f"🛑 *Stop Loss:* {entry_signal.stop_loss:,.0f} VNĐ "
    sl_pct = ((entry_signal.stop_loss - entry_signal.entry_price)/entry_signal.entry_price * 100)
    msg += f"({sl_pct:+.1f}%)\n\n"
    
    # Take profits
    msg += f"🎯 *Take Profit:*\n"
    for i, tp in enumerate(entry_signal.take_profit_targets[:2], 1):  # Show TP1, TP2
        tp_pct = ((tp - entry_signal.entry_price) / entry_signal.entry_price) * 100
        msg += f"  TP{i}: {tp:,.0f} (+{tp_pct:.1f}%)\n"
    
    # Top reasons
    if entry_signal.reasons:
        msg += f"\n✅ *Lý do:*\n"
        for reason in entry_signal.reasons[:2]:
            msg += f"• {reason}\n"
    
    # Warnings
    if entry_signal.warnings:
        msg += f"\n⚠️ *Cảnh báo:* {entry_signal.warnings[0]}\n"
    
    # Risk
    msg += f"\n💸 *Risk:* {position.max_loss:,.0f} VNĐ ({position.risk_percent:.2f}%)"
    
    return msg


# ================ FILE I/O ====================

def load_selected_tickers():
    """Load tickers đã chọn"""
    if os.path.exists(SELECTED_TICKERS_FILE):
        try:
            with open(SELECTED_TICKERS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data['tickers']
        except Exception as e:
            log_error(f"Lỗi load tickers: {e}")
    
    # Fallback
    return TICKERS[:10] if len(TICKERS) >= 10 else TICKERS


def load_active_positions():
    """Load active positions"""
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log_error(f"Lỗi load positions: {e}")
    
    return {}


def save_active_positions(positions):
    """Save active positions"""
    try:
        with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(positions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log_error(f"Lỗi save positions: {e}")


def save_pending_position(symbol, entry_signal, position):
    """Save position vừa có signal"""
    positions = load_active_positions()
    
    positions[symbol] = {
        'entry_date': datetime.now().isoformat(),
        'entry_price': entry_signal.entry_price,
        'stop_loss': entry_signal.stop_loss,
        'take_profit_targets': entry_signal.take_profit_targets,
        'shares': position.shares,
        'partial_exits': []
    }
    
    save_active_positions(positions)


def log_error(msg):
    """Log error"""
    try:
        with open(os.path.join(LOGS_DIR, 'bot_errors.log'), 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass
    print(f"⚠️ {msg}")


# ================ RUNNER ====================

async def run_bot():
    """Main runner"""
    await run_bot_with_context(bot, CHAT_ID)


def run_bot_sync():
    """Sync wrapper"""
    asyncio.run(run_bot())


# ================ MAIN ====================

if __name__ == "__main__":
    """Test chạy bot"""
    print("\n" + "="*70)
    print("🤖 TESTING BOT RUNNER")
    print("="*70 + "\n")
    
    # Test market check
    print("1️⃣ Testing market check...")
    can_trade, message = check_market_before_trading()
    print(f"   Can trade: {can_trade}")
    print(f"   Message: {message}\n")
    
    # Test load tickers
    print("2️⃣ Testing load tickers...")
    tickers = load_selected_tickers()
    print(f"   Loaded {len(tickers)} tickers\n")
    
    # Info
    print("✅ Bot runner OK!")
    print("\nĐể chạy bot thực tế:")
    print("  python -c 'from bot_runner_improved import run_bot_sync; run_bot_sync()'")
    print("\nHoặc integrate vào main.py")