# [file name]: bot_runner_improved.py
# [file content begin]
# -*- coding: utf-8 -*-

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from telegram import Bot
import pandas as pd

# ===== CONFIG IMPORTS =====
try:
    from config import (
        TICKERS, LOOKBACK, CHAT_ID, TELEGRAM_TOKEN,
        KIM_SECTOR, THUY_SECTOR, ALL_TICKERS
    )
    print("✅ Import config thành công")
except ImportError as e:
    print(f"❌ Lỗi import config: {e}")
    # Fallback values
    TICKERS = ['VNM', 'VCB', 'HPG', 'FPT', 'SSI']
    LOOKBACK = 100
    CHAT_ID = "5501113513"
    TELEGRAM_TOKEN = "234790554:AAFbdwZ3zi0ocpELA0gav6qeYqDKXbDg-yI"
    KIM_SECTOR = {}
    THUY_SECTOR = {}
    ALL_TICKERS = TICKERS

# ===== ORIGINAL MODULES =====
try:
    from data_loader import load_data
    print("✅ Import data_loader thành công")
except ImportError as e:
    print(f"❌ Lỗi import data_loader: {e}")
    def load_data(symbol, lookback):
        return pd.DataFrame()

try:
    from ml_signals import MLSignalGenerator
    ml_generator = MLSignalGenerator()
    print("✅ Import ml_signals thành công")
except ImportError as e:
    print(f"❌ Lỗi import ml_signals: {e}")
    class MLSignalGenerator:
        def analyze(self, df):
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'ML không khả dụng'}
    ml_generator = MLSignalGenerator()

# ===== NEW IMPROVED MODULES =====
try:
    from market_regime_proxy import ProxyMarketRegimeAnalyzer as MarketAnalyzer
    market_analyzer = MarketAnalyzer()
    print("✅ Using Proxy Market Analyzer (Blue-chip stocks)")
except ImportError as e:
    print(f"⚠️ No market analyzer available: {e}")
    market_analyzer = None

try:
    from improved_sector_analysis import EnhancedSectorAnalyzer
    sector_analyzer = EnhancedSectorAnalyzer(
        min_volume=500_000,
        min_price=10_000
    )
    print("✅ Import improved_sector_analysis thành công")
except ImportError as e:
    print(f"❌ Lỗi import improved_sector_analysis: {e}")
    sector_analyzer = None

try:
    from improved_entry_logic import ImprovedEntryLogic
    entry_logic = ImprovedEntryLogic(
        min_confidence=50,
        min_risk_reward=2.0,
        require_trend_alignment=True,
        require_volume_confirmation=False
    )
    print("✅ Import improved_entry_logic thành công")
except ImportError as e:
    print(f"❌ Lỗi import improved_entry_logic: {e}")
    entry_logic = None

try:
    from improved_position_sizing import ConservativePositionSizer
    position_sizer = ConservativePositionSizer(
        total_capital=100_000_000,
        max_risk_per_trade=0.02,
        max_position_size=0.10,
        max_total_exposure=0.60,
        min_positions=8
    )
    print("✅ Import improved_position_sizing thành công")
except ImportError as e:
    print(f"❌ Lỗi import improved_position_sizing: {e}")
    position_sizer = None

try:
    from improved_exit_logic import ImprovedExitStrategy
    exit_strategy = ImprovedExitStrategy()
    print("✅ Import improved_exit_logic thành công")
except ImportError as e:
    print(f"❌ Lỗi import improved_exit_logic: {e}")
    exit_strategy = None

try:
    from news_analyzer import analyze_news_trend, get_top_news
    print("✅ Import news_analyzer thành công")
except ImportError as e:
    print(f"⚠️ Không thể import news_analyzer: {e}")
    analyze_news_trend = None
    get_top_news = None

# Initialize bot
try:
    bot = Bot(token=TELEGRAM_TOKEN)
    print("✅ Telegram bot initialized")
except Exception as e:
    print(f"❌ Lỗi khởi tạo Telegram bot: {e}")
    bot = None

# Files
SELECTED_TICKERS_FILE = 'selected_tickers.json'
POSITIONS_FILE = 'active_positions.json'
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)

# Scan & risk configs
MAX_SCAN_UNIVERSE = int(os.getenv('MAX_SCAN_UNIVERSE', '40'))
SECTOR_CACHE_TTL_DAYS = int(os.getenv('SECTOR_CACHE_TTL_DAYS', '7'))
WATCHLIST_SIZE = int(os.getenv('WATCHLIST_SIZE', '5'))


# ======================================================
# HELPER FUNCTIONS - IMPROVED WITH ERROR HANDLING
# ======================================================

def check_market_before_trading():
    """
    Check market regime trước khi trade với error handling
    
    Returns:
        (can_trade: bool, message: str)
    """
    if not market_analyzer:
        return True, "⚠️ Không có market analyzer - trade thận trọng"
    
    try:
        result = market_analyzer.analyze_market_regime()
        return result['tradeable'], result['message']
    except Exception as e:
        log_error(f"Lỗi check market: {e}")
        return True, f"⚠️ Lỗi check market - trade thận trọng"


# ======================================================
# PHÂN TÍCH NGÀNH - IMPROVED VERSION
# ======================================================

def run_sector_analysis():
    """Phân tích ngành với enhanced analyzer và error handling"""
    print("🔍 Bắt đầu phân tích thị trường (Enhanced)...")
    
    if not sector_analyzer:
        print("❌ Sector analyzer không khả dụng")
        return TICKERS[:10]
    
    # Combine all sectors
    all_sectors = {}
    try:
        all_sectors.update({f"Kim_{k}": v for k, v in KIM_SECTOR.items()})
        all_sectors.update({f"Thuy_{k}": v for k, v in THUY_SECTOR.items()})
    except Exception as e:
        print(f"⚠️ Lỗi combine sectors: {e}")
        # Fallback to basic sectors
        all_sectors = {'Basic': TICKERS[:20]}
    
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
    """Gửi summary qua Telegram với error handling"""
    if not bot:
        print("❌ Bot không khả dụng, không gửi Telegram")
        return
        
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
# DATA & STRATEGY HELPERS
# ======================================================

def _parse_datetime(value):
    """Parse ISO datetime string safely"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def _load_sector_snapshot():
    """Đọc cached sector snapshot nếu còn hạn"""
    if not os.path.exists(SELECTED_TICKERS_FILE):
        return None
    try:
        with open(SELECTED_TICKERS_FILE, 'r', encoding='utf-8') as f:
            snapshot = json.load(f)
        analyzed_at = _parse_datetime(snapshot.get('selected_at'))
        if analyzed_at and datetime.now() - analyzed_at > timedelta(days=SECTOR_CACHE_TTL_DAYS):
            print("ℹ️ Sector snapshot đã quá hạn -> cần refresh")
            return None
        return snapshot
    except Exception as e:
        log_error(f"Lỗi đọc sector snapshot: {e}")
        return None


def get_selected_tickers(force_refresh=False, max_tickers=MAX_SCAN_UNIVERSE):
    """
    Trả về (tickers, snapshot) - ưu tiên sử dụng dữ liệu cache để tránh quét lặp.
    """
    snapshot = None if force_refresh else _load_sector_snapshot()

    if snapshot is None:
        selected = run_sector_analysis()
        snapshot = _load_sector_snapshot() or {
            'tickers': selected,
            'selected_at': datetime.now().isoformat()
        }

    tickers = snapshot.get('tickers') or TICKERS
    tickers = sorted(set(tickers)) or TICKERS

    if max_tickers and max_tickers > 0:
        tickers = tickers[:max_tickers]

    return tickers, snapshot


def apply_market_adjustments(market_regime):
    """Điều chỉnh tham số chiến lược theo market regime"""
    if not entry_logic or not position_sizer:
        return

    regime = (market_regime or {}).get('regime', 'UNKNOWN').upper()

    if regime == 'BULL':
        entry_logic.min_confidence = 55
        entry_logic.min_risk_reward = 1.8
        entry_logic.require_trend_alignment = True
        position_sizer.max_total_exposure = 0.70
        position_sizer.min_positions = 6
    elif regime == 'BEAR':
        entry_logic.min_confidence = 50
        entry_logic.min_risk_reward = 1.4
        entry_logic.require_trend_alignment = False
        position_sizer.max_total_exposure = 0.30
        position_sizer.min_positions = 2
    else:  # SIDEWAYS / UNKNOWN
        entry_logic.min_confidence = 55
        entry_logic.min_risk_reward = 1.6
        entry_logic.require_trend_alignment = True
        position_sizer.max_total_exposure = 0.50
        position_sizer.min_positions = 4

    print(
        "⚙️ Điều chỉnh chiến lược "
        f"(regime={regime}, min_conf={entry_logic.min_confidence}, "
        f"R:R>={entry_logic.min_risk_reward}, trend_required={entry_logic.require_trend_alignment}, "
        f"max_exposure={position_sizer.max_total_exposure*100:.0f}%)"
    )


def sync_position_sizer_with_active_positions(active_positions):
    """Đồng bộ position_sizer.current_positions với vị thế đang nắm"""
    if not position_sizer:
        return

    position_sizer.current_positions = {}
    for symbol, pos in active_positions.items():
        shares = pos.get('shares', 0)
        if shares <= 0:
            continue
        entry_price = pos.get('entry_price', 0)
        position_sizer.current_positions[symbol] = {
            'shares': shares,
            'entry_price': entry_price,
            'current_price': pos.get('current_price', entry_price),
            'unrealized_pnl': 0
        }


# ======================================================
# BOT RUNNER - IMPROVED VERSION WITH ERROR HANDLING
# ======================================================
async def check_portfolio_and_recommend(bot_instance, chat_id):
    """Kiểm tra portfolio và đề xuất mua/bán"""
    print("\n🔍 Kiểm tra portfolio hiện tại...")
    
    if not bot_instance:
        print("❌ Bot không khả dụng")
        return
    
    try:
        from portfolio_manager import PortfolioManager
        
        # Khởi tạo portfolio manager
        manager = PortfolioManager()
        
        # Lấy phân tích chi tiết
        analysis_report = manager.get_detailed_analysis()
        
        # Gửi qua Telegram
        if len(analysis_report) > 4000:
            # Chia nhỏ message nếu quá dài
            parts = [analysis_report[i:i+4000] for i in range(0, len(analysis_report), 4000)]
            for part in parts:
                await bot_instance.send_message(chat_id, part, parse_mode='Markdown')
        else:
            await bot_instance.send_message(chat_id, analysis_report, parse_mode='Markdown')
        
        print("✅ Đã gửi phân tích portfolio")
        
    except Exception as e:
        error_msg = f"❌ Lỗi kiểm tra portfolio: {e}"
        print(error_msg)
        await bot_instance.send_message(chat_id, error_msg)

async def run_bot_with_context(bot_instance, chat_id):
    """Bot runner với improved logic và error handling"""
    
    if not bot_instance:
        print("❌ Bot instance không khả dụng")
        return
    
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
    
    apply_market_adjustments(market_regime)

    # Đồng bộ các vị thế hiện tại
    active_positions = load_active_positions()
    existing_symbols = set(active_positions.keys())
    sync_position_sizer_with_active_positions(active_positions)

    # ===== CHECK 2: LOAD TICKERS =====
    current_tickers, sector_snapshot = get_selected_tickers()
    print(f"🔍 Quét {len(current_tickers)} mã...")
    
    signal_count = 0
    watchlist_candidates = []

    top_sectors = sector_snapshot.get('top_sectors')[:3] if sector_snapshot else []
    sector_text = "\n".join([f"   • {s}" for s in top_sectors]) if top_sectors else "   • N/A"

    try:
        regime_text = market_regime['regime'] if market_regime else 'UNKNOWN'
        await bot_instance.send_message(
            chat_id=chat_id,
            text=(
                f"🔍 Đang quét {len(current_tickers)} mã...\n"
                f"📊 Market: {regime_text}\n"
                f"🏆 Top sectors:\n{sector_text}"
            )
        )
    except Exception as e:
        log_error(f"Lỗi gửi Telegram: {e}")

    # ===== CHECK 3: CHECK EXITS TRƯỚC =====
    await check_active_positions(bot_instance, chat_id, market_regime)
    
    # ===== CHECK 4: SCAN FOR NEW ENTRIES =====
    for symbol in current_tickers:
        try:
            if symbol in existing_symbols:
                print(f"⏭️ Bỏ qua {symbol} (đã có vị thế)")
                continue

            df = load_data(symbol, LOOKBACK)
            if df.empty or len(df) < 50:
                continue
            
            # Get ML signal
            ml_signal = ml_generator.analyze(df)
            
            # Skip nếu không có entry logic
            if not entry_logic:
                continue
                
            # ===== IMPROVED ENTRY LOGIC =====
            entry_signal = entry_logic.analyze_entry(
                df=df,
                ml_signal=ml_signal,
                market_regime=market_regime
            )
            
            news_context = analyze_news_trend(symbol) if analyze_news_trend else None
            news_sentiment = news_context.get("sentiment_score", 0.0) if news_context else 0.0
            if news_context and news_context.get("articles"):
                if news_sentiment >= 0.5:
                    entry_signal.confidence = min(100, entry_signal.confidence + 5)
                    entry_signal.reasons.append(f"📰 Tin tức tích cực ({news_sentiment:+.2f})")
                elif news_sentiment <= -0.5:
                    entry_signal.confidence = max(0, entry_signal.confidence - 7)
                    entry_signal.warnings.append(f"📰 Tin tức tiêu cực ({news_sentiment:+.2f})")
                else:
                    entry_signal.reasons.append(f"📰 Tin tức trung lập ({news_sentiment:+.2f})")

            if entry_signal.should_enter and entry_signal.confidence < entry_logic.min_confidence:
                entry_signal.should_enter = False
                entry_signal.warnings.append(
                    f"Confidence giảm xuống dưới ngưỡng sau khi điều chỉnh tin tức ({entry_signal.confidence}%)"
                )

            if not entry_signal.should_enter:
                confidence_for_watchlist = max(ml_signal.get('confidence', 0), entry_signal.confidence)
                if confidence_for_watchlist >= max(0, entry_logic.min_confidence - 5):
                    reason = ", ".join(entry_signal.warnings) if entry_signal.warnings else "Không đạt bộ lọc"
                    top_headline = ""
                    if news_context and news_context.get("top_headlines"):
                        top_headline = news_context["top_headlines"][0]["title"]
                    watchlist_candidates.append({
                        'symbol': symbol,
                        'confidence': confidence_for_watchlist,
                        'reason': reason,
                        'sentiment': news_sentiment,
                        'headline': top_headline
                    })
                continue
            
            latest = df.iloc[-1]
            price = latest['close']
            
            # Skip nếu không có position sizer
            if not position_sizer:
                continue
                
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

            if position_sizer and symbol not in position_sizer.current_positions:
                position_sizer.add_position(symbol, position.shares, entry_signal.entry_price)
                existing_symbols.add(symbol)
            
            # ===== FORMAT MESSAGE =====
            msg = format_entry_recommendation(
                symbol, 
                entry_signal, 
                position,
                market_regime,
                news_context=news_context
            )
            
            await bot_instance.send_message(chat_id, msg, parse_mode='Markdown')
            
            # Save to pending positions
            save_pending_position(symbol, entry_signal, position)
            
            signal_count += 1
            print(f"✅ {symbol}: {entry_signal.signal_type} ({entry_signal.confidence}%)")
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            log_error(f"Lỗi quét {symbol}: {e}")
    
    # ===== CHECK 5: PORTFOLIO ANALYSIS =====
    print("\n📊 Kiểm tra portfolio...")
    await check_portfolio_and_recommend(bot_instance, chat_id)
    # Summary
    regime_text = market_regime['regime'] if market_regime else 'UNKNOWN'
    summary = f"""
✅ Hoàn thành quét {len(current_tickers)} mã
🎯 Tín hiệu hợp lệ: {signal_count}
📊 Market: {regime_text}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    try:
        await bot_instance.send_message(chat_id, text=summary)
    except Exception as e:
        log_error(f"Lỗi gửi summary: {e}")
    print(summary)

    if signal_count == 0:
        if watchlist_candidates:
            watchlist_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            top_watchlist = watchlist_candidates[:WATCHLIST_SIZE]
            lines = [
                "• {symbol}: {conf:.0f}% - {reason}{sentiment}{headline}".format(
                    symbol=item['symbol'],
                    conf=item['confidence'],
                    reason=item['reason'],
                    sentiment=f", sentiment {item.get('sentiment', 0.0):+.2f}" if 'sentiment' in item else "",
                    headline=f" | {item['headline']}" if item.get('headline') else ""
                )
                for item in top_watchlist
            ]
            watchlist_msg = "👀 *WATCHLIST* (chưa đủ điều kiện BUY):\n" + "\n".join(lines)
        else:
            watchlist_msg = "⚠️ Thị trường chưa có mã nào đạt điều kiện BUY. Tiếp tục quan sát."
        try:
            await bot_instance.send_message(chat_id, text=watchlist_msg, parse_mode='Markdown')
        except Exception as e:
            log_error(f"Lỗi gửi watchlist: {e}")


async def check_active_positions(bot_instance, chat_id, market_regime):
    """Check các positions đang active với error handling"""
    if not bot_instance or not exit_strategy:
        return
        
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
                    if position_sizer:
                        position_sizer.close_position(symbol, current_price)
                    del positions[symbol]
                    print(f"🔴 Đã đóng {symbol}: {exit_decision.exit_reason.value}")
                else:
                    pos_data.setdefault('partial_exits', []).append(current_price)
                    print(f"🟡 Chốt lời 1 phần {symbol}: {exit_decision.exit_type}")
                
                save_active_positions(positions)
            
            # Update price
            if position_sizer:
                position_sizer.update_position_price(symbol, current_price)
            
        except Exception as e:
            log_error(f"Lỗi check exit {symbol}: {e}")


def format_entry_recommendation(symbol, entry_signal, position, market_regime, news_context=None):
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

    if news_context and news_context.get("articles"):
        msg += f"\n\n📰 *News Sentiment:* {news_context['sentiment_label']} ({news_context['sentiment_score']:+.2f})\n"
        for article in news_context.get("top_headlines", [])[:2]:
            published = article.get("published_at", "")[:16].replace("T", " ")
            msg += f"  • {article['title']} ({article['source']}, {published})\n"
            if article.get("url"):
                msg += f"    {article['url']}\n"
    
    return msg


# ================ FILE I/O ====================

def load_selected_tickers():
    tickers, _ = get_selected_tickers(force_refresh=False)
    print(f"Quet {len(tickers)} ma tu snapshot/config")
    return tickers

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
    try:
        asyncio.run(run_bot())
    except Exception as e:
        log_error(f"Lỗi chạy bot: {e}")

# Thêm vào bot_runner_improved.py

def analyze_current_portfolio():
    """Phân tích portfolio hiện tại"""
    from portfolio_manager import PortfolioManager
    
    print("🔍 Phân tích portfolio hiện tại...")
    manager = PortfolioManager()
    
    # Lấy phân tích
    analysis_report = manager.get_detailed_analysis()
    
    # Gửi qua Telegram
    try:
        asyncio.run(send_telegram_message(analysis_report))
    except Exception as e:
        print(f"⚠️ Không gửi được Telegram: {e}")
    
    print(analysis_report)
    return analysis_report

async def send_telegram_message(message):
    """Gửi message qua Telegram"""
    if bot:
        try:
            await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
        except Exception as e:
            print(f"❌ Lỗi gửi Telegram: {e}")

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
# [file content end]