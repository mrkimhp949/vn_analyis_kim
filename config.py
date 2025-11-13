import os
import sys
from typing import List

# Fix encoding cho Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def get_env_list(key: str, default: List[str]) -> List[str]:
    """Lấy list từ environment variable"""
    value = os.getenv(key)
    if value:
        return [x.strip() for x in value.split(',')]
    return default

# Sử dụng env vars từ Render
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', "8234790554:AAFbdwZ3zi0ocpELA0gav6qeYqDKXbDg-yI")
CHAT_ID = os.getenv('CHAT_ID', "5501113513")

RESOLUTION = "1D"
LOOKBACK = 200

# Cấu hình cho ticker fetcher
# Mặc định tắt dynamic tickers, dùng danh sách static trong config
USE_DYNAMIC_TICKERS = os.getenv('USE_DYNAMIC_TICKERS', 'false').lower() == 'true'
MIN_VOLUME = int(os.getenv('MIN_VOLUME', '100000'))  # Volume tối thiểu

# ═══════════════════════════════════════════════════════════
# 🏦 NGÀNH KIM (Tài chính, Ngân hàng, Chứng khoán)
# ═══════════════════════════════════════════════════════════
KIM_SECTOR = {
    'banks_big4': ['VCB', 'CTG', 'BID', 'TCB'],
    'banks_other': ['MBB', 'ACB', 'VPB', 'STB', 'TPB', 'VIB', 'HDB', 'SHB', 'MSB', 'OCB'],
    'securities': ['SSI', 'VND', 'HCM', 'VCI', 'FTS', 'MBS', 'BSI'],  # Removed AGR (delisted)
    'insurance': ['BVH', 'BMI', 'PVI', 'PTI', 'BIC', 'PGI', 'VNR', 'MIG'],
    'finance': ['FLC', 'VCS', 'CTS', 'ORS', 'IFS', 'TVS', 'APS', 'WSS']
}

# ═══════════════════════════════════════════════════════════
# 💻 NGÀNH THỦY (Công nghệ, Dữ liệu, Logistics, Viễn thông)
# ═══════════════════════════════════════════════════════════
THUY_SECTOR = {
    'technology': ['FPT', 'CMG', 'VGI', 'SAM', 'ELC', 'ITD', 'VTP', 'SGT', 'CMT'],
    'telecom': ['CTR', 'FOX', 'VNZ', 'SGN', 'VGS', 'ICT', 'TIG'],
    'logistics': ['GMD', 'HAH', 'TMS', 'VSC', 'VOS', 'STG', 'PHP', 'SGP'],
    'ecommerce': ['MWG', 'FRT', 'PNJ', 'DGW', 'PET'],
    'digital': ['VNM', 'HDG', 'VNP', 'VTO', 'DAG']
}

def get_all_tickers(sectors_dict):
    """Lấy tất cả mã từ một dict sectors"""
    tickers = []
    for category, stocks in sectors_dict.items():
        tickers.extend(stocks)
    return sorted(list(set(tickers)))

# Danh sách mặc định (fallback nếu không dùng dynamic tickers)
KIM_TICKERS = get_all_tickers(KIM_SECTOR)
THUY_TICKERS = get_all_tickers(THUY_SECTOR)
ALL_TICKERS_STATIC = sorted(list(set(KIM_TICKERS + THUY_TICKERS)))

def get_tickers() -> List[str]:
    """
    Lấy danh sách mã cổ phiếu
    - Nếu USE_DYNAMIC_TICKERS=true: Lấy từ Yahoo Finance
    - Nếu không: Dùng danh sách static trong config
    """
    # Kiểm tra env override trước
    env_tickers = os.getenv('TICKERS')
    if env_tickers:
        tickers = [x.strip() for x in env_tickers.split(',')]
        print(f"📊 Sử dụng {len(tickers)} mã từ env variable")
        return tickers
    
    # Nếu dùng dynamic tickers
    if USE_DYNAMIC_TICKERS:
        try:
            from ticker_fetcher import get_active_tickers
            tickers = get_active_tickers(min_volume=MIN_VOLUME, use_cache=True)
            print(f"📊 Lấy {len(tickers)} mã từ TCBS API (volume >= {MIN_VOLUME:,})")
            return tickers
        except Exception as e:
            print(f"⚠️ Lỗi lấy dynamic tickers: {e}")
            print(f"📊 Fallback: Sử dụng {len(ALL_TICKERS_STATIC)} mã static")
            return ALL_TICKERS_STATIC
    
    # Dùng danh sách static
    print(f"📊 Sử dụng {len(ALL_TICKERS_STATIC)} mã static từ config")
    return ALL_TICKERS_STATIC

# Lấy danh sách tickers
TICKERS = get_tickers()