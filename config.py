"""
DEPRECATED: Use trading_config.py instead
This file is kept for backward compatibility only
"""
import os
import sys
from typing import List
import warnings

warnings.warn(
    "config.py is deprecated. Use 'from trading_config import get_config' instead",
    DeprecationWarning,
    stacklevel=2
)

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Install: pip install python-dotenv")

# Fix encoding cho Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Import from new config
from trading_config import get_config

_config = get_config()

# Backward compatibility exports
TELEGRAM_TOKEN = _config.telegram.token
CHAT_ID = _config.telegram.chat_id
RESOLUTION = "1D"
LOOKBACK = _config.data.lookback
USE_DYNAMIC_TICKERS = _config.data.use_dynamic_tickers
MIN_VOLUME = _config.data.min_volume

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
    env_tickers = os.getenv('TICKERS')
    if env_tickers:
        tickers = [x.strip() for x in env_tickers.split(',')]
        print(f"📊 Sử dụng {len(tickers)} mã từ env variable")
        return tickers
    
    if _config.data.use_dynamic_tickers:
        try:
            from ticker_fetcher import get_active_tickers
            tickers = get_active_tickers(min_volume=_config.data.min_volume, use_cache=True)
            print(f"📊 Lấy {len(tickers)} mã từ TCBS API (volume >= {_config.data.min_volume:,})")
            return tickers
        except Exception as e:
            print(f"⚠️ Lỗi lấy dynamic tickers: {e}")
            print(f"📊 Fallback: Sử dụng {len(ALL_TICKERS_STATIC)} mã static")
            return ALL_TICKERS_STATIC
    
    print(f"📊 Sử dụng {len(ALL_TICKERS_STATIC)} mã static từ config")
    return ALL_TICKERS_STATIC

# Lấy danh sách tickers
TICKERS = get_tickers()