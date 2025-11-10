import os
from typing import List

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

# ═══════════════════════════════════════════════════════════
# 🏦 NGÀNH KIM (Tài chính, Ngân hàng, Chứng khoán)
# ═══════════════════════════════════════════════════════════
KIM_SECTOR = {
    'banks_big4': ['VCB', 'CTG', 'BID', 'TCB'],
    'banks_other': ['MBB', 'ACB', 'VPB', 'STB', 'TPB', 'VIB', 'HDB', 'SHB', 'MSB', 'OCB'],
    'securities': ['SSI', 'VND', 'HCM', 'VCI', 'FTS', 'MBS', 'BSI', 'AGR'],
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

# Danh sách mặc định
KIM_TICKERS = get_all_tickers(KIM_SECTOR)
THUY_TICKERS = get_all_tickers(THUY_SECTOR)
ALL_TICKERS = sorted(list(set(KIM_TICKERS + THUY_TICKERS)))

# Có thể override tickers qua env (Render)
TICKERS = get_env_list('TICKERS', ALL_TICKERS[:10])  # Giới hạn 10 mã trên Render

print(f"📊 Đang theo dõi {len(TICKERS)} mã cổ phiếu trên Render")