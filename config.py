import os

TELEGRAM_TOKEN = "8234790554:AAFbdwZ3zi0ocpELA0gav6qeYqDKXbDg-yI"
CHAT_ID = "5501113513"

RESOLUTION = "1D"
LOOKBACK = 200  # Lấy 200 phiên gần nhất

# ═══════════════════════════════════════════════════════════
# 🏦 NGÀNH KIM (Tài chính, Ngân hàng, Chứng khoán)
# ═══════════════════════════════════════════════════════════
KIM_SECTOR = {
    # Ngân hàng lớn (Big 4)
    'banks_big4': ['VCB', 'CTG', 'BID', 'TCB'],
    
    # Ngân hàng khác
    'banks_other': ['MBB', 'ACB', 'VPB', 'STB', 'TPB', 'VIB', 'HDB', 'SHB', 'MSB', 'OCB'],
    
    # Chứng khoán
    'securities': ['SSI', 'VND', 'HCM', 'VCI', 'FTS', 'MBS', 'BSI', 'AGR'],
    
    # Bảo hiểm
    'insurance': ['BVH', 'BMI', 'PVI', 'PTI', 'BIC', 'PGI', 'VNR', 'MIG'],
    
    # Tài chính/Cho vay
    'finance': ['FLC', 'VCS', 'CTS', 'ORS', 'IFS', 'TVS', 'APS', 'WSS']
}

# ═══════════════════════════════════════════════════════════
# 💻 NGÀNH THỦY (Công nghệ, Dữ liệu, Logistics, Viễn thông)
# ═══════════════════════════════════════════════════════════
THUY_SECTOR = {
    # Công nghệ thông tin
    'technology': ['FPT', 'CMG', 'VGI', 'SAM', 'ELC', 'ITD', 'VTP', 'SGT', 'CMT'],
    
    # Viễn thông
    'telecom': ['CTR', 'FOX', 'VNZ', 'SGN', 'VGS', 'ICT', 'TIG'],
    
    # Logistics/Vận tải
    'logistics': ['GMD', 'HAH', 'TMS', 'VSC', 'VOS', 'STG', 'PHP', 'SGP'],
    
    # E-commerce/Bán lẻ công nghệ
    'ecommerce': ['MWG', 'FRT', 'PNJ', 'DGW', 'PET'],
    
    # Dữ liệu/Dịch vụ số
    'digital': ['VNM', 'HDG', 'VNP', 'VTO', 'DAG']
}

# ═══════════════════════════════════════════════════════════
# 📋 TẤT CẢ MÃ CỔ PHIẾU
# ═══════════════════════════════════════════════════════════

def get_all_tickers(sectors_dict):
    """Lấy tất cả mã từ một dict sectors"""
    tickers = []
    for category, stocks in sectors_dict.items():
        tickers.extend(stocks)
    return sorted(list(set(tickers)))  # Remove duplicates và sort

# Danh sách mặc định
KIM_TICKERS = get_all_tickers(KIM_SECTOR)
THUY_TICKERS = get_all_tickers(THUY_SECTOR)
ALL_TICKERS = sorted(list(set(KIM_TICKERS + THUY_TICKERS)))

# ═══════════════════════════════════════════════════════════
# ⚙️ CHỌN MÃ CẦN QUÉT (Thay đổi ở đây)
# ═══════════════════════════════════════════════════════════

# Option 1: Quét TẤT CẢ
TICKERS = ALL_TICKERS

# Option 2: Chỉ Kim (Tài chính)
# TICKERS = KIM_TICKERS

# Option 3: Chỉ Thủy (Công nghệ)
# TICKERS = THUY_TICKERS

# Option 4: Chọn từng nhóm cụ thể
# TICKERS = KIM_SECTOR['banks_big4'] + THUY_SECTOR['technology']

# Option 5: Tùy chỉnh
# TICKERS = ['VCB', 'TCB', 'FPT', 'MWG', 'SSI']

print(f"📊 Đang theo dõi {len(TICKERS)} mã cổ phiếu:")
print(f"  • Kim (Tài chính): {len(KIM_TICKERS)} mã")
print(f"  • Thủy (Công nghệ): {len(THUY_TICKERS)} mã")
print(f"  • Tổng: {len(ALL_TICKERS)} mã")