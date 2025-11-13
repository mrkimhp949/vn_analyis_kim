# Quản lý danh sách mã cổ phiếu

## Cách sử dụng (Mặc định)

Bot sử dụng danh sách mã cố định trong `config.py`:

```python
KIM_SECTOR = {
    'banks_big4': ['VCB', 'CTG', 'BID', 'TCB'],
    'securities': ['SSI', 'VND', 'HCM', ...],
    ...
}
```

### Cập nhật danh sách
1. Mở `config.py`
2. Thêm/xóa mã trong sector tương ứng
3. Restart bot

### Validate danh sách
```bash
python validate_tickers.py
```

## Tùy chọn nâng cao: Dynamic Tickers

Bot có thể tự động lấy danh sách từ TCBS API (tắt mặc định):

```bash
# Bật trong .env
USE_DYNAMIC_TICKERS=true
MIN_VOLUME=100000
```

## TCBS API

Bot sử dụng TCBS API (miễn phí) để validate và lấy dữ liệu:
- Base URL: `https://apipubaws.tcbs.com.vn`
- Không cần API key
- Hỗ trợ tốt cổ phiếu VN

## Xử lý lỗi

Bot tự động bỏ qua mã không hợp lệ:
```
⚠️ Bỏ qua AGR: Mã có thể đã bị hủy niêm yết
```

## Các file liên quan

- `config.py` - Cấu hình danh sách mã
- `ticker_fetcher.py` - Module lấy danh sách từ TCBS
- `validate_tickers.py` - Script kiểm tra mã hợp lệ
- `data_loader.py` - Load dữ liệu từ Yahoo Finance
