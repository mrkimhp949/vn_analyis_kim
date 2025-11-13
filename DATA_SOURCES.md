# Data Sources - Nguồn dữ liệu

## Tổng quan

Bot sử dụng 2 nguồn dữ liệu khác nhau:

### 1. Danh sách mã cổ phiếu (Ticker List)

**Nguồn**: Static list trong `config.py`

**Mặc định**: 71 mã từ KIM_SECTOR và THUY_SECTOR

**Tùy chọn**: TCBS API (khi `USE_DYNAMIC_TICKERS=true`)

```python
# config.py
KIM_SECTOR = {
    'banks_big4': ['VCB', 'CTG', 'BID', 'TCB'],
    'securities': ['SSI', 'VND', 'HCM', ...],
    ...
}
```

### 2. Dữ liệu giá (Price Data)

**Nguồn**: TCBS API ✅

**Lý do**: 
- Miễn phí, không cần API key
- Hỗ trợ TỐT cổ phiếu Việt Nam
- Dữ liệu realtime và lịch sử đầy đủ
- API ổn định

**Ưu điểm**:
- ✅ Hỗ trợ tất cả mã VN (VCB, FPT, VNM, HPG, ACB, BID...)
- ✅ Dữ liệu chính xác
- ✅ Không bị giới hạn rate limit nghiêm ngặt

**Xử lý**:
- Cache dữ liệu để giảm API calls
- Tự động retry khi timeout
- Validate dữ liệu trước khi sử dụng

## Workflow

```
1. Load danh sách mã từ config.py (71 mã static)
   ↓
2. Với mỗi mã, load dữ liệu giá từ TCBS API
   ↓
3. Nếu không có dữ liệu → Bỏ qua mã đó (hiếm khi xảy ra)
   ↓
4. Nếu có dữ liệu → Phân tích và trade
```

## Validation

### Kiểm tra mã hợp lệ (TCBS API)
```bash
python validate_tickers.py
```

### Kiểm tra nhanh
```bash
python check_invalid_tickers.py
```

## Cấu hình

### Environment Variables

```bash
# Danh sách mã
USE_DYNAMIC_TICKERS=false  # Dùng static list (khuyến nghị)
MIN_VOLUME=100000          # Volume tối thiểu (nếu dùng dynamic)

# Override thủ công
TICKERS=VCB,FPT,VNM,HPG    # Danh sách mã cụ thể
```

## API được sử dụng

### TCBS API (Chính) ✅
- **Mục đích**: Load dữ liệu giá OHLCV, validate mã
- **Endpoint**: `https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term`
- **Rate limit**: ~10 requests/second
- **Cost**: Miễn phí
- **Hỗ trợ**: Tất cả mã VN

### Yahoo Finance (Backup - Không dùng cho VN)
- **Mục đích**: Chỉ dùng cho mã quốc tế (nếu cần)
- **Hạn chế**: Không hỗ trợ mã VN
- **File backup**: `data_loader_yahoo_backup.py`

## Troubleshooting

### Mã không có dữ liệu
```
⏭️ CMG: Bỏ qua (không có dữ liệu)
```
→ Mã này không có trên Yahoo Finance, bot tự động bỏ qua

### Lỗi kết nối
```
⚠️ Lỗi tải dữ liệu: Connection timeout
```
→ Kiểm tra kết nối internet, thử lại sau

### Dữ liệu không đủ
```
ValueError: Dữ liệu không đủ cho XYZ: có 10 nến, cần ít nhất 50
```
→ Mã này có quá ít dữ liệu lịch sử, không thể phân tích

## Best Practices

1. **Dùng static list** cho production (ổn định)
2. **Validate định kỳ** để loại bỏ mã không hợp lệ
3. **Cache dữ liệu** để giảm API calls
4. **Monitor logs** để phát hiện mã có vấn đề
5. **Backup data** định kỳ

## Tài liệu liên quan

- `config.py` - Cấu hình danh sách mã
- `data_loader.py` - Load dữ liệu từ Yahoo Finance
- `ticker_fetcher.py` - Lấy danh sách từ TCBS API
- `validate_tickers.py` - Validate mã qua TCBS
- `TICKER_MANAGEMENT.md` - Hướng dẫn quản lý mã
