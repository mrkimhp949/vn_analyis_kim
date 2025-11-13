# Cleanup Messages - Checklist

## Đã sửa

### market_regime_proxy.py
- ✅ "LOI" → "⚠️" hoặc "⏭️"
- ✅ "Dang phan tich" → "📊 Phân tích"
- ✅ "Khong du du lieu" → "⏭️ Không đủ dữ liệu"
- ✅ "DA PHAN TICH" → "✅ Đã phân tích"
- ✅ "Khoi tao" → "🔧 Khởi tạo"
- ✅ "PHAN TICH THI TRUONG" → "📈 PHÂN TÍCH THỊ TRƯỜNG"

### bot_runner_improved.py
- ✅ "Quet" → "🔍 Quét"

### data_loader.py
- ✅ Suppress yfinance ERROR logging

## Cải thiện error handling

### ValueError từ Yahoo Finance
```python
except ValueError as e:
    error_msg = str(e)
    if "hủy niêm yết" in error_msg or "không tồn tại" in error_msg:
        # Bỏ qua mã không hợp lệ
        continue
```

## Best Practices

1. **Dùng emoji** cho messages dễ nhìn:
   - 📊 Phân tích
   - ✅ Thành công
   - ⚠️ Cảnh báo
   - ❌ Lỗi
   - ⏭️ Bỏ qua
   - 🔍 Tìm kiếm
   - 📈 Thị trường

2. **Có dấu tiếng Việt** cho messages người dùng

3. **Suppress logging** từ thư viện bên ngoài:
   - yfinance ERROR
   - TensorFlow warnings
   - Transformers warnings

4. **Error handling rõ ràng**:
   - Catch ValueError riêng
   - Check error message cụ thể
   - Bỏ qua mã không hợp lệ thay vì crash
