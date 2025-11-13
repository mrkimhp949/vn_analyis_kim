# Warnings Suppression

## Tổng quan

Module `suppress_warnings.py` tắt tất cả warnings không cần thiết từ các thư viện:
- TensorFlow/Keras deprecation warnings
- Transformers FutureWarnings
- oneDNN info messages
- Và các warnings khác

## Cách sử dụng

Import ở đầu file chính:

```python
import suppress_warnings  # noqa: F401
```

## Các file đã áp dụng

- `bot_runner_improved.py` - Main bot runner
- `main.py` - FastAPI server
- `logging_config.py` - Logging setup
- `ml_pipeline/sentiment_model.py` - Sentiment analysis
- `data_loader.py` - Data loading (yfinance)

## Environment Variables được set

```bash
TF_ENABLE_ONEDNN_OPTS=0          # Tắt oneDNN
TF_CPP_MIN_LOG_LEVEL=3           # Chỉ hiện ERROR
TF_CPP_MIN_VLOG_LEVEL=3          # Tắt verbose logging
```

## Warnings được suppress

### TensorFlow/Keras
- Deprecation warnings (tf.compat.v1.*)
- oneDNN custom operations messages
- Keras API deprecations

### Transformers
- FutureWarning về resume_download
- HuggingFace Hub warnings

### yfinance
- ERROR messages về mã không tồn tại
- "No timezone found" errors
- Delisted ticker warnings

### Python Standard
- DeprecationWarning
- UserWarning không quan trọng

## Thêm warnings mới

Chỉnh sửa `suppress_warnings.py`:

```python
# Thêm vào cuối file
warnings.filterwarnings('ignore', message='.*your_pattern.*')
```

## Troubleshooting

Nếu vẫn thấy warnings:
1. Kiểm tra `suppress_warnings` được import TRƯỚC các thư viện khác
2. Thêm pattern cụ thể vào `suppress_warnings.py`
3. Restart Python process

## Lưu ý

- Chỉ suppress warnings không ảnh hưởng đến logic
- Không suppress ERROR hoặc CRITICAL
- Giữ warnings quan trọng về security/data loss
