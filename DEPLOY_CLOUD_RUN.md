# 🚀 Hướng dẫn Deploy lên Google Cloud Run

## 📋 Yêu cầu

1. **Google Cloud Account** (Miễn phí)
   - Đăng ký tại: https://cloud.google.com/free
   - Có $300 credit miễn phí trong 90 ngày
   - Sau đó free tier: 2 triệu requests/tháng

2. **Google Cloud SDK**
   - Tải tại: https://cloud.google.com/sdk/docs/install
   - Hoặc dùng Cloud Shell trên web

---

## 🎯 Phương pháp 1: Deploy Tự động (Khuyến nghị)

### Bước 1: Cài đặt Google Cloud SDK

```bash
# Trên MacOS
brew install --cask google-cloud-sdk

# Trên Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Trên Windows
# Download từ: https://cloud.google.com/sdk/docs/install
```

### Bước 2: Đăng nhập và setup

```bash
# Đăng nhập
gcloud auth login

# Tạo project mới (hoặc chọn project có sẵn)
gcloud projects create vn-trading-bot-123 --name="VN Trading Bot"

# Set project ID
export GCP_PROJECT_ID=vn-trading-bot-123
gcloud config set project $GCP_PROJECT_ID

# Enable billing (bắt buộc - nhưng vẫn dùng free tier được)
# Vào: https://console.cloud.google.com/billing
# Link billing account với project
```

### Bước 3: Enable required APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### Bước 4: Deploy bằng script tự động

```bash
# Cho phép execute script
chmod +x deploy-cloudrun.sh

# Run script
./deploy-cloudrun.sh
```

Script sẽ tự động:
- ✅ Build Docker image
- ✅ Push lên Container Registry
- ✅ Deploy lên Cloud Run với 2GB RAM
- ✅ Configure region gần Việt Nam (Singapore)
- ✅ Return service URL

---

## 🎯 Phương pháp 2: Deploy Thủ công

### Build và push image

```bash
# Set project ID
export GCP_PROJECT_ID=your-project-id

# Build image
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/vn-trading-bot --timeout=30m

# Deploy
gcloud run deploy vn-trading-bot \
    --image gcr.io/$GCP_PROJECT_ID/vn-trading-bot \
    --platform managed \
    --region asia-southeast1 \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 1 \
    --allow-unauthenticated
```

---

## 🔐 Cấu hình Environment Variables

### Cách 1: Qua Cloud Console

1. Vào: https://console.cloud.google.com/run
2. Click vào service "vn-trading-bot"
3. Click "Edit & Deploy New Revision"
4. Tab "Variables & Secrets"
5. Thêm:
   - `TELEGRAM_TOKEN` = your_telegram_bot_token
   - `CHAT_ID` = your_telegram_chat_id
   - `TICKERS` = VNM,HPG,VIC (optional)

### Cách 2: Qua CLI

```bash
gcloud run services update vn-trading-bot \
    --region asia-southeast1 \
    --set-env-vars "TELEGRAM_TOKEN=your_token,CHAT_ID=your_chat_id"
```

### Cách 3: Từ file .env (Secrets)

```bash
# Tạo secret
gcloud secrets create telegram-token --data-file=<(echo -n "your_token")

# Mount vào Cloud Run
gcloud run services update vn-trading-bot \
    --region asia-southeast1 \
    --update-secrets TELEGRAM_TOKEN=telegram-token:latest
```

---

## 📊 Monitoring & Logs

### Xem logs

```bash
# Xem logs real-time
gcloud run services logs tail vn-trading-bot --region asia-southeast1

# Xem logs với filter
gcloud run services logs read vn-trading-bot \
    --region asia-southeast1 \
    --limit 50
```

### Xem metrics

Vào: https://console.cloud.google.com/run/detail/asia-southeast1/vn-trading-bot/metrics

Bạn sẽ thấy:
- Request count
- Request latency
- Container instance count
- Memory & CPU usage

---

## 💰 Chi phí ước tính (Sau khi hết free tier)

**Free Tier (MIỄN PHÍ):**
- 2 triệu requests/tháng
- 360,000 GB-seconds memory
- 180,000 vCPU-seconds

**Ước tính cho bot của bạn (24/7):**
- Memory: 2GB × 2,592,000 seconds = 5,184,000 GB-seconds
- vCPU: 1 × 2,592,000 = 2,592,000 vCPU-seconds

**Chi phí khi vượt free tier:**
- Memory: (5,184,000 - 360,000) × $0.0000025 = ~$12/tháng
- vCPU: (2,592,000 - 180,000) × $0.00001 = ~$24/tháng

**TỔNG: ~$36/tháng nếu chạy 24/7**

### 💡 Cách giảm chi phí:

1. **Min instances = 0** (mặc định)
   - Container sẽ shutdown khi không có traffic
   - Chi phí chỉ tính khi có requests

2. **Scheduled jobs thay vì 24/7**
   - Dùng Cloud Scheduler để trigger endpoints
   - Chi phí: ~$0.1/tháng

3. **Optimize memory**
   - Comment các dependencies không dùng trong `requirements-cloudrun.txt`
   - Giảm từ 2Gi xuống 1Gi nếu được

---

## 🔄 CI/CD với GitHub Actions (Optional)

Tạo file `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - id: 'auth'
      uses: 'google-github-actions/auth@v1'
      with:
        credentials_json: '${{ secrets.GCP_SA_KEY }}'

    - name: 'Set up Cloud SDK'
      uses: 'google-github-actions/setup-gcloud@v1'

    - name: 'Build and Deploy'
      run: |
        gcloud builds submit --config cloudbuild.yaml
```

**Setup:**
1. Tạo Service Account key: https://console.cloud.google.com/iam-admin/serviceaccounts
2. Download JSON key
3. Add vào GitHub Secrets với tên `GCP_SA_KEY`

---

## 🧪 Test deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe vn-trading-bot \
    --region asia-southeast1 \
    --format 'value(status.url)')

# Test health
curl $SERVICE_URL/health

# Test API docs
open $SERVICE_URL/docs
```

---

## ❌ Troubleshooting

### Lỗi: "Permission denied"
```bash
# Grant permissions
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="user:your-email@gmail.com" \
    --role="roles/run.admin"
```

### Lỗi: "Out of memory"
```bash
# Increase memory
gcloud run services update vn-trading-bot \
    --region asia-southeast1 \
    --memory 4Gi
```

### Lỗi: "Build timeout"
```bash
# Increase build timeout
gcloud builds submit --timeout=40m
```

### Container không start được
```bash
# Check logs
gcloud run services logs tail vn-trading-bot --region asia-southeast1

# Check revisions
gcloud run revisions list --service vn-trading-bot --region asia-southeast1
```

---

## 📚 Tài liệu tham khảo

- Cloud Run Docs: https://cloud.google.com/run/docs
- Pricing Calculator: https://cloud.google.com/products/calculator
- Best Practices: https://cloud.google.com/run/docs/tips/general
- Python on Cloud Run: https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service

---

## 🎉 Hoàn tất!

Sau khi deploy xong, bạn sẽ có:
- ✅ Trading bot chạy trên Cloud Run
- ✅ HTTPS URL public
- ✅ Auto-scaling (0 → 1 instances)
- ✅ 2GB RAM, 1 vCPU
- ✅ Free tier hoặc chi phí thấp

**Service URL mẫu:**
```
https://vn-trading-bot-abc123-as.a.run.app
```

Endpoints:
- `/` - Health check
- `/health` - Chi tiết health status
- `/docs` - API documentation
- `/run-bot` - Manual trigger
