# 🚀 Quick Deploy to Google Cloud Run

## Prerequisite

1. ✅ Install Google Cloud SDK
2. ✅ Create Google Cloud account (free tier)
3. ✅ Enable billing (still free up to 2M requests/month)

---

## Deploy in 3 Steps

### Step 1: Login and Setup

```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID (change this!)
export GCP_PROJECT_ID=vn-trading-bot-2024

# Configure gcloud
gcloud config set project $GCP_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com
```

### Step 2: Set Environment Variables (Important!)

Create a `.env.yaml` file in your project root:

```yaml
# .env.yaml
TELEGRAM_TOKEN: "your_telegram_bot_token_here"
CHAT_ID: "your_telegram_chat_id_here"
PORT: "8080"
```

**Security Note:** Add `.env.yaml` to `.gitignore`!

### Step 3: Deploy

```bash
# Option A: Using deploy script (recommended)
chmod +x deploy-cloudrun.sh
./deploy-cloudrun.sh

# Option B: Manual deploy
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/vn-trading-bot --timeout=30m

gcloud run deploy vn-trading-bot \
    --image gcr.io/$GCP_PROJECT_ID/vn-trading-bot \
    --platform managed \
    --region asia-southeast1 \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 1 \
    --allow-unauthenticated \
    --env-vars-file .env.yaml
```

---

## After Deployment

Your bot will be available at:
```
https://vn-trading-bot-xxxxxxxxxx-as.a.run.app
```

**Test endpoints:**
- Health check: `https://your-url/health`
- API docs: `https://your-url/docs`
- Manual trigger: `POST https://your-url/run-bot`

---

## Monitoring

```bash
# View logs
gcloud run services logs tail vn-trading-bot --region asia-southeast1

# View service details
gcloud run services describe vn-trading-bot --region asia-southeast1

# Update environment variables
gcloud run services update vn-trading-bot \
    --region asia-southeast1 \
    --set-env-vars "NEW_VAR=value"
```

---

## Cost Estimate

**Free Tier (No charge):**
- 2 million requests/month
- 360,000 GB-seconds memory
- 180,000 vCPU-seconds

**After free tier (~$36/month if running 24/7):**
- Memory: 2GB × 720 hours = ~$12/month
- CPU: 1 vCPU × 720 hours = ~$24/month

**💡 Pro tip:** Use Cloud Scheduler to trigger bot at specific times instead of running 24/7 = ~$0.10/month!

---

## Troubleshooting

### Build timeout
```bash
# Increase timeout
gcloud builds submit --timeout=40m
```

### Out of memory
```bash
# Increase to 4GB
gcloud run services update vn-trading-bot --memory 4Gi --region asia-southeast1
```

### Service won't start
```bash
# Check logs
gcloud run services logs read vn-trading-bot --region asia-southeast1 --limit 100
```

### Permission denied
```bash
# Add IAM role
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="user:your-email@gmail.com" \
    --role="roles/run.admin"
```

---

## Next Steps

1. ✅ Set up custom domain (optional)
2. ✅ Configure Cloud Scheduler for scheduled tasks
3. ✅ Set up monitoring and alerts
4. ✅ Configure CI/CD with GitHub Actions

Read full documentation: [DEPLOY_CLOUD_RUN.md](./DEPLOY_CLOUD_RUN.md)
