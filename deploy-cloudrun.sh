#!/bin/bash

# ========================================
# Google Cloud Run Deployment Script
# ========================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🚀 Google Cloud Run Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration
PROJECT_ID="${GCP_PROJECT_ID}"
SERVICE_NAME="vn-trading-bot"
REGION="asia-southeast1"  # Singapore
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: GCP_PROJECT_ID environment variable not set${NC}"
    echo -e "${YELLOW}Set it with: export GCP_PROJECT_ID=your-project-id${NC}"
    exit 1
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI not found${NC}"
    echo -e "${YELLOW}Install from: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# Check if user is authenticated
echo -e "\n${YELLOW}📋 Checking authentication...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${RED}❌ Not authenticated. Running gcloud auth login...${NC}"
    gcloud auth login
fi

# Set project
echo -e "\n${YELLOW}🔧 Setting project to ${PROJECT_ID}...${NC}"
gcloud config set project "${PROJECT_ID}"

# Enable required APIs
echo -e "\n${YELLOW}🔌 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build the Docker image
echo -e "\n${YELLOW}🏗️  Building Docker image...${NC}"
gcloud builds submit --tag "${IMAGE_NAME}:latest" --timeout=30m

# Deploy to Cloud Run
echo -e "\n${YELLOW}🚀 Deploying to Cloud Run...${NC}"
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}:latest" \
    --platform managed \
    --region "${REGION}" \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 1 \
    --allow-unauthenticated \
    --set-env-vars "PORT=8080"

# Get the service URL
echo -e "\n${YELLOW}🔍 Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --platform managed \
    --region "${REGION}" \
    --format 'value(status.url)')

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Deployment successful!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🌐 Service URL: ${SERVICE_URL}${NC}"
echo -e "${GREEN}📊 Health: ${SERVICE_URL}/health${NC}"
echo -e "${GREEN}📚 Docs: ${SERVICE_URL}/docs${NC}"
echo -e "${GREEN}========================================${NC}"

# Test health endpoint
echo -e "\n${YELLOW}🏥 Testing health endpoint...${NC}"
sleep 5
if curl -f "${SERVICE_URL}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Health check passed!${NC}"
else
    echo -e "${RED}⚠️  Health check failed. Check logs:${NC}"
    echo -e "${YELLOW}gcloud run services logs read ${SERVICE_NAME} --region ${REGION}${NC}"
fi

echo -e "\n${GREEN}🎉 All done!${NC}"
