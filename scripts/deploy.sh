#!/bin/bash

# Deployment script for Trading Bot
set -e

echo "🚀 Starting deployment..."

# Check environment variables
if [ -z "$TELEGRAM_TOKEN" ]; then
    echo "❌ TELEGRAM_TOKEN is not set"
    exit 1
fi

if [ -z "$CHAT_ID" ]; then
    echo "❌ CHAT_ID is not set" 
    exit 1
fi

# Create directories
mkdir -p logs backtest_results models data_cache

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run tests
echo "🧪 Running tests..."
python -c "
import sys
print(f'Python {sys.version}')
try:
    import pandas as pd
    import numpy as np
    from fastapi import FastAPI
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

echo "✅ Deployment preparation completed!"
echo "🔧 Starting application..."