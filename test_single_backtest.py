#!/usr/bin/env python3
"""
Quick test for single stock backtest to debug ML issues
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.run_backtest import Backtester

def main():
    print("🧪 Testing single stock backtest...")
    
    try:
        # Initialize backtester
        backtester = Backtester(initial_capital=100_000_000, commission=0.0015)
        print("✅ Backtester initialized")
        
        # Test with a single stock
        symbol = "VNM"  # Use a common stock
        print(f"🔍 Testing {symbol}...")
        
        result = backtester.run_backtest(
            symbol=symbol,
            lookback=100,  # Use smaller dataset for testing
            confidence_threshold=50
        )
        
        print("✅ Backtest completed!")
        print(f"📊 Total trades: {result['total_trades']}")
        print(f"📈 Return: {result['total_return']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()