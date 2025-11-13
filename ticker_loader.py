"""
Ticker Loader - Load tất cả tickers từ List.csv
Simple CSV loader - no sector filtering
"""
import pandas as pd
import os
from typing import List


class TickerLoader:
    """Load tất cả tickers từ List.csv"""
    
    def __init__(self, csv_file='List.csv'):
        self.csv_file = csv_file
        self.all_tickers = []
        self.load_from_csv()
    
    def load_from_csv(self):
        """Load tất cả tickers từ CSV"""
        if not os.path.exists(self.csv_file):
            print(f"⚠️ File {self.csv_file} không tồn tại")
            return
        
        try:
            # Read CSV with error handling
            df = pd.read_csv(
                self.csv_file, 
                encoding='utf-8',
                on_bad_lines='skip',  # Skip bad lines
                engine='python'  # More flexible parser
            )
            
            # Get ticker column (first column)
            self.all_tickers = df.iloc[:, 0].tolist()
            
            # Clean tickers
            self.all_tickers = [str(t).strip().upper() for t in self.all_tickers if pd.notna(t)]
            
            # Remove empty strings
            self.all_tickers = [t for t in self.all_tickers if t]
            
            print(f"📊 Loaded {len(self.all_tickers)} tickers from {self.csv_file}")
            
        except Exception as e:
            print(f"❌ Error loading {self.csv_file}: {e}")
            self.all_tickers = []


# Singleton instance
_loader = None

def get_ticker_loader() -> TickerLoader:
    """Get singleton instance"""
    global _loader
    if _loader is None:
        _loader = TickerLoader()
    return _loader


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING TICKER LOADER")
    print("=" * 60)
    
    loader = TickerLoader()
    
    print(f"\n📊 Total tickers: {len(loader.all_tickers)}")
    print(f"📋 Sample (first 20): {loader.all_tickers[:20]}")
    print(f"📋 Sample (last 20): {loader.all_tickers[-20:]}")
    
    print("\n✅ Test completed!")
