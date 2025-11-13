"""
Module để lấy danh sách mã cổ phiếu từ TCBS API (hỗ trợ tốt cổ phiếu VN)
"""
import requests
import pandas as pd
from typing import List, Dict, Optional
import time
import json
import os
from datetime import datetime

TICKERS_CACHE_FILE = 'tickers_cache.json'
TCBS_API_BASE = 'https://apipubaws.tcbs.com.vn'

class TickerFetcher:
    """Lấy danh sách mã cổ phiếu từ Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 86400  # 24 giờ
        self._load_cache_from_file()
    
    def _load_cache_from_file(self):
        """Load cache từ file nếu có"""
        if os.path.exists(TICKERS_CACHE_FILE):
            try:
                with open(TICKERS_CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                updated_at = cache_data.get('updated_at')
                if updated_at:
                    cache_time = datetime.fromisoformat(updated_at).timestamp()
                    cache_age = time.time() - cache_time
                    
                    if cache_age < self.cache_duration:
                        tickers = cache_data.get('tickers', [])
                        self.cache['vietnam_tickers'] = tickers
                        self.cache_time['vietnam_tickers'] = cache_time
                        print(f"📁 Load {len(tickers)} mã từ cache file (age: {cache_age/3600:.1f}h)")
            except Exception as e:
                print(f"⚠️ Không đọc được cache file: {e}")
    
    def get_vietnam_tickers(self, min_volume: float = 100000) -> List[str]:
        """
        Lấy danh sách mã cổ phiếu Việt Nam từ TCBS API
        
        Args:
            min_volume: Volume tối thiểu để lọc
            
        Returns:
            List các mã cổ phiếu hợp lệ
        """
        # Check memory cache
        if 'vietnam_tickers' in self.cache:
            cache_age = time.time() - self.cache_time.get('vietnam_tickers', 0)
            if cache_age < self.cache_duration:
                print(f"📁 Sử dụng cache danh sách mã ({len(self.cache['vietnam_tickers'])} mã)")
                return self.cache['vietnam_tickers']
        
        print("🔍 Đang lấy danh sách mã từ TCBS API...")
        
        # Lấy danh sách tất cả mã từ TCBS
        all_tickers = self._fetch_all_tickers_from_tcbs()
        
        if not all_tickers:
            print("⚠️ Không lấy được danh sách từ TCBS, dùng danh sách static")
            all_tickers = self._get_common_vietnam_tickers()
        
        # Validate và lọc theo volume
        valid_tickers = []
        
        for i, symbol in enumerate(all_tickers):
            try:
                print(f"  ({i+1}/{len(all_tickers)}) Kiểm tra {symbol}...", end='\r')
                
                # Lấy thông tin từ TCBS
                is_valid, volume = self._validate_ticker_tcbs(symbol)
                
                if is_valid:
                    if volume >= min_volume:
                        valid_tickers.append(symbol)
                        print(f"  ✅ {symbol}: Volume = {volume:,.0f}                    ")
                    else:
                        print(f"  ⏭️ {symbol}: Volume thấp ({volume:,.0f})                ")
                else:
                    print(f"  ❌ {symbol}: Không hợp lệ                    ")
                
                # Rate limiting
                time.sleep(0.05)
                
            except Exception as e:
                print(f"  ❌ {symbol}: {str(e)[:50]}                    ")
                continue
        
        print(f"\n✅ Tìm thấy {len(valid_tickers)} mã hợp lệ")
        
        # Cache kết quả
        self.cache['vietnam_tickers'] = valid_tickers
        self.cache_time['vietnam_tickers'] = time.time()
        
        return valid_tickers
    
    def _fetch_all_tickers_from_tcbs(self) -> List[str]:
        """Lấy danh sách tất cả mã từ TCBS API"""
        try:
            # TCBS API endpoint để lấy danh sách mã
            url = f"{TCBS_API_BASE}/stock/list"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract ticker symbols
                if isinstance(data, list):
                    tickers = [item.get('ticker') or item.get('symbol') for item in data]
                    tickers = [t for t in tickers if t]  # Remove None
                    print(f"  📊 TCBS API: {len(tickers)} mã")
                    return tickers
                elif isinstance(data, dict) and 'data' in data:
                    items = data['data']
                    tickers = [item.get('ticker') or item.get('symbol') for item in items]
                    tickers = [t for t in tickers if t]
                    print(f"  📊 TCBS API: {len(tickers)} mã")
                    return tickers
            
            print(f"  ⚠️ TCBS API trả về status {response.status_code}")
            return []
            
        except Exception as e:
            print(f"  ⚠️ Lỗi khi gọi TCBS API: {e}")
            return []
    
    def _validate_ticker_tcbs(self, symbol: str) -> tuple[bool, float]:
        """
        Validate mã cổ phiếu qua TCBS API
        
        Returns:
            (is_valid, avg_volume)
        """
        try:
            from datetime import datetime, timedelta
            
            # Lấy dữ liệu 10 ngày gần nhất
            url = f"{TCBS_API_BASE}/stock-insight/v1/stock/bars-long-term"
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=10)
            
            params = {
                'ticker': symbol,
                'type': 'stock',
                'resolution': 'D',
                'from': int(from_date.timestamp()),
                'to': int(to_date.timestamp())
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, dict) and 'data' in data:
                    bars = data['data']
                    
                    if bars and len(bars) >= 1:
                        # Tính volume trung bình
                        volumes = [bar.get('volume', 0) for bar in bars]
                        avg_volume = sum(volumes) / len(volumes) if volumes else 0
                        return True, avg_volume
            
            return False, 0
            
        except Exception:
            return False, 0
    
    def _get_common_vietnam_tickers(self) -> List[str]:
        """
        Danh sách các mã phổ biến Việt Nam (fallback)
        """
        # Import từ config
        try:
            from config import KIM_SECTOR, THUY_SECTOR
            
            tickers = []
            for sector_dict in [KIM_SECTOR, THUY_SECTOR]:
                for category, stocks in sector_dict.items():
                    tickers.extend(stocks)
            
            return sorted(list(set(tickers)))
            
        except Exception:
            # Fallback nếu không import được
            return [
                # Banks
                'VCB', 'CTG', 'BID', 'TCB', 'MBB', 'ACB', 'VPB', 'STB', 
                'TPB', 'VIB', 'HDB', 'SHB', 'MSB', 'OCB',
                
                # Securities
                'SSI', 'VND', 'HCM', 'VCI', 'FTS', 'MBS', 'BSI',
                
                # Real Estate
                'VHM', 'VIC', 'NVL', 'PDR', 'DXG', 'KDH', 'HDG',
                
                # Manufacturing
                'HPG', 'HSG', 'NKG',
                
                # Retail & Consumer
                'VNM', 'MSN', 'MWG', 'FRT', 'PNJ',
                
                # Technology
                'FPT', 'CMG', 'VGI',
                
                # Energy
                'GAS', 'POW', 'PLX',
                
                # Others
                'VJC', 'HVN',
            ]
    
    def get_sector_tickers(self, sector: str, min_volume: float = 100000) -> List[str]:
        """
        Lấy mã theo ngành cụ thể từ config
        
        Args:
            sector: Tên ngành (banks, securities, real_estate, etc.)
            min_volume: Volume tối thiểu
        """
        try:
            from config import KIM_SECTOR, THUY_SECTOR
            
            # Tìm sector trong config
            all_sectors = {**KIM_SECTOR, **THUY_SECTOR}
            
            # Map tên sector
            sector_key = sector.lower().replace(' ', '_')
            
            if sector_key in all_sectors:
                tickers = all_sectors[sector_key]
            else:
                # Tìm gần đúng
                for key, stocks in all_sectors.items():
                    if sector_key in key or key in sector_key:
                        tickers = stocks
                        break
                else:
                    print(f"⚠️ Không tìm thấy sector '{sector}'")
                    return []
            
            # Validate qua TCBS
            valid_tickers = []
            for symbol in tickers:
                is_valid, volume = self._validate_ticker_tcbs(symbol)
                if is_valid and volume >= min_volume:
                    valid_tickers.append(symbol)
            
            return valid_tickers
            
        except Exception as e:
            print(f"⚠️ Lỗi get_sector_tickers: {e}")
            return []
    
    def validate_ticker(self, symbol: str) -> bool:
        """Kiểm tra xem mã có hợp lệ không qua TCBS API"""
        is_valid, _ = self._validate_ticker_tcbs(symbol)
        return is_valid


# Singleton instance
_fetcher_instance = None

def get_ticker_fetcher() -> TickerFetcher:
    """Get singleton instance của TickerFetcher"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = TickerFetcher()
    return _fetcher_instance


def get_active_tickers(min_volume: float = 100000, use_cache: bool = True) -> List[str]:
    """
    Helper function để lấy danh sách mã active
    
    Args:
        min_volume: Volume tối thiểu
        use_cache: Có sử dụng cache không
        
    Returns:
        List các mã cổ phiếu hợp lệ
    """
    fetcher = get_ticker_fetcher()
    
    if not use_cache:
        fetcher.cache.clear()
        fetcher.cache_time.clear()
    
    return fetcher.get_vietnam_tickers(min_volume=min_volume)


if __name__ == "__main__":
    # Test
    print("="*60)
    print("🧪 TEST TICKER FETCHER")
    print("="*60)
    
    tickers = get_active_tickers(min_volume=100000, use_cache=False)
    
    print(f"\n📊 Tìm thấy {len(tickers)} mã hợp lệ:")
    for i, ticker in enumerate(tickers, 1):
        print(f"  {i}. {ticker}")
