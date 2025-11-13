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
    """Lấy danh sách mã cổ phiếu từ TCBS"""
    
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
    
    def get_vietnam_tickers(self, min_volume: float = 100000, max_tickers: int = 0, 
                           skip_validation: bool = False) -> List[str]:
        """
        Lấy danh sách mã cổ phiếu Việt Nam từ TCBS API
        
        Args:
            min_volume: Volume tối thiểu để lọc
            max_tickers: Số lượng mã tối đa (0 = không giới hạn)
            skip_validation: Bỏ qua validation để nhanh hơn (dùng cache)
            
        Returns:
            List các mã cổ phiếu hợp lệ
        """
        # Check memory cache
        cache_key = f'vietnam_tickers_{min_volume}_{max_tickers}'
        if cache_key in self.cache:
            cache_age = time.time() - self.cache_time.get(cache_key, 0)
            if cache_age < self.cache_duration:
                print(f"📁 Sử dụng cache danh sách mã ({len(self.cache[cache_key])} mã, age: {cache_age/3600:.1f}h)")
                return self.cache[cache_key]
        
        print("🔍 Đang lấy danh sách mã từ TCBS API...")
        
        # Lấy danh sách tất cả mã từ TCBS
        all_tickers = self._fetch_all_tickers_from_tcbs()
        
        if not all_tickers:
            print("⚠️ Không lấy được danh sách từ TCBS, dùng danh sách static")
            all_tickers = self._get_common_vietnam_tickers()
        
        print(f"📊 Tổng số mã từ TCBS: {len(all_tickers)}")
        
        # Nếu skip validation, return luôn (nhanh)
        if skip_validation:
            result = all_tickers[:max_tickers] if max_tickers > 0 else all_tickers
            self.cache[cache_key] = result
            self.cache_time[cache_key] = time.time()
            self._save_cache_to_file(result)
            print(f"✅ Trả về {len(result)} mã (skip validation)")
            return result
        
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
                
                # Limit số lượng nếu cần
                if max_tickers > 0 and len(valid_tickers) >= max_tickers:
                    print(f"\n⏹️ Đã đủ {max_tickers} mã, dừng quét")
                    break
                
                # Rate limiting
                time.sleep(0.05)
                
            except Exception as e:
                print(f"  ❌ {symbol}: {str(e)[:50]}                    ")
                continue
        
        print(f"\n✅ Tìm thấy {len(valid_tickers)} mã hợp lệ")
        
        # Cache kết quả
        self.cache[cache_key] = valid_tickers
        self.cache_time[cache_key] = time.time()
        self._save_cache_to_file(valid_tickers)
        
        return valid_tickers
    
    def _save_cache_to_file(self, tickers: List[str]):
        """Save cache to file"""
        try:
            cache_data = {
                'tickers': tickers,
                'updated_at': datetime.now().isoformat(),
                'count': len(tickers)
            }
            with open(TICKERS_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Đã lưu {len(tickers)} mã vào cache file")
        except Exception as e:
            print(f"⚠️ Không lưu được cache: {e}")
    
    def _fetch_all_tickers_from_tcbs(self) -> List[str]:
        """Lấy danh sách tất cả mã từ nhiều nguồn"""
        
        # Method 1: SSI iBoard API (Most reliable)
        try:
            print("  🔍 Thử SSI iBoard API...")
            url = "https://iboard.ssi.com.vn/dchart/api/1.1/defaultAllStock"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                tickers = []
                
                # SSI returns array of stock objects
                if isinstance(data, list):
                    for item in data:
                        ticker = item.get('stockCode') or item.get('code') or item.get('symbol')
                        if ticker:
                            tickers.append(ticker.upper())
                
                if tickers:
                    print(f"  ✅ SSI API: {len(tickers)} mã")
                    return tickers
        except Exception as e:
            print(f"  ⚠️ SSI API failed: {str(e)[:50]}")
        
        # Method 2: VNDirect API
        try:
            print("  🔍 Thử VNDirect API...")
            url = "https://finfo-api.vndirect.com.vn/v4/stocks"
            params = {
                'q': 'type:STOCK~floor:HOSE,HNX,UPCOM',
                'size': 2000,
                'page': 1
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'data' in data:
                    items = data['data']
                    tickers = [item.get('code') for item in items if item.get('code')]
                    if tickers:
                        print(f"  ✅ VNDirect API: {len(tickers)} mã")
                        return tickers
        except Exception as e:
            print(f"  ⚠️ VNDirect API failed: {str(e)[:50]}")
        
        # Method 3: TCBS API (backup)
        try:
            print("  🔍 Thử TCBS API...")
            endpoints = [
                f"{TCBS_API_BASE}/stock-insight/v1/stock/second-tc-price",
                f"{TCBS_API_BASE}/stock/list"
            ]
            
            for url in endpoints:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        tickers = []
                        
                        if isinstance(data, dict) and 'data' in data:
                            items = data['data']
                            if isinstance(items, list):
                                tickers = [
                                    item.get('t') or item.get('ticker') or 
                                    item.get('symbol') or item.get('code')
                                    for item in items
                                ]
                        elif isinstance(data, list):
                            tickers = [
                                item.get('ticker') or item.get('symbol') or item.get('code')
                                for item in data
                            ]
                        
                        tickers = [t.upper() for t in tickers if t and isinstance(t, str)]
                        if tickers:
                            print(f"  ✅ TCBS API: {len(tickers)} mã")
                            return tickers
                except Exception:
                    continue
        except Exception as e:
            print(f"  ⚠️ TCBS API failed: {str(e)[:50]}")
        
        # Method 4: Comprehensive static list (fallback)
        print("  ⚠️ Tất cả API failed, dùng comprehensive static list...")
        return self._get_comprehensive_vietnam_tickers()
    
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
        """Danh sách các mã phổ biến Việt Nam (fallback)"""
        try:
            from config import KIM_SECTOR, THUY_SECTOR
            
            tickers = []
            for sector_dict in [KIM_SECTOR, THUY_SECTOR]:
                for category, stocks in sector_dict.items():
                    tickers.extend(stocks)
            
            return sorted(list(set(tickers)))
            
        except Exception:
            return self._get_comprehensive_vietnam_tickers()
    
    def _get_comprehensive_vietnam_tickers(self) -> List[str]:
        """
        Comprehensive list of Vietnam stocks (300+ tickers)
        Updated: Nov 2025
        """
        return [
            # === HOSE - Blue Chips ===
            'VCB', 'VHM', 'VIC', 'VNM', 'HPG', 'FPT', 'MSN', 'MWG', 'VPB', 'GAS',
            'CTG', 'BID', 'TCB', 'MBB', 'ACB', 'STB', 'SSI', 'VRE', 'PLX', 'POW',
            
            # === Banks ===
            'VCB', 'CTG', 'BID', 'TCB', 'MBB', 'ACB', 'VPB', 'STB', 'TPB', 'VIB',
            'HDB', 'SHB', 'MSB', 'OCB', 'LPB', 'EIB', 'VBB', 'BVB', 'NVB', 'ABB',
            'BAB', 'KLB', 'NAB', 'PGB', 'SEAB', 'VAB', 'VBB',
            
            # === Securities ===
            'SSI', 'VND', 'HCM', 'VCI', 'FTS', 'MBS', 'BSI', 'VIX', 'SHS', 'APS',
            'CTS', 'ORS', 'IFS', 'TVS', 'WSS', 'BVS', 'AGR', 'EVS', 'PSI',
            
            # === Real Estate ===
            'VHM', 'VIC', 'VRE', 'NVL', 'PDR', 'DXG', 'KDH', 'HDG', 'DIG', 'NLG',
            'HDC', 'CEO', 'LDG', 'SCR', 'SZC', 'IDC', 'ITA', 'KBC', 'NBB', 'NTL',
            'PPI', 'QCG', 'SJS', 'TDH', 'TDC', 'UIC', 'VCG', 'VPI',
            
            # === Manufacturing & Steel ===
            'HPG', 'HSG', 'NKG', 'POM', 'TLH', 'VGS', 'DTL', 'HT1', 'SMC', 'TIS',
            'VCS', 'VIS', 'VNS', 'VSH', 'VCA', 'VHC', 'VTO',
            
            # === Retail & Consumer ===
            'VNM', 'MSN', 'MWG', 'FRT', 'PNJ', 'DGW', 'PET', 'SAB', 'VHC', 'MCH',
            'VCF', 'BBC', 'KDC', 'VNL', 'QNS', 'SBT', 'CAN', 'VTO', 'TNG',
            
            # === Technology ===
            'FPT', 'CMG', 'VGI', 'SAM', 'ELC', 'ITD', 'VTP', 'SGT', 'CMT', 'ICT',
            'TIG', 'SGN', 'VGS', 'FOX', 'VNZ', 'CTR', 'EFI', 'CMX',
            
            # === Energy & Utilities ===
            'GAS', 'POW', 'PLX', 'PVD', 'PVS', 'PVT', 'PVC', 'PVG', 'BSR', 'OIL',
            'NT2', 'REE', 'PC1', 'VSH', 'GEG', 'GEX', 'PGV', 'PGD',
            
            # === Logistics & Transportation ===
            'GMD', 'HAH', 'TMS', 'VSC', 'VOS', 'STG', 'PHP', 'SGP', 'VJC', 'HVN',
            'ACV', 'VTP', 'TCL', 'TRA', 'VFC', 'VIP', 'VTO', 'VST',
            
            # === Insurance ===
            'BVH', 'BMI', 'PVI', 'PTI', 'BIC', 'PGI', 'VNR', 'MIG', 'ABI', 'PRE',
            
            # === Construction & Materials ===
            'HBC', 'CTD', 'HT1', 'VCG', 'LCG', 'PC1', 'HU1', 'C32', 'C47', 'DPG',
            'FCN', 'HU3', 'HU4', 'LBM', 'NNC', 'PXI', 'SC5', 'SZL', 'TDW', 'THG',
            'VCR', 'VE1', 'VE2', 'VE3', 'VE4', 'VE8', 'VE9',
            
            # === Agriculture & Fishery ===
            'VHC', 'HNG', 'BAF', 'FMC', 'ANV', 'AAM', 'AGF', 'AGG', 'ASM', 'BFC',
            'CAV', 'CMV', 'DBC', 'FID', 'HAG', 'HVG', 'KSB', 'LAF', 'LSS', 'MKV',
            'NSC', 'SJF', 'TS4', 'VIF', 'VNH',
            
            # === Pharmaceuticals & Healthcare ===
            'DHG', 'DMC', 'IMP', 'TRA', 'DBD', 'DCL', 'DHT', 'DVN', 'PME', 'PPP',
            'SPM', 'TNG', 'TNH', 'DP1', 'DP2', 'DP3',
            
            # === Chemicals & Fertilizers ===
            'DPM', 'DCM', 'DGC', 'DDV', 'BFC', 'CSV', 'LAS', 'NCS', 'PHR', 'SFG',
            
            # === Textiles & Garments ===
            'VGT', 'TNG', 'MSH', 'STK', 'TCM', 'GIL', 'VGG', 'TET', 'VTL',
            
            # === Food & Beverage ===
            'VNM', 'SAB', 'MCH', 'VCF', 'BBC', 'KDC', 'VNL', 'QNS', 'SBT', 'CAN',
            'VTO', 'TNG', 'LSS', 'HNG', 'BAF', 'FMC', 'ANV',
            
            # === Others ===
            'VCS', 'VTO', 'DAG', 'VNP', 'HDG', 'CTR', 'FOX', 'VNZ', 'SGN', 'VGS',
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


def get_active_tickers(min_volume: float = 100000, use_cache: bool = True, 
                      max_tickers: int = 0, skip_validation: bool = False) -> List[str]:
    """
    Helper function để lấy danh sách mã active
    
    Args:
        min_volume: Volume tối thiểu
        use_cache: Có sử dụng cache không
        max_tickers: Số lượng mã tối đa (0 = không giới hạn)
        skip_validation: Bỏ qua validation để nhanh hơn
        
    Returns:
        List các mã cổ phiếu hợp lệ
    """
    fetcher = get_ticker_fetcher()
    
    if not use_cache:
        fetcher.cache.clear()
        fetcher.cache_time.clear()
    
    return fetcher.get_vietnam_tickers(
        min_volume=min_volume,
        max_tickers=max_tickers,
        skip_validation=skip_validation
    )


if __name__ == "__main__":
    # Test
    print("="*60)
    print("🧪 TEST TICKER FETCHER")
    print("="*60)
    
    tickers = get_active_tickers(min_volume=100000, use_cache=False)
    
    print(f"\n📊 Tìm thấy {len(tickers)} mã hợp lệ:")
    for i, ticker in enumerate(tickers, 1):
        print(f"  {i}. {ticker}")
