# Phân Tích TCBS API và Giải Pháp Lấy P/E, P/B Ratios

**Ngày:** 2025-11-20
**Vấn đề:** TCBS API không trả về đầy đủ P/E, P/B
**Status:** ⚠️ Tất cả API công khai bị chặn (403 Forbidden)

---

## 📊 Tình Huống Hiện Tại

### 1. Kiểm Tra Đã Thực Hiện

Đã test toàn bộ các API công khai cho Vietnamese stocks:

| API Provider | Status | Chi Tiết |
|--------------|--------|----------|
| **TCBS API** | ❌ 403 Forbidden | `apipubaws.tcbs.com.vn` - Tất cả endpoints fundamental data |
| **VNDirect API** | ❌ 403 Forbidden | `finfo-api.vndirect.com.vn` - stocks, ratios, financial_statements |
| **SSI iboard API** | ❌ 403 Forbidden | `iboard-api.ssi.com.vn` - company info, ratios, valuation |
| **vnstock Package** | ❌ 403 Forbidden | Python package cũng không lấy được data (phụ thuộc vào TCBS) |

### 2. Endpoints Đã Test

#### TCBS API (`apipubaws.tcbs.com.vn`):
```
❌ /stock-insight/v1/stock/overview/{symbol}
❌ /stock-insight/v1/stock/fundamental/{symbol}
❌ /stock-insight/v1/stock/ratios/{symbol}
❌ /tcanalysis/v1/company/{symbol}/overview
❌ /tcanalysis/v1/company/{symbol}/fundamental
❌ /tcanalysis/v1/finance/{symbol}/financialratio
```

#### VNDirect API (`finfo-api.vndirect.com.vn`):
```
❌ /v4/stocks?q=code:{symbol}
❌ /v4/ratios?q=code:{symbol}
❌ /v4/financial_statements?q=code:{symbol}
```

#### SSI iboard API (`iboard-api.ssi.com.vn`):
```
❌ /stock/v2/company/info?symbol={symbol}
❌ /stock/v2/stock/financial-info?symbol={symbol}
❌ /statistics/company/ratio?symbol={symbol}
```

### 3. Headers Tested

Đã thử nhiều header combinations:
- ✅ Browser-like headers (User-Agent, Referer, Origin)
- ✅ Mobile headers
- ✅ Minimal headers
- ✅ Accept: application/json

**Kết quả:** Tất cả đều trả về **403 Forbidden**

---

## 🔍 Nguyên Nhân

### Có 4 khả năng chính:

1. **API đã chuyển sang yêu cầu authentication:**
   - Các API provider đã khóa public access
   - Yêu cầu API key hoặc token

2. **Rate limiting / IP blocking:**
   - Server có thể đã block IP range
   - Quá nhiều requests trong thời gian ngắn

3. **API đã deprecated hoặc thay đổi:**
   - Endpoints đã cũ, không còn sử dụng
   - URL structure đã thay đổi

4. **Firewall / Network restrictions:**
   - Có thể do environment restrictions
   - Corporate firewall blocking

---

## ✅ Giải Pháp

### **Giải Pháp 1: Sử Dụng vnstock với VCI Source (RECOMMENDED)**

vnstock package hỗ trợ nhiều sources, trong đó **VCI (Viet Capital Securities)** có thể vẫn hoạt động:

```python
from vnstock import Vnstock

# Thử VCI source thay vì TCBS
stock = Vnstock().stock(symbol='VNM', source='VCI')
ratios = stock.finance.ratio(period='year', lang='vi')

# VCI có thể trả về P/E, P/B trong các cột như:
# - 'pe', 'pb', 'priceToEarning', 'priceToBook'
pe_ratio = ratios['pe'].iloc[-1] if 'pe' in ratios.columns else None
pb_ratio = ratios['pb'].iloc[-1] if 'pb' in ratios.columns else None
```

**Implementation:** File `src/data/tcbs_provider.py` (xem bên dưới)

---

### **Giải Pháp 2: Sử Dụng API Keys (Nếu Có)**

Nếu bạn có API keys từ các providers:

#### VNDirect API Key:
```python
# Đăng ký tại: https://developers.vndirect.com.vn/
headers = {
    'Authorization': 'Bearer YOUR_API_KEY'
}
```

#### SSI API Key:
```python
# Liên hệ SSI để đăng ký
headers = {
    'X-API-KEY': 'YOUR_SSI_KEY'
}
```

#### FiinTrade API (Paid Service):
```python
# https://fiintrade.vn/
# Professional data service với đầy đủ fundamental data
```

---

### **Giải Pháp 3: Web Scraping từ Public Websites**

Scrape từ các website công khai (legal và ethical):

#### cafef.vn:
```python
url = f"https://s.cafef.vn/{symbol}.chn"
# Parse HTML để lấy P/E, P/B từ bảng thông tin
```

#### vietstock.vn:
```python
url = f"https://finance.vietstock.vn/{symbol}/overview.htm"
# Parse financial ratios từ overview page
```

**Lưu ý:** Cần tuân thủ robots.txt và rate limiting

---

### **Giải Pháp 4: Manual CSV Updates**

Tạo và maintain CSV file với P/E, P/B data:

```csv
Symbol,Name,PE,PB,LastUpdate
VNM,Vinamilk,18.5,5.2,2025-11-20
VCB,Vietcombank,12.3,2.8,2025-11-20
FPT,FPT Corporation,15.7,4.1,2025-11-20
```

**Advantages:**
- ✅ Reliable, không phụ thuộc vào API
- ✅ Full control
- ✅ No rate limiting

**Disadvantages:**
- ❌ Manual updates required
- ❌ May not be real-time

---

### **Giải Pháp 5: Hybrid Approach (BEST)**

Kết hợp nhiều sources với fallback strategy:

```python
class TCBSProvider(FundamentalDataProvider):
    """TCBS Provider with multiple fallback sources"""

    def get_fundamental_data(self, symbol: str):
        # Try 1: vnstock with VCI
        try:
            return self._get_from_vnstock_vci(symbol)
        except:
            pass

        # Try 2: Manual CSV file
        try:
            return self._get_from_csv(symbol)
        except:
            pass

        # Try 3: Web scraping (last resort)
        try:
            return self._get_from_web_scrape(symbol)
        except:
            pass

        return None
```

---

## 🚀 Implementation

### File 1: `src/data/tcbs_provider.py`

Tôi đã tạo provider mới sử dụng vnstock với VCI source:

```python
"""
TCBS Provider for fundamental data using vnstock package
Fallback to VCI source when TCBS is unavailable
"""

import logging
from typing import Optional
from datetime import datetime

try:
    from vnstock import Vnstock
    VNSTOCK_AVAILABLE = True
except ImportError:
    VNSTOCK_AVAILABLE = False

from src.data.fundamental_data import FundamentalData, FundamentalDataProvider

logger = logging.getLogger(__name__)


class TCBSProvider(FundamentalDataProvider):
    """
    TCBS/VCI Provider using vnstock package
    Falls back to VCI when TCBS is unavailable
    """

    def __init__(self, timeout: int = 10):
        if not VNSTOCK_AVAILABLE:
            raise ImportError("vnstock package not installed. Run: pip install vnstock")

        self.timeout = timeout
        self.vnstock = Vnstock()

    def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental data using vnstock"""
        try:
            # Try VCI source (more reliable than TCBS)
            stock = self.vnstock.stock(symbol=symbol, source='VCI')

            # Get financial ratios
            ratios = stock.finance.ratio(period='year', lang='vi')

            if ratios.empty:
                logger.warning(f"No ratio data for {symbol} from VCI")
                return None

            # Get latest ratios
            latest = ratios.iloc[-1]

            # Map column names (VCI may use different names)
            pe_ratio = self._extract_value(latest, ['pe', 'priceToEarning', 'PE'])
            pb_ratio = self._extract_value(latest, ['pb', 'priceToBook', 'PB'])
            roe = self._extract_value(latest, ['roe', 'ROE'])
            roa = self._extract_value(latest, ['roa', 'ROA'])
            debt_to_equity = self._extract_value(latest, ['de', 'debtToEquity', 'DE'])
            eps = self._extract_value(latest, ['eps', 'EPS'])

            fundamental = FundamentalData(
                symbol=symbol,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                roe=roe,
                roa=roa,
                debt_to_equity=debt_to_equity,
                eps=eps,
                source="TCBS/VCI",
            )

            logger.info(f"✅ Got fundamental data for {symbol} from VCI")
            return fundamental

        except Exception as e:
            logger.warning(f"⚠️ VCI/TCBS error for {symbol}: {e}")
            return None

    def _extract_value(self, row, possible_keys: list) -> Optional[float]:
        """Extract value from row using possible key names"""
        for key in possible_keys:
            if key in row.index:
                val = row[key]
                if val is not None and str(val).lower() not in ['nan', 'none', '']:
                    try:
                        return float(val)
                    except:
                        pass
        return None

    def get_earnings_date(self, symbol: str) -> Optional[dict]:
        """Get earnings dates (not implemented for VCI)"""
        logger.warning("⚠️ Earnings dates not available from VCI source")
        return None
```

### File 2: Update `src/data/fundamental_data.py`

Thêm TCBSProvider vào manager:

```python
# Add to imports
from src.data.tcbs_provider import TCBSProvider

# Update FundamentalDataManager.__init__
def __init__(
    self,
    tcbs_enabled: bool = True,  # NEW
    vndirect_enabled: bool = True,
    ssi_enabled: bool = False,
    ...
):
    self.providers = []

    # Try TCBS/VCI first (most up-to-date)
    if tcbs_enabled:
        try:
            self.providers.append(TCBSProvider())
        except ImportError:
            logger.warning("⚠️ vnstock not available, skipping TCBS provider")

    # Fallback to VNDirect
    if vndirect_enabled:
        self.providers.append(VNDirectProvider())
    ...
```

---

## 🧪 Testing

### Test TCBS Provider:

```bash
python scripts/test_tcbs_provider.py
```

### Test trong code:

```python
from src.data.fundamental_data import get_fundamental_manager

# Initialize với TCBS enabled
manager = get_fundamental_manager(tcbs_enabled=True)

# Get data
data = manager.get_fundamental_data("VNM")

if data:
    print(f"P/E: {data.pe_ratio}")
    print(f"P/B: {data.pb_ratio}")
    print(f"Source: {data.source}")
```

---

## 📝 Summary

### Current Situation:
- ❌ TCBS public API: **403 Forbidden**
- ❌ VNDirect API: **403 Forbidden**
- ❌ SSI iboard API: **403 Forbidden**
- ⚠️ vnstock TCBS source: **403 Forbidden**

### Recommended Solution:
1. ✅ **Use vnstock with VCI source** (implemented)
2. ✅ **Fallback to manual CSV** (if VCI fails)
3. ✅ **Web scraping as last resort** (optional)

### Files Created:
- ✅ `src/data/tcbs_provider.py` - TCBS/VCI provider
- ✅ `scripts/test_tcbs_provider.py` - Testing script
- ✅ `TCBS_API_ANALYSIS_AND_SOLUTIONS.md` - This document

### Next Steps:
1. **Test vnstock with VCI source**
2. **If VCI works:** Integrate into system
3. **If VCI fails:** Implement CSV fallback or web scraping
4. **Consider:** Getting API keys from providers for production

---

## 🔗 Resources

- vnstock documentation: https://vnstocks.com/
- vnstock GitHub: https://github.com/thinh-vu/vnstock
- VNDirect API docs: https://developers.vndirect.com.vn/
- SSI API contact: support@ssi.com.vn
- FiinTrade: https://fiintrade.vn/

---

**Status:** Đang chờ test vnstock với VCI source để xác nhận giải pháp cuối cùng.
