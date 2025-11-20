"""
Test SSI iboard API - Public API cho Vietnamese stocks
SSI iboard là nền tảng công khai, không cần API key
"""

import requests
import json

SSI_BASE = "https://iboard-api.ssi.com.vn"

def test_ssi_api():
    """Test SSI iboard API"""

    test_symbol = "VNM"

    print("=" * 80)
    print(f"🔍 TESTING SSI IBOARD API FOR {test_symbol}")
    print("=" * 80)

    endpoints = [
        # Stock info
        f"/stock/v2/company/info?symbol={test_symbol}",
        f"/stock/v2/stock/financial-info?symbol={test_symbol}",
        f"/stock/v2/stock/valuation?symbol={test_symbol}",
        f"/stock/v2/stock/overview?symbol={test_symbol}",

        # Statistics
        f"/statistics/company/financial-indicator?symbol={test_symbol}",
        f"/statistics/company/ratio?symbol={test_symbol}",

        # Company data
        f"/company/stats?symbol={test_symbol}",
        f"/company/overview?symbol={test_symbol}",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    successful_endpoints = []

    for endpoint in endpoints:
        url = f"{SSI_BASE}{endpoint}"
        print(f"\n{'─'*80}")
        print(f"📡 Testing: {endpoint}")

        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ SUCCESS! Type: {type(data)}")

                    # Check for P/E, P/B
                    data_str = json.dumps(data, ensure_ascii=False).lower()

                    # More comprehensive search
                    pe_indicators = ['pe', 'p/e', 'peratio', 'pe_ratio', 'pricetoearning',
                                    'price_earning', 'price-earning', 'gia_thu_nhap']
                    pb_indicators = ['pb', 'p/b', 'pbratio', 'pb_ratio', 'pricetobook',
                                    'price_book', 'price-book', 'gia_so_sach']

                    has_pe = any(ind in data_str for ind in pe_indicators)
                    has_pb = any(ind in data_str for ind in pb_indicators)

                    if has_pe or has_pb:
                        print(f"   🎯 Contains P/E: {has_pe}, P/B: {has_pb}")
                        successful_endpoints.append((endpoint, data))

                        # Print sample data
                        print(f"   📊 Sample response:")
                        if isinstance(data, dict):
                            print(f"      Top-level keys: {list(data.keys())}")

                            # Look for data inside nested structure
                            if 'data' in data:
                                nested = data['data']
                                if isinstance(nested, dict):
                                    print(f"      Data keys: {list(nested.keys())[:20]}")
                                elif isinstance(nested, list) and len(nested) > 0:
                                    print(f"      First item keys: {list(nested[0].keys())[:20]}")

                except json.JSONDecodeError:
                    print(f"   ⚠️ Not JSON")
            elif response.status_code == 404:
                print(f"   ❌ 404 Not Found")
            elif response.status_code == 403:
                print(f"   ❌ 403 Forbidden")
            else:
                print(f"   ❌ {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    if successful_endpoints:
        print(f"\n✅ Found {len(successful_endpoints)} working endpoints:")
        for ep, data in successful_endpoints:
            print(f"\n   Endpoint: {ep}")
            print(f"   Full URL: {SSI_BASE}{ep}")

            # Show actual P/E, P/B values if found
            data_str = json.dumps(data, ensure_ascii=False)

            print(f"\n   Sample data structure:")
            if isinstance(data, dict):
                for k, v in list(data.items())[:5]:
                    print(f"      {k}: {str(v)[:100]}")

        return successful_endpoints[0]
    else:
        print("\n❌ No working endpoints found")
        print("\n💡 Will try alternative sources...")
        return None

if __name__ == "__main__":
    result = test_ssi_api()

    if result:
        print("\n✅ SSI iboard API is WORKING!")
        print("💡 Can integrate this into TCBSProvider alternative")
    else:
        print("\n⚠️ SSI iboard API also not accessible")
        print("💡 May need to use web scraping or paid APIs")
