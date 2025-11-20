"""
Tìm TCBS API endpoint cho financial ratios (P/E, P/B)
Dựa trên thông tin từ vnstock package
"""

import requests
import json

TCBS_API_BASE = "https://apipubaws.tcbs.com.vn"

def test_ratio_endpoints():
    """Test các endpoint có thể có cho financial ratios"""

    test_symbol = "VNM"

    print("=" * 80)
    print(f"🔍 FINDING TCBS FINANCIAL RATIO ENDPOINT FOR {test_symbol}")
    print("=" * 80)

    # Các endpoint có thể có dựa trên patterns phổ biến
    endpoints = [
        # Company endpoints
        f"/tcanalysis/v1/company/{test_symbol}/overview",
        f"/tcanalysis/v1/company/{test_symbol}/fundamental",
        f"/tcanalysis/v1/company/{test_symbol}/financialratio",
        f"/tcanalysis/v1/company/{test_symbol}/valuation",
        f"/tcanalysis/v1/company/{test_symbol}/snapshot",

        # Finance endpoints
        f"/tcanalysis/v1/finance/{test_symbol}/financialratio",
        f"/tcanalysis/v1/finance/{test_symbol}/ratio",
        f"/tcanalysis/v1/finance/{test_symbol}/ratios",

        # Stock endpoints
        f"/tcanalysis/v1/stock/{test_symbol}/ratio",
        f"/tcanalysis/v1/stock/{test_symbol}/financialratio",
        f"/tcanalysis/v1/stock/{test_symbol}/overview",
        f"/tcanalysis/v1/stock/{test_symbol}/fundamental",

        # Ticker endpoints
        f"/tcanalysis/v1/ticker/{test_symbol}/overview",
        f"/tcanalysis/v1/ticker/{test_symbol}/financial",
        f"/tcanalysis/v1/ticker/{test_symbol}/ratio",
    ]

    successful = []

    for endpoint in endpoints:
        url = f"{TCBS_API_BASE}{endpoint}"
        print(f"\n{'─' * 80}")
        print(f"📡 {endpoint}")

        try:
            response = requests.get(url, timeout=10)
            status = response.status_code

            if status == 200:
                try:
                    data = response.json()
                    print(f"   ✅ SUCCESS! Type: {type(data)}")

                    # Check for P/E, P/B
                    json_str = json.dumps(data, ensure_ascii=False).lower()

                    # Look for keys
                    pe_keys = ['pe', 'p/e', 'peratio', 'pe_ratio', 'pricetoearning', 'price_to_earning']
                    pb_keys = ['pb', 'p/b', 'pbratio', 'pb_ratio', 'pricetobook', 'price_to_book']

                    has_pe = any(key in json_str for key in pe_keys)
                    has_pb = any(key in json_str for key in pb_keys)

                    if has_pe or has_pb:
                        print(f"   🎯 P/E: {has_pe}, P/B: {has_pb}")
                        successful.append((endpoint, data))

                        # Show sample
                        print(f"   📊 Sample data:")
                        if isinstance(data, dict):
                            # Show first 15 keys
                            for i, (k, v) in enumerate(list(data.items())[:15]):
                                print(f"      {k}: {v}")
                        elif isinstance(data, list) and len(data) > 0:
                            print(f"      First item keys: {list(data[0].keys())[:15]}")
                    else:
                        print(f"   ⚠️ No P/E or P/B found")

                except Exception as e:
                    print(f"   ⚠️ Parse error: {str(e)[:100]}")
            elif status == 404:
                print(f"   ❌ 404")
            elif status == 403:
                print(f"   ❌ 403 Forbidden")
            else:
                print(f"   ❌ {status}")

        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    if successful:
        print(f"\n✅ Found {len(successful)} working endpoints:")
        for ep, data in successful:
            print(f"\n   Endpoint: {ep}")
            print(f"   URL: {TCBS_API_BASE}{ep}")
            print(f"   Sample keys: {list(data.keys())[:10] if isinstance(data, dict) else 'list'}")
    else:
        print("\n❌ No working endpoints found with P/E or P/B")
        print("\n💡 Next steps:")
        print("   1. Check vnstock package source code")
        print("   2. Use VNDirect API (already integrated in src/data/fundamental_data.py)")
        print("   3. Contact TCBS for API documentation")

if __name__ == "__main__":
    test_ratio_endpoints()
