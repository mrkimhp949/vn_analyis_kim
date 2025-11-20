"""
Test TCBS API với different headers để bypass 403
"""

import requests
import json

TCBS_API_BASE = "https://apipubaws.tcbs.com.vn"

def test_with_headers():
    """Test TCBS API với different headers"""

    test_symbol = "VNM"

    # Các endpoints có thể có
    endpoints = [
        f"/tcanalysis/v1/company/{test_symbol}/overview",
        f"/tcanalysis/v1/finance/{test_symbol}/financialratio",
        f"/tcanalysis/v1/stock/{test_symbol}/overview",
    ]

    # Different header combinations
    header_combinations = [
        {
            "name": "Browser-like headers",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
                "Referer": "https://www.tcbs.com.vn/",
                "Origin": "https://www.tcbs.com.vn",
            }
        },
        {
            "name": "Mobile headers",
            "headers": {
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X)",
                "Accept": "application/json",
            }
        },
        {
            "name": "Minimal headers",
            "headers": {
                "Accept": "application/json",
            }
        },
    ]

    print("=" * 80)
    print("🔍 TESTING TCBS API WITH DIFFERENT HEADERS")
    print("=" * 80)

    for header_combo in header_combinations:
        print(f"\n{'='*80}")
        print(f"🧪 Testing with: {header_combo['name']}")
        print(f"{'='*80}")

        for endpoint in endpoints:
            url = f"{TCBS_API_BASE}{endpoint}"
            print(f"\n📡 {endpoint}")

            try:
                response = requests.get(url, headers=header_combo['headers'], timeout=10)
                print(f"   Status: {response.status_code}")

                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"   ✅ SUCCESS!")

                        # Check for P/E, P/B
                        data_str = json.dumps(data).lower()
                        has_pe = 'pe' in data_str or 'pricetoearning' in data_str
                        has_pb = 'pb' in data_str or 'pricetobook' in data_str

                        if has_pe or has_pb:
                            print(f"   🎯 Has P/E: {has_pe}, P/B: {has_pb}")
                            print(f"   📊 Sample keys: {list(data.keys())[:10] if isinstance(data, dict) else 'list'}")
                            return url, header_combo['headers'], data

                    except Exception as e:
                        print(f"   ⚠️ Parse error: {str(e)[:100]}")
                elif response.status_code == 403:
                    print(f"   ❌ 403 Forbidden")
                else:
                    print(f"   ❌ {response.status_code}")

            except Exception as e:
                print(f"   ❌ Error: {str(e)[:100]}")

    print("\n" + "=" * 80)
    print("❌ No working combination found")
    print("=" * 80)
    return None, None, None

if __name__ == "__main__":
    result = test_with_headers()

    if result[0]:
        print(f"\n✅ Working endpoint found: {result[0]}")
        print(f"✅ Headers: {result[1]}")
    else:
        print("\n💡 TCBS API requires authentication or is not publicly accessible")
        print("💡 Alternative solutions:")
        print("   1. Use SSI iboard API (public)")
        print("   2. Use cafef.vn data")
        print("   3. Use vietstock.vn API")
        print("   4. Scrape from public websites")
