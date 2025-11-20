"""
Test TCBS API endpoints để lấy P/E, P/B và fundamental data
"""

import requests
import json

TCBS_API_BASE = "https://apipubaws.tcbs.com.vn"

def test_tcbs_endpoints():
    """Test các endpoints TCBS có thể có fundamental data"""

    test_symbol = "VNM"  # Vinamilk - cổ phiếu phổ biến

    print("=" * 80)
    print(f"🔍 TESTING TCBS API ENDPOINTS FOR {test_symbol}")
    print("=" * 80)

    # Danh sách endpoints để test
    endpoints = [
        # Stock info/overview
        f"/stock-insight/v1/stock/overview/{test_symbol}",
        f"/stock-insight/v1/stock/info/{test_symbol}",
        f"/stock-insight/v1/stock/details/{test_symbol}",
        f"/stock-insight/v1/stock/fundamental/{test_symbol}",
        f"/stock-insight/v1/stock/ratios/{test_symbol}",
        f"/stock-insight/v1/stock/valuation/{test_symbol}",

        # Stock snapshot
        f"/stock-insight/v1/snapshot/{test_symbol}",
        f"/stock-insight/v1/ticker/{test_symbol}",

        # Company data
        f"/stock-insight/v1/company/{test_symbol}/overview",
        f"/stock-insight/v1/company/{test_symbol}/fundamental",
        f"/stock-insight/v1/company/{test_symbol}/ratios",

        # Financial data
        f"/stock-insight/v1/finance/{test_symbol}/ratios",
        f"/stock-insight/v1/finance/{test_symbol}/fundamental",

        # Stock data với query param
        f"/stock-insight/v1/stock/data?ticker={test_symbol}",
        f"/stock-insight/v1/stock?ticker={test_symbol}",
    ]

    successful_endpoints = []

    for endpoint in endpoints:
        url = f"{TCBS_API_BASE}{endpoint}"
        print(f"\n{'─' * 80}")
        print(f"📡 Testing: {endpoint}")

        try:
            response = requests.get(url, timeout=10)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ SUCCESS! Response type: {type(data)}")

                    # Kiểm tra xem có P/E, P/B không
                    data_str = json.dumps(data).lower()
                    has_pe = any(key in data_str for key in ['pe', 'p/e', 'peratio', 'pe_ratio'])
                    has_pb = any(key in data_str for key in ['pb', 'p/b', 'pbratio', 'pb_ratio'])

                    if has_pe or has_pb:
                        print(f"   🎯 Contains P/E: {has_pe}, P/B: {has_pb}")
                        successful_endpoints.append(endpoint)

                        # Print sample data
                        print(f"   📊 Sample response:")
                        if isinstance(data, dict):
                            for key, value in list(data.items())[:10]:
                                print(f"      {key}: {value}")
                        elif isinstance(data, list) and len(data) > 0:
                            print(f"      First item: {data[0]}")
                    else:
                        print(f"   ⚠️ No P/E or P/B found in response")

                except json.JSONDecodeError:
                    print(f"   ⚠️ Response is not JSON")
            elif response.status_code == 404:
                print(f"   ❌ Not Found (404)")
            elif response.status_code == 401:
                print(f"   ❌ Unauthorized (401) - Requires API key")
            else:
                print(f"   ❌ Error: {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"   ⏱️ Timeout")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request Error: {str(e)[:100]}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    if successful_endpoints:
        print(f"\n✅ Found {len(successful_endpoints)} endpoints with P/E or P/B data:")
        for ep in successful_endpoints:
            print(f"   • {ep}")
    else:
        print("\n❌ No endpoints found with P/E or P/B data")
        print("\n💡 Recommendation:")
        print("   TCBS API may not provide fundamental data publicly.")
        print("   The existing VNDirect API integration already provides P/E and P/B.")
        print("   See: src/data/fundamental_data.py")

if __name__ == "__main__":
    test_tcbs_endpoints()
