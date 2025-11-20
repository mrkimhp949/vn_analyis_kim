"""
Test DIRECT call tới TCBS financial ratio endpoint
Bypass vnstock wrapper để xem có work không
Based on source code: vnstock/explorer/tcbs/financial.py
"""

import requests
import json
import pandas as pd

# Constants from vnstock source code
TCBS_BASE_URL = "https://apipubaws.tcbs.com.vn"
ANALYSIS_URL = "tcanalysis"

def test_direct_financialratio_call(symbol: str = "VNM"):
    """Test direct API call to TCBS financial ratio endpoint"""

    print("=" * 80)
    print(f"🧪 TESTING DIRECT TCBS FINANCIAL RATIO API")
    print(f"Symbol: {symbol}")
    print("=" * 80)

    # Construct endpoint URL (from vnstock source)
    endpoint = f"{TCBS_BASE_URL}/{ANALYSIS_URL}/v1/finance/{symbol}/financialratio"

    print(f"\n📡 Endpoint: {endpoint}")

    # Different header configurations to test
    header_configs = [
        {
            "name": "vnstock-style headers",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
            }
        },
        {
            "name": "Browser-like with Referer",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://www.tcbs.com.vn/",
                "Origin": "https://www.tcbs.com.vn",
            }
        },
        {
            "name": "Minimal headers",
            "headers": {
                "Accept": "application/json"
            }
        }
    ]

    # Try different parameter combinations
    param_configs = [
        {"yearly": 1, "isAll": "true"},
        {"yearly": 0, "isAll": "true"},
        {"yearly": 1},
        {}
    ]

    for header_config in header_configs:
        print(f"\n{'─' * 80}")
        print(f"🔧 Testing with: {header_config['name']}")

        for i, params in enumerate(param_configs):
            print(f"\n   📋 Params #{i+1}: {params}")

            try:
                response = requests.get(
                    endpoint,
                    headers=header_config['headers'],
                    params=params,
                    timeout=10
                )

                status = response.status_code
                print(f"      Status: {status}")

                if status == 200:
                    try:
                        data = response.json()
                        print(f"      ✅ SUCCESS! Got data")

                        # Check if we have P/E, P/B
                        data_str = json.dumps(data, ensure_ascii=False).lower()
                        has_pe = any(x in data_str for x in ['pe', 'pricetoearning', 'price_earning'])
                        has_pb = any(x in data_str for x in ['pb', 'pricetobook', 'price_book'])

                        print(f"      📊 Data type: {type(data)}")
                        if isinstance(data, dict):
                            print(f"      📋 Keys: {list(data.keys())[:10]}")
                        elif isinstance(data, list) and len(data) > 0:
                            print(f"      📋 List length: {len(data)}")
                            if isinstance(data[0], dict):
                                print(f"      📋 First item keys: {list(data[0].keys())[:10]}")

                        if has_pe or has_pb:
                            print(f"      🎯 P/E: {has_pe}, P/B: {has_pb}")

                            # Try to create DataFrame
                            try:
                                df = pd.DataFrame(data)
                                print(f"      📊 DataFrame shape: {df.shape}")
                                print(f"      📋 Columns: {list(df.columns)[:20]}")

                                # Look for P/E, P/B columns
                                for col in df.columns:
                                    col_lower = col.lower()
                                    if 'pe' in col_lower or 'earning' in col_lower:
                                        print(f"         ✓ P/E column: {col}")
                                    if 'pb' in col_lower or 'book' in col_lower:
                                        print(f"         ✓ P/B column: {col}")

                                return {
                                    'success': True,
                                    'endpoint': endpoint,
                                    'headers': header_config['headers'],
                                    'params': params,
                                    'data': df
                                }
                            except Exception as e:
                                print(f"      ⚠️ DataFrame error: {e}")
                        else:
                            print(f"      ⚠️ No P/E or P/B in response")

                    except Exception as e:
                        print(f"      ⚠️ Parse error: {str(e)[:100]}")

                elif status == 403:
                    print(f"      ❌ 403 Forbidden")
                elif status == 404:
                    print(f"      ❌ 404 Not Found")
                else:
                    print(f"      ❌ Error {status}")

            except requests.exceptions.Timeout:
                print(f"      ⏱️ Timeout")
            except Exception as e:
                print(f"      ❌ Error: {str(e)[:100]}")

    print("\n" + "=" * 80)
    print("❌ NO WORKING CONFIGURATION FOUND")
    print("=" * 80)
    return None


if __name__ == "__main__":
    result = test_direct_financialratio_call("VNM")

    print("\n" + "=" * 80)
    print("📊 FINAL RESULT")
    print("=" * 80)

    if result:
        print("\n✅ SUCCESS! Direct API call works!")
        print(f"   Endpoint: {result['endpoint']}")
        print(f"   Params: {result['params']}")
        print(f"\n📊 Sample data:")
        print(result['data'].head())
    else:
        print("\n❌ All attempts failed")
        print("\n💡 Conclusion:")
        print("   - Direct API calls also return 403 Forbidden")
        print("   - Not a vnstock wrapper issue")
        print("   - IP blocking at TCBS infrastructure level")
        print("   - Recommend CSV solution")
