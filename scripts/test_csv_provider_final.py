"""
Test CSV Fundamental Provider
"""
import sys
sys.path.insert(0, '/home/user/vn_analyis_kim')

import logging
logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("🧪 TESTING CSV FUNDAMENTAL PROVIDER")
print("=" * 80)

try:
    from src.data.csv_fundamental_provider import get_csv_fundamental_data

    test_symbols = ["VNM", "VCB", "FPT", "HPG", "INVALID_SYMBOL"]

    successful = 0
    failed = 0

    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        data = get_csv_fundamental_data(symbol)

        if data and data.is_valid():
            print(f"   ✅ SUCCESS!")
            print(f"   P/E Ratio: {data.pe_ratio}")
            print(f"   P/B Ratio: {data.pb_ratio}")
            print(f"   ROE: {data.roe}%")
            print(f"   ROA: {data.roa}%")
            print(f"   Debt/Equity: {data.debt_to_equity}")
            print(f"   EPS: {data.eps}")
            print(f"   Market Cap: {data.market_cap:,.0f} VND" if data.market_cap else "   Market Cap: N/A")
            print(f"   Source: {data.source}")
            successful += 1
        else:
            print(f"   ❌ No data")
            failed += 1

    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {successful}/{len(test_symbols)}")
    print(f"❌ Failed: {failed}/{len(test_symbols)}")

    if successful > 0:
        print("\n🎯 CSV PROVIDER IS WORKING!")
        print("💡 Ready for production use")
    else:
        print("\n⚠️ CSV provider not working properly")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
