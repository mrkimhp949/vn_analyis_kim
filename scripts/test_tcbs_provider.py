"""
Test TCBS Provider với VCI source
"""

import sys
import logging
sys.path.insert(0, '/home/user/vn_analyis_kim')

logging.basicConfig(level=logging.INFO)

print("\n" + "=" * 80)
print("🧪 TESTING TCBS PROVIDER WITH VCI SOURCE")
print("=" * 80 + "\n")

try:
    from src.data.tcbs_provider import TCBSProvider, get_tcbs_fundamental_data

    print("✅ Successfully imported TCBSProvider\n")

    # Test with VCI source
    test_symbols = ["VNM", "VCB", "FPT", "HPG", "VIC"]

    successful = []
    failed = []

    for symbol in test_symbols:
        print(f"{'─' * 80}")
        print(f"📊 Testing {symbol}...")

        try:
            data = get_tcbs_fundamental_data(symbol, source='VCI')

            if data and data.is_valid():
                print(f"✅ SUCCESS for {symbol}!")
                print(f"   Source: {data.source}")
                print(f"   P/E Ratio: {data.pe_ratio}")
                print(f"   P/B Ratio: {data.pb_ratio}")
                print(f"   ROE: {data.roe}%")
                print(f"   ROA: {data.roa}%")
                print(f"   Debt/Equity: {data.debt_to_equity}")
                print(f"   EPS: {data.eps}")
                if data.market_cap:
                    print(f"   Market Cap: {data.market_cap:,.0f} VND")
                successful.append(symbol)
            else:
                print(f"⚠️ No valid data for {symbol}")
                failed.append(symbol)

        except Exception as e:
            print(f"❌ Error for {symbol}: {e}")
            failed.append(symbol)

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    print(f"\n✅ Successful: {len(successful)}/{len(test_symbols)}")
    if successful:
        print(f"   {', '.join(successful)}")

    print(f"\n❌ Failed: {len(failed)}/{len(test_symbols)}")
    if failed:
        print(f"   {', '.join(failed)}")

    if len(successful) > 0:
        print("\n🎯 RESULT: VCI source is WORKING!")
        print("💡 Can integrate TCBSProvider into fundamental_data.py")
    else:
        print("\n⚠️ RESULT: VCI source also unavailable")
        print("💡 Need alternative solution (CSV, web scraping, or API keys)")

except Exception as e:
    print(f"❌ Error importing or running test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
