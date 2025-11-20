"""
Test vnstock với ALL available sources để tìm source nào work
Based on issue #186: VCI blocks cloud IPs, TCBS might work
"""

import sys
sys.path.insert(0, '/home/user/vn_analyis_kim')

import logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

print("=" * 80)
print("🧪 TESTING VNSTOCK - ALL DATA SOURCES")
print("=" * 80)

try:
    from vnstock import Vnstock
    print("\n✅ Successfully imported vnstock")

    # All available sources
    sources = ['TCBS', 'VCI', 'MSN']
    test_symbol = "VNM"  # Vinamilk - popular stock

    working_sources = []
    failed_sources = []

    for source in sources:
        print(f"\n{'═' * 80}")
        print(f"📡 TESTING SOURCE: {source}")
        print(f"{'═' * 80}")

        try:
            # Initialize with this source
            stock = Vnstock().stock(symbol=test_symbol, source=source)
            print(f"✅ Stock object created with {source} source")

            # Try to get financial ratios
            print(f"📈 Fetching financial ratios from {source}...")
            ratios = stock.finance.ratio(period='year', lang='vi', dropna=True)

            if ratios is not None and not ratios.empty and len(ratios) > 0:
                print(f"✅ SUCCESS! Got {len(ratios)} rows from {source}")
                print(f"📋 Columns: {list(ratios.columns)[:15]}...")

                # Look for P/E, P/B
                pe_found = False
                pb_found = False

                for col in ratios.columns:
                    col_lower = col.lower()
                    if any(x in col_lower for x in ['pe', 'pricetoearning', 'price_to_earning']):
                        pe_found = True
                        pe_value = ratios[col].iloc[-1] if len(ratios) > 0 else None
                        print(f"   🎯 P/E found in column '{col}': {pe_value}")

                    if any(x in col_lower for x in ['pb', 'pricetobook', 'price_to_book']):
                        pb_found = True
                        pb_value = ratios[col].iloc[-1] if len(ratios) > 0 else None
                        print(f"   🎯 P/B found in column '{col}': {pb_value}")

                if pe_found or pb_found:
                    print(f"\n✅ {source} WORKS! Has P/E: {pe_found}, Has P/B: {pb_found}")
                    working_sources.append(source)

                    # Show sample
                    print(f"\n📊 Sample data (first 5 columns):")
                    print(ratios.iloc[-1:, :5].to_string())
                else:
                    print(f"⚠️ {source} returned data but NO P/E or P/B columns")
                    print(f"   Available columns: {list(ratios.columns)}")
                    failed_sources.append(source)
            else:
                print(f"❌ {source} returned empty data")
                failed_sources.append(source)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ {source} FAILED: {error_msg[:100]}")
            failed_sources.append(source)

    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)

    print(f"\n✅ WORKING Sources ({len(working_sources)}/{len(sources)}):")
    if working_sources:
        for s in working_sources:
            print(f"   ✓ {s}")
        print("\n🎯 RESULT: CAN GET P/E, P/B FROM vnstock!")
        print(f"💡 Use source: {working_sources[0]}")
        print("💡 Ready to integrate into src/data/tcbs_provider.py")
    else:
        print("   None")

    print(f"\n❌ FAILED Sources ({len(failed_sources)}/{len(sources)}):")
    if failed_sources:
        for s in failed_sources:
            print(f"   ✗ {s}")

    if not working_sources:
        print("\n⚠️ RESULT: ALL SOURCES BLOCKED")
        print("💡 Likely cause: IP blocking (Cloud/VPS IPs)")
        print("💡 Solutions:")
        print("   1. Use CSV file (recommended)")
        print("   2. Use proxy server")
        print("   3. Run from local machine (not cloud)")
        print("   4. Contact VCI/TCBS for API access")

except Exception as e:
    print(f"\n❌ Fatal error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
