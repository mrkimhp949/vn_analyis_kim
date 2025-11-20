"""
Test vnstock3 với latest version theo documentation chính thức
Tham khảo: https://github.com/vnstock-official/vnstock
"""

import sys
sys.path.insert(0, '/home/user/vn_analyis_kim')

import logging
logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("🧪 TESTING VNSTOCK3 (OFFICIAL VERSION) - FINANCIAL RATIOS")
print("=" * 80)

try:
    from vnstock import Vnstock
    print("\n✅ Successfully imported vnstock")

    # Check version
    import vnstock
    if hasattr(vnstock, '__version__'):
        print(f"📦 vnstock version: {vnstock.__version__}")

    # Test với VCI source như documentation
    test_symbols = ["VCI", "VNM", "VCB", "FPT"]

    successful = []
    failed = []

    for symbol in test_symbols:
        print(f"\n{'─' * 80}")
        print(f"📊 Testing {symbol}...")

        try:
            # Khởi tạo stock object với VCI source
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            print(f"   ✅ Stock object created for {symbol}")

            # Lấy financial ratios (yearly)
            print(f"   📈 Fetching financial ratios...")
            ratios = stock.finance.ratio(period='year', lang='vi', dropna=True)

            if ratios is not None and not ratios.empty:
                print(f"   ✅ Got {len(ratios)} rows of ratio data")
                print(f"   📋 Columns ({len(ratios.columns)}): {list(ratios.columns)[:10]}...")

                # Tìm P/E và P/B columns
                pe_col = None
                pb_col = None

                for col in ratios.columns:
                    col_lower = col.lower()
                    if 'pe' in col_lower or 'pricetoearning' in col_lower or 'price_to_earning' in col_lower:
                        pe_col = col
                    if 'pb' in col_lower or 'pricetobook' in col_lower or 'price_to_book' in col_lower:
                        pb_col = col

                if pe_col or pb_col:
                    print(f"\n   🎯 FOUND P/E AND/OR P/B!")
                    if pe_col:
                        latest_pe = ratios[pe_col].iloc[-1] if len(ratios) > 0 else None
                        print(f"      P/E column: {pe_col}")
                        print(f"      Latest P/E: {latest_pe}")
                    if pb_col:
                        latest_pb = ratios[pb_col].iloc[-1] if len(ratios) > 0 else None
                        print(f"      P/B column: {pb_col}")
                        print(f"      Latest P/B: {latest_pb}")

                    successful.append(symbol)

                    # Show sample data
                    print(f"\n   📊 Latest data (showing first 10 columns):")
                    print(ratios.iloc[-1:].head().to_string())
                else:
                    print(f"   ⚠️ No P/E or P/B columns found")
                    print(f"   📋 Available columns: {list(ratios.columns)[:20]}")
                    failed.append(symbol)
            else:
                print(f"   ⚠️ Empty data returned")
                failed.append(symbol)

        except Exception as e:
            print(f"   ❌ Error: {e}")
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
        print("\n🎯 RESULT: vnstock3 IS WORKING!")
        print("💡 Can integrate into src/data/fundamental_data.py")
        print("\n📝 Next step: Update TCBSProvider to use this working method")
    else:
        print("\n⚠️ RESULT: vnstock3 still not working")
        print("💡 Recommend CSV solution as documented")

except ImportError as e:
    print(f"\n❌ Import error: {e}")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
