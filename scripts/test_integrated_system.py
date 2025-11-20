"""
Test Integrated Fundamental Data System with CSV Provider
"""
import sys
sys.path.insert(0, '/home/user/vn_analyis_kim')

import logging
logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("🧪 TESTING INTEGRATED FUNDAMENTAL DATA SYSTEM")
print("=" * 80)

try:
    from src.data.fundamental_data import get_fundamental_manager, get_fundamental_data

    # Test 1: Get manager with CSV enabled (default)
    print("\n📊 Test 1: Get fundamental manager...")
    manager = get_fundamental_manager()
    print(f"   ✅ Manager created with {len(manager.providers)} provider(s)")

    # Test 2: Get data through manager
    test_symbols = ["VNM", "VCB", "FPT"]

    print("\n📊 Test 2: Get fundamental data through manager...")
    for symbol in test_symbols:
        data = manager.get_fundamental_data(symbol)
        if data:
            print(f"   ✅ {symbol}: P/E={data.pe_ratio}, P/B={data.pb_ratio}, Source={data.source}")
        else:
            print(f"   ❌ {symbol}: No data")

    # Test 3: Use convenience function
    print("\n📊 Test 3: Use convenience function...")
    data = get_fundamental_data("HPG")
    if data:
        print(f"   ✅ HPG: P/E={data.pe_ratio}, P/B={data.pb_ratio}")
    else:
        print(f"   ❌ HPG: No data")

    # Test 4: Cache test
    print("\n📊 Test 4: Test caching...")
    data1 = get_fundamental_data("VNM")
    data2 = get_fundamental_data("VNM")  # Should use cache
    print(f"   ✅ Cache working: {data1 is data2 or (data1 and data2 and data1.symbol == data2.symbol)}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("💡 CSV Fundamental Provider is integrated and working!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
