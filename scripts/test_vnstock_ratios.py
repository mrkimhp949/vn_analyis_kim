"""
Test vnstock để lấy P/E, P/B từ TCBS
"""

try:
    from vnstock import Vnstock
    print("✅ vnstock imported successfully")
except ImportError as e:
    print(f"❌ Failed to import vnstock: {e}")
    exit(1)

def test_vnstock_ratios():
    """Test lấy financial ratios từ TCBS qua vnstock"""

    test_symbol = "VNM"

    print("=" * 80)
    print(f"🔍 TESTING VNSTOCK FOR {test_symbol}")
    print("=" * 80)

    try:
        # Initialize vnstock with TCBS source
        print(f"\n📊 Initializing Vnstock for {test_symbol}...")
        stock = Vnstock().stock(symbol=test_symbol, source='TCBS')
        print(f"   ✅ Stock object created")

        # Get financial ratios
        print(f"\n📈 Fetching financial ratios...")
        ratios = stock.finance.ratio(period='year', lang='en', dropna=True)

        print(f"   ✅ Got {len(ratios)} rows of ratio data")
        print(f"   📋 Columns: {list(ratios.columns)}")

        # Check for P/E and P/B
        print(f"\n🔍 Looking for P/E and P/B...")

        # Common column names for P/E
        pe_columns = ['pe', 'p_e', 'pe_ratio', 'peratio', 'priceToEarning', 'price_to_earning', 'PE']
        pb_columns = ['pb', 'p_b', 'pb_ratio', 'pbratio', 'priceToBook', 'price_to_book', 'PB']

        pe_col = None
        pb_col = None

        for col in ratios.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            if any(pe.lower().replace('_', '') in col_lower for pe in pe_columns):
                pe_col = col
            if any(pb.lower().replace('_', '') in col_lower for pb in pb_columns):
                pb_col = col

        if pe_col:
            print(f"   ✅ Found P/E column: {pe_col}")
            print(f"      Latest P/E: {ratios[pe_col].iloc[-1]}")
        else:
            print(f"   ❌ P/E not found")

        if pb_col:
            print(f"   ✅ Found P/B column: {pb_col}")
            print(f"      Latest P/B: {ratios[pb_col].iloc[-1]}")
        else:
            print(f"   ❌ P/B not found")

        # Show latest data
        print(f"\n📊 Latest financial ratios:")
        print(ratios.tail(1).T)

        # Return results
        return {
            'success': True,
            'ratios': ratios,
            'pe_col': pe_col,
            'pb_col': pb_col
        }

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    result = test_vnstock_ratios()

    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    if result['success']:
        print("\n✅ vnstock successfully retrieved financial ratios from TCBS!")
        print(f"   P/E column: {result['pe_col']}")
        print(f"   P/B column: {result['pb_col']}")
        print("\n💡 Ready to integrate into src/data/fundamental_data.py")
    else:
        print(f"\n❌ Failed: {result.get('error')}")
        print("\n💡 Fallback to VNDirect API (already integrated)")
