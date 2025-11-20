#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Fundamental Data Integration
Demo script to test fundamental provider and earnings calendar
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timedelta
from src.data.fundamental_provider import get_fundamental_provider
from src.data.earnings_calendar import get_earnings_calendar


def test_fundamental_provider():
    """Test fundamental data provider"""
    print("\n" + "=" * 70)
    print("🧪 TESTING FUNDAMENTAL DATA PROVIDER")
    print("=" * 70 + "\n")

    provider = get_fundamental_provider()

    # Test with popular Vietnamese stocks
    test_symbols = ["VNM", "VIC", "VHM", "HPG", "VCB"]

    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        fundamentals = provider.get_fundamentals(symbol)

        if fundamentals:
            print(f"✅ Got fundamental data for {symbol}:")
            print(f"   P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}")
            print(f"   P/B Ratio: {fundamentals.get('pb_ratio', 'N/A')}")
            print(f"   ROE: {fundamentals.get('roe', 'N/A')}%")
            print(f"   Debt/Equity: {fundamentals.get('debt_ratio', 'N/A')}")
            print(f"   Profit Margin: {fundamentals.get('profit_margin', 'N/A')}%")
            print(
                f"   Market Cap: {fundamentals.get('market_cap', 'N/A'):,.0f} VND"
                if fundamentals.get("market_cap")
                else "   Market Cap: N/A"
            )
        else:
            print(f"⚠️ No fundamental data available for {symbol}")

    print("\n" + "=" * 70)


def test_earnings_calendar():
    """Test earnings calendar provider"""
    print("\n" + "=" * 70)
    print("📅 TESTING EARNINGS CALENDAR")
    print("=" * 70 + "\n")

    calendar = get_earnings_calendar()

    # Test with a symbol
    symbol = "VNM"
    print(f"\n📊 Testing {symbol}...")

    # Get upcoming events
    events = calendar.get_upcoming_events(symbol, days_ahead=90)
    print(f"\n✅ Found {len(events)} upcoming events:")
    for event in events:
        print(f"   • {event['type']}: {event['date']} ({event['days_until']} days)")
        if event.get("description"):
            print(f"     Description: {event['description']}")

    # Check event proximity
    proximity = calendar.check_event_proximity(symbol)
    print(f"\n📍 Event Proximity Check:")
    print(f"   Too close to event: {proximity['too_close_to_event']}")
    print(f"   Event passed: {proximity['event_passed']}")
    if proximity["event_type"]:
        print(f"   Event type: {proximity['event_type']}")
        if proximity["days_until"]:
            print(f"   Days until: {proximity['days_until']}")
        if proximity["days_since"]:
            print(f"   Days since: {proximity['days_since']}")

    # Add a test event
    print(f"\n➕ Adding test event...")
    future_date = datetime.now().date() + timedelta(days=15)
    calendar.add_event(
        symbol="TEST", event_type="EARNINGS", event_date=future_date, description="Test Q4 Earnings"
    )

    # Check the test event
    test_events = calendar.get_upcoming_events("TEST", days_ahead=30)
    print(f"✅ Test event added: {len(test_events)} events for TEST")

    print("\n" + "=" * 70)


def test_entry_logic_integration():
    """Test integration with entry logic"""
    print("\n" + "=" * 70)
    print("🎯 TESTING ENTRY LOGIC INTEGRATION")
    print("=" * 70 + "\n")

    from src.strategies.entry_logic import ImprovedEntryLogic
    from src.data.loader import load_data

    entry_logic = ImprovedEntryLogic()

    symbol = "VNM"
    print(f"\n📊 Testing entry logic with {symbol}...")

    # Load data
    df = load_data(symbol, lookback=200)

    # Test earnings check
    print("\n📅 Testing earnings check...")
    earnings_check = entry_logic._check_earnings_events(df, symbol)
    print(f"   Too close to event: {earnings_check['too_close_to_event']}")
    print(f"   Event passed: {earnings_check['event_passed']}")
    if earnings_check["event_type"]:
        print(f"   Event type: {earnings_check['event_type']}")

    # Test fundamentals check
    print("\n💰 Testing fundamentals check...")
    current_price = df["close"].iloc[-1]
    fundamentals_check = entry_logic._check_fundamentals(df, symbol, current_price)
    print(f"   Poor fundamentals: {fundamentals_check['poor_fundamentals']}")
    print(f"   Good fundamentals: {fundamentals_check['good_fundamentals']}")
    if fundamentals_check["reason"]:
        print(f"   Reason: {fundamentals_check['reason']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 FUNDAMENTAL DATA INTEGRATION TEST")
    print("=" * 70)

    try:
        # Test fundamental provider
        test_fundamental_provider()

        # Test earnings calendar
        test_earnings_calendar()

        # Test entry logic integration
        test_entry_logic_integration()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
