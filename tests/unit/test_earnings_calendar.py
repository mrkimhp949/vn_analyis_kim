# -*- coding: utf-8 -*-
"""
Tests for Earnings Calendar Provider
"""

import pytest
from datetime import datetime, timedelta
from src.data.earnings_calendar import EarningsCalendarProvider, get_earnings_calendar


class TestEarningsCalendarProvider:
    """Test earnings calendar provider"""

    def test_provider_initialization(self):
        """Test provider can be initialized"""
        calendar = EarningsCalendarProvider()
        assert calendar is not None

    def test_get_upcoming_events_no_data(self):
        """Test getting upcoming events when no data"""
        calendar = EarningsCalendarProvider()
        events = calendar.get_upcoming_events("VNM", days_ahead=30)

        # Should return at least estimated quarterly earnings
        assert isinstance(events, list)
        # May have estimated earnings
        if len(events) > 0:
            assert "type" in events[0]
            assert "date" in events[0]
            assert "days_until" in events[0]

    def test_add_event(self):
        """Test adding an event"""
        calendar = EarningsCalendarProvider()

        future_date = datetime.now().date() + timedelta(days=10)
        calendar.add_event(
            symbol="VNM", event_type="EARNINGS", event_date=future_date, description="Q4 Earnings"
        )

        # Check event was added
        events = calendar.get_upcoming_events("VNM", days_ahead=30)
        assert len(events) > 0

        # Find our event
        our_event = [e for e in events if e.get("description") == "Q4 Earnings"]
        assert len(our_event) > 0

    def test_check_event_proximity_no_events(self):
        """Test checking event proximity when no events"""
        calendar = EarningsCalendarProvider()

        result = calendar.check_event_proximity("TEST_SYMBOL")

        assert isinstance(result, dict)
        assert "too_close_to_event" in result
        assert "event_passed" in result

    def test_check_event_proximity_close_event(self):
        """Test checking event proximity when event is close"""
        calendar = EarningsCalendarProvider()

        # Add event in 3 days
        future_date = datetime.now().date() + timedelta(days=3)
        calendar.add_event(
            symbol="TEST",
            event_type="EARNINGS",
            event_date=future_date,
            description="Test Earnings",
        )

        result = calendar.check_event_proximity("TEST", avoid_days_before=5)

        # Should detect event is too close
        assert result["too_close_to_event"] == True
        assert result["event_type"] == "EARNINGS"
        assert result["days_until"] == 3

    def test_check_event_proximity_past_event(self):
        """Test checking event proximity when event just passed"""
        calendar = EarningsCalendarProvider()

        # Add event 5 days ago
        past_date = datetime.now().date() - timedelta(days=5)
        calendar.add_event(
            symbol="TEST2", event_type="EARNINGS", event_date=past_date, description="Past Earnings"
        )

        result = calendar.check_event_proximity("TEST2", prefer_days_after=10)

        # Should detect event just passed
        assert result["event_passed"] == True
        assert result["event_type"] == "EARNINGS"
        assert result["days_since"] == 5

    def test_quarterly_earnings_estimation(self):
        """Test quarterly earnings estimation"""
        calendar = EarningsCalendarProvider()

        # Get events for symbol without data
        events = calendar.get_upcoming_events("UNKNOWN_SYMBOL", days_ahead=90)

        # Should have estimated quarterly earnings
        assert len(events) > 0

        # Check it's an earnings event
        earnings_events = [e for e in events if e["type"] == "EARNINGS"]
        assert len(earnings_events) > 0

    def test_singleton_pattern(self):
        """Test singleton pattern works"""
        calendar1 = get_earnings_calendar()
        calendar2 = get_earnings_calendar()

        # Should be same instance
        assert calendar1 is calendar2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
