# -*- coding: utf-8 -*-
"""
Earnings Calendar Provider
Provides earnings dates and major events for Vietnamese stocks
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class EarningsCalendarProvider:
    """
    Provider for earnings calendar and major events

    Data sources:
    1. Local calendar file (CSV/JSON)
    2. TCBS/VNDirect news API
    3. Fallback to quarterly estimation
    """

    def __init__(self, calendar_file: Optional[str] = None):
        """
        Args:
            calendar_file: Path to local calendar file (optional)
        """
        self.calendar_file = calendar_file
        self._calendar = {}  # {symbol: [event1, event2, ...]}
        self._load_calendar()

    def _load_calendar(self):
        """Load calendar from file if available"""
        if not self.calendar_file:
            return

        try:
            import os

            if os.path.exists(self.calendar_file):
                df = pd.read_csv(self.calendar_file)
                # Expected columns: symbol, event_type, event_date
                for _, row in df.iterrows():
                    symbol = row["symbol"]
                    if symbol not in self._calendar:
                        self._calendar[symbol] = []

                    self._calendar[symbol].append(
                        {
                            "type": row["event_type"],
                            "date": pd.to_datetime(row["event_date"]).date(),
                            "description": row.get("description", ""),
                        }
                    )

                logger.info(f"Loaded earnings calendar for {len(self._calendar)} symbols")
        except Exception as e:
            logger.warning(f"Error loading calendar file: {e}")

    def get_upcoming_events(self, symbol: str, days_ahead: int = 30) -> List[Dict]:
        """
        Get upcoming events for a symbol

        Args:
            symbol: Stock symbol
            days_ahead: Look ahead this many days

        Returns:
            List of events: [
                {
                    'type': 'EARNINGS' | 'DIVIDEND' | 'AGM' | 'OTHER',
                    'date': datetime.date,
                    'days_until': int,
                    'description': str
                }
            ]
        """
        today = datetime.now().date()
        cutoff = today + timedelta(days=days_ahead)

        events = []

        # Check local calendar
        if symbol in self._calendar:
            for event in self._calendar[symbol]:
                event_date = event["date"]
                if today <= event_date <= cutoff:
                    days_until = (event_date - today).days
                    events.append(
                        {
                            "type": event["type"],
                            "date": event_date,
                            "days_until": days_until,
                            "description": event.get("description", ""),
                        }
                    )

        # If no events in calendar, estimate quarterly earnings
        if not events:
            estimated = self._estimate_quarterly_earnings(symbol, today, days_ahead)
            if estimated:
                events.append(estimated)

        # Sort by date
        events.sort(key=lambda x: x["date"])

        return events

    def _estimate_quarterly_earnings(
        self, symbol: str, today: datetime.date, days_ahead: int
    ) -> Optional[Dict]:
        """
        Estimate next quarterly earnings date

        Vietnamese stocks typically report:
        - Q4: Mid-late January
        - Q1: Mid-late April
        - Q2: Mid-late July
        - Q3: Mid-late October
        """
        current_month = today.month
        current_day = today.day

        # Earnings reporting months (mid-month)
        earnings_schedule = [
            (1, 20, "Q4"),  # Jan 20 - Q4 results
            (4, 20, "Q1"),  # Apr 20 - Q1 results
            (7, 20, "Q2"),  # Jul 20 - Q2 results
            (10, 20, "Q3"),  # Oct 20 - Q3 results
        ]

        for month, day, quarter in earnings_schedule:
            # Calculate next earnings date
            if month > current_month or (month == current_month and day > current_day):
                # This year
                earnings_date = datetime(today.year, month, day).date()
            else:
                # Next year
                earnings_date = datetime(today.year + 1, month, day).date()

            days_until = (earnings_date - today).days

            # Check if within our window
            if 0 <= days_until <= days_ahead:
                return {
                    "type": "EARNINGS",
                    "date": earnings_date,
                    "days_until": days_until,
                    "description": f"{quarter} Earnings (estimated)",
                }

        return None

    def check_event_proximity(
        self, symbol: str, avoid_days_before: int = 5, prefer_days_after: int = 10
    ) -> Dict:
        """
        Check if symbol has events too close

        Args:
            symbol: Stock symbol
            avoid_days_before: Avoid buying this many days before event
            prefer_days_after: Prefer buying this many days after event

        Returns:
            {
                'too_close_to_event': bool,
                'event_passed': bool,
                'event_type': str,
                'days_until': int,
                'days_since': int,
                'description': str
            }
        """
        today = datetime.now().date()

        # Check upcoming events
        upcoming = self.get_upcoming_events(symbol, days_ahead=avoid_days_before)

        if upcoming:
            next_event = upcoming[0]
            days_until = next_event["days_until"]

            if days_until <= avoid_days_before:
                return {
                    "too_close_to_event": True,
                    "event_passed": False,
                    "event_type": next_event["type"],
                    "days_until": days_until,
                    "days_since": None,
                    "description": next_event["description"],
                }

        # Check recent past events
        past_events = self._get_past_events(symbol, days_back=prefer_days_after)

        if past_events:
            recent_event = past_events[0]
            days_since = (today - recent_event["date"]).days

            if days_since <= prefer_days_after:
                return {
                    "too_close_to_event": False,
                    "event_passed": True,
                    "event_type": recent_event["type"],
                    "days_until": None,
                    "days_since": days_since,
                    "description": recent_event["description"],
                }

        # No events nearby
        return {
            "too_close_to_event": False,
            "event_passed": False,
            "event_type": None,
            "days_until": None,
            "days_since": None,
            "description": None,
        }

    def _get_past_events(self, symbol: str, days_back: int = 30) -> List[Dict]:
        """Get past events within days_back"""
        today = datetime.now().date()
        cutoff = today - timedelta(days=days_back)

        events = []

        # Check local calendar
        if symbol in self._calendar:
            for event in self._calendar[symbol]:
                event_date = event["date"]
                if cutoff <= event_date < today:
                    days_since = (today - event_date).days
                    events.append(
                        {
                            "type": event["type"],
                            "date": event_date,
                            "days_since": days_since,
                            "description": event.get("description", ""),
                        }
                    )

        # Sort by date (most recent first)
        events.sort(key=lambda x: x["date"], reverse=True)

        return events

    def add_event(
        self, symbol: str, event_type: str, event_date: datetime.date, description: str = ""
    ):
        """
        Add an event to the calendar

        Args:
            symbol: Stock symbol
            event_type: 'EARNINGS', 'DIVIDEND', 'AGM', 'OTHER'
            event_date: Date of event
            description: Event description
        """
        if symbol not in self._calendar:
            self._calendar[symbol] = []

        self._calendar[symbol].append(
            {"type": event_type, "date": event_date, "description": description}
        )

        logger.info(f"Added {event_type} event for {symbol} on {event_date}")

    def save_calendar(self, output_file: str):
        """Save calendar to CSV file"""
        try:
            rows = []
            for symbol, events in self._calendar.items():
                for event in events:
                    rows.append(
                        {
                            "symbol": symbol,
                            "event_type": event["type"],
                            "event_date": event["date"],
                            "description": event.get("description", ""),
                        }
                    )

            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
            logger.info(f"Saved calendar to {output_file}")
        except Exception as e:
            logger.error(f"Error saving calendar: {e}")


# Singleton instance
_earnings_calendar = None


def get_earnings_calendar(calendar_file: Optional[str] = None) -> EarningsCalendarProvider:
    """Get singleton earnings calendar"""
    global _earnings_calendar
    if _earnings_calendar is None:
        _earnings_calendar = EarningsCalendarProvider(calendar_file)
    return _earnings_calendar
