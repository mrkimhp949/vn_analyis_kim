"""
Emergency Stop - Dừng trading khi market crash
Bảo vệ khỏi các sự kiện bất thường
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple


@dataclass
class EmergencyEvent:
    """Emergency event"""

    timestamp: str
    event_type: str
    reason: str
    vnindex_change: float
    duration_hours: int
    auto_resume: bool


class EmergencyStop:
    """
    Emergency Stop System

    Triggers:
    - Market crash (VNINDEX giảm > 5% trong ngày)
    - Flash crash (giảm > 3% trong 1 giờ)
    - Circuit breaker triggered
    - Manual emergency stop
    """

    def __init__(
        self,
        crash_threshold_daily: float = -0.05,  # -5% trong ngày
        crash_threshold_hourly: float = -0.03,  # -3% trong giờ
        auto_resume_hours: int = 24,
        events_file: str = "emergency_events.json",
    ):
        self.crash_threshold_daily = crash_threshold_daily
        self.crash_threshold_hourly = crash_threshold_hourly
        self.auto_resume_hours = auto_resume_hours
        self.events_file = events_file

        self.events = self._load_events()
        self._check_auto_resume()

    def _load_events(self) -> Dict:
        """Load events từ file"""
        if os.path.exists(self.events_file):
            try:
                with open(self.events_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return {"active_emergency": None, "history": []}

    def _save_events(self):
        """Save events vào file"""
        with open(self.events_file, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)

    def _check_auto_resume(self):
        """Check xem có thể auto resume không"""
        if not self.events["active_emergency"]:
            return

        event = EmergencyEvent(**self.events["active_emergency"])

        if not event.auto_resume:
            return  # Manual resume required

        # Check duration
        event_time = datetime.fromisoformat(event.timestamp)
        now = datetime.now()
        hours_passed = (now - event_time).total_seconds() / 3600

        if hours_passed >= event.duration_hours:
            # Auto resume
            self.resume("Auto resume after duration")

    def check_market_crash(self) -> Tuple[bool, Optional[str]]:
        """
        Check xem có market crash không

        Returns:
            (is_crash, reason)
        """
        try:
            from data_loader import load_data

            # Load VNINDEX
            vnindex = load_data("VNINDEX", lookback=10, use_cache=False)

            if vnindex.empty or len(vnindex) < 2:
                return False, None

            # Check 1: Daily change
            current = vnindex["close"].iloc[-1]
            prev_close = vnindex["close"].iloc[-2]
            daily_change = current / prev_close - 1

            if daily_change <= self.crash_threshold_daily:
                return True, f"Market crash: VNINDEX {daily_change*100:.2f}% in 1 day"

            # Check 2: Hourly change (if intraday data available)
            # For now, skip this check as we only have daily data

            return False, None

        except Exception as e:
            print(f"⚠️ Lỗi check market crash: {e}")
            return False, None

    def is_emergency_active(self) -> bool:
        """Check xem có emergency active không"""
        self._check_auto_resume()
        return self.events["active_emergency"] is not None

    def trigger_emergency(
        self,
        reason: str,
        event_type: str = "MARKET_CRASH",
        vnindex_change: float = 0.0,
        duration_hours: int = 24,
        auto_resume: bool = True,
    ):
        """
        Trigger emergency stop

        Args:
            reason: Lý do
            event_type: Loại event (MARKET_CRASH, FLASH_CRASH, MANUAL, etc.)
            vnindex_change: % thay đổi VNINDEX
            duration_hours: Thời gian emergency (giờ)
            auto_resume: Có tự động resume không
        """
        event = EmergencyEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            reason=reason,
            vnindex_change=vnindex_change,
            duration_hours=duration_hours,
            auto_resume=auto_resume,
        )

        self.events["active_emergency"] = asdict(event)
        self.events["history"].append(asdict(event))

        self._save_events()

        print(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")

    def resume(self, reason: str = "Manual resume"):
        """Resume trading sau emergency"""
        if not self.events["active_emergency"]:
            return

        event = self.events["active_emergency"]
        event["resume_timestamp"] = datetime.now().isoformat()
        event["resume_reason"] = reason

        # Move to history
        self.events["history"].append(event)
        self.events["active_emergency"] = None

        self._save_events()

        print(f"✅ EMERGENCY STOP RESUMED: {reason}")

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check xem có thể trade không

        Returns:
            (can_trade, reason)
        """
        # Check 1: Active emergency
        if self.is_emergency_active():
            event = EmergencyEvent(**self.events["active_emergency"])
            return False, f"🚨 EMERGENCY STOP: {event.reason}"

        # Check 2: Market crash
        is_crash, crash_reason = self.check_market_crash()
        if is_crash:
            # Auto trigger emergency
            vnindex_change = float(crash_reason.split()[2].replace("%", "")) / 100
            self.trigger_emergency(
                reason=crash_reason,
                event_type="MARKET_CRASH",
                vnindex_change=vnindex_change,
                duration_hours=self.auto_resume_hours,
                auto_resume=True,
            )
            return False, f"🚨 {crash_reason}"

        return True, "✅ OK to trade"

    def get_status_message(self) -> str:
        """Lấy status message"""
        msg = []
        msg.append("🚨 **EMERGENCY STOP STATUS**")
        msg.append("=" * 40)

        if self.is_emergency_active():
            event = EmergencyEvent(**self.events["active_emergency"])
            msg.append(f"⚠️ **EMERGENCY ACTIVE**")
            msg.append(f"Type: {event.event_type}")
            msg.append(f"Reason: {event.reason}")
            msg.append(f"Triggered: {event.timestamp[:16]}")
            msg.append(f"Duration: {event.duration_hours}h")
            msg.append(f"Auto resume: {'Yes' if event.auto_resume else 'No'}")

            # Time remaining
            event_time = datetime.fromisoformat(event.timestamp)
            resume_time = event_time + timedelta(hours=event.duration_hours)
            remaining = resume_time - datetime.now()

            if remaining.total_seconds() > 0:
                hours = int(remaining.total_seconds() / 3600)
                minutes = int((remaining.total_seconds() % 3600) / 60)
                msg.append(f"Time remaining: {hours}h {minutes}m")
        else:
            msg.append("✅ **NO EMERGENCY**")
            msg.append("Trading is allowed")

        # History
        if self.events["history"]:
            msg.append(f"\n📜 Recent events: {len(self.events['history'])}")
            for event in self.events["history"][-3:]:
                msg.append(f"  • {event['timestamp'][:16]}: {event['event_type']}")

        return "\n".join(msg)

    def get_history(self, limit: int = 10) -> list:
        """Lấy lịch sử emergency events"""
        return self.events["history"][-limit:]


# Global instance
_emergency_stop = None


def get_emergency_stop() -> EmergencyStop:
    """Get singleton instance"""
    global _emergency_stop
    if _emergency_stop is None:
        _emergency_stop = EmergencyStop()
    return _emergency_stop


# Test
if __name__ == "__main__":
    print("Testing Emergency Stop...")

    stop = EmergencyStop(crash_threshold_daily=-0.05, auto_resume_hours=24)

    # Test 1: Normal check
    print("\n1️⃣ Test normal check:")
    can_trade, reason = stop.can_trade()
    print(f"Can trade: {can_trade} - {reason}")

    # Test 2: Manual trigger
    print("\n2️⃣ Test manual trigger:")
    stop.trigger_emergency(
        reason="Test emergency", event_type="MANUAL", duration_hours=1, auto_resume=True
    )

    can_trade, reason = stop.can_trade()
    print(f"Can trade: {can_trade} - {reason}")

    # Test 3: Status
    print("\n3️⃣ Status:")
    print(stop.get_status_message())

    # Test 4: Resume
    print("\n4️⃣ Test resume:")
    stop.resume("Test completed")
    can_trade, reason = stop.can_trade()
    print(f"Can trade: {can_trade} - {reason}")

    print("\n✅ Test completed!")
