"""
Circuit Breaker - Giới hạn trades và loss per day
Bảo vệ khỏi lỗi logic hoặc market anomaly
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from threading import RLock
from typing import Dict, Tuple


@dataclass
class DailyStats:
    """Stats trong ngày"""

    date: str
    trades_count: int
    total_loss: float
    total_profit: float
    net_pnl: float
    last_updated: str


class CircuitBreaker:
    """
    Circuit Breaker để bảo vệ khỏi:
    - Trade quá nhiều trong 1 ngày
    - Loss quá nhiều trong 1 ngày
    - Consecutive losses
    - VNINDEX giảm sâu
    """

    def __init__(
        self,
        max_trades_per_day: int = 8,  # TIGHTENED: Max 8 trades/day
        max_loss_per_day_pct: float = 0.03,  # TIGHTENED: 3% max daily loss
        max_consecutive_losses: int = 3,  # TIGHTENED: 3 consecutive losses
        vnindex_drop_threshold: float = -2.5,  # -2.5% VNINDEX drop triggers stop
        # VN market specific: -2.5% is standard, but can be adjusted
        # Use vnindex_drop_threshold_conservative for stricter protection
        vnindex_drop_threshold_conservative: float = -2.0,  # Stricter threshold option
        use_conservative_threshold: bool = True,  # ENABLED: Use conservative mode by default
        total_capital: float = 100_000_000,
        stats_file: str = "circuit_breaker_stats.json",
        max_portfolio_heat: float = 0.60,  # TIGHTENED: Max 60% portfolio exposure
        volatility_multiplier: float = 1.5,  # Adjust thresholds by volatility
        # Gradual response levels - TIGHTENED
        warning_threshold_pct: float = -1.0,  # TIGHTENED: Warning at -1.0%
        caution_threshold_pct: float = -1.5,  # TIGHTENED: Caution at -1.5%
        # Max drawdown protection - TIGHTENED
        max_drawdown_pct: float = 0.12,  # TIGHTENED: 12% max drawdown from peak
        drawdown_warning_pct: float = 0.08,  # TIGHTENED: 8% drawdown warning
        # NEW: Per-session limits
        max_trades_per_session: int = 4,  # NEW: Max 4 trades per session (AM/PM)
        # NEW: Winning streak protection (avoid overconfidence)
        # IMPROVED v5.0: Increased from 5 to 7 - less aggressive pause
        # Rationale: 5 wins is too conservative, 7 wins gives more room while still protecting
        max_consecutive_wins: int = 7,  # Pause after 7 consecutive wins (was 5)
    ):
        self.max_trades_per_day = max_trades_per_day
        self.max_loss_per_day_pct = max_loss_per_day_pct
        self.base_max_loss_per_day_pct = max_loss_per_day_pct  # Store original
        self.max_consecutive_losses = max_consecutive_losses

        # IMPROVED: Configurable VNINDEX thresholds with conservative option
        self.use_conservative_threshold = use_conservative_threshold
        if use_conservative_threshold:
            self.vnindex_drop_threshold = vnindex_drop_threshold_conservative / 100.0
        else:
            self.vnindex_drop_threshold = vnindex_drop_threshold / 100.0
        self.base_vnindex_drop_threshold = self.vnindex_drop_threshold  # Store original

        # Gradual response thresholds - TIGHTENED
        self.warning_threshold = warning_threshold_pct / 100.0
        self.caution_threshold = caution_threshold_pct / 100.0
        self.caution_mode = False  # When True, reduce position sizes by 50%
        self.warning_mode = False  # NEW: Warning mode (reduce by 25%)

        self.total_capital = total_capital
        self.stats_file = stats_file
        self.max_portfolio_heat = max_portfolio_heat
        self.volatility_multiplier = volatility_multiplier

        # Drawdown protection - TIGHTENED
        self.max_drawdown_pct = max_drawdown_pct
        self.drawdown_warning_pct = drawdown_warning_pct
        self.peak_portfolio_value = total_capital  # Track peak value
        self.current_drawdown = 0.0

        # NEW: Per-session limits
        self.max_trades_per_session = max_trades_per_session
        self._session_trades = {"morning": 0, "afternoon": 0}
        self._last_session_date: str = ""  # Track date for session reset

        # NEW: Winning streak protection
        # Rationale: After 5 consecutive wins, traders often become overconfident
        # and take excessive risks. A brief pause helps maintain discipline.
        self.max_consecutive_wins = max_consecutive_wins
        self._consecutive_wins = 0

        self.stats = self._load_stats()
        self._check_new_day()
        self._reset_session_trades_if_new_day()

        # Trạng thái ngắt mạch
        self.tripped = False
        self.tripped_reason = ""

        # OPTIMIZATION: Cache last volatility to avoid redundant recalculations
        self._last_volatility = None

        # CRITICAL FIX: Thread safety
        self._lock = RLock()  # Reentrant lock for nested calls

    def _load_stats(self) -> Dict:
        """Load stats từ file"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Đảm bảo các key cần thiết tồn tại
                    data.setdefault("today", self._get_today_stats())
                    data.setdefault("consecutive_losses", 0)
                    data.setdefault("last_trade_date", None)
                    return data
            except Exception:
                pass

        return {
            "today": self._get_today_stats(),
            "consecutive_losses": 0,
            "last_trade_date": None,
        }

    def _save_stats(self):
        """Save stats vào file"""
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

    def _get_today_stats(self) -> Dict:
        """Tạo stats mới cho ngày hôm nay"""
        return asdict(
            DailyStats(
                date=date.today().isoformat(),
                trades_count=0,
                total_loss=0.0,
                total_profit=0.0,
                net_pnl=0.0,
                last_updated=datetime.now().isoformat(),
            )
        )

    def _check_new_day(self):
        """Check xem có phải ngày mới không, reset stats"""
        today = date.today().isoformat()
        if self.stats.get("today", {}).get("date") != today:
            # New day - reset
            self.stats["today"] = self._get_today_stats()
            # Reset trạng thái ngắt mạch mỗi ngày mới
            self.tripped = False
            self.tripped_reason = ""
            # OPTIMIZATION: Reset volatility cache for fresh calculation
            self._last_volatility = None
            # Reset session trades for new day
            self._reset_session_trades_if_new_day()
            self._save_stats()

    def _reset_session_trades_if_new_day(self):
        """Reset session trade counts if it's a new day."""
        today = date.today().isoformat()
        if self._last_session_date != today:
            self._session_trades = {"morning": 0, "afternoon": 0}
            self._last_session_date = today

    def _get_current_session(self) -> str:
        """
        Get current trading session based on Vietnam market hours.

        Vietnam market sessions:
        - Morning (ATO): 09:00 - 11:30
        - Afternoon (ATC): 13:00 - 14:45

        Returns:
            "morning", "afternoon", or "closed"
        """
        try:
            # Try to use session_trading module if available
            from src.market.session_trading import get_session_manager

            session_mgr = get_session_manager()
            session_info = session_mgr.get_current_session()

            # Handle both dict and object responses
            if hasattr(session_info, "session"):
                session_name = str(session_info.session).lower()
            elif isinstance(session_info, dict):
                session_name = session_info.get("session", "").lower()
            else:
                session_name = str(session_info).lower()

            if "morning" in session_name or "ato" in session_name or "am" in session_name:
                return "morning"
            elif "afternoon" in session_name or "atc" in session_name or "pm" in session_name:
                return "afternoon"
            return "closed"

        except (ImportError, AttributeError, Exception):
            # Fallback: Simple time-based detection
            from datetime import datetime

            try:
                import pytz

                vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
                now = datetime.now(vn_tz)
            except ImportError:
                now = datetime.now()

            hour = now.hour
            minute = now.minute
            current_time = hour * 60 + minute  # Minutes since midnight

            # Morning session: 09:00 - 11:30 (540 - 690 minutes)
            if 540 <= current_time <= 690:
                return "morning"
            # Afternoon session: 13:00 - 14:45 (780 - 885 minutes)
            elif 780 <= current_time <= 885:
                return "afternoon"
            else:
                return "closed"

    def check_session_limit(self) -> tuple:
        """
        Check if current session trade limit has been reached.

        Vietnam market has 2 sessions:
        - Morning: 09:00 - 11:30
        - Afternoon: 13:00 - 14:45

        Limiting trades per session helps:
        - Avoid overtrading during volatile periods
        - Spread risk across sessions
        - Allow time for market analysis between sessions

        Returns:
            (can_trade: bool, message: str)
        """
        self._reset_session_trades_if_new_day()

        current_session = self._get_current_session()

        if current_session == "closed":
            return False, "🚫 Market is closed - no trading allowed"

        session_trades = self._session_trades.get(current_session, 0)

        if session_trades >= self.max_trades_per_session:
            return (
                False,
                f"🚫 Session limit reached: {session_trades}/{self.max_trades_per_session} "
                f"trades in {current_session} session",
            )

        remaining = self.max_trades_per_session - session_trades
        return (
            True,
            f"✅ Session OK: {session_trades}/{self.max_trades_per_session} trades "
            f"({remaining} remaining in {current_session})",
        )

    def record_session_trade(self):
        """Record a trade in the current session."""
        self._reset_session_trades_if_new_day()
        current_session = self._get_current_session()

        if current_session in self._session_trades:
            self._session_trades[current_session] += 1

    def check_winning_streak(self) -> tuple:
        """
        Check if winning streak limit has been reached.

        Rationale for winning streak protection:
        - After 5+ consecutive wins, traders often become overconfident
        - Overconfidence leads to larger position sizes and excessive risk
        - A brief pause (1-2 trades skipped) helps maintain discipline
        - This is a "cooling off" period, not a full stop

        Returns:
            (can_trade: bool, message: str)
        """
        if self._consecutive_wins >= self.max_consecutive_wins:
            return (
                False,
                f"⚠️ Winning streak pause: {self._consecutive_wins} consecutive wins. "
                f"Take a break to avoid overconfidence. "
                f"(Limit: {self.max_consecutive_wins})",
            )

        if self._consecutive_wins >= self.max_consecutive_wins - 1:
            # Warning: approaching limit
            return (
                True,
                f"🟡 Winning streak warning: {self._consecutive_wins} consecutive wins. "
                f"Consider reducing position size.",
            )

        return (
            True,
            f"✅ Winning streak OK: {self._consecutive_wins}/{self.max_consecutive_wins}",
        )

    def reset_winning_streak(self):
        """Reset winning streak counter (called after a loss or manual reset)."""
        self._consecutive_wins = 0

    def get_session_stats(self) -> dict:
        """Get current session trading statistics."""
        self._reset_session_trades_if_new_day()
        current_session = self._get_current_session()

        return {
            "current_session": current_session,
            "morning_trades": self._session_trades.get("morning", 0),
            "afternoon_trades": self._session_trades.get("afternoon", 0),
            "max_per_session": self.max_trades_per_session,
            "session_limit_reached": (
                self._session_trades.get(current_session, 0) >= self.max_trades_per_session
                if current_session != "closed"
                else False
            ),
            "consecutive_wins": self._consecutive_wins,
            "max_consecutive_wins": self.max_consecutive_wins,
            "winning_streak_pause": self._consecutive_wins >= self.max_consecutive_wins,
        }

    def check_and_update(
        self,
        portfolio_pnl_pct: float,
        vnindex_change_pct: float,
        portfolio_heat: float = 0.0,
        market_volatility: float = 0.0,
    ) -> bool:
        """
        ENHANCED: Kiểm tra các điều kiện ngắt mạch với volatility adjustments

        CRITICAL FIX: Thread-safe with RLock to prevent race conditions.

        Args:
            portfolio_pnl_pct (float): P&L hiện tại của portfolio trong ngày (dạng float, vd: -0.01 cho -1%).
            vnindex_change_pct (float): % thay đổi của VNINDEX trong ngày.
            portfolio_heat (float): Portfolio exposure (0-1), 1 = fully invested
            market_volatility (float): Market volatility (ATR/Price ratio)

        Returns:
            bool: True nếu ngắt mạch được kích hoạt, False nếu không.
        """
        # CRITICAL FIX: Thread-safe check
        with self._lock:
            if self.tripped:
                return True  # Nếu đã ngắt thì không cần check lại

            # Validate input parameters
            if not isinstance(portfolio_pnl_pct, (int, float)):
                raise ValueError(
                    f"portfolio_pnl_pct phải là số, nhận được: {type(portfolio_pnl_pct)}"
                )
            if not isinstance(vnindex_change_pct, (int, float)):
                raise ValueError(
                    f"vnindex_change_pct phải là số, nhận được: {type(vnindex_change_pct)}"
                )

            # ENHANCEMENT: Adjust thresholds based on volatility
            self._adjust_thresholds_for_volatility(market_volatility)

            # Check 1: Max loss per day
            if portfolio_pnl_pct < 0 and abs(portfolio_pnl_pct) >= self.max_loss_per_day_pct:
                self.tripped = True
                self.tripped_reason = (
                    f"Lỗ trong ngày ({portfolio_pnl_pct:.2%}) "
                    f"vượt ngưỡng cho phép ({self.max_loss_per_day_pct:.2%})."
                )
                self._save_stats()
                return True

            # Check 2: VNINDEX giảm sâu - với gradual response
            # Level 1: Warning (-1.5%) - Log warning only
            if (
                vnindex_change_pct <= self.warning_threshold
                and vnindex_change_pct > self.caution_threshold
            ):
                print(f"⚠️ VNINDEX Warning: {vnindex_change_pct:.2%} - Monitoring closely")
                self.caution_mode = False  # Not in caution yet

            # Level 2: Caution (-2.0%) - Reduce position sizes
            elif (
                vnindex_change_pct <= self.caution_threshold
                and vnindex_change_pct > self.vnindex_drop_threshold
            ):
                if not self.caution_mode:
                    self.caution_mode = True
                    print(
                        f"🟡 VNINDEX Caution Mode: {vnindex_change_pct:.2%} - Reducing position sizes by 50%"
                    )

            # Above warning threshold - normal mode
            elif vnindex_change_pct > self.warning_threshold:
                self.caution_mode = False

            # Level 3: Circuit breaker trip (-2.5% default)
            if vnindex_change_pct < self.vnindex_drop_threshold:
                self.tripped = True
                self.tripped_reason = (
                    f"VNINDEX giảm sâu ({vnindex_change_pct:.2%}) "
                    f"vượt ngưỡng ({self.vnindex_drop_threshold:.2%})."
                )
                self._save_stats()
                return True

            # Check 3: Max trades per day
            if self.stats["today"]["trades_count"] >= self.max_trades_per_day:
                self.tripped = True
                self.tripped_reason = (
                    f"Số lệnh trong ngày ({self.stats['today']['trades_count']}) đạt giới hạn."
                )
                self._save_stats()
                return True

            # Check 4: Consecutive losses
            if self.stats["consecutive_losses"] >= self.max_consecutive_losses:
                self.tripped = True
                self.tripped_reason = (
                    f"Số lệnh thua liên tiếp ({self.stats['consecutive_losses']}) đạt giới hạn."
                )
                self._save_stats()
                return True

            # NEW Check 5: Per-session trade limit
            session_ok, session_msg = self.check_session_limit()
            if not session_ok and "limit reached" in session_msg:
                # Don't trip circuit breaker, just block this session
                # Trading can resume in next session
                print(f"⚠️ {session_msg}")
                # Note: We don't set self.tripped here as it's session-specific

            # NEW Check 6: Winning streak protection
            streak_ok, streak_msg = self.check_winning_streak()
            if not streak_ok:
                # Soft block - warn but don't trip circuit breaker
                print(f"⚠️ {streak_msg}")
                # Note: This is advisory, not a hard stop

            # Check 7: Portfolio heat (overexposure)
            if portfolio_heat > self.max_portfolio_heat:
                self.tripped = True
                self.tripped_reason = (
                    f"Portfolio heat quá cao ({portfolio_heat:.1%}) "
                    f"vượt ngưỡng ({self.max_portfolio_heat:.1%})."
                )
                self._save_stats()
                return True

            return False

    def _adjust_thresholds_for_volatility(self, market_volatility: float):
        """
        ENHANCEMENT: Adjust circuit breaker thresholds based on market volatility

        Logic:
        - High volatility (>3%): Tighten thresholds (more protective)
        - Normal volatility (1-3%): Use base thresholds
        - Low volatility (<1%): Relax thresholds slightly

        Args:
            market_volatility: Market volatility ratio (e.g., 0.02 = 2%)
        """
        # OPTIMIZATION: Skip recalculation if volatility hasn't changed
        if (
            self._last_volatility is not None
            and abs(market_volatility - self._last_volatility) < 0.0001
        ):
            return  # No change, skip recalculation

        self._last_volatility = market_volatility

        if market_volatility == 0.0:
            # No volatility data - use base thresholds
            self.max_loss_per_day_pct = self.base_max_loss_per_day_pct
            self.vnindex_drop_threshold = self.base_vnindex_drop_threshold
            return

        # Convert to percentage for easier comparison
        vol_pct = market_volatility * 100

        if vol_pct > 3.0:
            # High volatility - tighten limits
            volatility_factor = 0.75  # 25% tighter
            print(
                f"⚠️ High volatility detected ({vol_pct:.1f}%). "
                "Tightening circuit breaker thresholds by 25%"
            )
        elif vol_pct > 2.0:
            # Medium-high volatility - slightly tighter
            volatility_factor = 0.90  # 10% tighter
            print(
                f"📊 Medium-high volatility ({vol_pct:.1f}%). "
                "Tightening circuit breaker thresholds by 10%"
            )
        elif vol_pct < 1.0:
            # Low volatility - can relax slightly
            volatility_factor = 1.10  # 10% looser
            print(
                f"📉 Low volatility ({vol_pct:.1f}%). " "Relaxing circuit breaker thresholds by 10%"
            )
        else:
            # Normal volatility - use base
            volatility_factor = 1.0

        # Apply adjustments
        self.max_loss_per_day_pct = self.base_max_loss_per_day_pct * volatility_factor
        self.vnindex_drop_threshold = self.base_vnindex_drop_threshold * volatility_factor

        print(
            f"🔧 Circuit breaker adjusted: "
            f"max_loss={self.max_loss_per_day_pct:.2%}, "
            f"vnindex_threshold={self.vnindex_drop_threshold:.2%}"
        )

    def is_active(self) -> bool:
        """
        Kiểm tra xem circuit breaker có đang kích hoạt không.

        Returns:
            bool: True nếu circuit breaker đang kích hoạt, False nếu không.
        """
        return self.tripped

    def is_caution_mode(self) -> bool:
        """
        Check if caution mode is active (VNINDEX between -2% and -2.5%).
        When True, position sizes should be reduced by 50%.

        Returns:
            bool: True if caution mode active
        """
        return self.caution_mode

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on current market conditions.

        Returns:
            float: 1.0 for normal, 0.5 for caution mode, 0.0 if tripped
        """
        if self.tripped:
            return 0.0
        if self.caution_mode:
            return 0.5
        return 1.0

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check xem có thể vào lệnh mới không.

        IMPROVED v4.2: Includes session limits and winning streak checks.

        Returns:
            (can_trade: bool, reason: str)
        """
        if self.tripped:
            return False, self.tripped_reason

        today_stats = self.stats["today"]

        # Check 1: Max trades per day
        if today_stats["trades_count"] >= self.max_trades_per_day:
            return False, f"🚫 Max trades per day reached ({self.max_trades_per_day})"

        # Check 2: Consecutive losses
        if self.stats["consecutive_losses"] >= self.max_consecutive_losses:
            return (
                False,
                f"🚫 Too many consecutive losses ({self.stats['consecutive_losses']})",
            )

        # Check 3: Per-session limit
        session_ok, session_msg = self.check_session_limit()
        if not session_ok:
            return False, session_msg

        # Check 4: Winning streak (soft check - warning only)
        streak_ok, streak_msg = self.check_winning_streak()
        if not streak_ok:
            # This is a soft block - return warning but allow override
            return False, streak_msg

        return True, "✅ OK to trade"

    def record_trade(self, pnl: float):
        """
        Record một trade

        CRITICAL FIX: Thread-safe with RLock.
        IMPROVED v4.2: Tracks session trades and winning streak.

        Args:
            pnl: Profit/Loss (positive = profit, negative = loss)
        """
        with self._lock:
            self._check_new_day()

            today_stats = self.stats["today"]

            # Update counts
            today_stats["trades_count"] += 1
            today_stats["last_updated"] = datetime.now().isoformat()

            # NEW: Record session trade
            self.record_session_trade()

            # Update P&L and streak tracking
            if pnl > 0:
                today_stats["total_profit"] += pnl
                self.stats["consecutive_losses"] = 0  # Reset loss streak
                # NEW: Track winning streak
                self._consecutive_wins += 1
                if self._consecutive_wins >= self.max_consecutive_wins:
                    print(
                        f"🏆 Winning streak: {self._consecutive_wins} consecutive wins! "
                        f"Consider taking a break to avoid overconfidence."
                    )
            else:
                today_stats["total_loss"] += abs(pnl)
                self.stats["consecutive_losses"] += 1
                # NEW: Reset winning streak on loss
                if self._consecutive_wins > 0:
                    print(
                        f"📉 Winning streak ended at {self._consecutive_wins} wins. "
                        f"Resetting counter."
                    )
                self._consecutive_wins = 0

            today_stats["net_pnl"] = today_stats["total_profit"] - today_stats["total_loss"]

            # Update last trade date
            self.stats["last_trade_date"] = date.today().isoformat()

            self._save_stats()

    def record_pnl(self, portfolio_pnl_pct: float):
        """
        Ghi nhận PnL hiện tại của portfolio ngay lập tức.
        Được gọi sau khi thoát lệnh để cập nhật trạng thái circuit breaker.

        CRITICAL FIX: Thread-safe with RLock to prevent race conditions.

        Args:
            portfolio_pnl_pct (float): P&L hiện tại của portfolio (dạng float, vd: -0.01 cho -1%)
        """
        with self._lock:
            self._check_new_day()

            # Lưu PnL vào stats để tracking
            self.stats["today"]["last_updated"] = datetime.now().isoformat()

            # Kiểm tra ngay xem có cần kích hoạt circuit breaker không
            if portfolio_pnl_pct < 0 and abs(portfolio_pnl_pct) >= self.max_loss_per_day_pct:
                self.tripped = True
                self.tripped_reason = (
                    f"Lỗ trong ngày ({portfolio_pnl_pct:.2%}) "
                    f"vượt ngưỡng cho phép ({self.max_loss_per_day_pct:.2%})."
                )

            self._save_stats()

    def get_daily_stats(self) -> DailyStats:
        """Lấy stats của ngày hôm nay"""
        self._check_new_day()
        return DailyStats(**self.stats["today"])

    def get_status_message(self) -> str:
        """Lấy status message"""
        self._check_new_day()

        stats = self.get_daily_stats()
        session_stats = self.get_session_stats()

        msg = []
        msg.append("🔒 **CIRCUIT BREAKER STATUS**")
        msg.append("=" * 40)
        msg.append(f"📅 Date: {stats.date}")
        msg.append(f"🔄 Trades today: {stats.trades_count}/{self.max_trades_per_day}")
        msg.append(f"📉 Total loss: {stats.total_loss:,.0f} VNĐ")
        msg.append(f"📈 Total profit: {stats.total_profit:,.0f} VNĐ")
        msg.append(f"💰 Net P&L: {stats.net_pnl:+,.0f} VNĐ")
        msg.append(
            f"⚠️ Consecutive losses: {self.stats.get('consecutive_losses', 0)}/{self.max_consecutive_losses}"
        )
        msg.append("")

        # NEW: Session info
        msg.append("📊 **SESSION STATS**")
        msg.append(f"   Current session: {session_stats['current_session']}")
        msg.append(
            f"   Morning trades: {session_stats['morning_trades']}/{self.max_trades_per_session}"
        )
        msg.append(
            f"   Afternoon trades: {session_stats['afternoon_trades']}/{self.max_trades_per_session}"
        )
        msg.append("")

        # NEW: Winning streak info
        msg.append("🏆 **WINNING STREAK**")
        msg.append(
            f"   Consecutive wins: {session_stats['consecutive_wins']}/{self.max_consecutive_wins}"
        )
        if session_stats["winning_streak_pause"]:
            msg.append("   ⚠️ PAUSE RECOMMENDED - Take a break to avoid overconfidence")
        msg.append("")

        msg.append(f"Status: {'TRIPPED - ' + self.tripped_reason if self.tripped else 'OK'}")

        return "\n".join(msg)

    def reset(self):
        """Reset toàn bộ trạng thái (cho testing hoặc manual reset)"""
        self.stats["today"] = self._get_today_stats()
        self.stats["consecutive_losses"] = 0
        self.tripped = False
        self.tripped_reason = ""
        self._save_stats()
        print("Circuit breaker has been reset.")

    # ========================================================================
    # IMPROVEMENT #9: Drawdown Protection
    # ========================================================================

    def update_portfolio_value(self, current_value: float) -> Dict:
        """
        IMPROVEMENT #9: Update portfolio value and check drawdown

        Tracks peak portfolio value and calculates current drawdown.
        Triggers circuit breaker if drawdown exceeds max threshold.

        Args:
            current_value: Current portfolio value

        Returns:
            Dict with drawdown analysis
        """
        with self._lock:
            # Update peak if new high
            if current_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_value
                self.current_drawdown = 0.0
                return {
                    "new_peak": True,
                    "peak_value": self.peak_portfolio_value,
                    "current_drawdown": 0.0,
                    "drawdown_warning": False,
                    "drawdown_critical": False,
                }

            # Calculate drawdown from peak
            if self.peak_portfolio_value > 0:
                self.current_drawdown = (
                    self.peak_portfolio_value - current_value
                ) / self.peak_portfolio_value
            else:
                self.current_drawdown = 0.0

            # Check warning level
            drawdown_warning = self.current_drawdown >= self.drawdown_warning_pct
            drawdown_critical = self.current_drawdown >= self.max_drawdown_pct

            # Trip circuit breaker if max drawdown exceeded
            if drawdown_critical and not self.tripped:
                self.tripped = True
                self.tripped_reason = (
                    f"🚨 Max drawdown exceeded: {self.current_drawdown:.1%} "
                    f"(limit: {self.max_drawdown_pct:.1%}). "
                    f"Peak: {self.peak_portfolio_value:,.0f}, "
                    f"Current: {current_value:,.0f}"
                )
                self._save_stats()
                print(f"🚨 CIRCUIT BREAKER TRIPPED: {self.tripped_reason}")

            return {
                "new_peak": False,
                "peak_value": self.peak_portfolio_value,
                "current_value": current_value,
                "current_drawdown": self.current_drawdown,
                "drawdown_pct": self.current_drawdown * 100,
                "drawdown_warning": drawdown_warning,
                "drawdown_critical": drawdown_critical,
                "tripped": self.tripped,
            }

    def check_drawdown(self, current_value: float) -> tuple:
        """
        IMPROVEMENT #9: Check if drawdown is within acceptable limits

        Args:
            current_value: Current portfolio value

        Returns:
            (is_ok, warning_message)
        """
        result = self.update_portfolio_value(current_value)

        if result["drawdown_critical"]:
            return (
                False,
                f"🚨 CRITICAL: Drawdown {result['drawdown_pct']:.1f}% exceeds max {self.max_drawdown_pct*100:.1f}%",
            )

        if result["drawdown_warning"]:
            return (
                True,
                f"⚠️ WARNING: Drawdown {result['drawdown_pct']:.1f}% approaching limit "
                f"({self.max_drawdown_pct*100:.1f}%)",
            )

        return (True, None)

    def get_drawdown_status(self) -> Dict:
        """
        Get current drawdown status

        Returns:
            Dict with drawdown metrics
        """
        return {
            "peak_value": self.peak_portfolio_value,
            "current_drawdown_pct": self.current_drawdown * 100,
            "max_drawdown_pct": self.max_drawdown_pct * 100,
            "warning_threshold_pct": self.drawdown_warning_pct * 100,
            "is_warning": self.current_drawdown >= self.drawdown_warning_pct,
            "is_critical": self.current_drawdown >= self.max_drawdown_pct,
        }

    def reset_peak(self, new_peak: float = None):
        """
        Reset peak value (e.g., after capital injection or manual reset)

        Args:
            new_peak: New peak value (default: current total_capital)
        """
        with self._lock:
            self.peak_portfolio_value = new_peak or self.total_capital
            self.current_drawdown = 0.0
            print(f"📊 Peak portfolio value reset to {self.peak_portfolio_value:,.0f}")


# Global instance
_circuit_breaker = None


def get_circuit_breaker(total_capital: float = 100_000_000) -> CircuitBreaker:
    """Get singleton instance"""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker(total_capital=total_capital)
    return _circuit_breaker


# Test
if __name__ == "__main__":
    print("Testing Circuit Breaker...")

    breaker = CircuitBreaker(
        max_trades_per_day=5,
        max_loss_per_day_pct=0.05,
        max_consecutive_losses=3,
        total_capital=100_000_000,
    )

    # Test 1: Normal trades
    print("\n1️⃣ Test normal trades:")
    for i in range(3):
        can_trade, reason = breaker.can_trade()
        print(f"Trade {i+1}: {reason}")
        if can_trade:
            # Simulate profit
            breaker.record_trade(1_000_000)

    # Test 2: Consecutive losses
    print("\n2️⃣ Test consecutive losses:")
    for i in range(4):
        can_trade, reason = breaker.can_trade()
        print(f"Loss {i+1}: {reason}")
        if can_trade:
            # Simulate loss
            breaker.record_trade(-500_000)

    # Test 3: Status
    print("\n3️⃣ Status:")
    print(breaker.get_status_message())

    print("\n✅ Test completed!")
