# -*- coding: utf-8 -*-
"""
Margin Trading Manager for Vietnam Stock Market

Complete margin trading implementation with:
- Margin call simulation and monitoring
- Maintenance margin tracking
- Force liquidation logic
- Margin utilization optimization
- Real-time margin health monitoring

Vietnam Margin Rules (2024):
- Initial Margin: 50% (can borrow up to 50% of position value)
- Maintenance Margin: 30-35% (broker dependent)
- Margin Call: Triggered when equity < maintenance margin
- Force Liquidation: When equity < 25-30%
- T+0 allowed for margin accounts

Author: Trading Bot Team
Version: 2.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from threading import RLock
import json
import os

logger = logging.getLogger(__name__)


class MarginStatus(Enum):
    """Margin account status levels"""
    HEALTHY = "HEALTHY"           # Equity ratio > 50%
    NORMAL = "NORMAL"             # Equity ratio 40-50%
    WARNING = "WARNING"           # Equity ratio 35-40%
    MARGIN_CALL = "MARGIN_CALL"   # Equity ratio 30-35%
    FORCE_LIQUIDATION = "FORCE_LIQUIDATION"  # Equity ratio < 30%


class MarginCallType(Enum):
    """Types of margin calls"""
    DEPOSIT_CASH = "DEPOSIT_CASH"
    DEPOSIT_SECURITIES = "DEPOSIT_SECURITIES"
    REDUCE_POSITION = "REDUCE_POSITION"
    FORCE_SELL = "FORCE_SELL"


@dataclass
class MarginPosition:
    """Individual margin position"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    margin_ratio: float  # How much was borrowed (0.5 = 50% borrowed)
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost
    
    @property
    def borrowed_amount(self) -> float:
        return self.cost_basis * self.margin_ratio
    
    @property
    def equity_value(self) -> float:
        return self.market_value - self.borrowed_amount
    
    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis
    
    @property
    def equity_ratio(self) -> float:
        """Equity as percentage of market value"""
        if self.market_value <= 0:
            return 0.0
        return self.equity_value / self.market_value


@dataclass
class MarginCallEvent:
    """Margin call event record"""
    timestamp: datetime
    call_type: MarginCallType
    equity_ratio: float
    required_action: str
    amount_required: float
    deadline: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_action: Optional[str] = None


@dataclass
class MarginAccountState:
    """Complete margin account state"""
    # Account values
    cash_balance: float
    total_market_value: float
    total_borrowed: float
    total_equity: float
    
    # Ratios
    equity_ratio: float
    margin_utilization: float
    buying_power: float
    
    # Status
    status: MarginStatus
    margin_call_active: bool
    
    # Limits
    initial_margin_req: float
    maintenance_margin_req: float
    
    # Positions
    positions: List[MarginPosition] = field(default_factory=list)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return self.status in [MarginStatus.HEALTHY, MarginStatus.NORMAL]


class MarginManager:
    """
    Complete Margin Trading Manager for Vietnam Market
    
    Features:
    - Real-time margin monitoring
    - Margin call simulation
    - Force liquidation logic
    - Position-level margin tracking
    - Margin optimization suggestions
    
    Vietnam Margin Thresholds:
    - Initial Margin: 50% (borrow up to 50%)
    - Maintenance Margin: 35%
    - Margin Call: 30%
    - Force Liquidation: 25%
    """
    
    # Vietnam margin thresholds
    INITIAL_MARGIN = 0.50        # 50% initial margin requirement
    MAINTENANCE_MARGIN = 0.35   # 35% maintenance margin
    MARGIN_CALL_LEVEL = 0.30    # 30% triggers margin call
    FORCE_LIQUIDATION = 0.25    # 25% triggers force liquidation
    
    # Interest rates (annual)
    MARGIN_INTEREST_RATE = 0.12  # 12% annual interest on borrowed amount
    
    # Time limits
    MARGIN_CALL_DEADLINE_HOURS = 24  # 24 hours to meet margin call
    
    def __init__(
        self,
        initial_cash: float = 100_000_000,
        margin_limit: float = 200_000_000,
        initial_margin: float = INITIAL_MARGIN,
        maintenance_margin: float = MAINTENANCE_MARGIN,
        margin_call_level: float = MARGIN_CALL_LEVEL,
        force_liquidation_level: float = FORCE_LIQUIDATION,
        interest_rate: float = MARGIN_INTEREST_RATE,
        state_file: str = "margin_state.json",
    ):
        """
        Initialize Margin Manager.
        
        Args:
            initial_cash: Starting cash balance
            margin_limit: Maximum borrowing limit
            initial_margin: Initial margin requirement (default 50%)
            maintenance_margin: Maintenance margin requirement (default 35%)
            margin_call_level: Margin call trigger level (default 30%)
            force_liquidation_level: Force liquidation level (default 25%)
            interest_rate: Annual interest rate on borrowed amount
            state_file: File to persist margin state
        """
        self.cash_balance = initial_cash
        self.margin_limit = margin_limit
        self.initial_margin = initial_margin
        self.maintenance_margin = maintenance_margin
        self.margin_call_level = margin_call_level
        self.force_liquidation_level = force_liquidation_level
        self.interest_rate = interest_rate
        self.state_file = state_file
        
        # Position tracking
        self._positions: Dict[str, MarginPosition] = {}
        self._borrowed_amount: float = 0.0
        
        # Margin call tracking
        self._margin_calls: List[MarginCallEvent] = []
        self._active_margin_call: Optional[MarginCallEvent] = None
        
        # Interest tracking
        self._last_interest_date: date = date.today()
        self._accrued_interest: float = 0.0
        
        # Thread safety
        self._lock = RLock()
        
        # Load persisted state
        self._load_state()
        
        logger.info(
            f"✅ MarginManager initialized: "
            f"cash={initial_cash:,.0f}, limit={margin_limit:,.0f}, "
            f"initial={initial_margin:.0%}, maintenance={maintenance_margin:.0%}"
        )

    def _load_state(self):
        """Load persisted margin state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.cash_balance = data.get('cash_balance', self.cash_balance)
                    self._borrowed_amount = data.get('borrowed_amount', 0)
                    self._accrued_interest = data.get('accrued_interest', 0)
                    logger.info(f"📂 Loaded margin state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load margin state: {e}")
    
    def _save_state(self):
        """Persist margin state."""
        try:
            data = {
                'cash_balance': self.cash_balance,
                'borrowed_amount': self._borrowed_amount,
                'accrued_interest': self._accrued_interest,
                'last_updated': datetime.now().isoformat(),
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save margin state: {e}")
    
    # =========================================================================
    # CORE MARGIN CALCULATIONS
    # =========================================================================
    
    def get_account_state(self) -> MarginAccountState:
        """
        Get complete margin account state.
        
        Returns:
            MarginAccountState with all account metrics
        """
        with self._lock:
            # Calculate totals
            total_market_value = sum(p.market_value for p in self._positions.values())
            total_borrowed = self._borrowed_amount + self._accrued_interest
            total_equity = self.cash_balance + total_market_value - total_borrowed
            
            # Calculate ratios
            if total_market_value > 0:
                equity_ratio = total_equity / total_market_value
            else:
                equity_ratio = 1.0 if self.cash_balance > 0 else 0.0
            
            margin_utilization = total_borrowed / self.margin_limit if self.margin_limit > 0 else 0
            
            # Calculate buying power
            # Buying power = (Equity - Maintenance Requirement) / Initial Margin
            maintenance_req = total_market_value * self.maintenance_margin
            excess_equity = total_equity - maintenance_req
            buying_power = max(0, excess_equity / self.initial_margin)
            
            # Determine status
            status = self._determine_status(equity_ratio)
            
            # Check for active margin call
            margin_call_active = self._active_margin_call is not None
            
            # Generate warnings
            warnings = self._generate_warnings(equity_ratio, margin_utilization)
            
            return MarginAccountState(
                cash_balance=self.cash_balance,
                total_market_value=total_market_value,
                total_borrowed=total_borrowed,
                total_equity=total_equity,
                equity_ratio=equity_ratio,
                margin_utilization=margin_utilization,
                buying_power=buying_power,
                status=status,
                margin_call_active=margin_call_active,
                initial_margin_req=self.initial_margin,
                maintenance_margin_req=self.maintenance_margin,
                positions=list(self._positions.values()),
                warnings=warnings,
            )
    
    def _determine_status(self, equity_ratio: float) -> MarginStatus:
        """Determine margin status from equity ratio."""
        if equity_ratio >= 0.50:
            return MarginStatus.HEALTHY
        elif equity_ratio >= 0.40:
            return MarginStatus.NORMAL
        elif equity_ratio >= self.maintenance_margin:
            return MarginStatus.WARNING
        elif equity_ratio >= self.force_liquidation_level:
            return MarginStatus.MARGIN_CALL
        else:
            return MarginStatus.FORCE_LIQUIDATION
    
    def _generate_warnings(self, equity_ratio: float, margin_utilization: float) -> List[str]:
        """Generate warnings based on account state."""
        warnings = []
        
        if equity_ratio < self.force_liquidation_level:
            warnings.append(f"🚨 CRITICAL: Force liquidation imminent! Equity: {equity_ratio:.1%}")
        elif equity_ratio < self.margin_call_level:
            warnings.append(f"⚠️ MARGIN CALL: Deposit funds or reduce positions! Equity: {equity_ratio:.1%}")
        elif equity_ratio < self.maintenance_margin:
            warnings.append(f"⚠️ WARNING: Approaching margin call. Equity: {equity_ratio:.1%}")
        
        if margin_utilization > 0.90:
            warnings.append(f"⚠️ High margin utilization: {margin_utilization:.1%}")
        
        if self._accrued_interest > self.cash_balance * 0.1:
            warnings.append(f"⚠️ High accrued interest: {self._accrued_interest:,.0f} VND")
        
        return warnings
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def can_open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        use_margin: bool = True,
    ) -> Tuple[bool, str, Dict]:
        """
        Check if a new position can be opened.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Entry price
            use_margin: Whether to use margin
            
        Returns:
            (can_open, reason, details)
        """
        with self._lock:
            position_value = quantity * price
            
            if use_margin:
                # Calculate required cash (initial margin)
                required_cash = position_value * self.initial_margin
                borrow_amount = position_value - required_cash
                
                # Check cash availability
                if required_cash > self.cash_balance:
                    return False, f"Insufficient cash. Need: {required_cash:,.0f}, Have: {self.cash_balance:,.0f}", {}
                
                # Check margin limit
                new_borrowed = self._borrowed_amount + borrow_amount
                if new_borrowed > self.margin_limit:
                    return False, f"Exceeds margin limit. Need: {new_borrowed:,.0f}, Limit: {self.margin_limit:,.0f}", {}
                
                # Check post-trade equity ratio
                state = self.get_account_state()
                new_market_value = state.total_market_value + position_value
                new_equity = state.total_equity - required_cash + position_value - borrow_amount
                new_equity_ratio = new_equity / new_market_value if new_market_value > 0 else 0
                
                if new_equity_ratio < self.maintenance_margin:
                    return False, f"Would breach maintenance margin. Post-trade equity: {new_equity_ratio:.1%}", {}
                
                return True, "OK", {
                    "required_cash": required_cash,
                    "borrow_amount": borrow_amount,
                    "post_trade_equity_ratio": new_equity_ratio,
                }
            else:
                # Cash only
                if position_value > self.cash_balance:
                    return False, f"Insufficient cash. Need: {position_value:,.0f}, Have: {self.cash_balance:,.0f}", {}
                return True, "OK", {"required_cash": position_value, "borrow_amount": 0}
    
    def open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        use_margin: bool = True,
    ) -> Tuple[bool, str]:
        """
        Open a new margin position.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Entry price
            use_margin: Whether to use margin
            
        Returns:
            (success, message)
        """
        with self._lock:
            can_open, reason, details = self.can_open_position(symbol, quantity, price, use_margin)
            
            if not can_open:
                return False, reason
            
            position_value = quantity * price
            required_cash = details["required_cash"]
            borrow_amount = details["borrow_amount"]
            
            # Update cash and borrowed amount
            self.cash_balance -= required_cash
            self._borrowed_amount += borrow_amount
            
            # Create or update position
            margin_ratio = borrow_amount / position_value if position_value > 0 else 0
            
            if symbol in self._positions:
                # Average into existing position
                existing = self._positions[symbol]
                total_cost = existing.cost_basis + position_value
                total_qty = existing.quantity + quantity
                new_avg_cost = total_cost / total_qty
                new_margin_ratio = (existing.borrowed_amount + borrow_amount) / (existing.market_value + position_value)
                
                self._positions[symbol] = MarginPosition(
                    symbol=symbol,
                    quantity=total_qty,
                    avg_cost=new_avg_cost,
                    current_price=price,
                    margin_ratio=new_margin_ratio,
                )
            else:
                self._positions[symbol] = MarginPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=price,
                    current_price=price,
                    margin_ratio=margin_ratio,
                )
            
            self._save_state()
            
            logger.info(
                f"📈 Opened margin position: {symbol} {quantity} @ {price:,.0f} "
                f"(margin: {margin_ratio:.0%}, borrowed: {borrow_amount:,.0f})"
            )
            
            return True, f"Position opened. Borrowed: {borrow_amount:,.0f} VND"
    
    def close_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
    ) -> Tuple[bool, str, float]:
        """
        Close a margin position.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to close
            price: Exit price
            
        Returns:
            (success, message, realized_pnl)
        """
        with self._lock:
            if symbol not in self._positions:
                return False, f"No position found for {symbol}", 0.0
            
            position = self._positions[symbol]
            
            if quantity > position.quantity:
                return False, f"Insufficient shares. Have: {position.quantity}, Want: {quantity}", 0.0
            
            # Calculate P&L
            close_value = quantity * price
            cost_basis = quantity * position.avg_cost
            realized_pnl = close_value - cost_basis
            
            # Calculate borrowed amount to repay (proportional)
            borrowed_to_repay = position.borrowed_amount * (quantity / position.quantity)
            
            # Update cash (receive sale proceeds, repay borrowed)
            self.cash_balance += close_value - borrowed_to_repay
            self._borrowed_amount -= borrowed_to_repay
            
            # Update or remove position
            if quantity == position.quantity:
                del self._positions[symbol]
            else:
                remaining_qty = position.quantity - quantity
                remaining_borrowed = position.borrowed_amount - borrowed_to_repay
                new_margin_ratio = remaining_borrowed / (remaining_qty * position.avg_cost) if remaining_qty > 0 else 0
                
                self._positions[symbol] = MarginPosition(
                    symbol=symbol,
                    quantity=remaining_qty,
                    avg_cost=position.avg_cost,
                    current_price=price,
                    margin_ratio=new_margin_ratio,
                )
            
            self._save_state()
            
            logger.info(
                f"📉 Closed margin position: {symbol} {quantity} @ {price:,.0f} "
                f"(P&L: {realized_pnl:+,.0f}, repaid: {borrowed_to_repay:,.0f})"
            )
            
            return True, f"Position closed. P&L: {realized_pnl:+,.0f} VND", realized_pnl

    # =========================================================================
    # MARGIN CALL SIMULATION & HANDLING
    # =========================================================================
    
    def simulate_price_change(
        self,
        price_changes: Dict[str, float],
    ) -> Tuple[MarginAccountState, List[str]]:
        """
        Simulate account state after price changes.
        
        Args:
            price_changes: Dict of {symbol: new_price}
            
        Returns:
            (simulated_state, actions_required)
        """
        with self._lock:
            # Create temporary positions with new prices
            temp_positions = {}
            for symbol, pos in self._positions.items():
                new_price = price_changes.get(symbol, pos.current_price)
                temp_positions[symbol] = MarginPosition(
                    symbol=symbol,
                    quantity=pos.quantity,
                    avg_cost=pos.avg_cost,
                    current_price=new_price,
                    margin_ratio=pos.margin_ratio,
                )
            
            # Calculate new totals
            total_market_value = sum(p.market_value for p in temp_positions.values())
            total_borrowed = self._borrowed_amount + self._accrued_interest
            total_equity = self.cash_balance + total_market_value - total_borrowed
            
            equity_ratio = total_equity / total_market_value if total_market_value > 0 else 1.0
            status = self._determine_status(equity_ratio)
            
            # Determine required actions
            actions = []
            
            if status == MarginStatus.FORCE_LIQUIDATION:
                # Calculate how much to liquidate
                target_equity_ratio = self.margin_call_level + 0.05  # Target 35%
                required_equity = total_market_value * target_equity_ratio
                equity_shortfall = required_equity - total_equity
                
                actions.append(f"🚨 FORCE LIQUIDATION: Sell positions worth {equity_shortfall:,.0f} VND")
                actions.append(self._suggest_liquidation_order(temp_positions, equity_shortfall))
                
            elif status == MarginStatus.MARGIN_CALL:
                # Calculate deposit required
                target_equity_ratio = self.maintenance_margin + 0.05  # Target 40%
                required_equity = total_market_value * target_equity_ratio
                deposit_required = required_equity - total_equity
                
                actions.append(f"⚠️ MARGIN CALL: Deposit {deposit_required:,.0f} VND or reduce positions")
                actions.append(f"   Deadline: {self.MARGIN_CALL_DEADLINE_HOURS} hours")
                
            elif status == MarginStatus.WARNING:
                buffer = (equity_ratio - self.margin_call_level) * total_market_value
                actions.append(f"⚠️ WARNING: {buffer:,.0f} VND buffer before margin call")
            
            # Create simulated state
            simulated_state = MarginAccountState(
                cash_balance=self.cash_balance,
                total_market_value=total_market_value,
                total_borrowed=total_borrowed,
                total_equity=total_equity,
                equity_ratio=equity_ratio,
                margin_utilization=total_borrowed / self.margin_limit if self.margin_limit > 0 else 0,
                buying_power=0,  # Not relevant for simulation
                status=status,
                margin_call_active=status in [MarginStatus.MARGIN_CALL, MarginStatus.FORCE_LIQUIDATION],
                initial_margin_req=self.initial_margin,
                maintenance_margin_req=self.maintenance_margin,
                positions=list(temp_positions.values()),
                warnings=self._generate_warnings(equity_ratio, 0),
            )
            
            return simulated_state, actions
    
    def _suggest_liquidation_order(
        self,
        positions: Dict[str, MarginPosition],
        amount_needed: float,
    ) -> str:
        """Suggest which positions to liquidate first."""
        # Sort by: 1) Largest loss first, 2) Highest margin ratio
        sorted_positions = sorted(
            positions.values(),
            key=lambda p: (p.unrealized_pnl, -p.margin_ratio),
        )
        
        suggestions = []
        remaining = amount_needed
        
        for pos in sorted_positions:
            if remaining <= 0:
                break
            
            sell_value = min(pos.market_value, remaining)
            sell_qty = int(sell_value / pos.current_price)
            sell_qty = (sell_qty // 100) * 100  # Round to lot
            
            if sell_qty > 0:
                suggestions.append(f"   - Sell {pos.symbol}: {sell_qty} shares ({sell_qty * pos.current_price:,.0f} VND)")
                remaining -= sell_qty * pos.current_price
        
        return "\n".join(suggestions) if suggestions else "   No positions to liquidate"
    
    def check_margin_call(self) -> Optional[MarginCallEvent]:
        """
        Check if margin call should be triggered.
        
        Returns:
            MarginCallEvent if margin call triggered, None otherwise
        """
        with self._lock:
            state = self.get_account_state()
            
            if state.status == MarginStatus.FORCE_LIQUIDATION:
                # Create force liquidation event
                target_equity = state.total_market_value * (self.margin_call_level + 0.05)
                amount_required = target_equity - state.total_equity
                
                event = MarginCallEvent(
                    timestamp=datetime.now(),
                    call_type=MarginCallType.FORCE_SELL,
                    equity_ratio=state.equity_ratio,
                    required_action="Immediate position liquidation required",
                    amount_required=amount_required,
                    deadline=datetime.now(),  # Immediate
                )
                
                self._active_margin_call = event
                self._margin_calls.append(event)
                
                logger.critical(
                    f"🚨 FORCE LIQUIDATION triggered! "
                    f"Equity: {state.equity_ratio:.1%}, Need: {amount_required:,.0f} VND"
                )
                
                return event
                
            elif state.status == MarginStatus.MARGIN_CALL:
                # Create margin call event
                target_equity = state.total_market_value * self.maintenance_margin
                amount_required = target_equity - state.total_equity
                
                event = MarginCallEvent(
                    timestamp=datetime.now(),
                    call_type=MarginCallType.DEPOSIT_CASH,
                    equity_ratio=state.equity_ratio,
                    required_action=f"Deposit {amount_required:,.0f} VND or reduce positions",
                    amount_required=amount_required,
                    deadline=datetime.now() + timedelta(hours=self.MARGIN_CALL_DEADLINE_HOURS),
                )
                
                self._active_margin_call = event
                self._margin_calls.append(event)
                
                logger.warning(
                    f"⚠️ MARGIN CALL triggered! "
                    f"Equity: {state.equity_ratio:.1%}, Need: {amount_required:,.0f} VND, "
                    f"Deadline: {event.deadline}"
                )
                
                return event
            
            # Clear active margin call if status improved
            if self._active_margin_call and state.status in [MarginStatus.HEALTHY, MarginStatus.NORMAL]:
                self._active_margin_call.resolved = True
                self._active_margin_call.resolution_time = datetime.now()
                self._active_margin_call.resolution_action = "Account restored to healthy status"
                self._active_margin_call = None
                logger.info("✅ Margin call resolved - account healthy")
            
            return None
    
    def execute_force_liquidation(
        self,
        get_price_func,
    ) -> List[Tuple[str, int, float, float]]:
        """
        Execute force liquidation to restore margin.
        
        Args:
            get_price_func: Function to get current price for symbol
            
        Returns:
            List of (symbol, quantity, price, pnl) for liquidated positions
        """
        with self._lock:
            state = self.get_account_state()
            
            if state.status != MarginStatus.FORCE_LIQUIDATION:
                return []
            
            # Calculate how much to liquidate
            target_equity_ratio = self.margin_call_level + 0.05
            target_equity = state.total_market_value * target_equity_ratio
            equity_shortfall = target_equity - state.total_equity
            
            # Sort positions: liquidate losers first
            sorted_positions = sorted(
                self._positions.values(),
                key=lambda p: p.unrealized_pnl,
            )
            
            liquidations = []
            remaining = equity_shortfall
            
            for pos in sorted_positions:
                if remaining <= 0:
                    break
                
                current_price = get_price_func(pos.symbol) or pos.current_price
                
                # Calculate shares to sell
                shares_needed = int(remaining / current_price) + 100  # Extra buffer
                shares_to_sell = min(shares_needed, pos.quantity)
                shares_to_sell = (shares_to_sell // 100) * 100  # Round to lot
                
                if shares_to_sell > 0:
                    success, msg, pnl = self.close_position(pos.symbol, shares_to_sell, current_price)
                    if success:
                        liquidations.append((pos.symbol, shares_to_sell, current_price, pnl))
                        remaining -= shares_to_sell * current_price
                        
                        logger.warning(
                            f"🔴 Force liquidated: {pos.symbol} {shares_to_sell} @ {current_price:,.0f} "
                            f"(P&L: {pnl:+,.0f})"
                        )
            
            # Resolve margin call
            if self._active_margin_call:
                self._active_margin_call.resolved = True
                self._active_margin_call.resolution_time = datetime.now()
                self._active_margin_call.resolution_action = f"Force liquidated {len(liquidations)} positions"
                self._active_margin_call = None
            
            return liquidations
    
    # =========================================================================
    # INTEREST CALCULATION
    # =========================================================================
    
    def calculate_daily_interest(self) -> float:
        """
        Calculate and accrue daily interest on borrowed amount.
        
        Returns:
            Daily interest amount
        """
        with self._lock:
            today = date.today()
            
            if today <= self._last_interest_date:
                return 0.0
            
            # Calculate days since last interest
            days = (today - self._last_interest_date).days
            
            # Daily interest rate
            daily_rate = self.interest_rate / 365
            
            # Calculate interest
            interest = self._borrowed_amount * daily_rate * days
            
            # Accrue interest
            self._accrued_interest += interest
            self._last_interest_date = today
            
            self._save_state()
            
            if interest > 0:
                logger.info(f"💰 Accrued interest: {interest:,.0f} VND ({days} days)")
            
            return interest
    
    def pay_interest(self, amount: Optional[float] = None) -> Tuple[bool, float]:
        """
        Pay accrued interest.
        
        Args:
            amount: Amount to pay (default: all accrued)
            
        Returns:
            (success, amount_paid)
        """
        with self._lock:
            pay_amount = amount or self._accrued_interest
            pay_amount = min(pay_amount, self._accrued_interest)
            
            if pay_amount > self.cash_balance:
                return False, 0.0
            
            self.cash_balance -= pay_amount
            self._accrued_interest -= pay_amount
            
            self._save_state()
            
            logger.info(f"💳 Paid interest: {pay_amount:,.0f} VND")
            
            return True, pay_amount
    
    # =========================================================================
    # PRICE UPDATES
    # =========================================================================
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all positions.
        
        Args:
            prices: Dict of {symbol: current_price}
        """
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    self._positions[symbol] = MarginPosition(
                        symbol=pos.symbol,
                        quantity=pos.quantity,
                        avg_cost=pos.avg_cost,
                        current_price=price,
                        margin_ratio=pos.margin_ratio,
                    )
            
            # Check for margin call after price update
            self.check_margin_call()
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def get_status_report(self) -> str:
        """Get formatted status report."""
        state = self.get_account_state()
        
        lines = [
            "=" * 60,
            "📊 MARGIN ACCOUNT STATUS",
            "=" * 60,
            f"Status: {state.status.value}",
            "-" * 60,
            f"Cash Balance:      {state.cash_balance:>15,.0f} VND",
            f"Market Value:      {state.total_market_value:>15,.0f} VND",
            f"Borrowed Amount:   {state.total_borrowed:>15,.0f} VND",
            f"Total Equity:      {state.total_equity:>15,.0f} VND",
            "-" * 60,
            f"Equity Ratio:      {state.equity_ratio:>14.1%}",
            f"Margin Utilization:{state.margin_utilization:>14.1%}",
            f"Buying Power:      {state.buying_power:>15,.0f} VND",
            "-" * 60,
            f"Accrued Interest:  {self._accrued_interest:>15,.0f} VND",
        ]
        
        if state.positions:
            lines.append("-" * 60)
            lines.append("📈 POSITIONS:")
            for pos in state.positions:
                pnl_pct = (pos.unrealized_pnl / pos.cost_basis * 100) if pos.cost_basis > 0 else 0
                emoji = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
                lines.append(
                    f"   {emoji} {pos.symbol}: {pos.quantity} @ {pos.avg_cost:,.0f} → {pos.current_price:,.0f} "
                    f"({pnl_pct:+.1f}%) [Margin: {pos.margin_ratio:.0%}]"
                )
        
        if state.warnings:
            lines.append("-" * 60)
            lines.append("⚠️ WARNINGS:")
            for warning in state.warnings:
                lines.append(f"   {warning}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# SINGLETON & HELPERS
# =============================================================================

_margin_manager: Optional[MarginManager] = None


def get_margin_manager(
    initial_cash: float = 100_000_000,
    margin_limit: float = 200_000_000,
) -> MarginManager:
    """Get singleton margin manager instance."""
    global _margin_manager
    if _margin_manager is None:
        _margin_manager = MarginManager(
            initial_cash=initial_cash,
            margin_limit=margin_limit,
        )
    return _margin_manager


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 70)
    print("🧪 TESTING MARGIN MANAGER")
    print("=" * 70)
    
    # Create manager
    manager = MarginManager(
        initial_cash=100_000_000,
        margin_limit=200_000_000,
    )
    
    # Test opening position
    print("\n1️⃣ Opening margin position...")
    success, msg = manager.open_position("VNM", 1000, 80_000, use_margin=True)
    print(f"   Result: {success} - {msg}")
    
    print("\n" + manager.get_status_report())
    
    # Test price drop simulation
    print("\n2️⃣ Simulating 20% price drop...")
    sim_state, actions = manager.simulate_price_change({"VNM": 64_000})
    print(f"   Simulated equity ratio: {sim_state.equity_ratio:.1%}")
    print(f"   Simulated status: {sim_state.status.value}")
    for action in actions:
        print(f"   {action}")
    
    # Test margin call
    print("\n3️⃣ Applying price drop and checking margin call...")
    manager.update_prices({"VNM": 64_000})
    margin_call = manager.check_margin_call()
    if margin_call:
        print(f"   Margin call triggered: {margin_call.call_type.value}")
        print(f"   Amount required: {margin_call.amount_required:,.0f} VND")
    
    print("\n" + manager.get_status_report())
    
    print("\n" + "=" * 70)
    print("✅ Margin Manager test completed!")
    print("=" * 70)
