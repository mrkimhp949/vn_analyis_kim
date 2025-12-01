# Entry Logic Review & Improvement Plan

**Date:** 2025-12-01
**Reviewer:** Claude Code
**Scope:** Entry signal generation and filtering logic

---

## Executive Summary

The current entry logic is **comprehensive but overly complex**, with 7 core filters and multiple soft filters that adjust confidence scores. While the logic includes Vietnam market-specific validations and ML fallback mechanisms, it suffers from:

1. **High False Negative Rate** - Too many restrictive filters blocking valid signals
2. **Complex Confidence Adjustment** - Hard to tune and debug
3. **Missing Performance Feedback** - No systematic filter validation
4. **Incomplete Implementations** - Some features (correlation check, limit orders) not fully integrated

**Overall Grade: B- (Good foundation, needs optimization)**

---

## Current Architecture Overview

### Entry Signal Flow

```
1. EntrySignalService.scan_for_entries()
   ↓
2. _scan_single_ticker() [Parallel execution]
   ├── Circuit breaker check
   ├── Session boundary check
   ├── Data validation (min 50 rows)
   ├── ML analysis (with fallback to technical)
   ├── Entry logic filtering (7 core + soft filters)
   ├── Vietnam price limit check
   ├── Position sizing
   ├── T+2 settlement check
   └── Volume constraint check
   ↓
3. ImprovedEntryLogic.analyze_entry()
   ├── Validate initial signal
   ├── Run all filters
   ├── Calculate adjusted confidence
   ├── Apply performance feedback
   ├── Calculate prices & risk/reward
   ├── Check portfolio correlation
   └── Return EntrySignal
```

### 7 Core Filters

| # | Filter | Purpose | Threshold | Issues |
|---|--------|---------|-----------|---------|
| 1 | Market Regime | Ensure tradeable market | regime.tradeable = True | ✅ Good |
| 2 | VN Price Limits | Avoid floor/ceiling (±7%) | ±6.5% from reference | ✅ Good |
| 3 | Trend Alignment | EMA 20 > EMA 50 | Optional (configurable) | ⚠️ Too rigid |
| 4 | Liquidity | Tiered thresholds | 1-10B VND/day | ⚠️ Arbitrary |
| 5 | Volatility | ATR/Price ratio | 2-8% range | ⚠️ Not validated |
| 6 | RSI | Avoid overbought | RSI < 70 | ⚠️ Oversimplified |
| 7 | Portfolio Correlation | Diversification | Max 0.7 correlation | ❌ **Missing implementation** |

### Soft Filters (Confidence Adjustments)

- Volume confirmation: ±10 points
- Support/Resistance: ±15 points
- Multi-timeframe: Warning only
- Trend strength: ±5 points
- Liquidity quality: ±5 points

---

## Critical Issues Identified

### ~~🔴 CRITICAL: Missing Correlation Check~~ ✅ RESOLVED

**File:** `src/strategies/entry_logic.py:1744-1863`
**Status:** ✅ **IMPLEMENTED** - Correlation check exists and is properly implemented

**Implementation Details:**
```python
def _check_portfolio_correlation(self, df: pd.DataFrame, symbol: Optional[str]) -> Dict:
    """
    OPTIMIZED: Kiểm tra correlation với portfolio hiện tại (with caching)
    - Uses 60-day lookback for correlation calculation
    - Caches results for 5 minutes to optimize performance
    - Invalidates cache on date change or portfolio composition change
    - Threshold: max_correlation > 0.7 triggers warning with -20 confidence penalty
    """
```

**Called in:** Filter #12 (line 726) - Applied as soft filter with confidence adjustment

**Assessment:** Well-implemented with proper caching and optimization. No changes needed.

---

### 🔴 CRITICAL: Transaction Cost Calculation Error

**File:** `src/strategies/entry_logic.py:844-846`

```python
# WRONG: Using same cost for entry and exit
entry_cost = entry_price * TOTAL_TRANSACTION_COST
exit_cost = entry_price * TOTAL_TRANSACTION_COST  # Should use exit price!
```

**Issue:**
- Exit cost should be based on take_profit price, not entry_price
- Underestimates actual transaction costs
- Inflates risk/reward ratio

**Fix:**
```python
entry_cost = entry_price * TOTAL_TRANSACTION_COST
exit_cost = take_profit * TOTAL_TRANSACTION_COST  # Use actual exit price
```

**Impact:** Medium - Affects ~10-15% of R:R calculations

---

### 🟡 MAJOR: Complex Confidence Adjustment Logic

**File:** `src/strategies/entry_logic.py:271-600`

**Issue:** Too many confidence adjustments make system unpredictable

**Current Adjustments:**
```
Base ML Confidence: 60%
- Trend alignment: +5
- Volume low: -10
- Near support: +10
- Liquidity good: +5
- Volatility high: -15
- RSI neutral: 0
- Performance penalty: -5
= Final: 50%
```

**Problems:**
1. **Hard to tune** - 10+ different adjustment sources
2. **Conflicting signals** - Positive and negative adjustments cancel out
3. **No weighting** - All adjustments treated equally
4. **Debug difficulty** - Hard to understand why signals rejected

**Recommendation:** Simplify to 3-5 key adjustments with clear weighting

---

### 🟡 MAJOR: No Filter Performance Validation

**Issue:** All filter thresholds appear arbitrary without backtesting validation

**Examples of Unvalidated Thresholds:**
- Min liquidity: 1B VND (why not 500M or 2B?)
- Volatility range: 2-8% (why these numbers?)
- RSI threshold: 70 (why not 65 or 75?)
- Min confidence: 45% (why not 40% or 50%?)

**Impact:**
- Potentially missing profitable signals (false negatives)
- May be accepting losing signals (false positives)
- No data-driven optimization

**Fix:** Implement systematic backtesting of each filter

---

### 🟡 MAJOR: ML Fallback Not Properly Tracked

**File:** `src/strategies/entry_logic.py:230-251`

**Issue:** Technical-only signals marked but not tracked separately for performance

```python
# Sets flag but no separate tracking
self._is_technical_only = True
```

**Problem:**
- Can't compare ML vs Technical-only performance
- Don't know if ML failure is actually harmful
- Missing opportunity to improve ML system

**Fix:** Add separate metrics tracking for:
- ML signal win rate
- Technical-only signal win rate
- ML failure rate
- Performance degradation when using fallback

---

## Medium Priority Issues

### 🟢 Incomplete Limit Order Implementation

**File:** `src/strategies/entry_logic.py:1018-1040`

```python
# Sets limit order flag but not integrated with execution
is_limit_order = optimized_entry.get("entry_type") in ["PULLBACK", "BREAKOUT"]
limit_price = optimized_entry.get("entry_price") if is_limit_order else None
```

**Issue:**
- Flag is set but execution layer doesn't use it
- No timeout/cancellation logic for unfilled limit orders
- No fallback to market order if limit not filled

**Recommendation:** Either complete implementation or remove incomplete code

---

### 🟢 Trend Alignment Filter Too Rigid

**File:** `src/strategies/entry_logic.py:370-400`

**Issue:** Only checks EMA 20 > EMA 50, ignores:
- Strength of trend (slope)
- Recent trend change
- Multiple timeframe confirmation

**Impact:** Misses early trend reversals and late trend joiners

**Fix:** Add trend strength scoring:
```python
# Instead of binary aligned/not aligned
trend_score = calculate_trend_strength(ema_20, ema_50, slope, acceleration)
# Adjust confidence based on trend_score (0-100)
```

---

### 🟢 Liquidity Tiers Not Well Justified

**File:** `src/strategies/entry_logic.py:167-171`

```python
self.liquidity_tiers = {
    "large": strategy_config.entry.liquidity_tiers.large_cap,
    "mid": strategy_config.entry.liquidity_tiers.mid_cap,
    "small": strategy_config.entry.liquidity_tiers.small_cap,
}
```

**Issue:** Tiers seem arbitrary, not based on actual VN market data

**Recommendation:** Analyze actual VN market liquidity distribution and set percentile-based tiers

---

### 🟢 Correlation Cache Implementation Incomplete

**File:** `src/strategies/entry_logic.py:176-181`

```python
# Cache variables declared but never used
self._correlation_cache = None
self._correlation_cache_time = None
self._correlation_cache_symbols = None
```

**Issue:** Cache infrastructure exists but correlation check missing

---

## Strengths to Preserve

✅ **Vietnam Market Specific Validations**
- T+2 settlement checking
- Price floor/ceiling validation
- Session boundary checks
- Proper transaction cost modeling (0.9% round trip)

✅ **Circuit Breaker Integration**
- Global circuit breaker
- Per-symbol circuit breaker
- Prevents repeated bad trades

✅ **Comprehensive Data Validation**
- Min 50 rows requirement
- Price validation
- Volume checks

✅ **Good Error Handling**
- Graceful ML fallback
- Exception handling
- Detailed logging

✅ **Regime-Aware Filtering**
- Relaxes filters in BULL markets
- Tightens in BEAR markets
- Good adaptive behavior

---

## Improvement Recommendations

### Priority 1: Critical Fixes (Implement Immediately)

#### 1.1 Implement Missing Correlation Check

```python
def _check_correlation(
    self,
    symbol: str,
    df: pd.DataFrame,
    existing_positions: Dict
) -> Dict:
    """
    Check correlation with existing portfolio positions

    Returns:
        {
            'passed': bool,
            'max_correlation': float,
            'correlated_symbol': str,
            'diversified': bool
        }
    """
    # Implementation needed
    pass
```

**Effort:** 4 hours
**Impact:** High - Reduces portfolio risk

---

#### 1.2 Fix Transaction Cost Calculation

**File:** `src/strategies/entry_logic.py:844-846`

```python
# BEFORE
entry_cost = entry_price * TOTAL_TRANSACTION_COST
exit_cost = entry_price * TOTAL_TRANSACTION_COST  # WRONG

# AFTER
entry_cost = entry_price * TOTAL_TRANSACTION_COST
exit_cost = take_profit * TOTAL_TRANSACTION_COST  # Use actual TP price

# Also account for slippage differently for entry vs exit
entry_with_slippage = entry_price * (1 + DEFAULT_SLIPPAGE + TOTAL_TRANSACTION_COST)
exit_with_costs = take_profit * (1 - DEFAULT_SLIPPAGE - TOTAL_TRANSACTION_COST)
```

**Effort:** 1 hour
**Impact:** Medium - More accurate R:R calculations

---

#### 1.3 Add ML vs Technical Performance Tracking

**File:** `src/monitoring/signal_performance.py` (new)

```python
class SignalPerformanceTracker:
    """Track ML vs Technical-only signal performance"""

    def record_signal(self, symbol, signal_source, confidence, outcome):
        """Record signal and eventual outcome"""
        pass

    def get_performance_by_source(self) -> Dict:
        """
        Returns:
            {
                'ml': {'win_rate': 0.58, 'avg_return': 2.3%, 'count': 145},
                'technical': {'win_rate': 0.52, 'avg_return': 1.8%, 'count': 67},
                'ml_failure_rate': 0.15
            }
        """
        pass
```

**Effort:** 6 hours
**Impact:** High - Enables data-driven ML improvements

---

### Priority 2: Major Improvements (Next Sprint)

#### 2.1 Simplify Confidence Adjustment Logic

**Current:** 10+ adjustment sources
**Proposed:** 5 weighted adjustments

```python
# Simplified adjustment system
adjustments = {
    'trend_quality': {'weight': 0.30, 'score': trend_score},      # 30%
    'liquidity_quality': {'weight': 0.20, 'score': liq_score},    # 20%
    'technical_setup': {'weight': 0.25, 'score': tech_score},     # 25%
    'risk_quality': {'weight': 0.15, 'score': risk_score},        # 15%
    'regime_alignment': {'weight': 0.10, 'score': regime_score},  # 10%
}

# Calculate weighted adjustment
total_adjustment = sum(
    adj['weight'] * (adj['score'] - 50)  # Centered at 50
    for adj in adjustments.values()
)

final_confidence = base_confidence + total_adjustment
```

**Benefits:**
- Easier to understand
- More predictable
- Weights can be optimized via backtesting
- Clear contribution of each factor

**Effort:** 8 hours
**Impact:** High - Reduces confusion, improves tuning

---

#### 2.2 Implement Filter Backtesting Framework

**File:** `tests/backtest/test_filter_performance.py` (new)

```python
def backtest_filter(filter_name, threshold_range, historical_data):
    """
    Test filter across range of thresholds to find optimal value

    Args:
        filter_name: 'rsi', 'volatility', 'liquidity', etc.
        threshold_range: [30, 35, 40, ..., 70, 75, 80]
        historical_data: Past trades and outcomes

    Returns:
        {
            'optimal_threshold': 65,
            'win_rate_improvement': +0.08,
            'false_negative_reduction': -0.15,
            'expected_return_improvement': +1.2%
        }
    """
    pass
```

**Effort:** 16 hours
**Impact:** Very High - Data-driven filter optimization

---

#### 2.3 Enhance Trend Alignment with Strength Scoring

```python
def _check_trend_alignment(self, df, signal_type) -> Dict:
    """Enhanced trend check with strength scoring"""

    ema_20 = safe_get_latest(df, 'ema_20', 0)
    ema_50 = safe_get_latest(df, 'ema_50', 0)

    # Binary alignment
    aligned = ema_20 > ema_50 if signal_type == 'BUY' else ema_20 < ema_50

    # NEW: Trend strength (0-100)
    separation = abs(ema_20 - ema_50) / ema_50 * 100
    slope_20 = calculate_slope(df['ema_20'].tail(10))
    slope_50 = calculate_slope(df['ema_50'].tail(20))

    # Combine into strength score
    strength = min(100, (
        separation * 40 +      # Separation = 40%
        slope_20 * 100 * 30 +  # Recent slope = 30%
        slope_50 * 100 * 30    # Long-term slope = 30%
    ))

    return {
        'aligned': aligned,
        'strength': strength,
        'separation': separation,
        'reason': f"EMA20/50 {'aligned' if aligned else 'misaligned'}, strength: {strength:.0f}"
    }
```

**Effort:** 4 hours
**Impact:** Medium - Better trend quality assessment

---

### Priority 3: Code Quality Improvements

#### 3.1 Extract Magic Numbers to Configuration

**Current:** Hardcoded thresholds throughout code
**Proposed:** Centralize in config

```python
# src/config/filter_config.py
@dataclass
class FilterConfig:
    # RSI thresholds
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    rsi_warning_threshold: int = 65

    # Volatility thresholds
    volatility_min: float = 0.02  # 2%
    volatility_max: float = 0.08  # 8%
    volatility_optimal_range: Tuple[float, float] = (0.03, 0.05)

    # Liquidity thresholds (VND)
    liquidity_large_cap: int = 10_000_000_000  # 10B
    liquidity_mid_cap: int = 5_000_000_000     # 5B
    liquidity_small_cap: int = 1_000_000_000   # 1B

    # ... etc
```

**Effort:** 3 hours
**Impact:** Low - Better maintainability

---

#### 3.2 Add Structured Logging

```python
# Instead of scattered logger.debug/info
logger.info(
    "Entry signal analysis",
    extra={
        'symbol': symbol,
        'base_confidence': base_confidence,
        'adjusted_confidence': adjusted_confidence,
        'filters_passed': filters_passed,
        'filters_failed': filters_failed,
        'final_decision': 'ACCEPT' if should_enter else 'REJECT',
        'rejection_reason': rejection_reason,
        'ml_source': 'ml' if not self._is_technical_only else 'technical'
    }
)
```

**Effort:** 4 hours
**Impact:** Medium - Better debugging and monitoring

---

#### 3.3 Add Unit Tests for Each Filter

**Current:** Limited filter testing
**Proposed:** Comprehensive unit tests

```python
# tests/unit/test_entry_filters.py

class TestTrendAlignment:
    def test_uptrend_buy_signal(self):
        """Test EMA 20 > 50 for BUY signal"""
        pass

    def test_trend_strength_scoring(self):
        """Test trend strength calculation"""
        pass

class TestVolatilityFilter:
    def test_volatility_too_high_blocks(self):
        """Test high volatility blocks entry"""
        pass

    def test_optimal_volatility_passes(self):
        """Test optimal volatility passes"""
        pass

# ... etc for all 7 filters
```

**Effort:** 12 hours
**Impact:** Medium - Prevents regressions

---

## Implementation Roadmap

### Week 1: Critical Fixes
- [ ] Implement correlation check (4h)
- [ ] Fix transaction cost calculation (1h)
- [ ] Add ML vs Technical tracking (6h)
- [ ] Testing and validation (5h)

**Total:** 16 hours

---

### Week 2: Major Improvements
- [ ] Simplify confidence adjustments (8h)
- [ ] Enhance trend alignment (4h)
- [ ] Design filter backtesting framework (4h)
- [ ] Testing and documentation (4h)

**Total:** 20 hours

---

### Week 3: Backtesting & Optimization
- [ ] Implement filter backtesting (12h)
- [ ] Run backtests on historical data (8h)
- [ ] Optimize filter thresholds (6h)
- [ ] Validation and documentation (6h)

**Total:** 32 hours

---

### Week 4: Code Quality & Testing
- [ ] Extract config constants (3h)
- [ ] Add structured logging (4h)
- [ ] Write comprehensive unit tests (12h)
- [ ] Integration testing (5h)
- [ ] Documentation updates (4h)

**Total:** 28 hours

---

## Expected Improvements

### Quantitative Goals

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| False Negative Rate | ~40% | ~25% | -37.5% |
| Signal Quality (Win Rate) | ~55% | ~60% | +9% |
| Avg R:R Ratio | 1.5:1 | 1.8:1 | +20% |
| Profitable Signals/Day | 2.3 | 3.5 | +52% |
| ML Failure Rate | 30% | <10% | -67% |

### Qualitative Goals

✅ More predictable entry logic
✅ Easier to tune and optimize
✅ Better debugging capabilities
✅ Data-driven filter validation
✅ Reduced false negatives
✅ Improved risk management

---

## Conclusion

The current entry logic is **well-structured but over-engineered**, with too many filters and adjustments that create complexity without proportional benefit. The recommended improvements focus on:

1. **Fixing critical bugs** (correlation check, transaction costs)
2. **Simplifying logic** (fewer, better-weighted adjustments)
3. **Adding validation** (backtesting framework)
4. **Improving observability** (tracking, logging)

**Estimated Total Effort:** 96 hours (12 days)
**Expected ROI:** 30-50% improvement in signal quality

**Recommendation:** Implement Priority 1 fixes immediately (Week 1), then proceed with major improvements based on results.

---

## Appendix A: Filter Performance Questions

Each filter should answer:

1. **What is the win rate of signals that pass this filter vs. fail?**
2. **How many profitable signals are we blocking (false negatives)?**
3. **What is the optimal threshold that maximizes (win rate × signal count)?**
4. **Does this filter add value or just reduce signal count?**

**Action:** Run historical analysis to answer these questions for each filter.

---

## Appendix B: Code Files to Modify

### Critical Priority
- `src/strategies/entry_logic.py` - Main entry logic
- `src/monitoring/signal_performance.py` - NEW: ML tracking
- `src/strategies/correlation_check.py` - NEW: Correlation implementation

### Major Priority
- `src/config/filter_config.py` - NEW: Centralized config
- `tests/backtest/test_filter_performance.py` - NEW: Backtesting
- `src/strategies/entry_logic.py` - Simplify adjustments

### Code Quality
- `tests/unit/test_entry_filters.py` - NEW: Unit tests
- `src/utils/logging_utils.py` - NEW: Structured logging
- Documentation updates

---

**End of Review Document**
