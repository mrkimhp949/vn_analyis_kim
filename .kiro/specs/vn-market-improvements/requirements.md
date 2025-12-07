# Requirements Document

## Status: IMPLEMENTED ✅

All requirements in this document have been implemented and tested.
- Implementation Date: December 7, 2025
- Test File: `tests/unit/test_vn_market_improvements_v6.py`
- All 38 tests passing

## Introduction

This document specifies improvements to the Vietnam stock market trading system to address identified risks and gaps in the current implementation. The improvements focus on six key areas:

1. **Floor Bounce Logic** - Enhanced exit strategy with volume-based triggers instead of time-only
2. **DCA Strategy Documentation** - Clear documentation of why DCA is disabled for VN market
3. **Odd-lot Trading** - Implementation of odd-lot trading with spread premium handling
4. **Margin Trading Integration** - Complete margin management with position sizing
5. **T+0 Intraday Trading** - Broker API validation and implementation
6. **Warrant/ETF Specific Logic** - Special instrument handling with different limits

Additionally, this addresses potential risks:
- Floor price trap (overnight position lock)
- Gap risk (7% full limit gaps)
- Liquidity risk for small caps
- Foreign flow data dependency

## Glossary

- **Floor Price**: The minimum price a stock can trade at in a single day (reference price - 7% for HOSE)
- **Ceiling Price**: The maximum price a stock can trade at in a single day (reference price + 7% for HOSE)
- **Floor Bounce**: A price recovery after hitting the floor price
- **DCA (Dollar Cost Averaging)**: Strategy of buying more shares as price drops
- **Odd-lot**: Trading quantity less than 100 shares (1-99 shares)
- **T+0**: Same-day trading (buy and sell on same day)
- **T+2**: Settlement cycle where trades settle 2 business days after execution
- **Warrant**: Derivative instrument with ±50% daily price limit
- **ETF**: Exchange Traded Fund
- **VN_System**: The Vietnam stock market trading system
- **HOSE**: Ho Chi Minh Stock Exchange (±7% daily limit)
- **HNX**: Hanoi Stock Exchange (±10% daily limit)
- **UPCOM**: Unlisted Public Company Market (±15% daily limit)
- **Panic Selling**: High-volume selling pressure indicating market fear
- **Volume Ratio**: Current volume divided by average volume

## Requirements

### Requirement 1: Enhanced Floor Bounce Logic

**User Story:** As a trader, I want the system to use volume-based exit triggers when price hits floor, so that I can avoid being trapped in positions during panic selling.

#### Acceptance Criteria

1. WHEN price hits floor AND volume ratio exceeds 3.0 (panic threshold) THEN the VN_System SHALL trigger immediate exit without waiting for bounce
2. WHEN price hits floor AND volume ratio is between 1.5 and 3.0 THEN the VN_System SHALL extend wait time to 60 minutes before exit
3. WHEN price hits floor AND volume ratio is below 1.5 THEN the VN_System SHALL use standard 30-minute wait for bounce confirmation
4. WHEN floor bounce wait exceeds configured maximum time THEN the VN_System SHALL trigger exit with reason "Floor Bounce Timeout"
5. WHEN price recovers from floor by at least 1% THEN the VN_System SHALL cancel floor exit timer and resume normal monitoring

### Requirement 2: DCA Strategy Documentation

**User Story:** As a developer, I want clear documentation explaining why DCA is disabled for Vietnam market, so that future maintainers understand the rationale.

#### Acceptance Criteria

1. WHEN DCA_ENABLED constant is defined THEN the VN_System SHALL include inline documentation explaining the 1.48% round-trip cost impact
2. WHEN DCA configuration is accessed THEN the VN_System SHALL provide a method returning detailed cost-benefit analysis
3. WHEN DCA is attempted while disabled THEN the VN_System SHALL log a warning with the specific reason (transaction costs exceed expected benefit)

### Requirement 3: Odd-lot Trading Implementation

**User Story:** As a trader, I want to trade odd-lots (1-99 shares) with proper spread premium handling, so that I can manage small positions efficiently.

#### Acceptance Criteria

1. WHEN calculating order for quantity less than 100 shares THEN the VN_System SHALL apply 0.5% spread premium to expected execution price
2. WHEN odd-lot order is placed THEN the VN_System SHALL apply minimum commission of 11,000 VND
3. WHEN position sizing results in odd-lot quantity THEN the VN_System SHALL warn user about higher transaction costs
4. WHEN odd-lot trading is disabled in configuration THEN the VN_System SHALL round position to nearest lot (100 shares)

### Requirement 4: Margin Trading Integration

**User Story:** As a margin trader, I want position sizing to account for margin requirements, so that I can avoid margin calls.

#### Acceptance Criteria

1. WHEN calculating position size for margin account THEN the VN_System SHALL limit position based on 50% initial margin requirement
2. WHEN margin ratio falls below 40% (warning level) THEN the VN_System SHALL alert user and reduce new position sizes by 50%
3. WHEN margin ratio falls below 35% (maintenance level) THEN the VN_System SHALL block new positions and recommend reducing exposure
4. WHEN margin ratio falls below 30% (margin call level) THEN the VN_System SHALL trigger emergency position reduction
5. WHEN calculating available buying power THEN the VN_System SHALL account for both cash and margin capacity

### Requirement 5: T+0 Intraday Trading Validation

**User Story:** As an intraday trader, I want T+0 trading to be validated against broker API capabilities, so that I can execute same-day round trips reliably.

#### Acceptance Criteria

1. WHEN T+0 trading is enabled THEN the VN_System SHALL verify broker supports intraday trading via API
2. WHEN T+0 trade is attempted THEN the VN_System SHALL validate account meets minimum 50M VND requirement
3. WHEN daily T+0 loss exceeds 2% THEN the VN_System SHALL disable T+0 trading for remainder of day
4. WHEN T+0 trade count exceeds 20 per day THEN the VN_System SHALL block additional T+0 trades
5. WHEN T+0 position is held less than 5 minutes THEN the VN_System SHALL warn about potential wash trade detection

### Requirement 6: Warrant and ETF Specific Logic

**User Story:** As a trader, I want the system to handle warrants and ETFs with their specific rules, so that I can trade these instruments safely.

#### Acceptance Criteria

1. WHEN trading a warrant THEN the VN_System SHALL apply ±50% daily price limit instead of ±7%
2. WHEN trading a warrant THEN the VN_System SHALL use T+0 settlement instead of T+2
3. WHEN warrant has less than 3 days to expiry THEN the VN_System SHALL block new positions
4. WHEN warrant has less than 30 days to expiry THEN the VN_System SHALL warn user about time decay risk
5. WHEN trading an ETF with short-selling enabled THEN the VN_System SHALL allow short positions with appropriate margin

### Requirement 7: Enhanced Gap Risk Protection

**User Story:** As a trader, I want stronger protection against gap risk, so that I can limit losses from overnight gaps.

#### Acceptance Criteria

1. WHEN gap down exceeds 4% THEN the VN_System SHALL trigger emergency exit regardless of other conditions
2. WHEN gap down is between 2.5% and 4% AND position is profitable THEN the VN_System SHALL trigger profit protection exit
3. WHEN gap up exceeds 4% THEN the VN_System SHALL consider partial profit taking
4. WHEN overnight position exists AND market shows high gap probability THEN the VN_System SHALL warn user before market close

### Requirement 8: Improved Liquidity Risk Management

**User Story:** As a trader, I want better liquidity risk management for small caps, so that I can exit positions without excessive slippage.

#### Acceptance Criteria

1. WHEN average daily value is below 500M VND THEN the VN_System SHALL limit position to 2% of average daily volume
2. WHEN attempting to exit position larger than 5% of daily volume THEN the VN_System SHALL recommend splitting into multiple orders
3. WHEN liquidity drops below 300M VND (critical level) THEN the VN_System SHALL trigger liquidity warning and reduce position size multiplier to 0.5
4. WHEN calculating slippage for illiquid stocks THEN the VN_System SHALL use 1.0% slippage instead of standard 0.4%

### Requirement 9: Foreign Flow Data Fallback

**User Story:** As a trader, I want the system to handle missing foreign flow data gracefully, so that trading continues when data is unavailable.

#### Acceptance Criteria

1. WHEN foreign flow data is unavailable THEN the VN_System SHALL use neutral (zero) foreign flow score
2. WHEN foreign flow data is delayed more than 15 minutes THEN the VN_System SHALL mark data as stale and reduce its weight by 50%
3. WHEN foreign flow data becomes available after being unavailable THEN the VN_System SHALL log the recovery and resume normal weighting
4. WHEN making trading decisions without foreign flow data THEN the VN_System SHALL note this limitation in the decision metadata

