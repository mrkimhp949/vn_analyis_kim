# -*- coding: utf-8 -*-
"""
Tests for Entry Logic v3.0 and Position Sizing v3.0

Verifies:
1. Simplified filter pipeline works correctly
2. Adaptive thresholds by market regime
3. Vietnam market rules compliance
4. Transaction cost awareness
5. Position sizing accuracy
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestEntryLogicV3:
    """Tests for SimplifiedEntryLogicV3."""

    @pytest.fixture
    def entry_logic(self):
        """Create entry logic instance."""
        from src.strategies.entry_logic_v3 import SimplifiedEntryLogicV3

        return SimplifiedEntryLogicV3()

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV DataFrame."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        np.random.seed(42)

        close = 50000 + np.cumsum(np.random.randn(100) * 500)

        df = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(100) * 0.01),
                "high": close * (1 + abs(np.random.randn(100) * 0.02)),
                "low": close * (1 - abs(np.random.randn(100) * 0.02)),
                "close": close,
                "volume": np.random.randint(100000, 1000000, 100),
                "rsi": 30 + np.random.rand(100) * 40,  # RSI 30-70
            },
            index=dates,
        )

        return df

    def test_critical_filters_block_entry(self, entry_logic, sample_df):
        """Test that critical filter failures block entry."""
        # Test with non-tradeable market
        result = entry_logic.analyze_entry(
            symbol="VCB",
            df=sample_df,
            market_regime={"regime": "BEAR", "tradeable": False, "confidence": 80},
        )

        assert result.should_enter is False
        assert result.critical_filters_passed < result.critical_filters_total

    def test_adaptive_thresholds_bull_market(self, entry_logic, sample_df):
        """Test adaptive thresholds in BULL market."""
        # BULL market should have lower confidence requirement
        result = entry_logic.analyze_entry(
            symbol="VCB",
            df=sample_df,
            market_regime={"regime": "BULL", "tradeable": True, "confidence": 75},
            ml_signal={"signal": "BUY", "confidence": 55},
        )

        # In BULL, min confidence is 50, so 55 should pass
        # Position multiplier should be higher (1.2)
        assert result.position_multiplier >= 1.0

    def test_adaptive_thresholds_bear_market(self, entry_logic, sample_df):
        """Test adaptive thresholds in BEAR market."""
        result = entry_logic.analyze_entry(
            symbol="VCB",
            df=sample_df,
            market_regime={"regime": "BEAR", "tradeable": True, "confidence": 75},
            ml_signal={"signal": "BUY", "confidence": 65},
        )

        # In BEAR, min confidence is 70, position multiplier is 0.6
        assert result.position_multiplier <= 0.8
        assert "BEAR" in str(result.warnings) or result.position_multiplier < 1.0

    def test_liquidity_filter_blocks_illiquid(self, entry_logic):
        """Test that illiquid stocks are blocked."""
        # Create low volume DataFrame
        dates = pd.date_range(end=datetime.now(), periods=50, freq="D")
        df = pd.DataFrame(
            {
                "open": [50000] * 50,
                "high": [51000] * 50,
                "low": [49000] * 50,
                "close": [50000] * 50,
                "volume": [1000] * 50,  # Very low volume
            },
            index=dates,
        )

        result = entry_logic.analyze_entry(
            symbol="ILLIQUID",
            df=df,
            market_regime={"regime": "SIDEWAYS", "tradeable": True, "confidence": 60},
        )

        # Should fail liquidity filter
        assert result.should_enter is False
        assert any("liquidity" in w.lower() for w in result.warnings)

    def test_risk_reward_calculation_includes_costs(self, entry_logic, sample_df):
        """Test that R:R calculation includes transaction costs."""
        result = entry_logic.analyze_entry(
            symbol="VCB",
            df=sample_df,
            market_regime={"regime": "SIDEWAYS", "tradeable": True, "confidence": 60},
            ml_signal={"signal": "BUY", "confidence": 70},
        )

        # Expected return should account for 1.48% round trip cost
        if result.should_enter:
            # Net return should be less than gross return
            gross_return = (result.take_profit_1 - result.entry_price) / result.entry_price
            assert result.expected_return_after_costs < gross_return * 100

    def test_filter_summary_contains_all_filters(self, entry_logic, sample_df):
        """Test that filter summary contains all 5 filters."""
        result = entry_logic.analyze_entry(
            symbol="VCB",
            df=sample_df,
            market_regime={"regime": "SIDEWAYS", "tradeable": True, "confidence": 60},
        )

        filter_names = [f.name for f in result.filters_summary]

        assert "market_regime" in filter_names
        assert "liquidity" in filter_names
        assert "technical_score" in filter_names
        assert "risk_reward" in filter_names
        assert "timing_flow" in filter_names

    def test_stop_loss_within_vietnam_limits(self, entry_logic, sample_df):
        """Test that stop loss is within Vietnam ±7% limit."""
        result = entry_logic.analyze_entry(
            symbol="VCB",
            df=sample_df,
            market_regime={"regime": "SIDEWAYS", "tradeable": True, "confidence": 60},
            ml_signal={"signal": "BUY", "confidence": 70},
        )

        if result.should_enter and result.entry_price > 0:
            stop_pct = (result.entry_price - result.stop_loss) / result.entry_price
            # Stop should be between 3% and 7%
            assert 0.03 <= stop_pct <= 0.07


class TestPositionSizingV3:
    """Tests for EnhancedPositionSizerV3."""

    @pytest.fixture
    def position_sizer(self):
        """Create position sizer instance."""
        from src.strategies.position_sizing_v3 import EnhancedPositionSizerV3

        return EnhancedPositionSizerV3(total_capital=100_000_000)

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV DataFrame."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq="D")
        np.random.seed(42)

        close = 50000 + np.cumsum(np.random.randn(50) * 500)

        df = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(50) * 0.01),
                "high": close * (1 + abs(np.random.randn(50) * 0.02)),
                "low": close * (1 - abs(np.random.randn(50) * 0.02)),
                "close": close,
                "volume": np.random.randint(100000, 500000, 50),
            },
            index=dates,
        )

        return df

    def test_lot_size_enforcement(self, position_sizer):
        """Test that shares are rounded to lot size 100."""
        result = position_sizer.calculate_position_size(
            symbol="VCB",
            entry_price=50000,
            stop_loss=47500,  # 5% stop
            confidence=70,
        )

        assert result.shares % 100 == 0

    def test_max_position_limit(self, position_sizer):
        """Test that position doesn't exceed max limit."""
        result = position_sizer.calculate_position_size(
            symbol="VCB",
            entry_price=50000,
            stop_loss=49000,  # 2% stop - would give large position
            confidence=90,
        )

        # Position should not exceed 12% of capital
        max_value = 100_000_000 * 0.12
        assert result.value <= max_value * 1.01  # Allow 1% tolerance

    def test_regime_adjustment_bear(self, position_sizer):
        """Test position reduction in BEAR market."""
        # Use tighter stop to get larger base position (more room for adjustment)
        bull_result = position_sizer.calculate_position_size(
            symbol="VCB",
            entry_price=50000,
            stop_loss=48500,  # 3% stop - larger position
            confidence=70,
            market_regime={"regime": "BULL", "tradeable": True},
        )

        bear_result = position_sizer.calculate_position_size(
            symbol="VCB",
            entry_price=50000,
            stop_loss=48500,  # Same stop
            confidence=70,
            market_regime={"regime": "BEAR", "tradeable": True},
        )

        # BEAR position should be smaller than BULL (or equal if both hit minimum)
        # Check adjustments show different regime multipliers
        assert bear_result.adjustments.get("regime_mult", 1.0) < bull_result.adjustments.get(
            "regime_mult", 1.0
        )
        # BEAR should have warning about reduced position
        assert any("BEAR" in w for w in bear_result.warnings)

    def test_confidence_adjustment(self, position_sizer):
        """Test position adjustment by confidence."""
        high_conf = position_sizer.calculate_position_size(
            symbol="VCB",
            entry_price=50000,
            stop_loss=47500,
            confidence=85,
        )

        low_conf = position_sizer.calculate_position_size(
            symbol="VCB",
            entry_price=50000,
            stop_loss=47500,
            confidence=55,
        )

        # High confidence should give larger position
        assert high_conf.shares >= low_conf.shares

    def test_vn30_liquidity_tier(self, position_sizer):
        """Test VN30 stocks get higher position limits."""
        vn30_result = position_sizer.calculate_position_size(
            symbol="VCB",  # VN30 stock
            entry_price=50000,
            stop_loss=47500,
            confidence=70,
        )

        # VN30 should have tier detected
        assert vn30_result.adjustments.get("liquidity_tier") == "VN30"

    def test_minimum_position_check(self, position_sizer):
        """Test minimum viable position enforcement."""
        # Very tight stop would give tiny position
        result = position_sizer.calculate_position_size(
            symbol="VCB",
            entry_price=50000,
            stop_loss=49900,  # 0.2% stop - too tight
            confidence=50,
        )

        # Should either increase to minimum or reject
        if result.is_valid:
            min_value = 100_000_000 * 0.03  # 3% minimum
            assert result.value >= min_value * 0.9  # Allow 10% tolerance

    def test_risk_calculation_accuracy(self, position_sizer):
        """Test risk amount calculation."""
        result = position_sizer.calculate_position_size(
            symbol="VCB",
            entry_price=50000,
            stop_loss=47500,  # 2500 VND risk per share
            confidence=70,
        )

        if result.is_valid:
            expected_risk = result.shares * 2500
            assert abs(result.risk_amount - expected_risk) < 100  # Allow small rounding

    def test_trade_history_recording(self, position_sizer):
        """Test trade history for Kelly calculation."""
        # Record some trades
        for i in range(15):
            position_sizer.record_trade(
                {
                    "symbol": "VCB",
                    "entry_price": 50000,
                    "exit_price": 52000 if i % 2 == 0 else 48000,
                    "pnl": 2000 if i % 2 == 0 else -2000,
                    "pnl_pct": 4 if i % 2 == 0 else -4,
                }
            )

        stats = position_sizer.get_trade_statistics()

        assert stats["total_trades"] == 15
        assert 0 <= stats["win_rate"] <= 100
        assert stats["kelly_pct"] >= 0


class TestIntegration:
    """Integration tests for Entry + Position Sizing."""

    def test_entry_to_position_flow(self):
        """Test complete flow from entry signal to position size."""
        from src.strategies.entry_logic_v3 import SimplifiedEntryLogicV3
        from src.strategies.position_sizing_v3 import calculate_position_with_entry

        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        np.random.seed(42)
        close = 50000 + np.cumsum(np.random.randn(100) * 500)

        df = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(100) * 0.01),
                "high": close * (1 + abs(np.random.randn(100) * 0.02)),
                "low": close * (1 - abs(np.random.randn(100) * 0.02)),
                "close": close,
                "volume": np.random.randint(200000, 800000, 100),
                "rsi": 35 + np.random.rand(100) * 30,
            },
            index=dates,
        )

        # Get entry signal
        entry_logic = SimplifiedEntryLogicV3()
        entry_signal = entry_logic.analyze_entry(
            symbol="VCB",
            df=df,
            market_regime={"regime": "BULL", "tradeable": True, "confidence": 70},
            ml_signal={"signal": "BUY", "confidence": 72},
        )

        if entry_signal.should_enter:
            # Calculate position size
            position = calculate_position_with_entry(
                symbol="VCB",
                entry_signal=entry_signal,
                total_capital=100_000_000,
                market_regime={"regime": "BULL", "tradeable": True},
                df=df,
            )

            # Verify position is valid
            assert position.shares >= 100
            assert position.shares % 100 == 0
            assert position.risk_pct <= 2.0  # Max 2% risk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
