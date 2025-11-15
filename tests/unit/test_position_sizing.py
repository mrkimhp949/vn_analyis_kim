# -*- coding: utf-8 -*-
"""
Unit tests for ConservativePositionSizer
"""
import pytest
from improved_position_sizing import ConservativePositionSizer


class TestConservativePositionSizer:
    
    @pytest.fixture
    def sizer(self):
        return ConservativePositionSizer(
            total_capital=100_000_000,
            max_risk_per_trade=0.02,
            max_position_size=0.10,
            max_total_exposure=0.60
        )
    
    def test_initialization(self, sizer):
        """Test sizer initialization"""
        assert sizer.total_capital == 100_000_000
        assert sizer.max_risk_per_trade == 0.02
        assert sizer.max_position_size == 0.10
    
    def test_calculate_position_size_basic(self, sizer):
        """Test basic position size calculation"""
        position = sizer.calculate_position_size(
            symbol='VNM',
            entry_price=80000,
            stop_loss=76000,  # 5% risk
            confidence=75,
            signal_strength='STRONG'
        )
        
        assert position.shares > 0
        assert position.shares % 100 == 0  # Multiple of 100
        assert position.risk_percent <= 2.0  # Max 2% risk
        assert position.position_percent <= 10.0  # Max 10% position
    
    def test_invalid_stop_loss(self, sizer):
        """Test with invalid stop loss"""
        position = sizer.calculate_position_size(
            symbol='VNM',
            entry_price=80000,
            stop_loss=80000,  # Same as entry
            confidence=75,
            signal_strength='STRONG'
        )
        
        assert position.shares == 0
        assert 'Stop loss không hợp lệ' in position.warnings[0]
    
    def test_max_exposure_reached(self, sizer):
        """Test when max exposure is reached"""
        # Add existing positions
        sizer.current_positions = {
            'VCB': {'shares': 600, 'entry_price': 90000, 'current_price': 90000},
            'FPT': {'shares': 800, 'entry_price': 70000, 'current_price': 70000}
        }
        # Total: 60M (60% of capital)
        
        position = sizer.calculate_position_size(
            symbol='VNM',
            entry_price=80000,
            stop_loss=76000,
            confidence=75,
            signal_strength='STRONG'
        )
        
        assert position.shares == 0
        assert 'Exposure đã đạt limit' in position.warnings[0]
    
    def test_risk_multiplier_bull_market(self, sizer):
        """Test risk multiplier in bull market"""
        market_regime = {'regime': 'BULL', 'tradeable': True}
        
        multiplier = sizer._calculate_risk_multiplier(
            confidence=80,
            signal_strength='VERY_STRONG',
            market_regime=market_regime
        )
        
        assert multiplier > 1.0  # Should be aggressive in bull
        assert multiplier <= 1.2  # But capped at 1.2
    
    def test_risk_multiplier_bear_market(self, sizer):
        """Test risk multiplier in bear market"""
        market_regime = {'regime': 'BEAR', 'tradeable': False}
        
        multiplier = sizer._calculate_risk_multiplier(
            confidence=70,
            signal_strength='MODERATE',
            market_regime=market_regime
        )
        
        assert multiplier < 1.0  # Should be conservative in bear
        assert multiplier >= 0.5  # But not too small
    
    def test_dca_entries(self, sizer):
        """Test DCA entry calculation"""
        entries = sizer._calculate_dca_entries(80000, 1000)
        
        assert len(entries) == 3
        assert entries[0]['level'] == 1
        assert entries[0]['price'] < 80000  # Lower than base
        assert entries[0]['percent'] == 50
        assert sum(e['shares'] for e in entries) <= 1000
    
    def test_add_and_update_position(self, sizer):
        """Test adding and updating position"""
        sizer.add_position('VNM', 500, 80000)
        
        assert 'VNM' in sizer.current_positions
        assert sizer.current_positions['VNM']['shares'] == 500
        
        # Update price
        sizer.update_position_price('VNM', 82000)
        assert sizer.current_positions['VNM']['current_price'] == 82000
        assert sizer.current_positions['VNM']['unrealized_pnl'] == 1000000
    
    def test_close_position(self, sizer):
        """Test closing position"""
        sizer.add_position('VNM', 500, 80000)
        sizer.close_position('VNM', 85000)
        
        assert 'VNM' not in sizer.current_positions
        assert sizer.realized_pnl == 2500000  # 500 * (85000 - 80000)
    
    def test_portfolio_status(self, sizer):
        """Test portfolio status calculation"""
        sizer.add_position('VNM', 500, 80000)
        sizer.add_position('VCB', 300, 90000)
        sizer.update_position_price('VNM', 82000)
        sizer.update_position_price('VCB', 91000)
        
        status = sizer.get_portfolio_status()
        
        assert status['num_positions'] == 2
        assert status['invested'] > 0
        assert status['unrealized_pnl'] > 0
        assert status['exposure_percent'] < 100
