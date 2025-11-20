# -*- coding: utf-8 -*-
"""
Tests for Fundamental Data Provider
"""

import pytest
from datetime import datetime
from src.data.fundamental_provider import FundamentalDataProvider, get_fundamental_provider


class TestFundamentalDataProvider:
    """Test fundamental data provider"""

    def test_provider_initialization(self):
        """Test provider can be initialized"""
        provider = FundamentalDataProvider(cache_ttl_hours=24)
        assert provider is not None
        assert provider.cache_ttl_hours == 24

    def test_get_fundamentals_invalid_symbol(self):
        """Test getting fundamentals for invalid symbol"""
        provider = FundamentalDataProvider()
        result = provider.get_fundamentals("INVALID_SYMBOL_XYZ")
        # Should return None for invalid symbol
        assert result is None

    def test_get_fundamentals_valid_symbol(self):
        """Test getting fundamentals for valid symbol"""
        provider = FundamentalDataProvider()

        # Test with a real Vietnamese stock (VNM - Vinamilk)
        result = provider.get_fundamentals("VNM")

        # May return None if API is down or symbol not found
        # But should not raise exception
        if result:
            # If data is available, check structure
            assert isinstance(result, dict)
            assert "pe_ratio" in result
            assert "pb_ratio" in result
            assert "roe" in result
            assert "debt_ratio" in result
            assert "timestamp" in result

    def test_cache_functionality(self):
        """Test that cache works"""
        provider = FundamentalDataProvider(cache_ttl_hours=1)

        # First call
        result1 = provider.get_fundamentals("VNM")

        # Second call should use cache
        result2 = provider.get_fundamentals("VNM")

        # Both should return same result (from cache)
        assert result1 == result2

    def test_clear_cache(self):
        """Test cache clearing"""
        provider = FundamentalDataProvider()

        # Add to cache
        provider.get_fundamentals("VNM")

        # Clear cache
        provider.clear_cache()

        # Cache should be empty
        assert len(provider._cache) == 0

    def test_singleton_pattern(self):
        """Test singleton pattern works"""
        provider1 = get_fundamental_provider()
        provider2 = get_fundamental_provider()

        # Should be same instance
        assert provider1 is provider2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
