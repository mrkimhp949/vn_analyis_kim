"""
Unit tests for Portfolio Manager - Validation Logic Only
"""

import os
import sys

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPortfolioManagerValidation:
    """Test PortfolioManager validation logic (without database)"""

    def test_validation_imports(self):
        """Test that portfolio manager imports correctly"""
        from exceptions import PortfolioError
        from portfolio_manager import PortfolioManager

        # Just verify imports work
        assert PortfolioManager is not None
        assert PortfolioError is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
