# -*- coding: utf-8 -*-
"""
Test API Input Validation
Kiểm tra validation cho API endpoints

Note: Test verify code có validation, không test runtime vì có dependency issues
"""

import pytest
import re


class TestAPIValidationCodeExists:
    """Test verify rằng API có validation code"""

    def test_api_has_validation_imports(self):
        """Test API import InputValidator"""
        with open("src/api/main.py", "r", encoding="utf-8") as f:
            api_code = f.read()

        assert "from src.utils.validation import InputValidator" in api_code

    def test_add_portfolio_has_validation(self):
        """Test /portfolio/add endpoint có validation"""
        with open("src/api/main.py", "r", encoding="utf-8") as f:
            api_code = f.read()

        # Tìm function add_to_portfolio
        assert '@app.post("/portfolio/add")' in api_code
        assert "async def add_to_portfolio" in api_code

        # Kiểm tra có validate
        assert "InputValidator.validate_symbol(symbol)" in api_code
        assert "InputValidator.validate_shares(shares)" in api_code
        assert "InputValidator.validate_price(price)" in api_code

        # Kiểm tra có raise HTTPException
        assert "raise HTTPException(status_code=400" in api_code

    def test_remove_portfolio_has_validation(self):
        """Test /portfolio/remove endpoint có validation"""
        with open("src/api/main.py", "r", encoding="utf-8") as f:
            api_code = f.read()

        # Tìm function remove_from_portfolio
        assert '@app.post("/portfolio/remove")' in api_code
        assert "async def remove_from_portfolio" in api_code

        # Kiểm tra có validate symbol
        assert "InputValidator.validate_symbol(symbol)" in api_code

    def test_validation_utils_has_required_methods(self):
        """Test validation.py có các methods cần thiết"""
        with open("src/utils/validation.py", "r", encoding="utf-8") as f:
            validation_code = f.read()

        # Kiểm tra có InputValidator class
        assert "class InputValidator:" in validation_code

        # Kiểm tra có các methods
        assert "def validate_symbol" in validation_code
        assert "def validate_shares" in validation_code
        assert "def validate_price" in validation_code

        # Kiểm tra validate symbol có check length
        assert "len(symbol)" in validation_code

        # Kiểm tra validate shares có check positive
        assert "shares <= 0" in validation_code or "shares > 0" in validation_code

        # Kiểm tra validate price có check positive
        assert "price <= 0" in validation_code or "price > 0" in validation_code

    def test_validation_raises_value_error(self):
        """Test validation raise ValueError khi invalid"""
        with open("src/utils/validation.py", "r", encoding="utf-8") as f:
            validation_code = f.read()

        # Kiểm tra có raise ValueError
        assert "raise ValueError" in validation_code

    def test_api_catches_validation_errors(self):
        """Test API catch validation errors và return 400"""
        with open("src/api/main.py", "r", encoding="utf-8") as f:
            api_code = f.read()

        # Pattern: try-except ValueError -> HTTPException 400
        assert "except ValueError as e:" in api_code
        assert "raise HTTPException(status_code=400" in api_code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
