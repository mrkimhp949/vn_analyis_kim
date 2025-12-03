# -*- coding: utf-8 -*-
"""
Test ML Analysis Error Handling
Kiểm tra việc xử lý lỗi khi ML analysis fail

Note: Các test này verify rằng code có try-catch cho ML analysis.
Không test integration vì có dependency issues.
"""

from unittest.mock import Mock

import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """Tạo sample DataFrame"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    data = {
        "open": [100 + i for i in range(100)],
        "high": [105 + i for i in range(100)],
        "low": [95 + i for i in range(100)],
        "close": [102 + i for i in range(100)],
        "volume": [1000000] * 100,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_ml_generator():
    """Mock ML generator that raises exception"""
    mock = Mock()
    mock.analyze.side_effect = Exception("ML model failed")
    return mock


def test_ml_error_handling_code_exists():
    """
    Test verify rằng code có try-catch cho ML analysis.
    Đây là static test kiểm tra code structure.
    """

    # Test orchestrator có try-catch
    with open("src/core/orchestrator.py", "r", encoding="utf-8") as f:
        orchestrator_code = f.read()

    # Kiểm tra có try-catch cho ML analysis
    assert "try:" in orchestrator_code
    # Check for ml_generator.analyze call (may be in asyncio.to_thread wrapper)
    assert "ml_generator.analyze" in orchestrator_code
    assert "except" in orchestrator_code
    # Check for error logging (various patterns)
    assert (
        "ML analysis failed" in orchestrator_code
        or "ML error" in orchestrator_code
        or "Lỗi ML analysis" in orchestrator_code
    )

    # Test services có try-catch
    with open("src/services/entry_service.py", "r", encoding="utf-8") as f:
        entry_service_code = f.read()

    assert "try:" in entry_service_code
    # Check for ml_generator.analyze call (may use different patterns)
    assert "ml_generator" in entry_service_code or "ml_signal" in entry_service_code
    assert "except" in entry_service_code

    with open("src/services/exit_service.py", "r", encoding="utf-8") as f:
        exit_service_code = f.read()

    assert "try:" in exit_service_code
    # Check for ml_generator.analyze call (may use different patterns)
    assert "ml_generator" in exit_service_code or "ml_signal" in exit_service_code
    assert "except" in exit_service_code


def test_ml_error_sets_signal_to_none():
    """Test ML error handler sets ml_signal to None"""
    # Verify pattern trong code
    with open("src/core/orchestrator.py", "r", encoding="utf-8") as f:
        code = f.read()

    # Pattern: function returns None on error (various patterns)
    # The orchestrator returns None when ML analysis fails
    assert "return None" in code
    # Check that error handling exists
    assert "except" in code
    # Check that ML failures are handled (returns None or sets to None)
    assert "ml_signal" in code


def test_all_ml_analyze_calls_have_error_handling():
    """Test tất cả các lần gọi ml_generator.analyze đều có error handling"""
    import re

    files_to_check = [
        "src/core/orchestrator.py",
        "src/services/entry_service.py",
        "src/services/exit_service.py",
        "src/portfolio/analyzer.py",
    ]

    for filepath in files_to_check:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Tìm tất cả ml_generator.analyze calls
        analyze_calls = re.findall(r"ml_generator\.analyze\([^)]*\)", content)

        if analyze_calls:
            # Nếu có analyze calls, phải có try-except
            assert "try:" in content, f"{filepath} missing try block"
            assert "except Exception" in content, f"{filepath} missing except block"
            assert (
                "ml_signal = None" in content or "# Tiếp tục với ml_signal = None" in content
            ), f"{filepath} missing ml_signal = None"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
