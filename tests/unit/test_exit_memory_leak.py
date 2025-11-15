# -*- coding: utf-8 -*-
"""
Test Exit Strategy Memory Leak Fix
Kiểm tra việc dọn dẹp position_highs dictionary
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.strategies.exit_logic import ImprovedExitStrategy


@pytest.fixture
def exit_strategy():
    """Tạo exit strategy instance"""
    return ImprovedExitStrategy()


@pytest.fixture
def sample_df():
    """Tạo sample DataFrame cho testing"""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    data = {
        "open": [100 + i for i in range(50)],
        "high": [105 + i for i in range(50)],
        "low": [95 + i for i in range(50)],
        "close": [102 + i for i in range(50)],
        "volume": [1000000] * 50,
    }
    return pd.DataFrame(data, index=dates)


def test_position_tracking_added_on_check_exit(exit_strategy, sample_df):
    """Test position được track khi check_exit được gọi"""
    symbol = "VNM"

    # Ban đầu không có tracking
    assert symbol not in exit_strategy.position_highs

    # Gọi check_exit
    decision = exit_strategy.check_exit(
        symbol=symbol,
        entry_price=100,
        current_price=110,
        stop_loss=95,
        take_profit_targets=[105, 110, 115],
        entry_date=datetime.now() - timedelta(days=5),
        df=sample_df,
    )

    # Sau khi check_exit, position được track
    assert symbol in exit_strategy.position_highs
    assert exit_strategy.position_highs[symbol] == 110


def test_clear_position_tracking_removes_symbol(exit_strategy):
    """Test clear_position_tracking xóa symbol khỏi tracking"""
    symbol = "VNM"

    # Thêm tracking
    exit_strategy.position_highs[symbol] = 100
    assert symbol in exit_strategy.position_highs

    # Clear tracking
    exit_strategy.clear_position_tracking(symbol)

    # Kiểm tra đã xóa
    assert symbol not in exit_strategy.position_highs


def test_clear_position_tracking_nonexistent_symbol(exit_strategy):
    """Test clear_position_tracking với symbol không tồn tại"""
    symbol = "NONEXISTENT"

    # Không có lỗi khi clear symbol không tồn tại
    exit_strategy.clear_position_tracking(symbol)
    assert symbol not in exit_strategy.position_highs


def test_get_tracked_positions(exit_strategy):
    """Test lấy danh sách positions đang được track"""
    # Thêm một số positions
    exit_strategy.position_highs["VNM"] = 100
    exit_strategy.position_highs["VIC"] = 200
    exit_strategy.position_highs["HPG"] = 150

    tracked = exit_strategy.get_tracked_positions()

    assert len(tracked) == 3
    assert "VNM" in tracked
    assert "VIC" in tracked
    assert "HPG" in tracked


def test_clear_all_tracking(exit_strategy):
    """Test xóa toàn bộ tracking"""
    # Thêm nhiều positions
    exit_strategy.position_highs["VNM"] = 100
    exit_strategy.position_highs["VIC"] = 200
    exit_strategy.position_highs["HPG"] = 150

    assert len(exit_strategy.position_highs) == 3

    # Clear all
    exit_strategy.clear_all_tracking()

    assert len(exit_strategy.position_highs) == 0


def test_memory_leak_scenario(exit_strategy, sample_df):
    """Test scenario thực tế: nhiều positions được mở và đóng"""
    symbols = ["VNM", "VIC", "HPG", "VCB", "GAS", "MSN", "MWG", "FPT", "VHM", "BID"]

    # Mở 10 positions
    for symbol in symbols:
        exit_strategy.check_exit(
            symbol=symbol,
            entry_price=100,
            current_price=110,
            stop_loss=95,
            take_profit_targets=[105, 110, 115],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df,
        )

    # Kiểm tra tất cả đều được track
    assert len(exit_strategy.position_highs) == 10

    # Đóng 5 positions
    for symbol in symbols[:5]:
        exit_strategy.clear_position_tracking(symbol)

    # Kiểm tra chỉ còn 5
    assert len(exit_strategy.position_highs) == 5

    # Đóng hết
    for symbol in symbols[5:]:
        exit_strategy.clear_position_tracking(symbol)

    # Kiểm tra không còn gì
    assert len(exit_strategy.position_highs) == 0


def test_highest_price_tracking_updates(exit_strategy, sample_df):
    """Test highest price được cập nhật đúng"""
    symbol = "VNM"

    # Check exit lần 1 với giá 100
    exit_strategy.check_exit(
        symbol=symbol,
        entry_price=90,
        current_price=100,
        stop_loss=85,
        take_profit_targets=[95, 100, 105],
        entry_date=datetime.now() - timedelta(days=5),
        df=sample_df,
    )

    assert exit_strategy.position_highs[symbol] == 100

    # Check exit lần 2 với giá cao hơn
    exit_strategy.check_exit(
        symbol=symbol,
        entry_price=90,
        current_price=110,
        stop_loss=85,
        take_profit_targets=[95, 100, 105],
        entry_date=datetime.now() - timedelta(days=5),
        df=sample_df,
    )

    assert exit_strategy.position_highs[symbol] == 110

    # Check exit lần 3 với giá thấp hơn (không update)
    exit_strategy.check_exit(
        symbol=symbol,
        entry_price=90,
        current_price=105,
        stop_loss=85,
        take_profit_targets=[95, 100, 105],
        entry_date=datetime.now() - timedelta(days=5),
        df=sample_df,
    )

    # Vẫn giữ highest
    assert exit_strategy.position_highs[symbol] == 110


def test_partial_exit_keeps_tracking(exit_strategy, sample_df):
    """Test partial exit không xóa tracking (vì position còn)"""
    symbol = "VNM"

    # Thêm tracking
    exit_strategy.position_highs[symbol] = 100

    # Trong thực tế, orchestrator chỉ clear khi FULL exit
    # Partial exit không clear
    # Test này chỉ để document behavior

    assert symbol in exit_strategy.position_highs


def test_full_exit_should_clear_tracking(exit_strategy, sample_df):
    """Test FULL exit nên clear tracking"""
    symbol = "VNM"

    # Thêm tracking
    exit_strategy.position_highs[symbol] = 100

    # Giả lập FULL exit
    exit_strategy.clear_position_tracking(symbol)

    # Tracking đã bị xóa
    assert symbol not in exit_strategy.position_highs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
