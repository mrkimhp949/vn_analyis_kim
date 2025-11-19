"""
Repositories Module

Implements Repository Pattern for data access.

Benefits:
- Abstracts database access from business logic
- Easy to test (can mock repositories)
- Centralized query logic
- Easy to optimize queries
- Can swap database implementations
"""

from src.repositories.position_repository import PositionRepository
from src.repositories.trade_repository import TradeRepository

__all__ = ["PositionRepository", "TradeRepository"]
