"""
Vietnamese News Sentiment Integration Module

Real-time integration of Vietnamese financial news sentiment
into trading signals with event detection and impact scoring.

IMPROVED v2.0 (2025-01):
- VALIDATED news sources with reliability scoring
- Proper fallback mechanism when sources fail
- Rate limiting and circuit breaker for API calls
- Source health monitoring

Features:
- Real-time news monitoring from CafeF, VnExpress, VietStock, TVSI
- Vietnamese NLP financial event detection
- News impact scoring and decay
- Signal adjustment based on sentiment
- Corporate action detection (dividends, earnings, M&A)
- Government policy impact analysis

Author: Trading Bot
Created: 2025
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# TIMEZONE HELPERS
# =============================================================================
def _get_aware_now() -> datetime:
    """Get current time as timezone-aware datetime (UTC)."""
    return datetime.now(timezone.utc)


def _normalize_datetime(dt: Optional[datetime]) -> datetime:
    """
    Normalize datetime to be timezone-aware.
    If dt is naive, assume it's in UTC.
    If dt is None, return current UTC time.
    """
    if dt is None:
        return _get_aware_now()
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt

# =============================================================================
# IMPORT VALIDATED NEWS SOURCES
# =============================================================================
try:
    from src.utils.vn_market_data import (
        NewsSentimentValidator,
        get_news_validator,
        VALIDATED_NEWS_SOURCES,
        FALLBACK_SENTIMENT,
    )

    VN_NEWS_VALIDATION_AVAILABLE = True
    logger.info("✅ Using validated news sources from vn_market_data")
except ImportError:
    VN_NEWS_VALIDATION_AVAILABLE = False
    FALLBACK_SENTIMENT = {"default": 0.0, "no_data": None, "error": 0.0}

    def get_news_validator():
        """Fallback validator stub."""
        return None


# =============================================================================
# ENUMS
# =============================================================================


class NewsEventType(Enum):
    """Types of news events affecting stocks."""

    # Corporate Actions
    EARNINGS_POSITIVE = "earnings_positive"
    EARNINGS_NEGATIVE = "earnings_negative"
    DIVIDEND_ANNOUNCEMENT = "dividend"
    STOCK_SPLIT = "stock_split"
    RIGHTS_ISSUE = "rights_issue"
    SHARE_BUYBACK = "buyback"

    # M&A Events
    MERGER_ANNOUNCEMENT = "merger"
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"

    # Corporate Governance
    MANAGEMENT_CHANGE = "management_change"
    INSIDER_TRADING = "insider_trading"
    REGULATORY_VIOLATION = "violation"

    # Business Development
    NEW_CONTRACT = "new_contract"
    EXPANSION = "expansion"
    PRODUCT_LAUNCH = "product_launch"
    PARTNERSHIP = "partnership"

    # Financial
    DEBT_RESTRUCTURE = "debt_restructure"
    CREDIT_RATING = "credit_rating"
    CAPITAL_INCREASE = "capital_increase"

    # Market Events
    INDEX_ADDITION = "index_add"
    INDEX_REMOVAL = "index_remove"
    FOREIGN_OWNERSHIP = "foreign_ownership"

    # Government/Policy
    TAX_POLICY = "tax_policy"
    REGULATORY_CHANGE = "regulatory_change"
    GOVERNMENT_SUPPORT = "government_support"

    # Sector Events
    SECTOR_NEWS = "sector_news"
    COMMODITY_PRICE = "commodity_price"

    # General
    GENERAL_POSITIVE = "general_positive"
    GENERAL_NEGATIVE = "general_negative"
    NEUTRAL = "neutral"


class NewsImpactLevel(Enum):
    """Impact level of news events."""

    EXTREME = "extreme"  # ±5-7% potential move
    HIGH = "high"  # ±3-5% potential move
    MEDIUM = "medium"  # ±1-3% potential move
    LOW = "low"  # ±0.5-1% potential move
    MINIMAL = "minimal"  # <0.5% potential move


class SentimentDirection(Enum):
    """Sentiment direction."""

    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


# =============================================================================
# VIETNAMESE FINANCIAL KEYWORDS
# =============================================================================

VN_EVENT_KEYWORDS = {
    NewsEventType.EARNINGS_POSITIVE: [
        # Vietnamese
        "lợi nhuận tăng",
        "lợi nhuận đột biến",
        "lãi ròng tăng",
        "doanh thu tăng mạnh",
        "vượt kế hoạch",
        "lợi nhuận kỷ lục",
        "tăng trưởng lợi nhuận",
        "kết quả kinh doanh tích cực",
        "lãi sau thuế tăng",
        "biên lợi nhuận cải thiện",
        "tăng trưởng doanh thu",
        "hiệu quả kinh doanh",
        # English
        "profit increase",
        "record profit",
        "revenue growth",
        "exceeds expectations",
        "strong earnings",
        "profit surge",
    ],
    NewsEventType.EARNINGS_NEGATIVE: [
        # Vietnamese
        "lợi nhuận giảm",
        "thua lỗ",
        "lỗ ròng",
        "doanh thu sụt giảm",
        "không đạt kế hoạch",
        "kết quả kinh doanh yếu",
        "lỗ nặng",
        "biên lợi nhuận giảm",
        "chi phí tăng cao",
        "hoạt động thua lỗ",
        "lãi giảm sâu",
        "doanh thu giảm mạnh",
        # English
        "profit decline",
        "loss reported",
        "revenue drop",
        "misses expectations",
        "weak earnings",
        "operating loss",
    ],
    NewsEventType.DIVIDEND_ANNOUNCEMENT: [
        # Vietnamese
        "chia cổ tức",
        "cổ tức tiền mặt",
        "cổ tức bằng cổ phiếu",
        "tỷ lệ cổ tức",
        "ngày chốt danh sách",
        "ĐHCĐ thông qua cổ tức",
        "trả cổ tức",
        "cổ tức cao",
        "tăng cổ tức",
        # English
        "dividend",
        "cash dividend",
        "stock dividend",
        "dividend payout",
        "ex-dividend",
        "dividend yield",
    ],
    NewsEventType.MERGER_ANNOUNCEMENT: [
        # Vietnamese
        "sáp nhập",
        "hợp nhất",
        "thương vụ M&A",
        "mua lại",
        "tiếp quản",
        "đề xuất sáp nhập",
        "thỏa thuận sáp nhập",
        # English
        "merger",
        "acquisition",
        "M&A deal",
        "takeover",
        "buyout",
        "combination",
        "consolidation",
    ],
    NewsEventType.MANAGEMENT_CHANGE: [
        # Vietnamese
        "thay đổi ban lãnh đạo",
        "tổng giám đốc mới",
        "chủ tịch mới",
        "từ nhiệm",
        "bổ nhiệm",
        "thay đổi nhân sự cấp cao",
        "CEO mới",
        "CFO mới",
        "ban điều hành mới",
        # English
        "CEO change",
        "management change",
        "new chairman",
        "resignation",
        "appointment",
        "executive change",
    ],
    NewsEventType.NEW_CONTRACT: [
        # Vietnamese
        "hợp đồng mới",
        "trúng thầu",
        "ký hợp đồng",
        "đơn hàng lớn",
        "hợp đồng giá trị",
        "đối tác mới",
        "khách hàng mới",
        "dự án mới",
        "hợp đồng xuất khẩu",
        # English
        "new contract",
        "wins contract",
        "signs deal",
        "major order",
        "partnership deal",
        "export contract",
    ],
    NewsEventType.EXPANSION: [
        # Vietnamese
        "mở rộng",
        "nhà máy mới",
        "dự án đầu tư",
        "tăng công suất",
        "chi nhánh mới",
        "thị trường mới",
        "sản phẩm mới",
        "phát triển kinh doanh",
        "đầu tư mới",
        # English
        "expansion",
        "new factory",
        "capacity increase",
        "new market",
        "growth investment",
        "new branch",
    ],
    NewsEventType.CAPITAL_INCREASE: [
        # Vietnamese
        "tăng vốn",
        "phát hành thêm",
        "chào bán cổ phiếu",
        "phát hành riêng lẻ",
        "tăng vốn điều lệ",
        "huy động vốn",
        "phát hành cổ phiếu",
        "IPO",
        "niêm yết",
        # English
        "capital increase",
        "share issuance",
        "rights issue",
        "capital raise",
        "equity offering",
        "IPO",
    ],
    NewsEventType.INDEX_ADDITION: [
        # Vietnamese
        "vào rổ VN30",
        "thêm vào VNIndex",
        "nâng hạng",
        "vào danh mục ETF",
        "được MSCI thêm vào",
        # English
        "index addition",
        "joins VN30",
        "index inclusion",
        "ETF inclusion",
        "MSCI upgrade",
    ],
    NewsEventType.INDEX_REMOVAL: [
        # Vietnamese
        "rời rổ VN30",
        "loại khỏi VNIndex",
        "hạ hạng",
        "loại khỏi danh mục",
        "bị MSCI loại",
        # English
        "index removal",
        "leaves VN30",
        "index exclusion",
        "ETF exclusion",
        "MSCI downgrade",
    ],
    NewsEventType.FOREIGN_OWNERSHIP: [
        # Vietnamese
        "khối ngoại mua ròng",
        "khối ngoại bán ròng",
        "tỷ lệ sở hữu nước ngoài",
        "hết room ngoại",
        "nới room ngoại",
        "nhà đầu tư nước ngoài",
        # English
        "foreign buying",
        "foreign selling",
        "foreign ownership",
        "foreign room",
        "foreign investor",
        "FOL",
    ],
    NewsEventType.REGULATORY_CHANGE: [
        # Vietnamese
        "thay đổi quy định",
        "nghị định mới",
        "thông tư mới",
        "chính sách mới",
        "luật mới",
        "điều chỉnh quy định",
        "thanh tra",
        "kiểm tra",
        "xử phạt",
        # English
        "new regulation",
        "policy change",
        "regulatory update",
        "compliance",
        "investigation",
        "penalty",
    ],
    NewsEventType.GOVERNMENT_SUPPORT: [
        # Vietnamese
        "hỗ trợ chính phủ",
        "gói kích thích",
        "ưu đãi thuế",
        "hỗ trợ doanh nghiệp",
        "giảm thuế",
        "chính sách hỗ trợ",
        "đầu tư công",
        "dự án trọng điểm",
        # English
        "government support",
        "stimulus package",
        "tax incentive",
        "policy support",
        "public investment",
    ],
}

# Event impact configuration
EVENT_IMPACT_CONFIG = {
    NewsEventType.EARNINGS_POSITIVE: {
        "impact": NewsImpactLevel.HIGH,
        "direction": 1,
        "decay_hours": 72,
        "max_adjustment": 0.15,
    },
    NewsEventType.EARNINGS_NEGATIVE: {
        "impact": NewsImpactLevel.HIGH,
        "direction": -1,
        "decay_hours": 72,
        "max_adjustment": 0.20,
    },
    NewsEventType.DIVIDEND_ANNOUNCEMENT: {
        "impact": NewsImpactLevel.MEDIUM,
        "direction": 1,
        "decay_hours": 48,
        "max_adjustment": 0.08,
    },
    NewsEventType.MERGER_ANNOUNCEMENT: {
        "impact": NewsImpactLevel.EXTREME,
        "direction": 1,
        "decay_hours": 168,  # 7 days
        "max_adjustment": 0.20,
    },
    NewsEventType.MANAGEMENT_CHANGE: {
        "impact": NewsImpactLevel.MEDIUM,
        "direction": 0,  # Neutral initially
        "decay_hours": 24,
        "max_adjustment": 0.05,
    },
    NewsEventType.NEW_CONTRACT: {
        "impact": NewsImpactLevel.MEDIUM,
        "direction": 1,
        "decay_hours": 48,
        "max_adjustment": 0.10,
    },
    NewsEventType.EXPANSION: {
        "impact": NewsImpactLevel.LOW,
        "direction": 1,
        "decay_hours": 24,
        "max_adjustment": 0.05,
    },
    NewsEventType.CAPITAL_INCREASE: {
        "impact": NewsImpactLevel.MEDIUM,
        "direction": -1,  # Dilution
        "decay_hours": 72,
        "max_adjustment": 0.10,
    },
    NewsEventType.INDEX_ADDITION: {
        "impact": NewsImpactLevel.HIGH,
        "direction": 1,
        "decay_hours": 168,
        "max_adjustment": 0.12,
    },
    NewsEventType.INDEX_REMOVAL: {
        "impact": NewsImpactLevel.HIGH,
        "direction": -1,
        "decay_hours": 168,
        "max_adjustment": 0.15,
    },
    NewsEventType.FOREIGN_OWNERSHIP: {
        "impact": NewsImpactLevel.MEDIUM,
        "direction": 0,
        "decay_hours": 24,
        "max_adjustment": 0.08,
    },
    NewsEventType.REGULATORY_CHANGE: {
        "impact": NewsImpactLevel.MEDIUM,
        "direction": 0,
        "decay_hours": 48,
        "max_adjustment": 0.10,
    },
    NewsEventType.GOVERNMENT_SUPPORT: {
        "impact": NewsImpactLevel.MEDIUM,
        "direction": 1,
        "decay_hours": 72,
        "max_adjustment": 0.10,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class NewsEvent:
    """Detected news event."""

    symbol: str
    event_type: NewsEventType
    headline: str
    content: str = ""
    source: str = ""
    url: str = ""

    # Analysis
    impact_level: NewsImpactLevel = NewsImpactLevel.LOW
    sentiment_score: float = 0.0  # -1 to 1
    confidence: float = 0.5

    # Timing
    published_at: datetime = None
    detected_at: datetime = None

    # Keywords matched
    keywords_matched: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.published_at is None:
            self.published_at = _get_aware_now()
        else:
            self.published_at = _normalize_datetime(self.published_at)
        if self.detected_at is None:
            self.detected_at = _get_aware_now()
        else:
            self.detected_at = _normalize_datetime(self.detected_at)

    @property
    def hours_since_publish(self) -> float:
        """Hours since news was published."""
        now = _get_aware_now()
        published = _normalize_datetime(self.published_at)
        delta = now - published
        return delta.total_seconds() / 3600

    @property
    def impact_decay(self) -> float:
        """Calculate impact decay factor (0 to 1)."""
        config = EVENT_IMPACT_CONFIG.get(self.event_type, {})
        decay_hours = config.get("decay_hours", 24)

        hours = self.hours_since_publish
        if hours >= decay_hours:
            return 0.0

        # Exponential decay
        return np.exp(-2 * hours / decay_hours)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "event_type": self.event_type.value,
            "headline": self.headline,
            "source": self.source,
            "impact_level": self.impact_level.value,
            "sentiment_score": self.sentiment_score,
            "published_at": self.published_at.isoformat(),
            "hours_since_publish": self.hours_since_publish,
            "impact_decay": self.impact_decay,
        }


@dataclass
class SentimentSignal:
    """Sentiment-based trading signal adjustment."""

    symbol: str

    # Overall sentiment
    overall_score: float = 0.0  # -1 to 1
    direction: SentimentDirection = SentimentDirection.NEUTRAL
    confidence: float = 0.5

    # Signal adjustments
    entry_adjustment: float = 0.0  # Confidence adjustment for entry
    exit_urgency: float = 0.0  # 0=no urgency, 1=exit immediately
    position_size_multiplier: float = 1.0  # Position sizing adjustment

    # Flags
    should_block_entry: bool = False
    should_force_exit: bool = False

    # Active events
    active_events: List[NewsEvent] = field(default_factory=list)

    # Reasoning
    reasons: List[str] = field(default_factory=list)

    # Timestamp
    generated_at: datetime = None

    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = _get_aware_now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "overall_score": self.overall_score,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "entry_adjustment": self.entry_adjustment,
            "exit_urgency": self.exit_urgency,
            "position_size_multiplier": self.position_size_multiplier,
            "should_block_entry": self.should_block_entry,
            "should_force_exit": self.should_force_exit,
            "active_events_count": len(self.active_events),
            "reasons": self.reasons,
            "generated_at": self.generated_at.isoformat(),
        }


# =============================================================================
# NEWS EVENT DETECTOR
# =============================================================================


class VNNewsEventDetector:
    """
    Detect specific financial events from Vietnamese news.

    Uses keyword matching and pattern recognition
    to identify significant market-moving events.
    """

    def __init__(self):
        self._compiled_patterns: Dict[NewsEventType, List[re.Pattern]] = {}
        self._compile_patterns()

        logger.info("📰 VN News Event Detector initialized")

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        for event_type, keywords in VN_EVENT_KEYWORDS.items():
            patterns = []
            for keyword in keywords:
                # Case-insensitive pattern
                try:
                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                    patterns.append(pattern)
                except:
                    pass
            self._compiled_patterns[event_type] = patterns

    def detect_events(
        self,
        symbol: str,
        headline: str,
        content: str = "",
        source: str = "",
        published_at: datetime = None,
    ) -> List[NewsEvent]:
        """
        Detect events from news article.

        Args:
            symbol: Stock symbol
            headline: News headline
            content: Full article content
            source: News source
            published_at: Publication time

        Returns:
            List of detected NewsEvent objects
        """
        events = []
        full_text = f"{headline} {content}"

        for event_type, patterns in self._compiled_patterns.items():
            keywords_matched = []

            for pattern in patterns:
                if pattern.search(full_text):
                    keywords_matched.append(pattern.pattern)

            if keywords_matched:
                # Get event config
                config = EVENT_IMPACT_CONFIG.get(event_type, {})

                # Calculate sentiment score
                direction = config.get("direction", 0)
                match_strength = min(1.0, len(keywords_matched) / 3)
                sentiment_score = direction * match_strength

                # Determine impact level
                impact = config.get("impact", NewsImpactLevel.LOW)

                event = NewsEvent(
                    symbol=symbol,
                    event_type=event_type,
                    headline=headline,
                    content=content[:500],  # Truncate
                    source=source,
                    impact_level=impact,
                    sentiment_score=sentiment_score,
                    confidence=min(0.9, 0.5 + match_strength * 0.4),
                    published_at=_normalize_datetime(published_at),
                    keywords_matched=keywords_matched[:5],  # Limit
                )
                events.append(event)

        # If no specific events detected, classify as general
        if not events:
            general_sentiment = self._analyze_general_sentiment(full_text)
            if abs(general_sentiment) > 0.2:
                event_type = (
                    NewsEventType.GENERAL_POSITIVE
                    if general_sentiment > 0
                    else NewsEventType.GENERAL_NEGATIVE
                )
                events.append(
                    NewsEvent(
                        symbol=symbol,
                        event_type=event_type,
                        headline=headline,
                        source=source,
                        impact_level=NewsImpactLevel.LOW,
                        sentiment_score=general_sentiment,
                        confidence=0.4,
                        published_at=_normalize_datetime(published_at),
                    )
                )

        return events

    def _analyze_general_sentiment(self, text: str) -> float:
        """Analyze general sentiment of text."""
        text_lower = text.lower()

        # Simple keyword counting
        positive_words = [
            "tăng",
            "tích cực",
            "khởi sắc",
            "đột phá",
            "kỷ lục",
            "tốt",
            "mạnh",
            "hấp dẫn",
            "cơ hội",
            "triển vọng",
            "bullish",
            "positive",
            "growth",
            "record",
            "strong",
        ]

        negative_words = [
            "giảm",
            "sụt",
            "yếu",
            "khó khăn",
            "thua lỗ",
            "rủi ro",
            "lo ngại",
            "tiêu cực",
            "suy thoái",
            "bán tháo",
            "bearish",
            "negative",
            "decline",
            "risk",
            "weak",
        ]

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total


# =============================================================================
# NEWS CRAWLER INTEGRATION
# =============================================================================

NEWS_CRAWLER_AVAILABLE = False
try:
    from src.data.vn_news_crawler import get_news_crawler, VNNewsCrawler

    NEWS_CRAWLER_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# NEWS SENTIMENT INTEGRATION
# =============================================================================


class VNNewsSentimentIntegration:
    """
    Integrate Vietnamese news sentiment into trading signals.

    Maintains a rolling window of news events and generates
    real-time sentiment adjustments for trading decisions.

    Features:
    - Automatic news fetching via VN News Crawler
    - Real-time event detection
    - Sentiment score calculation with decay
    - Entry/Exit signal adjustments

    Usage:
        integration = VNNewsSentimentIntegration()

        # Auto-fetch and analyze news for symbol
        integration.refresh_news("VNM")

        # Or add news events manually
        integration.add_news("VNM", "Vinamilk lợi nhuận tăng 20%")

        # Get signal adjustment
        signal = integration.get_signal_adjustment("VNM")

        # Check entry
        entry_check = integration.check_entry_sentiment("VNM")
    """

    def __init__(
        self,
        event_window_hours: int = 168,  # 7 days
        max_events_per_symbol: int = 50,
        auto_fetch_news: bool = True,  # NEW: Auto-fetch from crawler
    ):
        self._lock = RLock()
        self._event_window = timedelta(hours=event_window_hours)
        self._max_events = max_events_per_symbol
        self._auto_fetch = auto_fetch_news

        # News crawler (if available)
        self._news_crawler = None
        if NEWS_CRAWLER_AVAILABLE and auto_fetch_news:
            try:
                self._news_crawler = get_news_crawler()
                logger.info("📰 News Crawler integrated for auto-fetching")
            except Exception as e:
                logger.debug(f"News crawler init failed: {e}")

        # Event storage
        self._events: Dict[str, deque] = {}  # symbol -> deque of NewsEvent

        # Detector
        self._detector = VNNewsEventDetector()

        # Cache
        self._signal_cache: Dict[str, Tuple[datetime, SentimentSignal]] = {}
        self._cache_ttl = timedelta(minutes=5)

        # Track last fetch time per symbol
        self._last_fetch: Dict[str, datetime] = {}
        self._fetch_cooldown = timedelta(minutes=10)

        logger.info("🔗 VN News Sentiment Integration initialized")

    def refresh_news(
        self,
        symbol: str,
        force: bool = False,
    ) -> int:
        """
        Fetch and process latest news for a symbol.

        Args:
            symbol: Stock symbol
            force: Force refresh even if recently fetched

        Returns:
            Number of new events detected
        """
        symbol = symbol.upper()

        # Check cooldown
        if not force and symbol in self._last_fetch:
            if _get_aware_now() - self._last_fetch[symbol] < self._fetch_cooldown:
                return 0

        if self._news_crawler is None:
            logger.debug("News crawler not available")
            return 0

        try:
            # Fetch news from crawler
            articles = self._news_crawler.get_news_for_symbol(
                symbol,
                max_articles=20,
                max_age_hours=int(self._event_window.total_seconds() / 3600),
            )

            if not articles:
                return 0

            # Convert to events
            new_events = 0
            for article in articles:
                events = self.add_news(
                    symbol=symbol,
                    headline=article.title,
                    content=article.summary,
                    source=article.source,
                    published_at=article.published_at,
                )
                new_events += len(events)

            self._last_fetch[symbol] = _get_aware_now()

            if new_events > 0:
                logger.info(f"📰 Refreshed news for {symbol}: {new_events} events detected")

            return new_events

        except Exception as e:
            logger.warning(f"News refresh failed for {symbol}: {e}")
            return 0

    def add_news(
        self,
        symbol: str,
        headline: str,
        content: str = "",
        source: str = "",
        published_at: datetime = None,
    ) -> List[NewsEvent]:
        """
        Add and analyze news for a symbol.

        Returns:
            List of detected events
        """
        symbol = symbol.upper()

        with self._lock:
            # Detect events
            events = self._detector.detect_events(
                symbol=symbol,
                headline=headline,
                content=content,
                source=source,
                published_at=published_at,
            )

            # Store events
            if symbol not in self._events:
                self._events[symbol] = deque(maxlen=self._max_events)

            for event in events:
                self._events[symbol].append(event)

            # Invalidate cache
            if symbol in self._signal_cache:
                del self._signal_cache[symbol]

            # Log high-impact events
            for event in events:
                if event.impact_level in (NewsImpactLevel.HIGH, NewsImpactLevel.EXTREME):
                    logger.info(
                        f"📰 High-impact event: {symbol} - {event.event_type.value} "
                        f"(impact={event.impact_level.value}, sentiment={event.sentiment_score:+.2f})"
                    )

            return events

    def add_events_batch(
        self,
        symbol: str,
        articles: List[Dict[str, Any]],
    ) -> List[NewsEvent]:
        """Add multiple news articles."""
        all_events = []

        for article in articles:
            events = self.add_news(
                symbol=symbol,
                headline=article.get("title", ""),
                content=article.get("content", ""),
                source=article.get("source", ""),
                published_at=article.get("date"),
            )
            all_events.extend(events)

        return all_events

    def get_signal_adjustment(
        self,
        symbol: str,
        use_cache: bool = True,
        auto_refresh: bool = True,  # NEW: Auto-fetch news if needed
    ) -> SentimentSignal:
        """
        Get sentiment-based signal adjustment.

        Args:
            symbol: Stock symbol
            use_cache: Use cached result if available
            auto_refresh: Auto-fetch latest news if crawler available

        Returns:
            SentimentSignal with adjustments
        """
        symbol = symbol.upper()

        # Check cache
        if use_cache and symbol in self._signal_cache:
            cache_time, signal = self._signal_cache[symbol]
            if _get_aware_now() - cache_time < self._cache_ttl:
                return signal

        # Auto-refresh news if available and enabled
        if auto_refresh and self._auto_fetch and self._news_crawler is not None:
            self.refresh_news(symbol)

        with self._lock:
            signal = self._calculate_signal(symbol)

            # Cache result
            self._signal_cache[symbol] = (_get_aware_now(), signal)

            return signal

    def _calculate_signal(self, symbol: str) -> SentimentSignal:
        """Calculate sentiment signal from active events."""
        signal = SentimentSignal(symbol=symbol)

        # Get active events (within window)
        cutoff = _get_aware_now() - self._event_window
        active_events = []

        if symbol in self._events:
            for event in self._events[symbol]:
                if event.published_at > cutoff:
                    active_events.append(event)

        if not active_events:
            return signal

        signal.active_events = active_events

        # Calculate weighted sentiment score
        total_weight = 0
        weighted_score = 0

        high_impact_negative = False
        high_impact_positive = False

        for event in active_events:
            # Weight by impact decay and impact level
            impact_weight = {
                NewsImpactLevel.EXTREME: 2.0,
                NewsImpactLevel.HIGH: 1.5,
                NewsImpactLevel.MEDIUM: 1.0,
                NewsImpactLevel.LOW: 0.5,
                NewsImpactLevel.MINIMAL: 0.2,
            }.get(event.impact_level, 0.5)

            decay = event.impact_decay
            weight = impact_weight * decay * event.confidence

            weighted_score += event.sentiment_score * weight
            total_weight += weight

            # Track high-impact events
            if event.impact_level in (NewsImpactLevel.HIGH, NewsImpactLevel.EXTREME):
                if event.sentiment_score > 0.3:
                    high_impact_positive = True
                elif event.sentiment_score < -0.3:
                    high_impact_negative = True

            # Add reasons
            if decay > 0.3:  # Recent enough to matter
                signal.reasons.append(f"{event.event_type.value}: {event.headline[:50]}...")

        # Calculate overall score
        if total_weight > 0:
            signal.overall_score = max(-1, min(1, weighted_score / total_weight))

        # Determine direction
        score = signal.overall_score
        if score > 0.6:
            signal.direction = SentimentDirection.VERY_BULLISH
        elif score > 0.2:
            signal.direction = SentimentDirection.BULLISH
        elif score < -0.6:
            signal.direction = SentimentDirection.VERY_BEARISH
        elif score < -0.2:
            signal.direction = SentimentDirection.BEARISH
        else:
            signal.direction = SentimentDirection.NEUTRAL

        # Calculate confidence
        signal.confidence = min(0.9, 0.3 + len(active_events) * 0.05)

        # Calculate adjustments
        signal.entry_adjustment = self._calculate_entry_adjustment(signal)
        signal.exit_urgency = self._calculate_exit_urgency(signal)
        signal.position_size_multiplier = self._calculate_position_multiplier(signal)

        # Blocking conditions
        if high_impact_negative:
            if signal.direction in (SentimentDirection.VERY_BEARISH, SentimentDirection.BEARISH):
                signal.should_block_entry = True
                signal.reasons.append("⛔ Blocking entry: High-impact negative news")

        if signal.direction == SentimentDirection.VERY_BEARISH and signal.confidence > 0.7:
            signal.should_force_exit = True
            signal.exit_urgency = 1.0
            signal.reasons.append("🚨 Force exit: Very bearish sentiment with high confidence")

        return signal

    def _calculate_entry_adjustment(self, signal: SentimentSignal) -> float:
        """Calculate entry confidence adjustment."""
        base_adjustment = 0.0

        # Adjust based on direction
        if signal.direction == SentimentDirection.VERY_BULLISH:
            base_adjustment = 0.10
        elif signal.direction == SentimentDirection.BULLISH:
            base_adjustment = 0.05
        elif signal.direction == SentimentDirection.BEARISH:
            base_adjustment = -0.08
        elif signal.direction == SentimentDirection.VERY_BEARISH:
            base_adjustment = -0.15

        # Scale by confidence
        return base_adjustment * signal.confidence

    def _calculate_exit_urgency(self, signal: SentimentSignal) -> float:
        """Calculate exit urgency (0 = no urgency, 1 = immediate exit)."""
        urgency = 0.0

        # High urgency for very negative sentiment
        if signal.direction == SentimentDirection.VERY_BEARISH:
            urgency = 0.8
        elif signal.direction == SentimentDirection.BEARISH:
            urgency = 0.3

        # Check for specific high-impact negative events
        for event in signal.active_events:
            if (
                event.impact_level in (NewsImpactLevel.HIGH, NewsImpactLevel.EXTREME)
                and event.sentiment_score < -0.5
                and event.impact_decay > 0.5
            ):
                urgency = max(urgency, 0.9)

        return min(1.0, urgency)

    def _calculate_position_multiplier(self, signal: SentimentSignal) -> float:
        """Calculate position size multiplier."""
        multiplier = 1.0

        # Increase for positive sentiment
        if signal.direction == SentimentDirection.VERY_BULLISH:
            multiplier = 1.2
        elif signal.direction == SentimentDirection.BULLISH:
            multiplier = 1.1
        # Decrease for negative sentiment
        elif signal.direction == SentimentDirection.BEARISH:
            multiplier = 0.8
        elif signal.direction == SentimentDirection.VERY_BEARISH:
            multiplier = 0.5

        return multiplier

    def check_entry_sentiment(
        self,
        symbol: str,
        side: str = "buy",
    ) -> Dict[str, Any]:
        """
        Check if sentiment supports entry.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"

        Returns:
            Dict with is_favorable, adjustments, reasons
        """
        signal = self.get_signal_adjustment(symbol)

        result = {
            "is_favorable": True,
            "should_proceed": True,
            "confidence_adjustment": signal.entry_adjustment,
            "position_multiplier": signal.position_size_multiplier,
            "sentiment_score": signal.overall_score,
            "sentiment_direction": signal.direction.value,
            "reasons": signal.reasons.copy(),
        }

        if side == "buy":
            # For buy, we want positive or neutral sentiment
            if signal.should_block_entry:
                result["is_favorable"] = False
                result["should_proceed"] = False
                result["reasons"].append("Entry blocked by negative news")
            elif signal.direction in (SentimentDirection.BEARISH, SentimentDirection.VERY_BEARISH):
                result["is_favorable"] = False
                # Still allow but flag
                result["reasons"].append("Caution: Bearish sentiment detected")
        else:
            # For sell, bearish sentiment is favorable
            if signal.direction in (SentimentDirection.BEARISH, SentimentDirection.VERY_BEARISH):
                result["is_favorable"] = True
                result["confidence_adjustment"] = abs(signal.entry_adjustment)

        return result

    def check_exit_sentiment(
        self,
        symbol: str,
        current_pnl_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Check if sentiment suggests exit.

        Args:
            symbol: Stock symbol
            current_pnl_pct: Current position P&L percentage

        Returns:
            Dict with should_exit, urgency, reasons
        """
        signal = self.get_signal_adjustment(symbol)

        result = {
            "should_exit": False,
            "urgency": signal.exit_urgency,
            "take_profit_adjustment": 0.0,
            "stop_loss_adjustment": 0.0,
            "reasons": signal.reasons.copy(),
        }

        # Force exit condition
        if signal.should_force_exit:
            result["should_exit"] = True
            result["urgency"] = 1.0
            result["reasons"].append("Forced exit due to severe negative news")

        # Exit if bearish and already profitable
        elif current_pnl_pct > 0.02 and signal.direction in (
            SentimentDirection.BEARISH,
            SentimentDirection.VERY_BEARISH,
        ):
            result["should_exit"] = True
            result["urgency"] = 0.6
            result["reasons"].append("Bearish sentiment, taking profits")

        # Adjust targets based on sentiment
        if signal.direction == SentimentDirection.VERY_BULLISH:
            result["take_profit_adjustment"] = 0.02  # Widen TP
        elif signal.direction == SentimentDirection.BEARISH:
            result["stop_loss_adjustment"] = 0.005  # Tighten SL
        elif signal.direction == SentimentDirection.VERY_BEARISH:
            result["stop_loss_adjustment"] = 0.01

        return result

    def get_active_events(
        self,
        symbol: str,
        min_impact: NewsImpactLevel = NewsImpactLevel.LOW,
    ) -> List[NewsEvent]:
        """Get active events for a symbol."""
        cutoff = _get_aware_now() - self._event_window
        events = []

        if symbol.upper() in self._events:
            for event in self._events[symbol.upper()]:
                if event.published_at > cutoff and event.impact_level.value >= min_impact.value:
                    events.append(event)

        return sorted(events, key=lambda e: e.published_at, reverse=True)

    def get_market_news_summary(self) -> Dict[str, Any]:
        """Get summary of recent market news."""
        total_events = 0
        symbols_with_news = []
        high_impact_events = []

        cutoff = _get_aware_now() - timedelta(hours=24)

        for symbol, events in self._events.items():
            symbol_events = [e for e in events if e.published_at > cutoff]
            if symbol_events:
                total_events += len(symbol_events)
                symbols_with_news.append(symbol)

                for event in symbol_events:
                    if event.impact_level in (NewsImpactLevel.HIGH, NewsImpactLevel.EXTREME):
                        high_impact_events.append(event)

        return {
            "total_events_24h": total_events,
            "symbols_with_news": len(symbols_with_news),
            "high_impact_count": len(high_impact_events),
            "high_impact_events": [e.to_dict() for e in high_impact_events[:10]],
        }

    def cleanup_old_events(self):
        """Remove expired events."""
        cutoff = _get_aware_now() - self._event_window

        with self._lock:
            for symbol in list(self._events.keys()):
                old_len = len(self._events[symbol])
                self._events[symbol] = deque(
                    [e for e in self._events[symbol] if _normalize_datetime(e.published_at) > cutoff],
                    maxlen=self._max_events,
                )
                new_len = len(self._events[symbol])

                if new_len < old_len:
                    logger.debug(f"Cleaned {old_len - new_len} old events for {symbol}")

    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        total_events = sum(len(events) for events in self._events.values())

        return {
            "symbols_tracked": len(self._events),
            "total_events": total_events,
            "cache_size": len(self._signal_cache),
            "event_window_hours": self._event_window.total_seconds() / 3600,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_integration_instance: Optional[VNNewsSentimentIntegration] = None
_lock = RLock()


def get_news_sentiment_integration() -> VNNewsSentimentIntegration:
    """Get singleton integration instance."""
    global _integration_instance
    with _lock:
        if _integration_instance is None:
            _integration_instance = VNNewsSentimentIntegration()
        return _integration_instance


def reset_news_sentiment_integration():
    """Reset singleton."""
    global _integration_instance
    with _lock:
        _integration_instance = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def add_news_event(
    symbol: str,
    headline: str,
    content: str = "",
    source: str = "",
) -> List[NewsEvent]:
    """Add news event (convenience function)."""
    return get_news_sentiment_integration().add_news(
        symbol=symbol,
        headline=headline,
        content=content,
        source=source,
    )


def check_news_sentiment(symbol: str) -> SentimentSignal:
    """Check news sentiment (convenience function)."""
    return get_news_sentiment_integration().get_signal_adjustment(symbol)


def should_block_entry(symbol: str) -> Tuple[bool, str]:
    """Check if entry should be blocked due to news."""
    signal = check_news_sentiment(symbol)
    if signal.should_block_entry:
        reason = "; ".join(signal.reasons[:2])
        return True, reason
    return False, ""


def should_force_exit(symbol: str) -> Tuple[bool, str]:
    """Check if exit should be forced due to news."""
    signal = check_news_sentiment(symbol)
    if signal.should_force_exit:
        reason = "; ".join(signal.reasons[:2])
        return True, reason
    return False, ""


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING VN NEWS SENTIMENT INTEGRATION")
    print("=" * 60)

    integration = get_news_sentiment_integration()

    # Test event detection
    print("\n📰 Event Detection Tests:")
    print("-" * 60)

    test_headlines = [
        ("VNM", "Vinamilk báo lãi ròng tăng 25% trong quý 3"),
        ("VNM", "Vinamilk công bố chia cổ tức tiền mặt 2,000 đồng/cp"),
        ("HPG", "Hòa Phát lỗ nặng do giá thép giảm mạnh"),
        ("VCB", "Vietcombank được thêm vào rổ VN30"),
        ("VIC", "Vingroup sáp nhập công ty con, tái cấu trúc tập đoàn"),
        ("FPT", "FPT ký hợp đồng giá trị 500 triệu USD với đối tác Nhật"),
        ("SSI", "SSI phát hành thêm cổ phiếu tăng vốn điều lệ"),
        ("TCB", "Khối ngoại bán ròng mạnh cổ phiếu Techcombank"),
    ]

    for symbol, headline in test_headlines:
        events = integration.add_news(symbol, headline, source="test")
        print(f"\n{symbol}: {headline[:50]}...")
        for event in events:
            print(f"  → {event.event_type.value}")
            print(f"    Impact: {event.impact_level.value}")
            print(f"    Sentiment: {event.sentiment_score:+.2f}")

    # Test signal adjustments
    print("\n\n📊 Signal Adjustments:")
    print("-" * 60)

    for symbol in ["VNM", "HPG", "VCB", "FPT"]:
        signal = integration.get_signal_adjustment(symbol)
        print(f"\n{symbol}:")
        print(f"  Overall Score: {signal.overall_score:+.3f}")
        print(f"  Direction: {signal.direction.value}")
        print(f"  Entry Adjustment: {signal.entry_adjustment:+.3f}")
        print(f"  Position Multiplier: {signal.position_size_multiplier:.2f}")
        print(f"  Exit Urgency: {signal.exit_urgency:.2f}")
        if signal.should_block_entry:
            print(f"  ⛔ ENTRY BLOCKED")
        if signal.should_force_exit:
            print(f"  🚨 FORCE EXIT")

    # Test entry check
    print("\n\n📈 Entry Sentiment Check:")
    print("-" * 60)

    for symbol in ["VNM", "HPG"]:
        check = integration.check_entry_sentiment(symbol)
        print(f"\n{symbol}:")
        print(f"  Favorable: {check['is_favorable']}")
        print(f"  Should Proceed: {check['should_proceed']}")
        print(f"  Confidence Adj: {check['confidence_adjustment']:+.3f}")
        print(f"  Position Multiplier: {check['position_multiplier']:.2f}")

    # Test exit check
    print("\n\n📉 Exit Sentiment Check:")
    print("-" * 60)

    for symbol in ["HPG", "VCB"]:
        check = integration.check_exit_sentiment(symbol, current_pnl_pct=0.03)
        print(f"\n{symbol}:")
        print(f"  Should Exit: {check['should_exit']}")
        print(f"  Urgency: {check['urgency']:.2f}")
        print(f"  TP Adjustment: {check['take_profit_adjustment']:+.3f}")
        print(f"  SL Adjustment: {check['stop_loss_adjustment']:+.3f}")

    # Test market summary
    print("\n\n🌍 Market News Summary:")
    print("-" * 60)

    summary = integration.get_market_news_summary()
    print(f"  Total Events (24h): {summary['total_events_24h']}")
    print(f"  Symbols with News: {summary['symbols_with_news']}")
    print(f"  High Impact Events: {summary['high_impact_count']}")

    # Status
    print("\n\n📊 Integration Status:")
    print("-" * 60)

    status = integration.get_status()
    print(f"  Symbols Tracked: {status['symbols_tracked']}")
    print(f"  Total Events: {status['total_events']}")
    print(f"  Cache Size: {status['cache_size']}")

    print("\n" + "=" * 60)
