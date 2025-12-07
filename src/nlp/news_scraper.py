# -*- coding: utf-8 -*-
"""
Vietnamese Financial News Scraper

Supports:
- CafeF (cafef.vn)
- VnExpress Business (vnexpress.net/kinh-doanh)
- VietStock (vietstock.vn)
- TCBS News

Dependencies:
    pip install requests beautifulsoup4 aiohttp
"""

import asyncio
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# Optional imports
try:
    import requests
    from bs4 import BeautifulSoup

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests/beautifulsoup4 not installed")

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class NewsArticle:
    """Container for a news article."""

    id: str
    title: str
    content: str
    url: str
    source: str
    published_at: Optional[datetime] = None
    symbols: List[str] = field(default_factory=list)
    category: str = ""
    author: str = ""
    scraped_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "symbols": self.symbols,
            "category": self.category,
        }


# =============================================================================
# BASE SCRAPER
# =============================================================================


class BaseNewsScraper(ABC):
    """Base class for news scrapers."""

    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize scraper.

        Args:
            rate_limit: Minimum seconds between requests
        """
        self.rate_limit = rate_limit
        self._last_request_time = 0
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        }

    @abstractmethod
    def get_source_name(self) -> str:
        """Get source name."""
        pass

    @abstractmethod
    def scrape_latest(self, limit: int = 20) -> List[NewsArticle]:
        """Scrape latest news articles."""
        pass

    @abstractmethod
    def scrape_by_symbol(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """Scrape news for a specific stock symbol."""
        pass

    def _rate_limit_wait(self):
        """Wait to respect rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content."""
        if not REQUESTS_AVAILABLE:
            logger.error("requests not available")
            return None

        self._rate_limit_wait()

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _generate_id(self, url: str) -> str:
        """Generate unique ID from URL."""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        # Pattern for Vietnamese stock symbols (3 uppercase letters)
        pattern = r"\b([A-Z]{3})\b"
        matches = re.findall(pattern, text)

        # Filter to likely stock symbols
        common_words = {"THE", "AND", "FOR", "VND", "USD", "CEO", "CFO", "IPO"}
        symbols = [m for m in matches if m not in common_words]

        return list(set(symbols))


# =============================================================================
# CAFEF SCRAPER
# =============================================================================


class CafeFScraper(BaseNewsScraper):
    """Scraper for CafeF (cafef.vn)."""

    BASE_URL = "https://cafef.vn"
    STOCK_URL = "https://cafef.vn/thi-truong-chung-khoan.chn"

    def get_source_name(self) -> str:
        return "CafeF"

    def scrape_latest(self, limit: int = 20) -> List[NewsArticle]:
        """Scrape latest stock market news from CafeF."""
        articles = []

        html = self._fetch_page(self.STOCK_URL)
        if not html:
            return articles

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Find news items
            news_items = soup.select(".tlitem, .box-category-item, .item-news")[:limit]

            for item in news_items:
                try:
                    # Extract title and URL
                    title_elem = item.select_one("h3 a, .title a, a.title")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    url = urljoin(self.BASE_URL, title_elem.get("href", ""))

                    if not title or not url:
                        continue

                    # Extract summary/content
                    summary_elem = item.select_one(".sapo, .summary, p")
                    content = summary_elem.get_text(strip=True) if summary_elem else title

                    # Extract date
                    date_elem = item.select_one(".time, .date, time")
                    published_at = self._parse_date(date_elem.get_text() if date_elem else "")

                    article = NewsArticle(
                        id=self._generate_id(url),
                        title=title,
                        content=content,
                        url=url,
                        source=self.get_source_name(),
                        published_at=published_at,
                        symbols=self._extract_symbols(title + " " + content),
                        category="stock_market",
                    )
                    articles.append(article)

                except Exception as e:
                    logger.debug(f"Error parsing CafeF item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error scraping CafeF: {e}")

        return articles

    def scrape_by_symbol(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """Scrape news for a specific symbol from CafeF."""
        url = f"https://cafef.vn/tim-kiem.chn?keywords={symbol}"
        articles = []

        html = self._fetch_page(url)
        if not html:
            return articles

        try:
            soup = BeautifulSoup(html, "html.parser")
            news_items = soup.select(".tlitem, .item-news")[:limit]

            for item in news_items:
                try:
                    title_elem = item.select_one("h3 a, .title a")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    article_url = urljoin(self.BASE_URL, title_elem.get("href", ""))

                    summary_elem = item.select_one(".sapo, p")
                    content = summary_elem.get_text(strip=True) if summary_elem else title

                    article = NewsArticle(
                        id=self._generate_id(article_url),
                        title=title,
                        content=content,
                        url=article_url,
                        source=self.get_source_name(),
                        symbols=[symbol],
                        category="stock_news",
                    )
                    articles.append(article)

                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Error scraping CafeF for {symbol}: {e}")

        return articles

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse CafeF date format."""
        if not date_str:
            return None

        try:
            # Format: "12/06/2024 - 10:30"
            date_str = date_str.strip()
            for fmt in ["%d/%m/%Y - %H:%M", "%d/%m/%Y", "%H:%M %d/%m/%Y"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
        except Exception:
            pass

        return None


# =============================================================================
# VNEXPRESS SCRAPER
# =============================================================================


class VnExpressScraper(BaseNewsScraper):
    """Scraper for VnExpress Business."""

    BASE_URL = "https://vnexpress.net"
    BUSINESS_URL = "https://vnexpress.net/kinh-doanh/chung-khoan"

    def get_source_name(self) -> str:
        return "VnExpress"

    def scrape_latest(self, limit: int = 20) -> List[NewsArticle]:
        """Scrape latest business news from VnExpress."""
        articles = []

        html = self._fetch_page(self.BUSINESS_URL)
        if not html:
            return articles

        try:
            soup = BeautifulSoup(html, "html.parser")

            news_items = soup.select("article.item-news, .item-news-common")[:limit]

            for item in news_items:
                try:
                    title_elem = item.select_one("h3 a, .title-news a")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    url = title_elem.get("href", "")

                    if not url.startswith("http"):
                        url = urljoin(self.BASE_URL, url)

                    desc_elem = item.select_one(".description, p.description")
                    content = desc_elem.get_text(strip=True) if desc_elem else title

                    article = NewsArticle(
                        id=self._generate_id(url),
                        title=title,
                        content=content,
                        url=url,
                        source=self.get_source_name(),
                        symbols=self._extract_symbols(title + " " + content),
                        category="business",
                    )
                    articles.append(article)

                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Error scraping VnExpress: {e}")

        return articles

    def scrape_by_symbol(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """Search VnExpress for symbol news."""
        url = f"https://timkiem.vnexpress.net/?q={symbol}&media_type=all&fromdate=0&todate=0&latest=&cate_code=&date_format=all&cate_code_ex="
        articles = []

        html = self._fetch_page(url)
        if not html:
            return articles

        try:
            soup = BeautifulSoup(html, "html.parser")
            news_items = soup.select("article.item-news")[:limit]

            for item in news_items:
                try:
                    title_elem = item.select_one("h3 a")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    article_url = title_elem.get("href", "")

                    desc_elem = item.select_one("p.description")
                    content = desc_elem.get_text(strip=True) if desc_elem else title

                    article = NewsArticle(
                        id=self._generate_id(article_url),
                        title=title,
                        content=content,
                        url=article_url,
                        source=self.get_source_name(),
                        symbols=[symbol],
                        category="search_result",
                    )
                    articles.append(article)

                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Error searching VnExpress for {symbol}: {e}")

        return articles


# =============================================================================
# VIETSTOCK SCRAPER
# =============================================================================


class VietStockScraper(BaseNewsScraper):
    """Scraper for VietStock."""

    BASE_URL = "https://vietstock.vn"
    NEWS_URL = "https://vietstock.vn/chung-khoan.htm"

    def get_source_name(self) -> str:
        return "VietStock"

    def scrape_latest(self, limit: int = 20) -> List[NewsArticle]:
        """Scrape latest news from VietStock."""
        articles = []

        html = self._fetch_page(self.NEWS_URL)
        if not html:
            return articles

        try:
            soup = BeautifulSoup(html, "html.parser")

            news_items = soup.select(".news-item, .item-news, article")[:limit]

            for item in news_items:
                try:
                    title_elem = item.select_one("h3 a, .title a, a.title")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    url = urljoin(self.BASE_URL, title_elem.get("href", ""))

                    desc_elem = item.select_one(".sapo, .description, p")
                    content = desc_elem.get_text(strip=True) if desc_elem else title

                    article = NewsArticle(
                        id=self._generate_id(url),
                        title=title,
                        content=content,
                        url=url,
                        source=self.get_source_name(),
                        symbols=self._extract_symbols(title + " " + content),
                        category="stock_market",
                    )
                    articles.append(article)

                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Error scraping VietStock: {e}")

        return articles

    def scrape_by_symbol(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """Scrape news for symbol from VietStock."""
        url = f"https://vietstock.vn/search?q={symbol}"
        return self._scrape_search_results(url, symbol, limit)

    def _scrape_search_results(self, url: str, symbol: str, limit: int) -> List[NewsArticle]:
        """Scrape search results page."""
        articles = []

        html = self._fetch_page(url)
        if not html:
            return articles

        try:
            soup = BeautifulSoup(html, "html.parser")
            news_items = soup.select(".search-item, .item-news")[:limit]

            for item in news_items:
                try:
                    title_elem = item.select_one("a")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    article_url = urljoin(self.BASE_URL, title_elem.get("href", ""))

                    article = NewsArticle(
                        id=self._generate_id(article_url),
                        title=title,
                        content=title,
                        url=article_url,
                        source=self.get_source_name(),
                        symbols=[symbol],
                    )
                    articles.append(article)

                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Error scraping VietStock search: {e}")

        return articles


# =============================================================================
# NEWS SCRAPER MANAGER
# =============================================================================


class NewsScraperManager:
    """
    Manager for multiple news scrapers.

    Aggregates news from multiple sources and provides
    unified interface for news retrieval.
    """

    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize news scraper manager.

        Args:
            rate_limit: Rate limit for each scraper
        """
        self.scrapers: List[BaseNewsScraper] = []
        self._cache: Dict[str, List[NewsArticle]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=15)

        # Initialize scrapers
        if REQUESTS_AVAILABLE:
            self.scrapers = [
                CafeFScraper(rate_limit),
                VnExpressScraper(rate_limit),
                VietStockScraper(rate_limit),
            ]
            logger.info(f"Initialized {len(self.scrapers)} news scrapers")
        else:
            logger.warning("News scrapers not available (requests not installed)")

    def get_latest_news(
        self,
        limit_per_source: int = 10,
        use_cache: bool = True,
    ) -> List[NewsArticle]:
        """
        Get latest news from all sources.

        Args:
            limit_per_source: Max articles per source
            use_cache: Use cached results

        Returns:
            List of NewsArticle from all sources
        """
        cache_key = "latest"

        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        all_articles = []

        for scraper in self.scrapers:
            try:
                articles = scraper.scrape_latest(limit_per_source)
                all_articles.extend(articles)
                logger.info(f"Scraped {len(articles)} articles from {scraper.get_source_name()}")
            except Exception as e:
                logger.error(f"Error scraping {scraper.get_source_name()}: {e}")

        # Sort by date (newest first)
        all_articles.sort(key=lambda a: a.published_at or datetime.min, reverse=True)

        # Cache results
        self._cache[cache_key] = all_articles
        self._cache_time[cache_key] = datetime.now()

        return all_articles

    def get_news_for_symbol(
        self,
        symbol: str,
        limit_per_source: int = 5,
        use_cache: bool = True,
    ) -> List[NewsArticle]:
        """
        Get news for a specific stock symbol.

        Args:
            symbol: Stock symbol (e.g., "VNM", "HPG")
            limit_per_source: Max articles per source
            use_cache: Use cached results

        Returns:
            List of NewsArticle related to the symbol
        """
        cache_key = f"symbol_{symbol}"

        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        all_articles = []

        for scraper in self.scrapers:
            try:
                articles = scraper.scrape_by_symbol(symbol, limit_per_source)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error scraping {scraper.get_source_name()} for {symbol}: {e}")

        # Also filter latest news for symbol mentions
        latest = self.get_latest_news(limit_per_source=20, use_cache=True)
        symbol_mentions = [a for a in latest if symbol in a.symbols or symbol in a.title.upper()]

        # Combine and deduplicate
        seen_ids = set()
        unique_articles = []

        for article in all_articles + symbol_mentions:
            if article.id not in seen_ids:
                seen_ids.add(article.id)
                unique_articles.append(article)

        # Cache results
        self._cache[cache_key] = unique_articles
        self._cache_time[cache_key] = datetime.now()

        return unique_articles

    def get_news_for_symbols(
        self,
        symbols: List[str],
        limit_per_symbol: int = 3,
    ) -> Dict[str, List[NewsArticle]]:
        """
        Get news for multiple symbols.

        Args:
            symbols: List of stock symbols
            limit_per_symbol: Max articles per symbol

        Returns:
            Dict mapping symbol to list of articles
        """
        result = {}

        for symbol in symbols:
            articles = self.get_news_for_symbol(symbol, limit_per_symbol)
            result[symbol] = articles[:limit_per_symbol]

        return result

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache is still valid."""
        if key not in self._cache or key not in self._cache_time:
            return False

        age = datetime.now() - self._cache_time[key]
        return age < self._cache_ttl

    def clear_cache(self):
        """Clear all cached news."""
        self._cache.clear()
        self._cache_time.clear()


# =============================================================================
# SINGLETON
# =============================================================================

_news_scraper: Optional[NewsScraperManager] = None


def get_news_scraper() -> NewsScraperManager:
    """Get singleton news scraper instance."""
    global _news_scraper
    if _news_scraper is None:
        _news_scraper = NewsScraperManager()
    return _news_scraper


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TESTING NEWS SCRAPER")
    print("=" * 70 + "\n")

    if not REQUESTS_AVAILABLE:
        print("❌ requests not installed. Install: pip install requests beautifulsoup4")
        exit(1)

    manager = NewsScraperManager(rate_limit=2.0)

    # Test latest news
    print("📰 Fetching latest news...")
    latest = manager.get_latest_news(limit_per_source=5)
    print(f"Found {len(latest)} articles\n")

    for article in latest[:5]:
        print(f"📄 {article.source}: {article.title[:60]}...")
        print(f"   Symbols: {article.symbols}")
        print()

    # Test symbol-specific news
    print("\n📰 Fetching news for VNM...")
    vnm_news = manager.get_news_for_symbol("VNM", limit_per_source=3)
    print(f"Found {len(vnm_news)} articles for VNM\n")

    for article in vnm_news[:3]:
        print(f"📄 {article.title[:60]}...")

    print("\n✅ Testing complete!")
