import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

try:
    from config import TICKERS
except ImportError:
    TICKERS = []

NEWS_CACHE_FILE = "news_cache.json"
DEFAULT_LOOKBACK_HOURS = 48

RSS_SOURCES = [
    {
        "name": "VnExpress Chứng khoán",
        "url": "https://vnexpress.net/rss/kinh-doanh/chung-khoan.rss",
    },
    {
        "name": "CafeF",
        "url": "https://cafef.vn/trang-chu.rss",
    },
    {
        "name": "Vietstock",
        "url": "https://vietstock.vn/feed/chung-khoan.rss",
    },
]

POSITIVE_KEYWORDS = [
    "tăng trưởng",
    "vượt",
    "lợi nhuận",
    "kỷ lục",
    "mua vào",
    "khuyến nghị mua",
    "tích cực",
    "bứt phá",
    "cao nhất",
    "thuận lợi",
    "tăng mạnh",
]

NEGATIVE_KEYWORDS = [
    "giảm",
    "lỗ",
    "bán ra",
    "khuyến nghị bán",
    "tiêu cực",
    "điều tra",
    "phạt",
    "rủi ro",
    "tụt",
    "thua lỗ",
    "suy giảm",
]


def _load_cache() -> Dict:
    if not os.path.exists(NEWS_CACHE_FILE):
        return {"last_updated": None, "articles": {}}
    try:
        with open(NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_updated": None, "articles": {}}


def _save_cache(cache: Dict) -> None:
    with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _parse_rss(xml_text: str, source: str) -> List[Dict]:
    items = []
    try:
        from xml.etree import ElementTree as ET

        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            title = item.findtext("title", default="").strip()
            description = item.findtext("description", default="").strip()
            pub_date = item.findtext("pubDate", default="")
            link = item.findtext("link", default="")

            try:
                published = datetime.strptime(pub_date[:25], "%a, %d %b %Y %H:%M:%S")
            except Exception:
                published = datetime.utcnow()

            items.append(
                {
                    "source": source,
                    "title": title,
                    "summary": re.sub("<[^<]+?>", "", description),
                    "published_at": published.isoformat(),
                    "url": link,
                }
            )
    except Exception:
        pass
    return items


def _fetch_source(source: Dict) -> List[Dict]:
    try:
        resp = requests.get(source["url"], timeout=10)
        resp.raise_for_status()
        return _parse_rss(resp.text, source["name"])
    except Exception:
        return []


def _extract_symbols(text: str, whitelist: Optional[List[str]] = None) -> List[str]:
    if not text:
        return []
    whitelist_set = {sym.upper() for sym in (whitelist or TICKERS)}
    pattern = r"\b[A-Z]{3,5}\b"
    matches = set(re.findall(pattern, text.upper()))
    return sorted(list(matches & whitelist_set))


def _score_sentiment(text: str) -> float:
    text_lower = text.lower()
    score = 0.0
    for word in POSITIVE_KEYWORDS:
        if word in text_lower:
            score += 1.0
    for word in NEGATIVE_KEYWORDS:
        if word in text_lower:
            score -= 1.0
    return score


def update_news_cache(symbols: Optional[List[str]] = None, lookback_hours: int = DEFAULT_LOOKBACK_HOURS) -> Dict:
    """Fetch latest news, update cache and return it."""
    symbols = symbols or TICKERS
    cache = _load_cache()
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

    if "articles" not in cache:
        cache["articles"] = {}

    fetched_articles = []
    for source in RSS_SOURCES:
        fetched_articles.extend(_fetch_source(source))

    for article in fetched_articles:
        published = datetime.fromisoformat(article["published_at"])
        if published < cutoff:
            continue

        summary_text = f"{article['title']} {article['summary']}"
        mentioned_symbols = _extract_symbols(summary_text, symbols)
        sentiment = _score_sentiment(summary_text)

        article_data = {
            "source": article["source"],
            "title": article["title"],
            "summary": article["summary"],
            "url": article["url"],
            "published_at": article["published_at"],
            "sentiment": sentiment,
        }

        if not mentioned_symbols:
            cache["articles"].setdefault("_market", []).append(article_data)
        else:
            for sym in mentioned_symbols:
                cache["articles"].setdefault(sym, []).append(article_data)

    # Clean old entries
    for sym, articles in list(cache["articles"].items()):
        filtered = []
        for article in articles:
            try:
                published = datetime.fromisoformat(article["published_at"])
            except Exception:
                published = cutoff
            if published >= cutoff:
                filtered.append(article)
        cache["articles"][sym] = filtered
        if not filtered:
            del cache["articles"][sym]

    cache["last_updated"] = datetime.utcnow().isoformat()
    _save_cache(cache)
    return cache


def _ensure_cache(symbols: Optional[List[str]] = None) -> Dict:
    cache = _load_cache()
    last_updated = cache.get("last_updated")
    needs_refresh = True
    if last_updated:
        try:
            last_time = datetime.fromisoformat(last_updated)
            needs_refresh = datetime.utcnow() - last_time > timedelta(hours=2)
        except Exception:
            needs_refresh = True
    if needs_refresh:
        cache = update_news_cache(symbols=symbols)
    return cache


def analyze_news_trend(symbol: str, lookback_hours: int = DEFAULT_LOOKBACK_HOURS) -> Dict:
    """Return sentiment summary for a symbol."""
    cache = _ensure_cache()
    articles = cache.get("articles", {}).get(symbol.upper(), [])

    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
    filtered = []
    for article in articles:
        try:
            published = datetime.fromisoformat(article["published_at"])
        except Exception:
            published = cutoff
        if published >= cutoff:
            filtered.append(article)

    if not filtered:
        return {
            "symbol": symbol.upper(),
            "sentiment_score": 0.0,
            "sentiment_label": "NEUTRAL",
            "articles": [],
            "top_headlines": [],
        }

    sentiment_total = sum(article.get("sentiment", 0.0) for article in filtered)
    sentiment_avg = sentiment_total / max(len(filtered), 1)

    if sentiment_avg >= 1.0:
        label = "STRONGLY_POSITIVE"
    elif sentiment_avg >= 0.3:
        label = "POSITIVE"
    elif sentiment_avg <= -1.0:
        label = "STRONGLY_NEGATIVE"
    elif sentiment_avg <= -0.3:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    sorted_articles = sorted(
        filtered,
        key=lambda a: a.get("published_at", ""),
        reverse=True,
    )

    return {
        "symbol": symbol.upper(),
        "sentiment_score": round(sentiment_avg, 2),
        "sentiment_label": label,
        "articles": sorted_articles,
        "top_headlines": sorted_articles[:5],
    }


def get_top_news(symbol: str, limit: int = 3) -> List[Dict]:
    trend = analyze_news_trend(symbol)
    return trend.get("top_headlines", [])[:limit]


def format_news_brief(symbol: str, limit: int = 3) -> str:
    trend = analyze_news_trend(symbol)
    headlines = trend.get("top_headlines", [])[:limit]
    if not headlines:
        return f"Không tìm thấy tin tức đáng chú ý cho {symbol.upper()} trong {DEFAULT_LOOKBACK_HOURS}h qua."

    msg_lines = [
        f"📰 Tin tức {symbol.upper()} ({trend['sentiment_label']}, score={trend['sentiment_score']:+.2f})",
    ]
    for article in headlines:
        published = article.get("published_at", "")[:16].replace("T", " ")
        msg_lines.append(f"• {article['title']} ({article['source']}, {published})")
        if article.get("url"):
            msg_lines.append(f"  {article['url']}")
    return "\n".join(msg_lines)


if __name__ == "__main__":
    print("🧪 Refreshing news cache...")
    update_news_cache()
    sample_symbol = (TICKERS or ["VNM"])[0]
    summary = analyze_news_trend(sample_symbol)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

