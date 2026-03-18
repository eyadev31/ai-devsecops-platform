"""
Hybrid Intelligence Portfolio System — News Collector
=======================================================
Agent 5: Multi-source financial news ingestion engine.

Collects news from multiple sources with retry, rate limiting,
fallback chains, and deduplication. Supports mock mode for testing
without API keys.

Sources:
  1. NewsAPI (headlines + search)
  2. RSS Feeds (Reuters, Bloomberg, CNBC, FT, etc.)
  3. GDELT (global event database)
  4. CryptoPanic (crypto-specific)
"""

import logging
import hashlib
import time
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class NewsCollector:
    """
    Multi-source financial news collector.
    
    Implements a fallback chain: if primary sources fail,
    secondary sources are attempted. All failures are logged
    but never crash the pipeline.
    """

    def __init__(self):
        from config.settings import NewsConfig
        self._config = NewsConfig
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "HybridIntelligencePortfolioSystem/1.0"
        })
        self._collection_stats = {
            "sources_queried": [],
            "sources_failed": [],
            "total_raw": 0,
            "duplicates_removed": 0,
        }

    @property
    def stats(self) -> dict:
        return self._collection_stats.copy()

    def collect(self, max_articles: int = 100, max_age_hours: int = 168) -> list[dict]:
        """
        Collect news from all available sources.
        
        Args:
            max_articles: Maximum articles to return
            max_age_hours: Maximum article age in hours
            
        Returns:
            List of raw article dicts with: title, summary, source, url, published_at
        """
        self._collection_stats = {
            "sources_queried": [],
            "sources_failed": [],
            "total_raw": 0,
            "duplicates_removed": 0,
        }

        all_articles = []

        # --- Source 1: RSS Feeds (always available, no API key needed) ---
        rss_articles = self._collect_rss()
        all_articles.extend(rss_articles)

        # --- Source 2: NewsAPI (requires API key) ---
        if self._config.NEWS_API_KEY:
            newsapi_articles = self._collect_newsapi()
            all_articles.extend(newsapi_articles)
        else:
            logger.info("NewsAPI key not configured, skipping.")

        # --- Source 3: GDELT (free, no API key needed) ---
        gdelt_articles = self._collect_gdelt()
        all_articles.extend(gdelt_articles)

        # --- Source 4: CryptoPanic (optional API key) ---
        cryptopanic_articles = self._collect_cryptopanic()
        all_articles.extend(cryptopanic_articles)

        self._collection_stats["total_raw"] = len(all_articles)

        # Deduplicate by title similarity
        unique_articles = self._deduplicate(all_articles)
        self._collection_stats["duplicates_removed"] = len(all_articles) - len(unique_articles)

        # Filter by age
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        filtered = []
        for article in unique_articles:
            pub_time = self._parse_datetime(article.get("published_at", ""))
            if pub_time and pub_time >= cutoff:
                filtered.append(article)
            elif not pub_time:
                filtered.append(article)  # Keep if can't parse date

        # Sort by recency and limit
        filtered.sort(key=lambda a: a.get("published_at", ""), reverse=True)
        result = filtered[:max_articles]

        logger.info(
            f"News Collection Complete: {len(result)} articles "
            f"(raw: {self._collection_stats['total_raw']}, "
            f"dedup: {self._collection_stats['duplicates_removed']}, "
            f"sources: {len(self._collection_stats['sources_queried'])})"
        )

        return result

    def collect_mock(self) -> list[dict]:
        """Return pre-built mock articles for testing without API keys."""
        now = datetime.now(timezone.utc)
        self._collection_stats = {
            "sources_queried": ["mock"],
            "sources_failed": [],
            "total_raw": 15,
            "duplicates_removed": 0,
        }

        return [
            {
                "title": "Federal Reserve signals possible rate cuts in 2026 amid cooling inflation",
                "summary": "The Federal Reserve indicated that interest rate reductions could begin later this year as inflation continues to moderate toward the 2% target. Chair Powell emphasized a data-dependent approach while noting encouraging trends in core PCE.",
                "source": "reuters",
                "url": "https://reuters.com/example/fed-rate-cuts",
                "published_at": (now - timedelta(hours=1)).isoformat(),
            },
            {
                "title": "Bitcoin surges past $95,000 following spot ETF inflows reaching record highs",
                "summary": "Bitcoin rallied sharply as institutional investors poured over $2 billion into spot Bitcoin ETFs in a single week, the highest inflow since launch. Analysts cite growing institutional adoption and favorable regulatory environment.",
                "source": "coindesk",
                "url": "https://coindesk.com/example/bitcoin-etf-surge",
                "published_at": (now - timedelta(hours=2)).isoformat(),
            },
            {
                "title": "SEC announces new cryptocurrency regulatory framework targeting DeFi protocols",
                "summary": "The Securities and Exchange Commission unveiled a comprehensive regulatory framework for decentralized finance protocols. The rules require DeFi platforms to register as securities exchanges, creating uncertainty in the crypto market.",
                "source": "bloomberg",
                "url": "https://bloomberg.com/example/sec-defi",
                "published_at": (now - timedelta(hours=3)).isoformat(),
            },
            {
                "title": "S&P 500 hits all-time high as tech earnings exceed expectations",
                "summary": "The S&P 500 index reached a new record high, driven by better-than-expected earnings from major technology companies including NVIDIA, Apple, and Microsoft. AI-related revenue growth continues to dominate.",
                "source": "cnbc",
                "url": "https://cnbc.com/example/sp500-record",
                "published_at": (now - timedelta(hours=4)).isoformat(),
            },
            {
                "title": "Gold prices retreat as dollar strengthens on robust US economic data",
                "summary": "Gold prices fell 1.2% as the US dollar rallied following stronger-than-expected GDP and employment data. The move suggests markets are recalibrating rate cut expectations despite the Fed's dovish tone.",
                "source": "reuters",
                "url": "https://reuters.com/example/gold-retreat",
                "published_at": (now - timedelta(hours=5)).isoformat(),
            },
            {
                "title": "European Central Bank holds rates steady, signals gradual easing path",
                "summary": "The ECB kept interest rates unchanged at its March meeting but signaled that gradual rate reductions are likely starting in the second quarter. Eurozone inflation remains slightly above the 2% target.",
                "source": "financial times",
                "url": "https://ft.com/example/ecb-rates",
                "published_at": (now - timedelta(hours=8)).isoformat(),
            },
            {
                "title": "NVIDIA reports record quarterly revenue driven by AI chip demand",
                "summary": "NVIDIA posted record quarterly revenue of $38 billion, up 85% year-over-year, driven by insatiable demand for its AI training and inference chips. The company raised guidance citing continued data center buildout.",
                "source": "cnbc",
                "url": "https://cnbc.com/example/nvidia-earnings",
                "published_at": (now - timedelta(hours=10)).isoformat(),
            },
            {
                "title": "Oil prices spike on Middle East tensions as shipping routes disrupted",
                "summary": "Crude oil prices jumped 3.5% as escalating tensions in the Middle East threatened key shipping routes through the Strait of Hormuz. Analysts warn of potential supply disruptions affecting global energy markets.",
                "source": "reuters",
                "url": "https://reuters.com/example/oil-tensions",
                "published_at": (now - timedelta(hours=12)).isoformat(),
            },
            {
                "title": "Treasury yields fall as investors seek safe haven amid geopolitical uncertainty",
                "summary": "US Treasury yields declined sharply with the 10-year falling below 4.0% as investors rotated into safe-haven assets. The flight to safety reflects growing concerns about geopolitical risks and their impact on global growth.",
                "source": "wall street journal",
                "url": "https://wsj.com/example/treasury-yields",
                "published_at": (now - timedelta(hours=14)).isoformat(),
            },
            {
                "title": "Ethereum completes major network upgrade, gas fees drop significantly",
                "summary": "Ethereum successfully completed its latest network upgrade, reducing gas fees by approximately 60% and improving transaction throughput. The upgrade is expected to accelerate DeFi and NFT adoption on the network.",
                "source": "cointelegraph",
                "url": "https://cointelegraph.com/example/eth-upgrade",
                "published_at": (now - timedelta(hours=18)).isoformat(),
            },
            {
                "title": "China stimulus measures boost emerging market sentiment",
                "summary": "China announced a new round of fiscal stimulus measures including tax cuts and infrastructure spending, boosting sentiment across emerging markets. Asian equities rallied while commodities gained on expected demand increase.",
                "source": "bloomberg",
                "url": "https://bloomberg.com/example/china-stimulus",
                "published_at": (now - timedelta(hours=24)).isoformat(),
            },
            {
                "title": "US unemployment claims rise unexpectedly, labor market showing signs of cooling",
                "summary": "Initial jobless claims rose to 240,000, above the expected 210,000, suggesting the labor market may be softening. Continuing claims also increased, supporting the case for Federal Reserve rate cuts.",
                "source": "marketwatch",
                "url": "https://marketwatch.com/example/jobless-claims",
                "published_at": (now - timedelta(hours=30)).isoformat(),
            },
            {
                "title": "Major bank upgrades outlook on commodities citing supply constraints",
                "summary": "Goldman Sachs upgraded its commodity outlook citing persistent supply constraints in copper, lithium, and agricultural markets. The bank forecasts a commodity supercycle driven by energy transition demand.",
                "source": "bloomberg",
                "url": "https://bloomberg.com/example/commodity-outlook",
                "published_at": (now - timedelta(hours=48)).isoformat(),
            },
            {
                "title": "Japan's central bank surprises with policy shift, yen strengthens sharply",
                "summary": "The Bank of Japan unexpectedly raised interest rates and signaled an end to its yield curve control policy, causing the yen to strengthen 2% against the dollar. The move has implications for global carry trades.",
                "source": "financial times",
                "url": "https://ft.com/example/boj-surprise",
                "published_at": (now - timedelta(hours=72)).isoformat(),
            },
            {
                "title": "New US tariffs on Chinese goods create supply chain concerns for tech sector",
                "summary": "The US announced new tariffs on Chinese semiconductor and technology imports, creating supply chain disruption fears. Tech stocks declined as companies assess the impact on component costs and delivery timelines.",
                "source": "reuters",
                "url": "https://reuters.com/example/tariffs-tech",
                "published_at": (now - timedelta(hours=96)).isoformat(),
            },
        ]

    # ═════════════════════════════════════════════════════
    #  SOURCE COLLECTORS
    # ═════════════════════════════════════════════════════

    def _collect_rss(self) -> list[dict]:
        """Collect articles from RSS feeds."""
        articles = []
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed. Skipping RSS collection.")
            self._collection_stats["sources_failed"].append("rss_feeds")
            return articles

        for feed_name, feed_url in self._config.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:15]:  # Max 15 per feed
                    pub_date = getattr(entry, "published", "")
                    if not pub_date:
                        pub_date = getattr(entry, "updated", datetime.now(timezone.utc).isoformat())

                    articles.append({
                        "title": getattr(entry, "title", ""),
                        "summary": getattr(entry, "summary", getattr(entry, "description", "")),
                        "source": self._normalize_source(feed_name),
                        "url": getattr(entry, "link", ""),
                        "published_at": pub_date,
                    })

                self._collection_stats["sources_queried"].append(feed_name)
                logger.debug(f"RSS [{feed_name}]: {len(feed.entries)} entries")

            except Exception as e:
                logger.warning(f"RSS feed failed [{feed_name}]: {e}")
                self._collection_stats["sources_failed"].append(feed_name)

        return articles

    def _collect_newsapi(self) -> list[dict]:
        """Collect articles from NewsAPI."""
        articles = []
        try:
            # Top financial headlines
            params = {
                "apiKey": self._config.NEWS_API_KEY,
                "category": "business",
                "language": "en",
                "pageSize": 30,
            }
            resp = self._session.get(
                f"{self._config.NEWS_API_BASE_URL}/top-headlines",
                params=params,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("articles", []):
                    articles.append({
                        "title": item.get("title", ""),
                        "summary": item.get("description", "") or item.get("content", ""),
                        "source": self._normalize_source(item.get("source", {}).get("name", "newsapi")),
                        "url": item.get("url", ""),
                        "published_at": item.get("publishedAt", ""),
                    })
                self._collection_stats["sources_queried"].append("newsapi_headlines")
                logger.debug(f"NewsAPI headlines: {len(data.get('articles', []))} articles")
            else:
                logger.warning(f"NewsAPI headlines failed: {resp.status_code}")
                self._collection_stats["sources_failed"].append("newsapi_headlines")

            # Financial keyword search
            time.sleep(0.5)  # Rate limiting
            search_params = {
                "apiKey": self._config.NEWS_API_KEY,
                "q": "finance OR stock OR crypto OR federal reserve OR economy",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
            }
            resp = self._session.get(
                f"{self._config.NEWS_API_BASE_URL}/everything",
                params=search_params,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("articles", []):
                    articles.append({
                        "title": item.get("title", ""),
                        "summary": item.get("description", "") or item.get("content", ""),
                        "source": self._normalize_source(item.get("source", {}).get("name", "newsapi")),
                        "url": item.get("url", ""),
                        "published_at": item.get("publishedAt", ""),
                    })
                self._collection_stats["sources_queried"].append("newsapi_search")
            else:
                self._collection_stats["sources_failed"].append("newsapi_search")

        except Exception as e:
            logger.warning(f"NewsAPI collection failed: {e}")
            self._collection_stats["sources_failed"].append("newsapi")

        return articles

    def _collect_gdelt(self) -> list[dict]:
        """Collect articles from GDELT Project API."""
        articles = []
        try:
            params = {
                "query": "finance economy market",
                "mode": "artlist",
                "maxrecords": "20",
                "format": "json",
                "sourcelang": "eng",
                "timespan": "24h",
            }
            resp = self._session.get(
                self._config.GDELT_API_URL,
                params=params,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("articles", []):
                    articles.append({
                        "title": item.get("title", ""),
                        "summary": item.get("seendate", ""),
                        "source": self._normalize_source(item.get("domain", "gdelt")),
                        "url": item.get("url", ""),
                        "published_at": item.get("seendate", ""),
                    })
                self._collection_stats["sources_queried"].append("gdelt")
                logger.debug(f"GDELT: {len(data.get('articles', []))} articles")
            else:
                self._collection_stats["sources_failed"].append("gdelt")

        except Exception as e:
            logger.warning(f"GDELT collection failed: {e}")
            self._collection_stats["sources_failed"].append("gdelt")

        return articles

    def _collect_cryptopanic(self) -> list[dict]:
        """Collect from CryptoPanic API."""
        articles = []
        try:
            params = {"auth_token": self._config.CRYPTOPANIC_API_KEY} if self._config.CRYPTOPANIC_API_KEY else {}
            params["public"] = "true"
            params["kind"] = "news"

            resp = self._session.get(
                self._config.CRYPTOPANIC_API_URL,
                params=params,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", []):
                    articles.append({
                        "title": item.get("title", ""),
                        "summary": item.get("title", ""),  # CryptoPanic has short titles only
                        "source": self._normalize_source(item.get("source", {}).get("title", "cryptopanic")),
                        "url": item.get("url", ""),
                        "published_at": item.get("published_at", ""),
                    })
                self._collection_stats["sources_queried"].append("cryptopanic")
            else:
                self._collection_stats["sources_failed"].append("cryptopanic")

        except Exception as e:
            logger.warning(f"CryptoPanic collection failed: {e}")
            self._collection_stats["sources_failed"].append("cryptopanic")

        return articles

    # ═════════════════════════════════════════════════════
    #  UTILITIES
    # ═════════════════════════════════════════════════════

    def _deduplicate(self, articles: list[dict]) -> list[dict]:
        """Remove duplicate articles based on title fingerprint."""
        seen = set()
        unique = []
        for article in articles:
            title = article.get("title", "").strip().lower()
            if not title:
                continue
            # Create fingerprint from cleaned title
            fingerprint = hashlib.md5(
                re.sub(r"[^a-z0-9\s]", "", title).encode()
            ).hexdigest()
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(article)
        return unique

    @staticmethod
    def _normalize_source(source: str) -> str:
        """Normalize source name for consistent matching."""
        if not source:
            return "unknown"
        source = source.lower().strip()
        # Map common variations
        mapping = {
            "reuters_markets": "reuters",
            "cnbc_finance": "cnbc",
            "ft_markets": "financial times",
            "yahoo_finance": "yahoo finance",
            "investing_com": "investing.com",
        }
        return mapping.get(source, source)

    @staticmethod
    def _parse_datetime(dt_str: str) -> Optional[datetime]:
        """Parse various datetime formats to UTC datetime."""
        if not dt_str:
            return None
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y%m%d%H%M%S",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return None
