"""
AlphaDesk — News Radar

Monitors global news via RSS feeds every 30 minutes, scores items for market
impact, and triggers alerts / regime updates when critical events occur.

Design:
- 12 feed sources covering macro, geopolitics, central banks, commodities
- Keyword-based scoring (tuned for market impact, not pure news relevance)
- Deduplication via SHA1(title) state persisted to JSON
- Telegram alerts for score >= 5
- DB log for all scored items (future ML training)
- Regime detector integration: when score >= 5 and regime-relevant, force refresh

Usage:
    radar = NewsRadar(notifier, db_path)
    report = await radar.scan()
    # report = {"scanned": N, "alerted": M, "critical": [items]}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

try:
    import feedparser  # type: ignore
except ImportError:
    feedparser = None

try:
    import aiohttp  # type: ignore
except ImportError:
    aiohttp = None


logger = logging.getLogger("alphadesk.news_radar")


# ══════════════════════════════════════════════════════════════════════
# FEED SOURCES
# ══════════════════════════════════════════════════════════════════════

RSS_FEEDS: Dict[str, str] = {
    # Macro & markets
    "reuters_business":    "https://feeds.reuters.com/reuters/businessNews",
    "cnbc_markets":        "https://www.cnbc.com/id/20409666/device/rss/rss.html",
    "ft_markets":          "https://www.ft.com/markets?format=rss",
    "marketwatch_top":     "https://feeds.marketwatch.com/marketwatch/topstories/",
    "wsj_markets":         "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    # Central banks & policy
    "fed_press":           "https://www.federalreserve.gov/feeds/press_all.xml",
    "ecb_press":           "https://www.ecb.europa.eu/rss/press.html",
    # Geopolitics (war, sanctions)
    "aljazeera_all":       "https://www.aljazeera.com/xml/rss/all.xml",
    "reuters_world":       "https://feeds.reuters.com/Reuters/worldNews",
    # Commodities (energy)
    "oilprice":            "https://oilprice.com/rss/main",
    # Tech (for our NVDA/META/GOOGL/MSFT positions)
    "cnbc_tech":           "https://www.cnbc.com/id/19854910/device/rss/rss.html",
    # Economic data (macro surprises)
    "bls":                 "https://www.bls.gov/feed/news_release.rss",
}


# ══════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════
# Rules:
#   - Score each matching keyword in title (x2) and summary (x1)
#   - Portfolio tickers get higher weight (direct exposure)
#   - Crisis/tail keywords get highest weight
#   - Total score determines action:
#     >= 7   CRITICAL → Telegram alert + regime refresh
#     4-6   HIGH     → Log + update sentiment
#     2-3   NOTEWORTHY → Log only
#     < 2   IGNORE

# Critical/tail events — highest impact
CRISIS_KEYWORDS: Dict[str, int] = {
    "bankruptcy": 5, "default": 5, "collapse": 5, "crash": 5,
    "circuit breaker": 5, "trading halt": 5, "liquidation": 4,
    "contagion": 4, "systemic": 4, "bailout": 4,
    "flash crash": 5, "freeze": 3, "run on": 5,
    "emergency meeting": 4, "emergency rate": 5,
}

# War / geopolitics
GEOPOLITICAL_KEYWORDS: Dict[str, int] = {
    "iran": 3, "israel": 3, "hormuz": 4, "russia": 2, "ukraine": 2,
    "china": 2, "taiwan": 3, "north korea": 3,
    "nuclear": 3, "missile": 3, "blockade": 4, "ceasefire": 4,
    "escalation": 3, "sanctions": 3, "tariff": 3,
    "war": 3, "strike": 2, "attack": 2, "invasion": 4,
    "hostages": 3, "hezbollah": 2, "hamas": 2, "houthi": 2,
}

# Central banks / rates
CENTRAL_BANK_KEYWORDS: Dict[str, int] = {
    "fomc": 4, "fed": 2, "powell": 3, "lagarde": 2, "ecb": 2,
    "boj": 2, "rate cut": 4, "rate hike": 4, "hawkish": 3, "dovish": 3,
    "quantitative tightening": 3, "quantitative easing": 3,
    "basis points": 2, "dot plot": 3, "rate decision": 4,
    "forward guidance": 2, "balance sheet": 2,
}

# Macro data
MACRO_KEYWORDS: Dict[str, int] = {
    "cpi": 4, "ppi": 3, "nfp": 4, "nonfarm payrolls": 4,
    "jobless claims": 2, "retail sales": 2,
    "gdp": 3, "recession": 4, "stagflation": 4, "inflation": 2,
    "unemployment": 2, "pmi": 2,
    "consumer confidence": 1, "housing starts": 1,
    "trade balance": 1, "consumer sentiment": 1,
}

# Commodities
COMMODITY_KEYWORDS: Dict[str, int] = {
    "opec": 3, "oil": 1, "crude": 2, "wti": 2, "brent": 2,
    "refinery": 2, "pipeline": 2, "natgas": 2, "lng": 2,
    "gold": 1, "silver": 1, "copper": 1,
    "supply shock": 4, "production cut": 3, "output cut": 3,
    "strategic petroleum reserve": 3, "spr": 2,
}

# Portfolio tickers (get higher weight for direct exposure)
PORTFOLIO_TICKERS: Dict[str, int] = {
    "xle": 3, "xom": 3, "slb": 3, "exxon": 3, "schlumberger": 3,
    "jnj": 3, "johnson & johnson": 3, "unh": 3, "unitedhealth": 3,
    "pfe": 3, "pfizer": 3,
    "msft": 3, "microsoft": 3, "nvda": 3, "nvidia": 3,
    "googl": 3, "alphabet": 3, "meta": 3,
    "aapl": 3, "apple": 3, "amzn": 3, "amazon": 3, "nflx": 3,
    "dis": 2, "disney": 2, "bac": 2, "bank of america": 2,
    "jpm": 2, "jpmorgan": 2, "v": 2, "visa": 2,
    "gld": 2, "spy": 2,
}

# Combine all keyword dicts
ALL_KEYWORDS: Dict[str, int] = {
    **CRISIS_KEYWORDS,
    **GEOPOLITICAL_KEYWORDS,
    **CENTRAL_BANK_KEYWORDS,
    **MACRO_KEYWORDS,
    **COMMODITY_KEYWORDS,
    **PORTFOLIO_TICKERS,
}

CRITICAL_THRESHOLD = 7
HIGH_THRESHOLD = 4
NOTEWORTHY_THRESHOLD = 2


# ══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class NewsItem:
    source: str
    title: str
    summary: str
    link: str
    published: Optional[datetime] = None
    score: int = 0
    matched_keywords: List[str] = field(default_factory=list)

    @property
    def item_hash(self) -> str:
        """Stable hash for dedup (title + link)."""
        h = hashlib.sha1(f"{self.title}|{self.link}".encode()).hexdigest()
        return h[:16]

    @property
    def severity(self) -> str:
        if self.score >= CRITICAL_THRESHOLD:
            return "CRITICAL"
        if self.score >= HIGH_THRESHOLD:
            return "HIGH"
        if self.score >= NOTEWORTHY_THRESHOLD:
            return "NOTEWORTHY"
        return "LOW"


# ══════════════════════════════════════════════════════════════════════
# NEWS RADAR
# ══════════════════════════════════════════════════════════════════════

class NewsRadar:
    """Monitors news feeds and scores items for market impact."""

    STATE_FILE = "/root/AlphaDesk/data/news_radar_state.json"
    MAX_STATE_SIZE = 5000  # LRU: keep last N hashes
    FETCH_TIMEOUT = 10
    MAX_CONCURRENT = 5  # concurrent HTTP fetches
    ITEM_AGE_MAX_HOURS = 6  # ignore items older than this

    def __init__(self, notifier=None, db_path: Optional[str] = None):
        self.notifier = notifier
        self.db_path = db_path
        self._seen_hashes: Set[str] = set()
        self._load_state()
        self._ensure_db_schema()

    # ── State persistence ──

    def _load_state(self):
        """Load seen item hashes from disk."""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    self._seen_hashes = set(data.get("hashes", []))
                    logger.debug(f"Loaded {len(self._seen_hashes)} seen hashes")
        except Exception as e:
            logger.warning(f"Could not load state: {e}")

    def _save_state(self):
        """Persist seen hashes to disk (LRU-truncated)."""
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            # Keep only last N hashes
            hashes = list(self._seen_hashes)[-self.MAX_STATE_SIZE:]
            with open(self.STATE_FILE, "w") as f:
                json.dump({"hashes": hashes, "updated": datetime.utcnow().isoformat()}, f)
        except Exception as e:
            logger.warning(f"Could not save state: {e}")

    def _ensure_db_schema(self):
        """Create news_events table if needed."""
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS news_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT DEFAULT (datetime('now')),
                        source TEXT,
                        title TEXT,
                        link TEXT,
                        published TEXT,
                        score INTEGER,
                        severity TEXT,
                        matched_keywords TEXT,
                        item_hash TEXT UNIQUE
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_news_score ON news_events(score DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news_events(timestamp DESC)"
                )
        except Exception as e:
            logger.warning(f"Could not create news_events table: {e}")

    # ── Scoring ──

    @staticmethod
    def score_item(title: str, summary: str) -> Tuple[int, List[str]]:
        """Score an item by keyword matching. Returns (score, matched_keywords)."""
        title_lower = (title or "").lower()
        summary_lower = (summary or "").lower()

        score = 0
        matched = []

        for keyword, weight in ALL_KEYWORDS.items():
            if keyword in title_lower:
                score += weight * 2  # Title matches weight 2x
                matched.append(keyword)
            elif keyword in summary_lower:
                score += weight
                matched.append(keyword)

        return score, matched

    # ── Fetching ──

    async def _fetch_feed(self, session, source: str, url: str) -> List[NewsItem]:
        """Fetch and parse a single RSS feed."""
        items: List[NewsItem] = []
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.FETCH_TIMEOUT),
                                     headers={"User-Agent": "Mozilla/5.0 (AlphaDesk NewsRadar/1.0)"}) as resp:
                if resp.status != 200:
                    logger.debug(f"[{source}] status {resp.status}")
                    return []
                raw = await resp.read()

            if feedparser is None:
                return []

            feed = feedparser.parse(raw)
            cutoff = datetime.utcnow() - timedelta(hours=self.ITEM_AGE_MAX_HOURS)

            for entry in feed.entries[:30]:  # Max 30 per feed
                title = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")
                link = entry.get("link", "")
                pub_parsed = entry.get("published_parsed") or entry.get("updated_parsed")

                published = None
                if pub_parsed:
                    try:
                        published = datetime(*pub_parsed[:6])
                        if published < cutoff:
                            continue  # Too old
                    except Exception:
                        pass

                score, matched = self.score_item(title, summary)
                if score < NOTEWORTHY_THRESHOLD:
                    continue  # Below noise floor

                items.append(NewsItem(
                    source=source,
                    title=title[:300],
                    summary=summary[:500],
                    link=link[:500],
                    published=published,
                    score=score,
                    matched_keywords=matched,
                ))
        except asyncio.TimeoutError:
            logger.debug(f"[{source}] timeout")
        except Exception as e:
            logger.debug(f"[{source}] error: {e}")

        return items

    async def fetch_all(self) -> List[NewsItem]:
        """Fetch all feeds concurrently."""
        if aiohttp is None:
            logger.error("aiohttp not installed")
            return []

        connector = aiohttp.TCPConnector(limit=self.MAX_CONCURRENT)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._fetch_feed(session, source, url)
                for source, url in RSS_FEEDS.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_items: List[NewsItem] = []
        for r in results:
            if isinstance(r, list):
                all_items.extend(r)

        return all_items

    # ── Persistence & Alerting ──

    def _log_to_db(self, item: NewsItem):
        """Persist item to DB."""
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO news_events
                       (source, title, link, published, score, severity, matched_keywords, item_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (item.source, item.title, item.link,
                     item.published.isoformat() if item.published else None,
                     item.score, item.severity,
                     ",".join(item.matched_keywords[:10]),
                     item.item_hash),
                )
        except Exception as e:
            logger.debug(f"DB log error: {e}")

    async def _send_alert(self, item: NewsItem):
        """Send Telegram alert for critical item."""
        if not self.notifier or not getattr(self.notifier, "enabled", False):
            return
        try:
            msg = (
                f"📡 *NEWS RADAR — {item.severity}*\n"
                f"Score: {item.score} | {item.source}\n\n"
                f"*{item.title[:200]}*\n\n"
                f"Matched: `{', '.join(item.matched_keywords[:6])}`\n"
                f"[link]({item.link})"
            )
            if hasattr(self.notifier, "send_message"):
                await self.notifier.send_message(msg)
            elif hasattr(self.notifier, "send"):
                await self.notifier.send(msg, parse_mode="Markdown")
        except Exception as e:
            logger.debug(f"Alert send error: {e}")

    # ── Main scan ──

    async def scan(self) -> dict:
        """Fetch all feeds, score items, alert on critical. Returns summary report."""
        start = time.time()
        items = await self.fetch_all()

        # Deduplicate against state
        new_items = [it for it in items if it.item_hash not in self._seen_hashes]

        # Log all new items to DB
        for item in new_items:
            self._log_to_db(item)
            self._seen_hashes.add(item.item_hash)

        # Sort by score desc
        new_items.sort(key=lambda x: x.score, reverse=True)

        # Alert on critical
        critical = [it for it in new_items if it.score >= CRITICAL_THRESHOLD]
        alerted = 0
        for item in critical[:3]:  # Cap at 3 alerts per scan to avoid spam
            await self._send_alert(item)
            alerted += 1

        self._save_state()

        elapsed = time.time() - start
        report = {
            "scanned_items": len(items),
            "new_items": len(new_items),
            "critical": len(critical),
            "alerted": alerted,
            "elapsed_sec": round(elapsed, 1),
            "top_items": [
                {"source": it.source, "score": it.score,
                 "title": it.title[:100], "keywords": it.matched_keywords[:5]}
                for it in new_items[:5]
            ],
        }
        logger.info(
            f"Scan: {report['new_items']} new / {report['scanned_items']} total, "
            f"{report['critical']} critical, {report['alerted']} alerted ({report['elapsed_sec']}s)"
        )
        return report

    def get_recent_events(self, hours: int = 24, min_score: int = HIGH_THRESHOLD) -> List[dict]:
        """Query recent events from DB (for agents to use as context)."""
        if not self.db_path:
            return []
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """SELECT source, title, score, severity, matched_keywords, timestamp
                       FROM news_events
                       WHERE timestamp >= datetime('now', ?) AND score >= ?
                       ORDER BY score DESC, timestamp DESC
                       LIMIT 50""",
                    (f"-{hours} hours", min_score),
                ).fetchall()
            return [
                {"source": r[0], "title": r[1], "score": r[2],
                 "severity": r[3], "keywords": r[4], "timestamp": r[5]}
                for r in rows
            ]
        except Exception as e:
            logger.debug(f"get_recent_events error: {e}")
            return []
