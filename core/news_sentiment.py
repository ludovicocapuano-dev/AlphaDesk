"""
AlphaDesk — News Sentiment Module
Lightweight sentiment analysis using VADER + public RSS feeds.

Provides macro sentiment, per-stock sentiment, and Fed/monetary policy
sentiment. Designed to complement the RegimeDetector with qualitative
news signals.

No API keys required — uses only public RSS feeds and nltk's VADER.
"""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("alphadesk.sentiment")

# ---------------------------------------------------------------------------
# VADER setup with graceful fallback
# ---------------------------------------------------------------------------

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    logger.warning(
        "nltk/VADER not installed — sentiment analysis disabled. "
        "Install with: pip install nltk"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RSS_FEEDS: Dict[str, str] = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "cnbc_economy": (
        "https://search.cnbc.com/rs/search/combinedcms/view.xml"
        "?partnerId=wrss01&id=20910258"
    ),
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
}

# RSS fetch timeout in seconds
_RSS_TIMEOUT = 10

# Cache TTL in seconds (30 minutes)
_CACHE_TTL = 30 * 60

# User-Agent for RSS requests (some feeds reject bare urllib)
_USER_AGENT = "AlphaDesk/2.0 (+https://github.com/ludovicocapuano-dev/AlphaDesk)"

# Company name mapping for the US_EQUITIES universe
COMPANY_NAMES: Dict[str, List[str]] = {
    "AAPL": ["Apple", "AAPL"],
    "MSFT": ["Microsoft", "MSFT"],
    "GOOGL": ["Google", "Alphabet", "GOOGL"],
    "AMZN": ["Amazon", "AMZN"],
    "NVDA": ["Nvidia", "NVDA"],
    "META": ["Meta", "Facebook", "META"],
    "NFLX": ["Netflix", "NFLX"],
    "XOM": ["Exxon", "ExxonMobil", "XOM"],
    "HD": ["Home Depot", "HD"],
    "TSLA": ["Tesla", "TSLA"],
    "DIS": ["Disney", "DIS"],
    "JPM": ["JPMorgan", "JP Morgan", "JPM"],
    "BAC": ["Bank of America", "BofA", "BAC"],
    "V": ["Visa"],
    "JNJ": ["Johnson & Johnson", "J&J", "JNJ"],
    "UNH": ["UnitedHealth", "UNH"],
    "PFE": ["Pfizer", "PFE"],
    "WMT": ["Walmart", "WMT"],
    "PG": ["Procter & Gamble", "P&G", "PG"],
    "KO": ["Coca-Cola", "Coca Cola", "KO"],
}

# Keywords for Fed / monetary-policy filtering
FED_KEYWORDS: List[str] = [
    "fed", "fomc", "federal reserve", "interest rate",
    "monetary policy", "rate hike", "rate cut", "powell",
    "quantitative tightening", "quantitative easing",
    "basis points", "dot plot", "treasury yield",
]


def _neutral_aggregate() -> dict:
    """Return a neutral aggregate sentiment result."""
    return {
        "score": 0.0,
        "n_articles": 0,
        "bullish_pct": 0.0,
        "bearish_pct": 0.0,
        "top_bullish": "",
        "top_bearish": "",
    }


# ---------------------------------------------------------------------------
# NewsSentiment class
# ---------------------------------------------------------------------------


class NewsSentiment:
    """
    Lightweight news sentiment analysis using VADER + RSS feeds.
    Provides macro sentiment (Fed, economy) and per-stock sentiment.
    """

    def __init__(self, feeds: Optional[Dict[str, str]] = None):
        """
        Args:
            feeds: Optional override for RSS feed URLs.
        """
        self.feeds = feeds or RSS_FEEDS
        self._analyzer = SentimentIntensityAnalyzer() if HAS_VADER else None

        # Simple in-memory cache: {"headlines": (timestamp, data)}
        self._cache: Dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # RSS fetching
    # ------------------------------------------------------------------

    def fetch_headlines(self, max_age_hours: int = 24) -> List[dict]:
        """
        Fetch and parse headlines from all configured RSS feeds.

        Results are cached for 30 minutes to avoid hammering feed servers.

        Args:
            max_age_hours: Only return headlines published within this window.

        Returns:
            List of dicts with keys: title, source, published, link.
        """
        cache_key = "headlines"
        now = time.time()

        # Check cache
        if cache_key in self._cache:
            cached_ts, cached_data = self._cache[cache_key]
            if now - cached_ts < _CACHE_TTL:
                logger.debug("Returning %d cached headlines", len(cached_data))
                return cached_data

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        headlines: List[dict] = []

        for source_name, url in self.feeds.items():
            try:
                items = self._parse_feed(url, source_name)
                for item in items:
                    if item["published"] >= cutoff:
                        headlines.append(item)
            except Exception as exc:
                logger.warning("Failed to fetch feed %s: %s", source_name, exc)

        # Sort by published descending
        headlines.sort(key=lambda h: h["published"], reverse=True)

        # Update cache
        self._cache[cache_key] = (now, headlines)
        logger.info("Fetched %d headlines from %d feeds", len(headlines), len(self.feeds))
        return headlines

    def _parse_feed(self, url: str, source_name: str) -> List[dict]:
        """Parse a single RSS feed and return headline dicts."""
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(req, timeout=_RSS_TIMEOUT) as resp:
            raw = resp.read()

        root = ET.fromstring(raw)
        items: List[dict] = []

        # Standard RSS 2.0: channel/item
        for item_el in root.iter("item"):
            title = self._text(item_el, "title")
            if not title:
                continue

            pub_str = self._text(item_el, "pubDate")
            published = self._parse_date(pub_str)

            items.append({
                "title": title,
                "source": source_name,
                "published": published,
                "link": self._text(item_el, "link") or "",
            })

        # Atom feeds: entry/title
        for entry_el in root.iter("{http://www.w3.org/2005/Atom}entry"):
            title = self._text(entry_el, "{http://www.w3.org/2005/Atom}title")
            if not title:
                continue

            pub_str = (
                self._text(entry_el, "{http://www.w3.org/2005/Atom}published")
                or self._text(entry_el, "{http://www.w3.org/2005/Atom}updated")
            )
            published = self._parse_date(pub_str)

            link_el = entry_el.find("{http://www.w3.org/2005/Atom}link")
            link = link_el.get("href", "") if link_el is not None else ""

            items.append({
                "title": title,
                "source": source_name,
                "published": published,
                "link": link,
            })

        return items

    @staticmethod
    def _text(element: ET.Element, tag: str) -> str:
        """Safely extract text from an XML child element."""
        child = element.find(tag)
        return child.text.strip() if child is not None and child.text else ""

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """Best-effort parse of RSS date strings."""
        if not date_str:
            return datetime.utcnow()

        # RFC 822 (most RSS feeds)
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                # Normalize to naive UTC
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                continue

        # Last resort: strip timezone abbreviation and retry
        parts = date_str.rsplit(" ", 1)
        if len(parts) == 2:
            try:
                return datetime.strptime(parts[0].strip(), "%a, %d %b %Y %H:%M:%S")
            except ValueError:
                pass

        logger.debug("Could not parse date '%s', using now()", date_str)
        return datetime.utcnow()

    # ------------------------------------------------------------------
    # Sentiment analysis
    # ------------------------------------------------------------------

    def analyze_headline(self, text: str) -> dict:
        """
        Run VADER sentiment on a single headline.

        Args:
            text: Headline text.

        Returns:
            Dict with keys: compound, pos, neg, neu (all floats).
            Returns neutral scores if VADER is unavailable.
        """
        if not HAS_VADER or self._analyzer is None:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        scores = self._analyzer.polarity_scores(text)
        return {
            "compound": scores["compound"],
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
        }

    def _aggregate(self, headlines: List[dict]) -> dict:
        """Compute aggregate sentiment from a list of headline dicts."""
        if not headlines:
            return _neutral_aggregate()

        compounds = []
        best_bull = ("", -1.0)
        best_bear = ("", 1.0)

        for h in headlines:
            scores = self.analyze_headline(h["title"])
            c = scores["compound"]
            compounds.append(c)

            if c > best_bull[1]:
                best_bull = (h["title"], c)
            if c < best_bear[1]:
                best_bear = (h["title"], c)

        n = len(compounds)
        bullish = sum(1 for c in compounds if c >= 0.05)
        bearish = sum(1 for c in compounds if c <= -0.05)

        return {
            "score": sum(compounds) / n,
            "n_articles": n,
            "bullish_pct": bullish / n if n else 0.0,
            "bearish_pct": bearish / n if n else 0.0,
            "top_bullish": best_bull[0],
            "top_bearish": best_bear[0],
        }

    # ------------------------------------------------------------------
    # Public high-level methods
    # ------------------------------------------------------------------

    def get_macro_sentiment(self) -> dict:
        """
        Analyze all recent headlines for overall macro sentiment.

        Returns:
            Dict with score (-1 to 1), n_articles, bullish_pct,
            bearish_pct, top_bullish headline, top_bearish headline.
        """
        if not HAS_VADER:
            logger.warning("VADER unavailable — returning neutral macro sentiment")
            return _neutral_aggregate()

        try:
            headlines = self.fetch_headlines()
        except Exception as exc:
            logger.error("Failed to fetch headlines for macro sentiment: %s", exc)
            return _neutral_aggregate()

        result = self._aggregate(headlines)
        logger.info(
            "Macro sentiment: score=%.3f, n=%d, bull=%.0f%%, bear=%.0f%%",
            result["score"], result["n_articles"],
            result["bullish_pct"] * 100, result["bearish_pct"] * 100,
        )
        return result

    def get_stock_sentiment(self, symbol: str) -> dict:
        """
        Get sentiment for a specific stock by filtering headlines that
        mention the ticker or company name.

        Args:
            symbol: Stock ticker (e.g. "AAPL").

        Returns:
            Aggregate sentiment dict. Returns neutral if no matching
            headlines are found.
        """
        if not HAS_VADER:
            logger.warning("VADER unavailable — returning neutral stock sentiment")
            return _neutral_aggregate()

        try:
            headlines = self.fetch_headlines()
        except Exception as exc:
            logger.error("Failed to fetch headlines for %s: %s", symbol, exc)
            return _neutral_aggregate()

        # Build search terms for this symbol
        search_terms = COMPANY_NAMES.get(symbol, [symbol])
        if symbol not in search_terms:
            search_terms.append(symbol)

        # Case-insensitive matching
        lower_terms = [t.lower() for t in search_terms]

        matching = [
            h for h in headlines
            if any(term in h["title"].lower() for term in lower_terms)
        ]

        result = self._aggregate(matching)
        if matching:
            logger.info(
                "%s sentiment: score=%.3f from %d headlines",
                symbol, result["score"], result["n_articles"],
            )
        else:
            logger.debug("No headlines found mentioning %s", symbol)

        return result

    def get_fed_sentiment(self) -> dict:
        """
        Get sentiment for Fed / monetary policy headlines.
        Useful as an input to regime detection (rate_regime).

        Returns:
            Aggregate sentiment dict filtered to Fed-related headlines.
        """
        if not HAS_VADER:
            logger.warning("VADER unavailable — returning neutral Fed sentiment")
            return _neutral_aggregate()

        try:
            headlines = self.fetch_headlines()
        except Exception as exc:
            logger.error("Failed to fetch headlines for Fed sentiment: %s", exc)
            return _neutral_aggregate()

        matching = [
            h for h in headlines
            if any(kw in h["title"].lower() for kw in FED_KEYWORDS)
        ]

        result = self._aggregate(matching)
        if matching:
            logger.info(
                "Fed sentiment: score=%.3f from %d headlines",
                result["score"], result["n_articles"],
            )
        else:
            logger.debug("No Fed-related headlines found")

        return result
