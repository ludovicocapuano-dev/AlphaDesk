"""
AlphaDesk — Unusual Whales API Client

Provides institutional-grade alternative data:
- Dark pool trades (institutional accumulation/distribution)
- Congress trades (smart money following)
- Insider transactions (corporate insider signals)
- Market tide (options flow sentiment)
- Market correlations (cross-asset regime)
- Institutional holdings & activity
- Options flow alerts per ticker

Integrated into:
- Macro Agent (Layer 1): market tide, correlations, insider aggregate
- Value/Contrarian Agents (Layer 2): congress, insider, dark pool per ticker
- Risk Agent (Layer 3): dark pool volume, institutional positioning
- Regime Detector: correlation + volatility spike overlay
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger("alphadesk.unusual_whales")

BASE_URL = "https://api.unusualwhales.com/api"


class UnusualWhalesClient:
    """Async client for Unusual Whales API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("UNUSUAL_WHALES_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, dict] = {}  # Simple TTL cache
        self._cache_ts: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=15)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, path: str, params: dict = None, cache_key: str = None) -> dict:
        """Make authenticated GET request with optional caching."""
        if cache_key and cache_key in self._cache:
            if datetime.utcnow() - self._cache_ts[cache_key] < self._cache_ttl:
                return self._cache[cache_key]

        session = await self._get_session()
        url = f"{BASE_URL}{path}"
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if cache_key:
                        self._cache[cache_key] = data
                        self._cache_ts[cache_key] = datetime.utcnow()
                    return data
                elif resp.status == 429:
                    logger.warning("UW rate limited on %s", path)
                    return {"data": [], "error": "rate_limited"}
                else:
                    text = await resp.text()
                    logger.warning("UW %s returned %d: %s", path, resp.status, text[:200])
                    return {"data": [], "error": f"http_{resp.status}"}
        except Exception as e:
            logger.warning("UW request failed %s: %s", path, e)
            return {"data": [], "error": str(e)}

    # ── Market-wide signals ──

    async def get_market_tide(self) -> List[dict]:
        """
        Options flow sentiment (net call vs put premium).
        Bullish when net_call_premium >> net_put_premium.
        """
        result = await self._get("/market/market-tide", cache_key="market_tide")
        return result.get("data", [])

    async def get_market_correlations(self, tickers: List[str]) -> dict:
        """Cross-asset correlations for regime detection."""
        params = {"tickers": ",".join(tickers)}
        result = await self._get("/market/correlations", params=params,
                                 cache_key=f"corr_{'_'.join(sorted(tickers))}")
        return result

    async def get_market_spike(self) -> List[dict]:
        """Volatility spike indicator."""
        result = await self._get("/market/spike", cache_key="spike")
        return result.get("data", [])

    async def get_insider_aggregate(self) -> dict:
        """Total insider buy vs sell across market."""
        result = await self._get("/market/insider-buy-sells", cache_key="insider_agg")
        return result.get("data", {})

    async def get_sector_etfs(self) -> List[dict]:
        """Sector ETF performance and flow data."""
        result = await self._get("/market/sector-etfs", cache_key="sector_etfs")
        return result.get("data", [])

    async def get_economic_calendar(self) -> List[dict]:
        """Upcoming economic events."""
        result = await self._get("/market/economic-calendar", cache_key="econ_cal")
        return result.get("data", [])

    # ── Congress trades ──

    async def get_congress_trades(self) -> List[dict]:
        """Recent congressional trading activity."""
        result = await self._get("/congress/recent-trades", cache_key="congress")
        return result.get("data", [])

    async def get_congress_late_reports(self) -> List[dict]:
        """Late-filed congressional reports (potential alpha)."""
        result = await self._get("/congress/late-reports", cache_key="congress_late")
        return result.get("data", [])

    # ── Dark pool ──

    async def get_darkpool_recent(self) -> List[dict]:
        """Recent dark pool trades across all tickers."""
        result = await self._get("/darkpool/recent", cache_key="dp_recent")
        return result.get("data", [])

    async def get_darkpool_ticker(self, ticker: str) -> List[dict]:
        """Dark pool trades for a specific ticker."""
        result = await self._get(f"/darkpool/{ticker}",
                                 cache_key=f"dp_{ticker}")
        return result.get("data", [])

    # ── Insider transactions ──

    async def get_insider_transactions(self) -> List[dict]:
        """Recent insider buy/sell transactions."""
        result = await self._get("/insider/transactions", cache_key="insider_txn")
        return result.get("data", [])

    async def get_insider_ticker(self, ticker: str) -> List[dict]:
        """Insider activity for a specific ticker."""
        result = await self._get(f"/insider/{ticker}",
                                 cache_key=f"insider_{ticker}")
        return result.get("data", [])

    # ── Institutional ──

    async def get_institutional_activity(self) -> List[dict]:
        """Recent institutional trading activity."""
        result = await self._get("/institution/activity-v2", cache_key="inst_act")
        return result.get("data", [])

    async def get_institutional_holdings(self, ticker: str) -> List[dict]:
        """Institutional holders for a ticker."""
        result = await self._get(f"/institution/{ticker}/holdings",
                                 cache_key=f"inst_hold_{ticker}")
        return result.get("data", [])

    # ── Options flow per ticker ──

    async def get_flow_alerts(self, ticker: str) -> List[dict]:
        """Options flow alerts for a ticker (unusual activity)."""
        result = await self._get(f"/stock/{ticker}/flow-alerts",
                                 cache_key=f"flow_{ticker}")
        return result.get("data", [])

    async def get_greek_exposure(self, ticker: str) -> dict:
        """Gamma/delta exposure for a ticker."""
        result = await self._get(f"/stock/{ticker}/greek-exposure",
                                 cache_key=f"gex_{ticker}")
        return result.get("data", {})

    # ── ETF flows ──

    async def get_etf_flows(self, ticker: str) -> List[dict]:
        """ETF inflow/outflow data."""
        result = await self._get(f"/etfs/{ticker}/in-outflow",
                                 cache_key=f"etf_flow_{ticker}")
        return result.get("data", [])

    # ── Screener ──

    async def get_analyst_ratings(self) -> List[dict]:
        """Recent analyst rating changes."""
        result = await self._get("/screener/analyst-ratings", cache_key="analyst")
        return result.get("data", [])

    # ── Seasonality ──

    async def get_market_seasonality(self) -> dict:
        """Historical market seasonality patterns."""
        result = await self._get("/seasonality/market", cache_key="seasonality")
        return result.get("data", {})

    # ── Short interest ──

    async def get_short_data(self, ticker: str) -> dict:
        """Short interest and volume data for a ticker."""
        result = await self._get(f"/short/{ticker}/data",
                                 cache_key=f"short_{ticker}")
        return result.get("data", {})

    # ── Aggregated signals for AlphaDesk ──

    async def get_macro_snapshot(self) -> dict:
        """
        Aggregate macro snapshot for the Macro Agent.
        Returns a dict with market_tide, insider_sentiment, and sector_flows.
        """
        tide = await self.get_market_tide()
        insider_agg = await self.get_insider_aggregate()
        sectors = await self.get_sector_etfs()

        # Compute tide sentiment from latest data
        tide_sentiment = "neutral"
        if tide:
            latest = tide[-1]
            net_call = float(latest.get("net_call_premium", 0))
            net_put = float(latest.get("net_put_premium", 0))
            net = net_call + net_put  # put premium is already negative
            if net > 50_000_000:
                tide_sentiment = "strong_bullish"
            elif net > 10_000_000:
                tide_sentiment = "bullish"
            elif net < -50_000_000:
                tide_sentiment = "strong_bearish"
            elif net < -10_000_000:
                tide_sentiment = "bearish"

        return {
            "tide_sentiment": tide_sentiment,
            "tide_latest": tide[-1] if tide else {},
            "insider_aggregate": insider_agg,
            "sector_etfs": sectors[:10] if sectors else [],
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_ticker_signals(self, ticker: str) -> dict:
        """
        Aggregate alternative data signals for a specific ticker.
        Used by Value/Contrarian agents during signal evaluation.
        """
        darkpool = await self.get_darkpool_ticker(ticker)
        insiders = await self.get_insider_ticker(ticker)
        flow = await self.get_flow_alerts(ticker)

        # Dark pool sentiment: net premium direction
        dp_premium = 0
        dp_count = 0
        for trade in (darkpool or [])[:50]:
            dp_premium += float(trade.get("premium", 0))
            dp_count += 1

        # Insider sentiment: net buys vs sells
        insider_buys = 0
        insider_sells = 0
        for txn in (insiders or [])[:20]:
            txn_type = (txn.get("transaction_type") or txn.get("txn_type") or "").lower()
            if "buy" in txn_type or "purchase" in txn_type:
                insider_buys += 1
            elif "sell" in txn_type or "sale" in txn_type:
                insider_sells += 1

        # Options flow: unusual activity count
        unusual_flow_count = len(flow or [])

        return {
            "ticker": ticker,
            "darkpool_premium": dp_premium,
            "darkpool_trades": dp_count,
            "insider_buys": insider_buys,
            "insider_sells": insider_sells,
            "insider_net": insider_buys - insider_sells,
            "unusual_flow_count": unusual_flow_count,
            "flow_alerts": (flow or [])[:5],
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_congress_signals(self) -> List[dict]:
        """
        Parse congress trades into actionable signals.
        Returns list of {ticker, direction, politician, amount, date}.
        """
        trades = await self.get_congress_trades()
        signals = []
        for t in (trades or []):
            ticker = t.get("ticker")
            if not ticker:
                continue
            txn = (t.get("txn_type") or "").lower()
            direction = "buy" if "buy" in txn or "purchase" in txn else "sell"
            signals.append({
                "ticker": ticker,
                "direction": direction,
                "politician": t.get("name", "Unknown"),
                "amount": t.get("amounts", ""),
                "date": t.get("transaction_date", ""),
                "filed": t.get("filed_at_date", ""),
            })
        return signals
