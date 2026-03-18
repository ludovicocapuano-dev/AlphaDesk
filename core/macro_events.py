"""
AlphaDesk — Macro Event Trading (BLS/FRED Data Releases)

Polls government data releases (NFP, CPI, PPI) at publication time,
compares actual vs consensus, and executes directional trades on
eToro CFDs (SPX500, NSDQ100, GOLD, EURUSD).

Inspired by: "19 lines of python that reads it 0.8 seconds after publication"
Adapted for eToro's ~5s API latency — we catch the 15-minute move, not the 2-second one.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd

logger = logging.getLogger("alphadesk.macro_events")

# ── Event Definitions ──

MACRO_EVENTS = {
    "NFP": {
        "name": "Nonfarm Payrolls",
        "fred_series": "PAYEMS",  # Total nonfarm, thousands
        "bls_url": "https://www.bls.gov/news.release/empsit.nr0.htm",
        "surprise_threshold": 50,  # ±50K from consensus = tradeable
        "instruments": {
            "positive_surprise": [  # Strong economy
                {"symbol": "SPX500", "direction": "BUY", "weight": 0.5},
                {"symbol": "USDJPY", "direction": "BUY", "weight": 0.3},
                {"symbol": "GOLD", "direction": "SELL", "weight": 0.2},
            ],
            "negative_surprise": [  # Weak economy
                {"symbol": "SPX500", "direction": "SELL", "weight": 0.3},
                {"symbol": "GOLD", "direction": "BUY", "weight": 0.4},
                {"symbol": "NSDQ100", "direction": "SELL", "weight": 0.3},
            ],
        },
    },
    "CPI": {
        "name": "Consumer Price Index",
        "fred_series": "CPIAUCSL",
        "bls_url": "https://www.bls.gov/news.release/cpi.nr0.htm",
        "surprise_threshold": 0.1,  # ±0.1% from consensus
        "instruments": {
            "positive_surprise": [  # Hotter inflation → hawkish
                {"symbol": "NSDQ100", "direction": "SELL", "weight": 0.4},
                {"symbol": "GOLD", "direction": "SELL", "weight": 0.3},
                {"symbol": "EURUSD", "direction": "SELL", "weight": 0.3},
            ],
            "negative_surprise": [  # Cooler inflation → dovish
                {"symbol": "NSDQ100", "direction": "BUY", "weight": 0.4},
                {"symbol": "GOLD", "direction": "BUY", "weight": 0.3},
                {"symbol": "EURUSD", "direction": "BUY", "weight": 0.3},
            ],
        },
    },
    "PPI": {
        "name": "Producer Price Index",
        "fred_series": "PPIACO",
        "bls_url": "https://www.bls.gov/news.release/ppi.nr0.htm",
        "surprise_threshold": 0.2,  # ±0.2% from consensus
        "instruments": {
            "positive_surprise": [
                {"symbol": "NSDQ100", "direction": "SELL", "weight": 0.5},
                {"symbol": "GOLD", "direction": "SELL", "weight": 0.5},
            ],
            "negative_surprise": [
                {"symbol": "NSDQ100", "direction": "BUY", "weight": 0.5},
                {"symbol": "GOLD", "direction": "BUY", "weight": 0.5},
            ],
        },
    },
}

# eToro instrument IDs for macro trading
MACRO_INSTRUMENT_IDS = {
    "SPX500": 260017,   # S&P 500 CFD
    "NSDQ100": 260019,  # NASDAQ 100 CFD
    "GOLD": 5002,       # Gold CFD
    "EURUSD": 1,        # EUR/USD
    "USDJPY": 3,        # USD/JPY
    "OIL": 5001,        # Crude Oil CFD
}


class MacroEventTrader:
    """Monitors and trades government data releases."""

    def __init__(self, etoro_client, notifier, max_risk_pct: float = 0.03):
        self.etoro = etoro_client
        self.notifier = notifier
        self.max_risk_pct = max_risk_pct  # Max 3% of portfolio per event
        self._fred_key = os.getenv("FRED_API_KEY", "")
        self._schedule_cache = None
        self._consensus_cache: Dict[str, float] = {}

    # ── Schedule Management ──

    def get_upcoming_events(self, days_ahead: int = 7) -> List[dict]:
        """Get upcoming BLS releases from the ICS calendar."""
        try:
            resp = requests.get(
                "https://www.bls.gov/schedule/news_release/bls.ics",
                timeout=10,
                headers={"User-Agent": "AlphaDesk/2.0"},
            )
            if resp.status_code != 200:
                logger.warning(f"BLS calendar fetch failed: {resp.status_code}")
                return []

            events = []
            now = datetime.utcnow()
            cutoff = now + timedelta(days=days_ahead)

            # Parse ICS manually (avoid icalendar dependency)
            current_event = {}
            for line in resp.text.split("\n"):
                line = line.strip()
                if line.startswith("BEGIN:VEVENT"):
                    current_event = {}
                elif line.startswith("END:VEVENT"):
                    if current_event.get("dtstart"):
                        dt = current_event["dtstart"]
                        if now <= dt <= cutoff:
                            events.append(current_event)
                    current_event = {}
                elif line.startswith("SUMMARY:"):
                    current_event["summary"] = line[8:]
                elif line.startswith("DTSTART"):
                    # Format: DTSTART;VALUE=DATE:20260403 or DTSTART:20260403T083000Z
                    date_str = line.split(":")[-1]
                    try:
                        if "T" in date_str:
                            dt = datetime.strptime(
                                date_str.replace("Z", ""), "%Y%m%dT%H%M%S"
                            )
                            # BLS times are ET — convert to UTC
                            # EDT (Mar-Nov): UTC-4, EST (Nov-Mar): UTC-5
                            if dt.month >= 3 and dt.month <= 11:
                                dt = dt + timedelta(hours=4)  # EDT → UTC
                            else:
                                dt = dt + timedelta(hours=5)  # EST → UTC
                            current_event["dtstart"] = dt
                        else:
                            # Date only — default to 8:30 AM ET
                            dt = datetime.strptime(date_str, "%Y%m%d")
                            if dt.month >= 3 and dt.month <= 11:
                                dt = dt.replace(hour=12, minute=30)  # 8:30 EDT = 12:30 UTC
                            else:
                                dt = dt.replace(hour=13, minute=30)  # 8:30 EST = 13:30 UTC
                            current_event["dtstart"] = dt
                    except ValueError:
                        pass

            # Filter for events we care about
            tradeable = []
            for ev in events:
                summary = ev.get("summary", "").lower()
                for event_key, event_def in MACRO_EVENTS.items():
                    if event_def["name"].lower() in summary:
                        tradeable.append({
                            "event": event_key,
                            "name": event_def["name"],
                            "datetime": ev["dtstart"],
                            "summary": ev.get("summary", ""),
                        })

            return sorted(tradeable, key=lambda x: x["datetime"])

        except Exception as e:
            logger.error(f"Failed to fetch BLS schedule: {e}")
            return []

    # ── Consensus Estimates ──

    async def fetch_consensus(self, event_key: str) -> Optional[float]:
        """Fetch consensus estimate from Unusual Whales economic calendar."""
        # Try UW first
        try:
            uw_key = os.getenv("UNUSUAL_WHALES_API_KEY", "")
            if uw_key:
                resp = requests.get(
                    "https://api.unusualwhales.com/api/economic-calendar",
                    headers={"Authorization": f"Bearer {uw_key}"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    events = data.get("data", data) if isinstance(data, dict) else data
                    if isinstance(events, list):
                        event_name = MACRO_EVENTS[event_key]["name"].lower()
                        for ev in events:
                            name = str(ev.get("name", ev.get("event", ""))).lower()
                            if event_name in name:
                                forecast = ev.get("forecast", ev.get("consensus"))
                                if forecast is not None:
                                    self._consensus_cache[event_key] = float(forecast)
                                    return float(forecast)
        except Exception as e:
            logger.debug(f"UW consensus fetch failed: {e}")

        return self._consensus_cache.get(event_key)

    # ── Data Polling ──

    def fetch_actual_fred(self, series_id: str) -> Optional[Tuple[float, float]]:
        """Fetch latest actual value from FRED. Returns (latest, previous)."""
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self._fred_key or "DEMO_KEY",
                "file_type": "json",
                "sort_order": "desc",
                "limit": 2,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return None

            obs = resp.json().get("observations", [])
            if len(obs) >= 2:
                latest = float(obs[0]["value"])
                previous = float(obs[1]["value"])
                return (latest, previous)
        except Exception as e:
            logger.error(f"FRED fetch failed for {series_id}: {e}")
        return None

    def poll_bls_page(self, url: str, max_attempts: int = 60,
                      interval_ms: int = 200) -> Optional[str]:
        """Poll BLS page every interval_ms until content appears.
        Used right at release time (e.g., 8:30 AM ET).
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (AlphaDesk Quant System)",
            "Accept": "text/html",
        }
        for attempt in range(max_attempts):
            try:
                resp = requests.get(url, headers=headers,
                                    timeout=max(0.5, interval_ms / 1000))
                if resp.status_code == 200 and len(resp.text) > 1000:
                    return resp.text
            except requests.RequestException:
                pass
            import time
            time.sleep(interval_ms / 1000)

        return None

    def extract_nfp_from_html(self, html: str) -> Optional[int]:
        """Extract nonfarm payroll change from BLS HTML report."""
        # Pattern: "Total nonfarm payroll employment [verb] by X,XXX"
        patterns = [
            r'nonfarm payroll employment\s+\w+\s+(?:by\s+)?([\d,]+)',
            r'payroll employment\s+\w+\s+(?:by\s+)?([\d,]+)',
            r'([\d,]+)\s+(?:jobs|workers)\s+(?:were|was)\s+(?:added|lost)',
        ]
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                num_str = match.group(1).replace(",", "")
                value = int(num_str)
                # Check if it was a decrease
                if any(word in html[:match.start()].lower()
                       for word in ["decreased", "fell", "lost", "down"]):
                    value = -value
                return value
        return None

    # ── Trade Execution ──

    async def execute_macro_trade(self, event_key: str, surprise: float,
                                   portfolio_value: float) -> List[dict]:
        """Execute trades based on macro surprise direction and magnitude."""
        event_def = MACRO_EVENTS[event_key]
        threshold = event_def["surprise_threshold"]

        if abs(surprise) < threshold:
            logger.info(f"[{event_key}] Surprise {surprise} below threshold {threshold} — no trade")
            return []

        # Determine direction
        if surprise > 0:
            trades_config = event_def["instruments"]["positive_surprise"]
        else:
            trades_config = event_def["instruments"]["negative_surprise"]

        # Size: proportional to surprise magnitude, capped at max_risk_pct
        surprise_magnitude = min(abs(surprise) / threshold, 3.0)  # Cap at 3x
        total_risk = self.max_risk_pct * surprise_magnitude / 3.0
        total_amount = portfolio_value * total_risk

        executed = []
        for trade in trades_config:
            symbol = trade["symbol"]
            instrument_id = MACRO_INSTRUMENT_IDS.get(symbol)
            if not instrument_id:
                continue

            amount = total_amount * trade["weight"]
            amount = max(amount, 50)  # eToro minimum

            direction = trade["direction"]
            try:
                result = await self.etoro.open_trade(
                    instrument_id=instrument_id,
                    amount=amount,
                    is_buy=(direction == "BUY"),
                )
                order_id = result.get("orderId")
                logger.info(
                    f"[MACRO {event_key}] {direction} ${amount:.0f} {symbol} "
                    f"(surprise={surprise:+.1f}) → order {order_id}"
                )
                executed.append({
                    "symbol": symbol,
                    "direction": direction,
                    "amount": amount,
                    "order_id": order_id,
                    "surprise": surprise,
                })
            except Exception as e:
                logger.error(f"[MACRO {event_key}] Failed to {direction} {symbol}: {e}")

        return executed

    # ── Main Event Handler ──

    async def handle_event(self, event_key: str, portfolio_value: float) -> dict:
        """Full event handling: poll → extract → compare → trade → notify."""
        event_def = MACRO_EVENTS[event_key]
        logger.info(f"[MACRO] Handling {event_def['name']} release...")

        # 1. Get consensus
        consensus = await self.fetch_consensus(event_key)

        # 2. Poll for actual value
        if event_key == "NFP":
            # Try BLS HTML first (fastest)
            html = self.poll_bls_page(event_def["bls_url"],
                                       max_attempts=60, interval_ms=200)
            actual = self.extract_nfp_from_html(html) if html else None

            if actual is None:
                # Fallback to FRED (may have slight delay)
                fred_data = self.fetch_actual_fred(event_def["fred_series"])
                if fred_data:
                    latest, previous = fred_data
                    actual = latest - previous  # Month-over-month change

        else:
            # CPI/PPI: use FRED
            fred_data = self.fetch_actual_fred(event_def["fred_series"])
            if fred_data:
                latest, previous = fred_data
                actual = ((latest - previous) / previous) * 100  # M/M % change

        if actual is None:
            msg = f"⚠️ [MACRO] {event_def['name']}: could not extract actual value"
            logger.warning(msg)
            await self.notifier.send_message(msg)
            return {"event": event_key, "status": "failed", "reason": "no_data"}

        # 3. Compute surprise
        if consensus is not None:
            surprise = actual - consensus
        else:
            # No consensus — use previous month's change as baseline
            logger.warning(f"[MACRO] No consensus for {event_key}, using previous as baseline")
            surprise = actual  # Treat the raw number as the surprise signal

        # 4. Notify
        direction = "above" if surprise > 0 else "below"
        msg = (
            f"🏛️ *{event_def['name']} RELEASED*\n\n"
            f"📊 Actual: {actual:+.1f}\n"
            f"📋 Consensus: {consensus if consensus else 'N/A'}\n"
            f"⚡ Surprise: {surprise:+.1f} ({direction} expectations)\n"
            f"📈 Threshold: ±{event_def['surprise_threshold']}"
        )

        # 5. Trade if surprise is significant
        trades = []
        if abs(surprise) >= event_def["surprise_threshold"]:
            msg += f"\n\n🔥 *TRADING* — surprise exceeds threshold"
            trades = await self.execute_macro_trade(
                event_key, surprise, portfolio_value
            )
            if trades:
                trade_lines = "\n".join(
                    f"  {'🟢' if t['direction']=='BUY' else '🔴'} "
                    f"{t['direction']} ${t['amount']:.0f} {t['symbol']}"
                    for t in trades
                )
                msg += f"\n\n{trade_lines}"
        else:
            msg += f"\n\n😐 No trade — surprise within normal range"

        await self.notifier.send_message(msg)

        return {
            "event": event_key,
            "actual": actual,
            "consensus": consensus,
            "surprise": surprise,
            "trades": trades,
            "status": "traded" if trades else "no_trade",
        }
