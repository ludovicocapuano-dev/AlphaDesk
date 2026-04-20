"""
AlphaDesk — Oil Futures Monitor

Tracks WTI and Brent crude oil futures in real-time to inform energy
position decisions. Critical for our 46% energy exposure.

Data source: yfinance (CL=F for WTI, BZ=F for Brent).
Runs every 15 minutes during market hours, every hour off-hours.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("alphadesk.oil_monitor")


@dataclass
class OilSnapshot:
    timestamp: datetime
    wti: float
    brent: float
    wti_change_1d: float
    wti_change_5d: float
    brent_change_1d: float
    wti_level_regime: str  # low/medium/high/extreme
    divergence: float      # brent - wti spread
    signal: str            # NEUTRAL/BULLISH/BEARISH/EXTREME_MOVE


class OilMonitor:
    """Real-time oil futures tracking for energy position risk management."""

    STATE_FILE = "/root/AlphaDesk/data/oil_monitor_state.json"

    # Regime thresholds (WTI)
    WTI_EXTREME_LOW = 60
    WTI_LOW = 75
    WTI_MEDIUM = 85
    WTI_HIGH = 100
    WTI_EXTREME_HIGH = 115

    # Alert thresholds
    BIG_MOVE_THRESHOLD_PCT = 0.05   # ±5% in a session = alert
    DIVERGENCE_ALERT = 15           # Brent-WTI spread > $15 = alert (unusual)

    def __init__(self, db_path: Optional[str] = None, notifier=None):
        self.db_path = db_path
        self.notifier = notifier
        self._last_snapshot: Optional[OilSnapshot] = None
        self._last_alert: Optional[datetime] = None
        self._ensure_db_schema()
        self._load_state()

    def _ensure_db_schema(self):
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS oil_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT DEFAULT (datetime('now')),
                        wti REAL,
                        brent REAL,
                        wti_change_1d REAL,
                        wti_change_5d REAL,
                        brent_change_1d REAL,
                        divergence REAL,
                        regime TEXT,
                        signal TEXT
                    )
                """)
        except Exception as e:
            logger.debug(f"Oil monitor schema error: {e}")

    def _load_state(self):
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    ts = data.get("last_alert_timestamp")
                    if ts:
                        self._last_alert = datetime.fromisoformat(ts)
        except Exception:
            pass

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "last_alert_timestamp":
                        self._last_alert.isoformat() if self._last_alert else None,
                }, f)
        except Exception:
            pass

    # ── Data fetching ──

    def fetch_current_prices(self) -> Optional[Tuple[float, float, pd.Series, pd.Series]]:
        """Fetch latest WTI (CL=F) and Brent (BZ=F) futures prices.

        Returns: (wti_current, brent_current, wti_history_5d, brent_history_5d)
        """
        try:
            import yfinance as yf

            # Fetch last 10 days for both (extra buffer for weekend gaps)
            data = yf.download(
                tickers=["CL=F", "BZ=F"],
                period="10d",
                interval="1h",
                progress=False,
                auto_adjust=True,
                threads=True,
                group_by="column",
            )

            if data is None or data.empty:
                return None

            # Extract close prices per ticker
            if "Close" in data.columns.get_level_values(0):
                closes = data["Close"]
            else:
                return None

            wti_series = closes["CL=F"].dropna()
            brent_series = closes["BZ=F"].dropna()

            if wti_series.empty or brent_series.empty:
                return None

            return (
                float(wti_series.iloc[-1]),
                float(brent_series.iloc[-1]),
                wti_series,
                brent_series,
            )
        except Exception as e:
            logger.warning(f"Oil price fetch failed: {e}")
            return None

    # ── Analysis ──

    def _compute_regime(self, wti: float) -> str:
        if wti < self.WTI_EXTREME_LOW:
            return "extreme_low"
        if wti < self.WTI_LOW:
            return "low"
        if wti < self.WTI_MEDIUM:
            return "medium"
        if wti < self.WTI_HIGH:
            return "high"
        if wti < self.WTI_EXTREME_HIGH:
            return "very_high"
        return "extreme_high"

    def _compute_signal(self, wti: float, wti_1d: float,
                        wti_5d: float, divergence: float) -> str:
        """Signal for energy position management."""
        # Extreme move
        if abs(wti_1d) >= self.BIG_MOVE_THRESHOLD_PCT:
            return "EXTREME_MOVE"

        # Bearish conditions: WTI down 5d + low regime + normalizing spread
        if wti_5d < -0.10 and wti < self.WTI_MEDIUM:
            return "BEARISH"

        # Bullish: WTI up 5d + high regime persists
        if wti_5d > 0.10 and wti > self.WTI_HIGH:
            return "BULLISH"

        # Divergence alert (relationship broken)
        if abs(divergence) > self.DIVERGENCE_ALERT:
            return "DIVERGENCE"

        return "NEUTRAL"

    async def snapshot(self) -> Optional[OilSnapshot]:
        """Take a snapshot of current oil market state."""
        fetched = self.fetch_current_prices()
        if not fetched:
            return None

        wti, brent, wti_series, brent_series = fetched

        # Price changes
        wti_24h_ago = wti_series.iloc[-24] if len(wti_series) >= 24 else wti_series.iloc[0]
        wti_5d_ago = wti_series.iloc[-120] if len(wti_series) >= 120 else wti_series.iloc[0]
        brent_24h_ago = brent_series.iloc[-24] if len(brent_series) >= 24 else brent_series.iloc[0]

        wti_1d = (wti - float(wti_24h_ago)) / float(wti_24h_ago) if wti_24h_ago else 0
        wti_5d = (wti - float(wti_5d_ago)) / float(wti_5d_ago) if wti_5d_ago else 0
        brent_1d = (brent - float(brent_24h_ago)) / float(brent_24h_ago) if brent_24h_ago else 0

        divergence = brent - wti

        snap = OilSnapshot(
            timestamp=datetime.utcnow(),
            wti=wti,
            brent=brent,
            wti_change_1d=wti_1d,
            wti_change_5d=wti_5d,
            brent_change_1d=brent_1d,
            wti_level_regime=self._compute_regime(wti),
            divergence=divergence,
            signal=self._compute_signal(wti, wti_1d, wti_5d, divergence),
        )

        self._last_snapshot = snap
        self._log_snapshot(snap)
        await self._maybe_alert(snap)
        return snap

    def _log_snapshot(self, snap: OilSnapshot):
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO oil_snapshots
                       (wti, brent, wti_change_1d, wti_change_5d,
                        brent_change_1d, divergence, regime, signal)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (snap.wti, snap.brent, snap.wti_change_1d,
                     snap.wti_change_5d, snap.brent_change_1d,
                     snap.divergence, snap.wti_level_regime, snap.signal),
                )
        except Exception as e:
            logger.debug(f"Oil snapshot log error: {e}")

    async def _maybe_alert(self, snap: OilSnapshot):
        """Telegram alert on extreme moves (with cooloff)."""
        if snap.signal == "NEUTRAL":
            return
        if not self.notifier:
            return

        # Cooloff: 4h between alerts of same signal
        if self._last_alert:
            hours = (datetime.utcnow() - self._last_alert).total_seconds() / 3600
            if hours < 4:
                return

        icon = {
            "EXTREME_MOVE": "⚡",
            "BEARISH": "🔻",
            "BULLISH": "🔺",
            "DIVERGENCE": "⚠️",
        }.get(snap.signal, "📊")

        msg = (
            f"{icon} *OIL MONITOR — {snap.signal}*\n"
            f"WTI: ${snap.wti:.2f} ({snap.wti_change_1d:+.1%} 1d, "
            f"{snap.wti_change_5d:+.1%} 5d)\n"
            f"Brent: ${snap.brent:.2f} ({snap.brent_change_1d:+.1%} 1d)\n"
            f"Spread: ${snap.divergence:.2f} | Regime: {snap.wti_level_regime}\n\n"
            f"Energy portfolio exposure: ~46% (XLE+XOM+SLB)"
        )

        try:
            if hasattr(self.notifier, "send_message"):
                await self.notifier.send_message(msg)
            elif hasattr(self.notifier, "send"):
                await self.notifier.send(msg, parse_mode="Markdown")
            self._last_alert = datetime.utcnow()
            self._save_state()
        except Exception as e:
            logger.debug(f"Oil alert error: {e}")

    def get_current_recommendation(self) -> Dict:
        """Called by main.py before rebalance execution. Returns action hint."""
        if not self._last_snapshot:
            return {"action": "NEUTRAL", "reason": "no data"}

        snap = self._last_snapshot
        age_minutes = (datetime.utcnow() - snap.timestamp).total_seconds() / 60
        if age_minutes > 30:
            return {"action": "NEUTRAL", "reason": f"stale data ({age_minutes:.0f}min)"}

        # Decision matrix for energy position management
        if snap.wti < 75 and snap.wti_change_5d < -0.08:
            return {
                "action": "TRIM_ENERGY_AGGRESSIVE",
                "reason": f"WTI ${snap.wti:.2f} and down {snap.wti_change_5d:.1%} over 5d",
                "suggested_trim": 0.40,  # Trim 40% of energy exposure
            }

        if snap.wti < 85 and snap.wti_change_5d < -0.05:
            return {
                "action": "TRIM_ENERGY_MODERATE",
                "reason": f"WTI weakening, ${snap.wti:.2f} and {snap.wti_change_5d:.1%} 5d",
                "suggested_trim": 0.20,
            }

        if snap.wti > 100 and snap.wti_change_5d > 0.08:
            return {
                "action": "HOLD_ENERGY",
                "reason": f"WTI strong ${snap.wti:.2f}, +{snap.wti_change_5d:.1%} 5d",
                "suggested_trim": 0.0,
            }

        return {
            "action": "NEUTRAL",
            "reason": f"WTI ${snap.wti:.2f}, stable range",
            "suggested_trim": 0.10,  # Baseline trim per rebalance plan
        }
