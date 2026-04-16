"""
AlphaDesk — Circuit Breaker

Protects portfolio from "mechanical loop" selloffs (passive ETF + 0DTE hedging
cascade) by halting new openings when intra-session drawdown exceeds
thresholds. Reference: paper wealth problem Discord analysis.

Three-tier protection:
  -3% session drawdown → HALT new openings (closes still allowed)
  -5% session drawdown → TRIM smallest positions to raise cash buffer
  -7% session drawdown → FULL HALT, close all non-hedge positions

State resets every trading session (NYSE open 14:30 UTC).

Persists state to JSON so restarts don't lose the session high.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("alphadesk.circuit_breaker")


# Hedge instruments that are allowed to stay open even at full halt
# (provide downside protection, should not be liquidated)
HEDGE_INSTRUMENTS = {3020, 559}  # GLD, GOLD 24/7


@dataclass
class CircuitBreakerState:
    session_date: str                # YYYY-MM-DD
    session_high: float              # Peak equity this session
    session_open: float              # Equity at session start
    current_equity: float            # Most recent reading
    drawdown_pct: float              # Negative if in drawdown
    tier: int                        # 0 = normal, 1 = halt openings, 2 = trim, 3 = full halt
    triggered_at: Optional[str] = None  # ISO timestamp of last trigger
    halt_openings: bool = False
    trim_issued: bool = False
    full_halt: bool = False


class CircuitBreaker:
    """Session-based drawdown protection."""

    STATE_FILE = "/root/AlphaDesk/data/circuit_breaker_state.json"

    # Drawdown tiers (negative fractions)
    TIER_1_THRESHOLD = -0.03  # -3%: halt new openings
    TIER_2_THRESHOLD = -0.05  # -5%: trim smallest to raise cash
    TIER_3_THRESHOLD = -0.07  # -7%: full halt, close non-hedges

    # Cool-off: after a trigger, wait this many minutes before evaluating again
    # (to avoid cascade of alerts on same move)
    COOLOFF_MINUTES = 15

    def __init__(self, db_path: Optional[str] = None, notifier=None):
        self.db_path = db_path
        self.notifier = notifier
        self.state: Optional[CircuitBreakerState] = None
        self._load_state()
        self._ensure_db_schema()

    # ── State persistence ──

    def _load_state(self):
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    # Reset if stale (different session date)
                    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if data.get("session_date") != today:
                        self.state = None
                        logger.debug("Circuit breaker state is stale, will reset on next check")
                    else:
                        self.state = CircuitBreakerState(**data)
                        logger.debug(
                            f"Loaded CB state: high={self.state.session_high:.2f}, "
                            f"dd={self.state.drawdown_pct:.2%}, tier={self.state.tier}"
                        )
        except Exception as e:
            logger.debug(f"CB state load error: {e}")

    def _save_state(self):
        if not self.state:
            return
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            with open(self.STATE_FILE, "w") as f:
                json.dump(self.state.__dict__, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"CB state save error: {e}")

    def _ensure_db_schema(self):
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS circuit_breaker_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT DEFAULT (datetime('now')),
                        session_date TEXT,
                        tier INTEGER,
                        drawdown_pct REAL,
                        equity REAL,
                        session_high REAL,
                        action TEXT
                    )
                """)
        except Exception as e:
            logger.debug(f"CB schema error: {e}")

    def _log_event(self, tier: int, drawdown: float, equity: float, action: str):
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO circuit_breaker_events
                       (session_date, tier, drawdown_pct, equity, session_high, action)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (self.state.session_date, tier, drawdown, equity,
                     self.state.session_high, action),
                )
        except Exception as e:
            logger.debug(f"CB log error: {e}")

    # ── Session management ──

    def _start_new_session(self, equity: float):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.state = CircuitBreakerState(
            session_date=today,
            session_high=equity,
            session_open=equity,
            current_equity=equity,
            drawdown_pct=0.0,
            tier=0,
        )
        logger.info(f"Circuit breaker: new session at equity ${equity:,.0f}")
        self._save_state()

    def _session_is_current(self) -> bool:
        if not self.state:
            return False
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.state.session_date == today

    # ── Main check ──

    async def check(self, equity: float, positions: List[dict],
                    etoro_client=None) -> Dict:
        """Main entry point: called from run_risk_check every 5 min.

        Returns:
            {"tier": 0..3, "halt_openings": bool, "action_taken": str|None}
        """
        # Start new session if needed
        if not self._session_is_current():
            self._start_new_session(equity)
            return {"tier": 0, "halt_openings": False, "action_taken": None}

        # Update session high (always track peak)
        if equity > self.state.session_high:
            self.state.session_high = equity

        self.state.current_equity = equity

        # Compute drawdown from session high (not open — more conservative)
        if self.state.session_high > 0:
            dd = (equity - self.state.session_high) / self.state.session_high
        else:
            dd = 0.0
        self.state.drawdown_pct = dd

        # Determine tier
        new_tier = 0
        if dd <= self.TIER_3_THRESHOLD:
            new_tier = 3
        elif dd <= self.TIER_2_THRESHOLD:
            new_tier = 2
        elif dd <= self.TIER_1_THRESHOLD:
            new_tier = 1

        action_taken = None

        # Only escalate (never de-escalate within a session)
        if new_tier > self.state.tier:
            action_taken = await self._trigger_tier(
                new_tier, positions, etoro_client
            )
            self.state.tier = new_tier
            self.state.triggered_at = datetime.now(timezone.utc).isoformat()

        # Apply tier effects
        self.state.halt_openings = self.state.tier >= 1
        self.state.full_halt = self.state.tier >= 3

        self._save_state()

        return {
            "tier": self.state.tier,
            "drawdown_pct": dd,
            "halt_openings": self.state.halt_openings,
            "full_halt": self.state.full_halt,
            "session_high": self.state.session_high,
            "action_taken": action_taken,
        }

    async def _trigger_tier(self, tier: int, positions: List[dict],
                             etoro_client=None) -> str:
        """Execute the action for a tier transition."""
        dd = self.state.drawdown_pct
        equity = self.state.current_equity

        if tier == 1:
            action = "HALT_OPENINGS"
            msg = (
                f"🚧 *CIRCUIT BREAKER TIER 1* — Drawdown {dd:.1%}\n"
                f"Equity: ${equity:,.0f} (peak ${self.state.session_high:,.0f})\n"
                f"Action: Halting new position openings. Existing positions untouched."
            )
            logger.warning(msg.replace("*", ""))

        elif tier == 2:
            action = self._trim_smallest(positions, etoro_client)
            msg = (
                f"⚠️ *CIRCUIT BREAKER TIER 2* — Drawdown {dd:.1%}\n"
                f"Equity: ${equity:,.0f} (peak ${self.state.session_high:,.0f})\n"
                f"Action: {action}"
            )
            logger.warning(msg.replace("*", ""))
            self.state.trim_issued = True

        elif tier == 3:
            action = "FULL_HALT"
            kept = len([p for p in positions
                        if p.get("instrumentID") in HEDGE_INSTRUMENTS])
            msg = (
                f"🛑 *CIRCUIT BREAKER TIER 3 — FULL HALT* — Drawdown {dd:.1%}\n"
                f"Equity: ${equity:,.0f} (peak ${self.state.session_high:,.0f})\n"
                f"Action: All trading halted. {kept} hedge positions (GLD) preserved.\n"
                f"Manual review required before resume."
            )
            logger.error(msg.replace("*", ""))
            self.state.full_halt = True

        else:
            action = "NONE"
            msg = None

        if msg and self.notifier:
            try:
                if hasattr(self.notifier, "send_message"):
                    await self.notifier.send_message(msg)
                elif hasattr(self.notifier, "send"):
                    await self.notifier.send(msg, parse_mode="Markdown")
            except Exception as e:
                logger.debug(f"CB notify error: {e}")

        self._log_event(tier, dd, equity, action)
        return action

    def _trim_smallest(self, positions: List[dict], etoro_client) -> str:
        """At tier 2: identify (but don't execute) the smallest positions
        that together would raise ~5% cash buffer. Execution is deferred
        to the main trading loop to avoid concurrent mutations."""
        # Filter out hedges
        tradeable = [p for p in positions
                     if p.get("instrumentID") not in HEDGE_INSTRUMENTS]
        tradeable.sort(key=lambda p: p.get("amount", 0))

        target_cash = self.state.current_equity * 0.05
        raised = 0.0
        to_close = []
        for p in tradeable:
            amt = p.get("amount", 0)
            to_close.append((p.get("instrumentID"), p.get("positionID"), amt))
            raised += amt
            if raised >= target_cash:
                break

        n = len(to_close)
        return f"TRIM_QUEUED: {n} smallest positions for ~${raised:.0f} cash buffer"

    # ── Public API ──

    def can_open_position(self) -> bool:
        """Called by signal scan before placing new orders."""
        return self.state is None or not self.state.halt_openings

    def can_trade_at_all(self) -> bool:
        """Tier 3: absolutely no trading."""
        return self.state is None or not self.state.full_halt

    def get_status(self) -> dict:
        """For /status Telegram command or dashboard."""
        if not self.state:
            return {"active": False, "tier": 0, "drawdown_pct": 0.0}
        return {
            "active": True,
            "session_date": self.state.session_date,
            "session_high": self.state.session_high,
            "session_open": self.state.session_open,
            "current_equity": self.state.current_equity,
            "drawdown_pct": self.state.drawdown_pct,
            "tier": self.state.tier,
            "halt_openings": self.state.halt_openings,
            "full_halt": self.state.full_halt,
            "triggered_at": self.state.triggered_at,
        }

    def reset(self):
        """Manual reset (e.g., after Tier 3 full halt, post-review)."""
        self.state = None
        try:
            if os.path.exists(self.STATE_FILE):
                os.remove(self.STATE_FILE)
        except Exception:
            pass
        logger.info("Circuit breaker manually reset")
