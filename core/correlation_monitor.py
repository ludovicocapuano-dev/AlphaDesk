"""
AlphaDesk — Correlation Monitor

Detects when portfolio concentration risk becomes elevated by monitoring
realized correlation between position returns. Reference: paper wealth problem
Discord intel — "passive funds + Mag7 concentration = mechanical selloff loop."

Compute:
  - 20-day rolling correlation matrix of daily returns
  - Avg off-diagonal correlation (the "systemic" measure)
  - Effective number of bets = exp(-sum(p*log(p))) for eigenvalue spread
  - Max position correlation (if any pair > 0.85 flag as redundant)

Alert thresholds:
  avg_corr > 0.5  →  HIGH concentration warning
  avg_corr > 0.7  →  CRITICAL, suggest decorrelated add (GLD, TLT, cash)

Runs every 4 hours (same as drift monitor).
Data source: yfinance (no eToro dependency, works weekends).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("alphadesk.correlation_monitor")


HIGH_CORRELATION_THRESHOLD = 0.5
CRITICAL_CORRELATION_THRESHOLD = 0.7
REDUNDANT_PAIR_THRESHOLD = 0.85


# Decorrelated assets we'd want to add if concentration is too high
DECORRELATED_SUGGESTIONS = [
    "GLD",   # Gold (negative correlation to equities in crisis)
    "TLT",   # Long treasuries
    "VIXY",  # VIX ETF
    "DBMF",  # Managed futures
]


@dataclass
class CorrelationReport:
    timestamp: datetime
    tickers: List[str]
    weights: Dict[str, float]         # position weight in portfolio
    avg_correlation: float            # avg off-diagonal
    max_correlation: float            # highest pair correlation
    max_pair: Tuple[str, str]         # the redundant pair
    effective_bets: float             # 1/HHI-like measure
    severity: str                     # LOW/HIGH/CRITICAL
    redundant_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class CorrelationMonitor:
    """Monitors portfolio concentration risk via rolling correlations."""

    LOOKBACK_DAYS = 60   # fetch 60 days of data to compute 20d rolling
    WINDOW = 20          # correlation window
    MIN_POSITIONS = 3    # need ≥3 positions to compute meaningful stats
    MIN_WEIGHT = 0.01    # ignore positions < 1% of portfolio

    def __init__(self, db_path: Optional[str] = None, notifier=None):
        self.db_path = db_path
        self.notifier = notifier
        self._last_alert: Optional[datetime] = None
        self._ensure_db_schema()

    def _ensure_db_schema(self):
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS correlation_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT DEFAULT (datetime('now')),
                        avg_correlation REAL,
                        max_correlation REAL,
                        max_pair TEXT,
                        effective_bets REAL,
                        severity TEXT,
                        n_positions INTEGER,
                        tickers TEXT
                    )
                """)
        except Exception as e:
            logger.debug(f"Correlation schema error: {e}")

    # ── Data fetching ──

    def _fetch_returns(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        """Fetch daily returns for all tickers. Uses yfinance."""
        try:
            import yfinance as yf
            end = datetime.utcnow()
            start = end - timedelta(days=self.LOOKBACK_DAYS + 10)

            # Download as multi-ticker (adjusted close)
            data = yf.download(
                tickers=tickers,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
                threads=True,
                group_by="column",
            )

            if data is None or data.empty:
                return None

            if len(tickers) == 1:
                # Single ticker: yfinance returns different shape
                if "Close" in data.columns:
                    prices = data[["Close"]].rename(columns={"Close": tickers[0]})
                else:
                    return None
            else:
                # Multi ticker: pick the 'Close' level
                if "Close" in data.columns.get_level_values(0):
                    prices = data["Close"]
                else:
                    return None

            # Daily log returns (more stable than pct_change)
            returns = np.log(prices / prices.shift(1)).dropna(how="all")
            return returns
        except Exception as e:
            logger.warning(f"yfinance fetch error: {e}")
            return None

    # ── Compute ──

    def _compute_stats(self, returns: pd.DataFrame,
                        weights: Dict[str, float]) -> Optional[dict]:
        """Compute correlation matrix + aggregate stats."""
        if returns is None or returns.empty:
            return None

        # Keep only tickers present in both
        common = [c for c in returns.columns if c in weights]
        if len(common) < self.MIN_POSITIONS:
            return None

        rets = returns[common].tail(self.WINDOW).dropna()
        if len(rets) < 10:  # need at least 10 days of overlap
            return None

        corr = rets.corr()

        # Extract upper triangle (off-diagonal)
        n = len(common)
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        off_diag = corr.values[upper_mask]

        # Stats
        avg_corr = float(np.nanmean(off_diag))
        max_corr = float(np.nanmax(off_diag))

        # Find max pair
        max_idx = np.unravel_index(np.nanargmax(np.where(upper_mask, corr.values, -np.inf)),
                                    corr.shape)
        max_pair = (common[max_idx[0]], common[max_idx[1]])

        # Effective number of bets (weight-weighted)
        w = np.array([weights.get(c, 0) for c in common])
        w_sum = w.sum()
        if w_sum > 0:
            w_norm = w / w_sum
            # Herfindahl-inverse
            hhi = np.sum(w_norm ** 2)
            effective_bets = 1.0 / hhi if hhi > 0 else 0.0
        else:
            effective_bets = 0.0

        # Find redundant pairs (>0.85 correlation)
        redundant = []
        for i in range(n):
            for j in range(i + 1, n):
                c = corr.iloc[i, j]
                if not np.isnan(c) and c >= REDUNDANT_PAIR_THRESHOLD:
                    redundant.append((common[i], common[j], float(c)))
        redundant.sort(key=lambda x: -x[2])

        return {
            "corr_matrix": corr,
            "avg_corr": avg_corr,
            "max_corr": max_corr,
            "max_pair": max_pair,
            "effective_bets": effective_bets,
            "redundant_pairs": redundant,
            "tickers": common,
        }

    def _severity(self, avg_corr: float) -> str:
        if avg_corr >= CRITICAL_CORRELATION_THRESHOLD:
            return "CRITICAL"
        if avg_corr >= HIGH_CORRELATION_THRESHOLD:
            return "HIGH"
        return "LOW"

    # ── Main entry ──

    async def analyze(self, positions: List[dict]) -> Optional[CorrelationReport]:
        """Compute correlation report for current portfolio.

        Args:
            positions: list of dicts with 'symbol' and 'amount'
        """
        # Build weights
        total = sum(p.get("amount", 0) for p in positions)
        if total <= 0:
            return None

        weights = {}
        for p in positions:
            sym = p.get("symbol") or p.get("ticker")
            if not sym:
                continue
            amt = p.get("amount", 0) / total
            if amt >= self.MIN_WEIGHT:
                # Aggregate if duplicates
                weights[sym] = weights.get(sym, 0) + amt

        if len(weights) < self.MIN_POSITIONS:
            logger.debug(f"Correlation: only {len(weights)} positions >= {self.MIN_WEIGHT:.0%}, skipping")
            return None

        tickers = list(weights.keys())
        returns = self._fetch_returns(tickers)
        stats = self._compute_stats(returns, weights) if returns is not None else None

        if stats is None:
            logger.debug("Correlation: insufficient return data")
            return None

        severity = self._severity(stats["avg_corr"])

        # Build suggestions if concentrated
        suggestions = []
        if severity in ("HIGH", "CRITICAL"):
            for tic in DECORRELATED_SUGGESTIONS:
                if tic not in weights:
                    suggestions.append(tic)

        report = CorrelationReport(
            timestamp=datetime.utcnow(),
            tickers=stats["tickers"],
            weights={t: weights[t] for t in stats["tickers"]},
            avg_correlation=stats["avg_corr"],
            max_correlation=stats["max_corr"],
            max_pair=stats["max_pair"],
            effective_bets=stats["effective_bets"],
            severity=severity,
            redundant_pairs=stats["redundant_pairs"][:5],
            suggestions=suggestions,
        )

        self._log_report(report)
        await self._maybe_alert(report)

        return report

    def _log_report(self, report: CorrelationReport):
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO correlation_reports
                       (avg_correlation, max_correlation, max_pair,
                        effective_bets, severity, n_positions, tickers)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (report.avg_correlation, report.max_correlation,
                     f"{report.max_pair[0]}/{report.max_pair[1]}",
                     report.effective_bets, report.severity,
                     len(report.tickers), ",".join(report.tickers)),
                )
        except Exception as e:
            logger.debug(f"Correlation log error: {e}")

    async def _maybe_alert(self, report: CorrelationReport):
        """Send Telegram alert if concentration is critical + cooloff."""
        if report.severity == "LOW":
            return
        if not self.notifier:
            return

        # Cooloff: don't re-alert more than once per 24h
        if self._last_alert:
            hours = (datetime.utcnow() - self._last_alert).total_seconds() / 3600
            if hours < 24 and report.severity != "CRITICAL":
                return

        icon = "🔴" if report.severity == "CRITICAL" else "🟡"
        lines = [
            f"{icon} *CONCENTRATION ALERT* — {report.severity}",
            f"Avg correlation: {report.avg_correlation:.2f}",
            f"Effective bets: {report.effective_bets:.1f} (of {len(report.tickers)} positions)",
            f"Max pair: {report.max_pair[0]}/{report.max_pair[1]} "
            f"({report.max_correlation:.2f})",
        ]

        if report.redundant_pairs:
            lines.append("\nRedundant pairs (>0.85):")
            for a, b, c in report.redundant_pairs[:3]:
                lines.append(f"  • {a}/{b}: {c:.2f}")

        if report.suggestions:
            lines.append(f"\nSuggestion: add decorrelated assets")
            lines.append(f"Candidates: {', '.join(report.suggestions)}")

        msg = "\n".join(lines)
        try:
            if hasattr(self.notifier, "send_message"):
                await self.notifier.send_message(msg)
            elif hasattr(self.notifier, "send"):
                await self.notifier.send(msg, parse_mode="Markdown")
            self._last_alert = datetime.utcnow()
        except Exception as e:
            logger.debug(f"Correlation alert error: {e}")

    def get_last_report(self) -> Optional[dict]:
        """Latest report from DB (for /status command)."""
        if not self.db_path:
            return None
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    """SELECT timestamp, avg_correlation, max_correlation,
                              max_pair, effective_bets, severity, n_positions
                       FROM correlation_reports
                       ORDER BY id DESC LIMIT 1"""
                ).fetchone()
            if not row:
                return None
            return {
                "timestamp": row[0],
                "avg_correlation": row[1],
                "max_correlation": row[2],
                "max_pair": row[3],
                "effective_bets": row[4],
                "severity": row[5],
                "n_positions": row[6],
            }
        except Exception:
            return None
