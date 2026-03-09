"""
AlphaDesk — Portfolio Rebalancer
Analyzes portfolio concentration and generates rebalancing recommendations.

This module is advisory-only: it produces structured recommendations
that the user must approve before any trades are executed.

Checks:
  - Sector concentration (max 40% per sector)
  - Single position concentration (max 25% per position)
  - Strategy allocation drift vs. config targets (±10%)
  - Minimum cash buffer (5% of equity)
  - Correlation risk between held positions
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config.instruments import US_EQUITIES, ETFS, COMMODITIES, FX_PAIRS, INDICES
from config.settings import config

logger = logging.getLogger("alphadesk.risk.rebalancer")

# ─── Limits ───
MAX_SECTOR_EXPOSURE = 0.40       # 40% of portfolio in one sector
MAX_SINGLE_POSITION = 0.25       # 25% of portfolio in one position
STRATEGY_TOLERANCE = 0.10        # ±10% from config target
MIN_CASH_BUFFER = 0.05           # 5% of equity must stay in cash

# ─── Sector lookup (merged from all instrument dicts) ───
_SECTOR_MAP: Dict[str, str] = {}
for _universe in (US_EQUITIES, ETFS, COMMODITIES, INDICES):
    for _sym, _meta in _universe.items():
        _SECTOR_MAP[_sym] = _meta.get("sector", "Unknown")
# FX pairs get their own "sector"
for _sym in FX_PAIRS:
    _SECTOR_MAP[_sym] = "FX"


def _get_sector(symbol: str) -> str:
    """Look up sector for a symbol, falling back to 'Unknown'."""
    clean = symbol.replace("=X", "").replace("/", "").upper()
    return _SECTOR_MAP.get(clean, "Unknown")


# ─── Correlation matrix (static, based on typical cross-asset correlations) ───
# In a live system this would be computed from rolling returns; for now we use
# a lookup of well-known high-correlation pairs.
_HIGH_CORRELATION_GROUPS = [
    {"XLE", "OIH", "XOM", "OIL"},         # Energy cluster
    {"GDX", "GLD", "GOLD", "SLV", "SILVER"},  # Precious metals cluster
    {"AAPL", "MSFT", "GOOGL", "NVDA", "META", "NFLX"},  # Big Tech
    {"JPM", "BAC", "V"},                   # Financials
    {"JNJ", "UNH", "PFE"},                 # Healthcare
    {"SPX500", "NSDQ100"},                 # US indices
]


def _correlation_group(symbol: str) -> Optional[int]:
    """Return the index of the correlation group this symbol belongs to, or None."""
    clean = symbol.replace("=X", "").replace("/", "").upper()
    for idx, group in enumerate(_HIGH_CORRELATION_GROUPS):
        if clean in group:
            return idx
    return None


# ═══════════════════════════════════════════════════════════════════
#  Main analysis
# ═══════════════════════════════════════════════════════════════════

class PortfolioRebalancer:
    """
    Produces a rebalancing recommendation based on current positions.

    Usage:
        rebalancer = PortfolioRebalancer()
        report = rebalancer.analyze(equity, cash, positions)
        # report is a dict suitable for Telegram formatting
    """

    def __init__(
        self,
        max_sector: float = MAX_SECTOR_EXPOSURE,
        max_position: float = MAX_SINGLE_POSITION,
        strategy_tolerance: float = STRATEGY_TOLERANCE,
        min_cash: float = MIN_CASH_BUFFER,
    ):
        self.max_sector = max_sector
        self.max_position = max_position
        self.strategy_tolerance = strategy_tolerance
        self.min_cash = min_cash

    # ── public API ──────────────────────────────────────────────

    def analyze(
        self,
        equity: float,
        cash: float,
        positions: List[dict],
    ) -> dict:
        """
        Run full concentration analysis and return a structured report.

        Args:
            equity: Total account equity (cash + invested).
            cash: Available cash.
            positions: List of position dicts as returned by eToro /portfolio.
                       Expected keys: symbol (or instrumentID), initialAmountInDollars,
                       investedAmount, netProfit, strategy_tag.

        Returns:
            dict with keys:
              - summary          (high-level numbers)
              - sector_analysis  (per-sector breakdown)
              - position_analysis (per-position concentration)
              - strategy_analysis (per-strategy vs. target)
              - correlation_analysis
              - cash_analysis
              - actions          (list of recommended actions)
              - severity         ("OK" | "WARNING" | "CRITICAL")
              - timestamp
        """
        if equity <= 0:
            return self._empty_report("Equity is zero — cannot analyze")

        # ── 1. Sector concentration ──
        sector_analysis = self._analyze_sectors(equity, positions)

        # ── 2. Single-position concentration ──
        position_analysis = self._analyze_positions(equity, positions)

        # ── 3. Strategy allocation ──
        strategy_analysis = self._analyze_strategies(equity, positions)

        # ── 4. Correlation risk ──
        correlation_analysis = self._analyze_correlations(positions)

        # ── 5. Cash buffer ──
        cash_analysis = self._analyze_cash(equity, cash)

        # ── 6. Build action list ──
        actions = []
        actions.extend(sector_analysis["actions"])
        actions.extend(position_analysis["actions"])
        actions.extend(strategy_analysis["actions"])
        actions.extend(correlation_analysis["actions"])
        actions.extend(cash_analysis["actions"])

        # Determine severity
        critical = any(a["severity"] == "CRITICAL" for a in actions)
        warning = any(a["severity"] == "WARNING" for a in actions)
        severity = "CRITICAL" if critical else ("WARNING" if warning else "OK")

        report = {
            "summary": {
                "equity": equity,
                "cash": cash,
                "cash_pct": cash / equity,
                "num_positions": len(positions),
                "num_actions": len(actions),
                "severity": severity,
            },
            "sector_analysis": sector_analysis,
            "position_analysis": position_analysis,
            "strategy_analysis": strategy_analysis,
            "correlation_analysis": correlation_analysis,
            "cash_analysis": cash_analysis,
            "actions": actions,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Rebalance analysis: {severity}, {len(actions)} actions, "
            f"equity=${equity:,.2f}, cash=${cash:,.2f}"
        )
        return report

    # ── sector ──────────────────────────────────────────────────

    def _analyze_sectors(self, equity: float, positions: List[dict]) -> dict:
        """Compute per-sector exposure and flag breaches."""
        sector_totals: Dict[str, float] = {}
        sector_positions: Dict[str, List[str]] = {}

        for pos in positions:
            symbol = pos.get("symbol", "?")
            amount = abs(pos.get("initialAmountInDollars", pos.get("investedAmount", 0)))
            sector = _get_sector(symbol)

            sector_totals[sector] = sector_totals.get(sector, 0) + amount
            sector_positions.setdefault(sector, []).append(symbol)

        # Convert to percentages
        sector_pcts: Dict[str, float] = {
            s: v / equity for s, v in sector_totals.items()
        }

        actions = []
        for sector, pct in sector_pcts.items():
            if pct > self.max_sector:
                excess_dollars = sector_totals[sector] - (equity * self.max_sector)
                actions.append({
                    "type": "TRIM_SECTOR",
                    "severity": "CRITICAL" if pct > 0.60 else "WARNING",
                    "sector": sector,
                    "current_pct": pct,
                    "limit_pct": self.max_sector,
                    "excess_dollars": round(excess_dollars, 2),
                    "symbols": sector_positions[sector],
                    "message": (
                        f"Sector '{sector}' is {pct:.0%} of portfolio "
                        f"(limit {self.max_sector:.0%}). "
                        f"Trim ~${excess_dollars:,.0f} from {sector_positions[sector]}."
                    ),
                })

        return {
            "sector_totals": {s: round(v, 2) for s, v in sector_totals.items()},
            "sector_pcts": {s: round(v, 4) for s, v in sector_pcts.items()},
            "sector_positions": sector_positions,
            "breaches": [s for s, p in sector_pcts.items() if p > self.max_sector],
            "actions": actions,
        }

    # ── single position ────────────────────────────────────────

    def _analyze_positions(self, equity: float, positions: List[dict]) -> dict:
        """Flag positions exceeding single-name concentration limit."""
        pos_data = []
        actions = []

        for pos in positions:
            symbol = pos.get("symbol", "?")
            amount = abs(pos.get("initialAmountInDollars", pos.get("investedAmount", 0)))
            pnl = pos.get("netProfit", 0)
            pct = amount / equity if equity > 0 else 0
            strategy = pos.get("strategy_tag", "unknown")

            pos_data.append({
                "symbol": symbol,
                "amount": round(amount, 2),
                "pct": round(pct, 4),
                "pnl": round(pnl, 2),
                "strategy": strategy,
                "sector": _get_sector(symbol),
            })

            if pct > self.max_position:
                target_amount = equity * self.max_position
                trim_amount = amount - target_amount
                actions.append({
                    "type": "TRIM_POSITION",
                    "severity": "CRITICAL" if pct > 0.40 else "WARNING",
                    "symbol": symbol,
                    "current_pct": pct,
                    "limit_pct": self.max_position,
                    "trim_dollars": round(trim_amount, 2),
                    "message": (
                        f"{symbol} is {pct:.0%} of portfolio "
                        f"(limit {self.max_position:.0%}). "
                        f"Trim ~${trim_amount:,.0f}."
                    ),
                })

        # Sort by weight descending
        pos_data.sort(key=lambda x: x["pct"], reverse=True)

        return {
            "positions": pos_data,
            "actions": actions,
        }

    # ── strategy allocation ────────────────────────────────────

    def _analyze_strategies(self, equity: float, positions: List[dict]) -> dict:
        """Compare actual strategy exposure vs. config targets."""
        targets = {
            "momentum": config.allocation.momentum,
            "mean_reversion": config.allocation.mean_reversion,
            "factor_model": config.allocation.factor_model,
            "fx_carry": config.allocation.fx_carry,
        }

        # Actual exposure per strategy
        actual: Dict[str, float] = {}
        for pos in positions:
            strategy = pos.get("strategy_tag", "unknown")
            amount = abs(pos.get("initialAmountInDollars", pos.get("investedAmount", 0)))
            actual[strategy] = actual.get(strategy, 0) + amount

        actual_pcts: Dict[str, float] = {
            s: v / equity for s, v in actual.items()
        }

        actions = []
        drift_details = {}

        for strat, target in targets.items():
            current = actual_pcts.get(strat, 0.0)
            drift = current - target
            drift_details[strat] = {
                "target_pct": target,
                "actual_pct": round(current, 4),
                "drift": round(drift, 4),
                "actual_dollars": round(actual.get(strat, 0), 2),
                "target_dollars": round(equity * target, 2),
            }

            if abs(drift) > self.strategy_tolerance:
                direction = "overweight" if drift > 0 else "underweight"
                delta_dollars = abs(drift) * equity
                actions.append({
                    "type": "STRATEGY_DRIFT",
                    "severity": "WARNING",
                    "strategy": strat,
                    "direction": direction,
                    "current_pct": current,
                    "target_pct": target,
                    "drift": drift,
                    "delta_dollars": round(delta_dollars, 2),
                    "message": (
                        f"Strategy '{strat}' is {direction}: "
                        f"{current:.0%} vs. target {target:.0%} "
                        f"(delta ${delta_dollars:,.0f})."
                    ),
                })

        # Flag unknown strategies (positions not tagged to any known strategy)
        for strat, pct in actual_pcts.items():
            if strat not in targets and strat != "unknown":
                drift_details[strat] = {
                    "target_pct": 0,
                    "actual_pct": round(pct, 4),
                    "drift": round(pct, 4),
                    "actual_dollars": round(actual.get(strat, 0), 2),
                    "target_dollars": 0,
                }

        return {
            "targets": targets,
            "actual": {s: round(v, 4) for s, v in actual_pcts.items()},
            "drift_details": drift_details,
            "actions": actions,
        }

    # ── correlation ────────────────────────────────────────────

    def _analyze_correlations(self, positions: List[dict]) -> dict:
        """Identify clusters of highly correlated positions."""
        # Group held symbols by correlation cluster
        cluster_holdings: Dict[int, List[dict]] = {}
        unclustered = []

        for pos in positions:
            symbol = pos.get("symbol", "?")
            amount = abs(pos.get("initialAmountInDollars", pos.get("investedAmount", 0)))
            grp = _correlation_group(symbol)
            if grp is not None:
                cluster_holdings.setdefault(grp, []).append({
                    "symbol": symbol,
                    "amount": amount,
                })
            else:
                unclustered.append(symbol)

        actions = []
        cluster_details = []

        for grp_idx, holdings in cluster_holdings.items():
            if len(holdings) >= 2:
                symbols = [h["symbol"] for h in holdings]
                total = sum(h["amount"] for h in holdings)
                group_name = ", ".join(sorted(
                    _HIGH_CORRELATION_GROUPS[grp_idx]
                ))

                cluster_details.append({
                    "group": group_name,
                    "held_symbols": symbols,
                    "combined_amount": round(total, 2),
                    "count": len(holdings),
                })

                if len(holdings) >= 3:
                    actions.append({
                        "type": "HIGH_CORRELATION",
                        "severity": "WARNING",
                        "symbols": symbols,
                        "group": group_name,
                        "combined_amount": round(total, 2),
                        "message": (
                            f"High correlation cluster: {symbols} "
                            f"(total ${total:,.0f}). "
                            f"Consider reducing to 1-2 names."
                        ),
                    })

        return {
            "clusters": cluster_details,
            "unclustered": unclustered,
            "actions": actions,
        }

    # ── cash buffer ────────────────────────────────────────────

    def _analyze_cash(self, equity: float, cash: float) -> dict:
        """Check that minimum cash buffer is maintained."""
        cash_pct = cash / equity if equity > 0 else 0
        min_cash_dollars = equity * self.min_cash

        actions = []
        if cash_pct < self.min_cash:
            shortfall = min_cash_dollars - cash
            actions.append({
                "type": "LOW_CASH",
                "severity": "CRITICAL" if cash_pct < 0.01 else "WARNING",
                "current_cash": round(cash, 2),
                "current_pct": round(cash_pct, 4),
                "required_pct": self.min_cash,
                "shortfall": round(shortfall, 2),
                "message": (
                    f"Cash is {cash_pct:.1%} of equity "
                    f"(minimum {self.min_cash:.0%}). "
                    f"Free up ~${shortfall:,.0f}."
                ),
            })

        return {
            "cash": round(cash, 2),
            "cash_pct": round(cash_pct, 4),
            "min_required": round(min_cash_dollars, 2),
            "is_adequate": cash_pct >= self.min_cash,
            "actions": actions,
        }

    # ── helpers ─────────────────────────────────────────────────

    def _empty_report(self, reason: str) -> dict:
        return {
            "summary": {"error": reason},
            "actions": [],
            "severity": "ERROR",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Telegram formatting ────────────────────────────────────

    @staticmethod
    def format_telegram(report: dict) -> str:
        """
        Format the rebalance report as an HTML message for Telegram.
        Keeps it within Telegram's 4096-char limit.
        """
        if "error" in report.get("summary", {}):
            return f"<b>REBALANCE ERROR</b>\n{report['summary']['error']}"

        s = report["summary"]
        severity_emoji = {
            "OK": "✅", "WARNING": "⚠️", "CRITICAL": "🔴", "ERROR": "❌"
        }
        sev = report["severity"]
        lines = [
            f"{severity_emoji.get(sev, '❓')} <b>REBALANCE ANALYSIS</b> [{sev}]",
            f"{'━' * 28}",
            f"💼 Equity: <b>${s['equity']:,.2f}</b>",
            f"💵 Cash: ${s['cash']:,.2f} ({s['cash_pct']:.1%})",
            f"📊 Positions: {s['num_positions']}",
            f"{'━' * 28}",
        ]

        # Sector breakdown
        sa = report.get("sector_analysis", {})
        if sa.get("sector_pcts"):
            lines.append("<b>Sector Exposure:</b>")
            for sector, pct in sorted(
                sa["sector_pcts"].items(), key=lambda x: -x[1]
            ):
                bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
                breach = " ❗" if sector in sa.get("breaches", []) else ""
                lines.append(f"  {sector}: {pct:.0%} {bar}{breach}")

        # Position weights
        pa = report.get("position_analysis", {})
        if pa.get("positions"):
            lines.append("")
            lines.append("<b>Position Weights:</b>")
            for p in pa["positions"]:
                pnl_emoji = "🟢" if p["pnl"] >= 0 else "🔴"
                over = " ❗" if p["pct"] > MAX_SINGLE_POSITION else ""
                lines.append(
                    f"  {pnl_emoji} {p['symbol']}: ${p['amount']:,.0f} "
                    f"({p['pct']:.0%}){over} [{p['strategy']}]"
                )

        # Strategy drift
        strat = report.get("strategy_analysis", {})
        if strat.get("drift_details"):
            lines.append("")
            lines.append("<b>Strategy Allocation:</b>")
            for name, d in strat["drift_details"].items():
                arrow = "↑" if d["drift"] > 0 else ("↓" if d["drift"] < 0 else "→")
                lines.append(
                    f"  {name}: {d['actual_pct']:.0%} "
                    f"(target {d['target_pct']:.0%}) {arrow}"
                )

        # Correlation clusters
        corr = report.get("correlation_analysis", {})
        if corr.get("clusters"):
            lines.append("")
            lines.append("<b>Correlation Clusters:</b>")
            for c in corr["clusters"]:
                lines.append(
                    f"  🔗 {c['held_symbols']} — ${c['combined_amount']:,.0f}"
                )

        # Actions
        actions = report.get("actions", [])
        if actions:
            lines.append("")
            lines.append(f"<b>Recommended Actions ({len(actions)}):</b>")
            for i, a in enumerate(actions, 1):
                sev_mark = "🔴" if a["severity"] == "CRITICAL" else "⚠️"
                lines.append(f"  {sev_mark} {i}. {a['message']}")
        else:
            lines.append("")
            lines.append("✅ Portfolio is within all limits.")

        ts = report.get("timestamp", "")[:19].replace("T", " ")
        lines.append(f"\n⏰ {ts} UTC")

        return "\n".join(lines)
