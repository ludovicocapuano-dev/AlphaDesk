"""
AlphaDesk — AI Multi-Agent Investment Analysis System

Five specialist investor agents powered by Claude LLM, each with a distinct
analytical personality, plus an AI Portfolio Manager that aggregates their
confidence-weighted votes to approve/reject/resize trade signals.

Inspired by virattt/ai-hedge-fund.  Designed as a validation layer that sits
between signal generation and order execution.

Integration point: call AIPortfolioManager.evaluate_signal() from main.py
after ML ensemble but before risk check.
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("alphadesk.ai_agents")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentSignal:
    """Structured output from a single investor agent."""
    signal: str        # "bullish", "bearish", "neutral"
    confidence: int    # 0-100
    reasoning: str

    @property
    def numeric(self) -> float:
        """Map signal to numeric: bullish=1, neutral=0, bearish=-1."""
        return {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}.get(self.signal, 0.0)


@dataclass
class AIDecision:
    """Final aggregated decision from the AI Portfolio Manager."""
    approved: bool
    action: str                    # "approve", "reject", "reduce_size"
    confidence: float              # 0.0 – 1.0
    final_score: float             # -1.0 – 1.0 (confidence-weighted vote)
    agents: List[Dict[str, Any]]   # per-agent results
    reasoning: str
    size_multiplier: float = 1.0   # 1.0 = full size, 0.5 = half, etc.
    tokens_used: int = 0
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# LLM Client wrapper
# ---------------------------------------------------------------------------

_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_SONNET_MODEL = "claude-sonnet-4-6"

# Pricing (USD per 1M tokens) — Claude haiku-4-5 / sonnet-4-6
_PRICING = {
    _HAIKU_MODEL:  {"input": 1.00, "output": 5.00},
    _SONNET_MODEL: {"input": 3.00, "output": 15.00},
}


class LLMClient:
    """
    Thin wrapper around the Anthropic Python SDK with:
    - Retry logic (exponential backoff)
    - Token budget enforcement
    - Cost logging to SQLite
    """

    def __init__(self, db_path: str, daily_token_budget: int = 50_000):
        self._client = None
        self._db_path = db_path
        self._daily_budget = daily_token_budget
        self._today_tokens = 0
        self._today_date: Optional[str] = None
        self._ensure_table()

    # -- lazy init so import doesn't fail without API key --
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
            except Exception as e:
                logger.error("Failed to initialise Anthropic client: %s", e)
                raise
        return self._client

    def _ensure_table(self):
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ai_agent_costs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        model TEXT NOT NULL,
                        ticker TEXT,
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        cost_usd REAL NOT NULL
                    )
                """)
        except Exception as e:
            logger.warning("Could not create ai_agent_costs table: %s", e)

    # -- budget --

    def _refresh_daily_counter(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self._today_date != today:
            self._today_date = today
            try:
                with sqlite3.connect(self._db_path) as conn:
                    row = conn.execute(
                        "SELECT COALESCE(SUM(total_tokens), 0) FROM ai_agent_costs "
                        "WHERE timestamp LIKE ?",
                        (f"{today}%",),
                    ).fetchone()
                    self._today_tokens = row[0] if row else 0
            except Exception:
                self._today_tokens = 0

    def budget_remaining(self) -> int:
        self._refresh_daily_counter()
        return max(0, self._daily_budget - self._today_tokens)

    def has_budget(self) -> bool:
        return self.budget_remaining() > 0

    # -- call --

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = _HAIKU_MODEL,
        max_tokens: int = 500,
        agent_name: str = "unknown",
        ticker: str = "",
        retries: int = 3,
    ) -> Tuple[str, int]:
        """
        Call Claude and return (response_text, total_tokens).

        Retries with exponential backoff on transient errors.
        Raises RuntimeError if budget exhausted.
        """
        if not self.has_budget():
            raise RuntimeError(
                f"Daily AI token budget exhausted ({self._daily_budget} tokens). "
                "Remaining calls will fall back to neutral."
            )

        client = self._get_client()
        last_err = None

        for attempt in range(retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                text = response.content[0].text
                inp_tok = response.usage.input_tokens
                out_tok = response.usage.output_tokens
                total = inp_tok + out_tok

                # Cost
                pricing = _PRICING.get(model, _PRICING[_HAIKU_MODEL])
                cost = (inp_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000

                # Log
                self._log_cost(agent_name, model, ticker, inp_tok, out_tok, total, cost)
                self._today_tokens += total

                return text, total

            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, retries, e, wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")

    def _log_cost(self, agent: str, model: str, ticker: str,
                  inp: int, out: int, total: int, cost: float):
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO ai_agent_costs "
                    "(timestamp, agent_name, model, ticker, input_tokens, output_tokens, "
                    "total_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (datetime.utcnow().isoformat(), agent, model, ticker, inp, out, total, cost),
                )
        except Exception as e:
            logger.warning("Failed to log AI cost: %s", e)


# ---------------------------------------------------------------------------
# Analysis cache (same ticker within 1 hour → reuse)
# ---------------------------------------------------------------------------

class AnalysisCache:
    """Simple in-memory cache keyed by (agent_name, ticker) with 1-hour TTL."""

    def __init__(self, ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, AgentSignal]] = {}
        self._hits = 0
        self._misses = 0

    def _key(self, agent: str, ticker: str) -> str:
        return f"{agent}:{ticker}"

    def get(self, agent: str, ticker: str) -> Optional[AgentSignal]:
        k = self._key(agent, ticker)
        entry = self._store.get(k)
        if entry is None:
            self._misses += 1
            return None
        ts, sig = entry
        if time.time() - ts > self._ttl:
            del self._store[k]
            self._misses += 1
            return None
        self._hits += 1
        return sig

    def put(self, agent: str, ticker: str, sig: AgentSignal):
        self._store[self._key(agent, ticker)] = (time.time(), sig)

    def clear(self):
        self._store.clear()


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseInvestorAgent(ABC):
    """Abstract investor agent.  Subclasses define personality via prompts."""

    name: str = "base"
    weight: float = 1.0  # relative weight in aggregation

    def __init__(self, llm: LLMClient, cache: AnalysisCache):
        self._llm = llm
        self._cache = cache

    # -- subclass hooks --

    @abstractmethod
    def _system_prompt(self) -> str:
        """Return the system prompt that defines this agent's personality."""
        ...

    @abstractmethod
    def _build_user_prompt(self, ticker: str, market_data: dict,
                           portfolio_context: dict) -> str:
        """Build the compact user prompt with relevant metrics."""
        ...

    # -- public API --

    def analyze(
        self,
        ticker: str,
        market_data: dict,
        portfolio_context: dict,
        model: str = _HAIKU_MODEL,
    ) -> AgentSignal:
        """
        Analyse a ticker and return a structured signal.

        Falls back to neutral on any error (never blocks trading).
        """
        # Cache check
        cached = self._cache.get(self.name, ticker)
        if cached is not None:
            logger.debug("[%s] Cache hit for %s", self.name, ticker)
            return cached

        try:
            user_prompt = self._build_user_prompt(ticker, market_data, portfolio_context)
            text, _ = self._llm.call(
                system_prompt=self._system_prompt(),
                user_prompt=user_prompt,
                model=model,
                max_tokens=500,
                agent_name=self.name,
                ticker=ticker,
            )
            result = self._parse_response(text)
            self._cache.put(self.name, ticker, result)
            return result

        except Exception as e:
            logger.warning("[%s] Analysis failed for %s: %s — returning neutral", self.name, ticker, e)
            return AgentSignal(signal="neutral", confidence=0, reasoning=f"Agent error: {e}")

    def _parse_response(self, text: str) -> AgentSignal:
        """Parse JSON from LLM response.  Robust to markdown fences."""
        clean = text.strip()
        if clean.startswith("```"):
            # Strip markdown code fences
            lines = clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            clean = "\n".join(lines).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract JSON object from surrounding text
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(clean[start:end])
            else:
                raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")

        signal = str(data.get("signal", "neutral")).lower()
        if signal not in ("bullish", "bearish", "neutral"):
            signal = "neutral"

        confidence = int(max(0, min(100, data.get("confidence", 50))))
        reasoning = str(data.get("reasoning", ""))[:500]

        return AgentSignal(signal=signal, confidence=confidence, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Concrete Agents
# ---------------------------------------------------------------------------

class ValueAgent(BaseInvestorAgent):
    """Warren Buffett / Benjamin Graham style value investor."""

    name = "value_agent"
    weight = 1.0

    def _system_prompt(self) -> str:
        return (
            "You are a value investor in the style of Warren Buffett and Benjamin Graham. "
            "You seek companies trading below intrinsic value with durable competitive "
            "advantages (moats), strong balance sheets, and consistent earnings.\n\n"
            "Analyse the provided data and respond ONLY with a JSON object:\n"
            '{"signal": "bullish"|"bearish"|"neutral", '
            '"confidence": 0-100, '
            '"reasoning": "brief explanation"}\n\n'
            "Be bullish if: P/E is below sector average, P/B < 1.5, ROE > 15%, "
            "debt/equity manageable, clear margin of safety.\n"
            "Be bearish if: overvalued, deteriorating fundamentals, high debt.\n"
            "Be neutral if: data insufficient or fair value."
        )

    def _build_user_prompt(self, ticker: str, market_data: dict,
                           portfolio_context: dict) -> str:
        md = market_data or {}
        parts = [f"Ticker: {ticker}"]
        parts.append(f"Price: {md.get('price', 'N/A')}")
        parts.append(f"P/E: {md.get('pe_ratio', 'N/A')}")
        parts.append(f"P/B: {md.get('pb_ratio', 'N/A')}")
        parts.append(f"Debt/Equity: {md.get('debt_equity', 'N/A')}")
        parts.append(f"ROE: {md.get('roe', 'N/A')}")
        parts.append(f"Profit Margin: {md.get('profit_margin', 'N/A')}")
        parts.append(f"Revenue Growth: {md.get('revenue_growth', 'N/A')}")
        parts.append(f"Free Cash Flow: {md.get('fcf', 'N/A')}")
        parts.append(f"Sector: {md.get('sector', 'N/A')}")
        parts.append(f"Regime: {md.get('regime', 'N/A')}")
        return "\n".join(parts)


class MomentumAgent(BaseInvestorAgent):
    """Stanley Druckenmiller style momentum / macro trader."""

    name = "momentum_agent"
    weight = 1.0

    def _system_prompt(self) -> str:
        return (
            "You are a momentum trader in the style of Stanley Druckenmiller. "
            "You follow strong trends backed by macro tailwinds, volume confirmation, "
            "and technical breakouts. You cut losers fast and ride winners.\n\n"
            "Analyse the provided data and respond ONLY with a JSON object:\n"
            '{"signal": "bullish"|"bearish"|"neutral", '
            '"confidence": 0-100, '
            '"reasoning": "brief explanation"}\n\n'
            "Be bullish if: strong uptrend, rising volume, macro supports direction, "
            "SMA alignment (50 > 200), MACD positive.\n"
            "Be bearish if: trend breaking down, volume divergence, macro headwinds.\n"
            "Be neutral if: choppy, no clear trend."
        )

    def _build_user_prompt(self, ticker: str, market_data: dict,
                           portfolio_context: dict) -> str:
        md = market_data or {}
        parts = [f"Ticker: {ticker}"]
        parts.append(f"Price: {md.get('price', 'N/A')}")
        parts.append(f"RSI(14): {md.get('rsi', 'N/A')}")
        parts.append(f"MACD: {md.get('macd', 'N/A')}")
        parts.append(f"MACD Signal: {md.get('macd_signal', 'N/A')}")
        parts.append(f"SMA50: {md.get('sma_50', 'N/A')}")
        parts.append(f"SMA200: {md.get('sma_200', 'N/A')}")
        parts.append(f"Volume Ratio (vs 20d avg): {md.get('volume_ratio', 'N/A')}")
        parts.append(f"Momentum 3M: {md.get('momentum_3m', 'N/A')}")
        parts.append(f"ATR%: {md.get('atr_pct', 'N/A')}")
        parts.append(f"BB Position: {md.get('bb_position', 'N/A')}")
        parts.append(f"Trend Regime: {md.get('trend_regime', 'N/A')}")
        parts.append(f"Volatility Regime: {md.get('volatility_regime', 'N/A')}")
        return "\n".join(parts)


class ContrarianAgent(BaseInvestorAgent):
    """Michael Burry style contrarian / deep value sceptic."""

    name = "contrarian_agent"
    weight = 0.8  # slightly lower weight — contrarian often disagrees

    def _system_prompt(self) -> str:
        return (
            "You are a contrarian investor in the style of Michael Burry. "
            "You look for overcrowded trades, sentiment extremes, bubble indicators, "
            "and assets where consensus is dangerously wrong. You are sceptical of "
            "momentum and look for mean-reversion opportunities.\n\n"
            "Analyse the provided data and respond ONLY with a JSON object:\n"
            '{"signal": "bullish"|"bearish"|"neutral", '
            '"confidence": 0-100, '
            '"reasoning": "brief explanation"}\n\n'
            "Be bearish if: consensus too bullish, RSI > 75, overcrowded long, "
            "valuation stretched, bubble indicators present.\n"
            "Be bullish if: consensus too bearish, RSI < 25, panic selling, "
            "extreme undervaluation after sell-off.\n"
            "Be neutral if: no extreme positioning."
        )

    def _build_user_prompt(self, ticker: str, market_data: dict,
                           portfolio_context: dict) -> str:
        md = market_data or {}
        parts = [f"Ticker: {ticker}"]
        parts.append(f"Price: {md.get('price', 'N/A')}")
        parts.append(f"RSI(14): {md.get('rsi', 'N/A')}")
        parts.append(f"P/E: {md.get('pe_ratio', 'N/A')}")
        parts.append(f"P/B: {md.get('pb_ratio', 'N/A')}")
        parts.append(f"BB Position: {md.get('bb_position', 'N/A')}")
        parts.append(f"Volume Ratio: {md.get('volume_ratio', 'N/A')}")
        parts.append(f"Momentum 3M: {md.get('momentum_3m', 'N/A')}")
        parts.append(f"52-Week Range Position: {md.get('range_52w_pct', 'N/A')}")
        parts.append(f"News Sentiment: {md.get('news_sentiment', 'N/A')}")
        parts.append(f"Correlation Regime: {md.get('correlation_regime', 'N/A')}")
        return "\n".join(parts)


class MacroAgent(BaseInvestorAgent):
    """Ray Dalio style macro / all-weather analyst."""

    name = "macro_agent"
    weight = 1.2  # slightly higher — macro context is high-value

    def _system_prompt(self) -> str:
        return (
            "You are a macro strategist in the style of Ray Dalio. "
            "You analyse the economic machine: growth, inflation, interest rates, "
            "credit cycles, and regime shifts. You focus on how macro conditions "
            "affect this specific asset.\n\n"
            "Analyse the provided data and respond ONLY with a JSON object:\n"
            '{"signal": "bullish"|"bearish"|"neutral", '
            '"confidence": 0-100, '
            '"reasoning": "brief explanation"}\n\n'
            "Be bullish if: favourable regime for this asset class, low VIX, "
            "sector rotation tailwind, easing monetary policy.\n"
            "Be bearish if: risk-off regime, high VIX, tightening policy, "
            "sector headwinds.\n"
            "Be neutral if: mixed signals or transition period."
        )

    def _build_user_prompt(self, ticker: str, market_data: dict,
                           portfolio_context: dict) -> str:
        md = market_data or {}
        regime = portfolio_context.get("regime", {}) if portfolio_context else {}
        parts = [f"Ticker: {ticker}"]
        parts.append(f"Price: {md.get('price', 'N/A')}")
        parts.append(f"Sector: {md.get('sector', 'N/A')}")
        parts.append(f"Volatility Regime: {regime.get('volatility_regime', md.get('volatility_regime', 'N/A'))}")
        parts.append(f"Trend Regime: {regime.get('trend_regime', md.get('trend_regime', 'N/A'))}")
        parts.append(f"Liquidity Regime: {regime.get('liquidity_regime', md.get('liquidity_regime', 'N/A'))}")
        parts.append(f"Rate Regime: {regime.get('rate_regime', md.get('rate_regime', 'N/A'))}")
        parts.append(f"Correlation Regime: {regime.get('correlation_regime', md.get('correlation_regime', 'N/A'))}")
        parts.append(f"HMM Regime: {regime.get('hmm_regime', md.get('hmm_regime', 'N/A'))}")
        parts.append(f"VIX: {md.get('vix', 'N/A')}")
        parts.append(f"News Sentiment: {md.get('news_sentiment', 'N/A')}")
        parts.append(f"Fed Sentiment: {md.get('fed_sentiment', 'N/A')}")
        return "\n".join(parts)


class RiskAgent(BaseInvestorAgent):
    """Portfolio risk management specialist."""

    name = "risk_agent"
    weight = 1.5  # highest weight — risk veto is important

    def _system_prompt(self) -> str:
        return (
            "You are a portfolio risk manager. Your job is to protect capital. "
            "You analyse portfolio concentration, correlation risk, drawdown levels, "
            "position sizing, and whether adding this trade increases systemic risk.\n\n"
            "Analyse the provided data and respond ONLY with a JSON object:\n"
            '{"signal": "bullish"|"bearish"|"neutral", '
            '"confidence": 0-100, '
            '"reasoning": "brief explanation"}\n\n'
            "Be bullish if: position adds diversification, portfolio drawdown is low, "
            "risk/reward is attractive, sizing is conservative.\n"
            "Be bearish if: concentrated exposure, high drawdown, correlated positions, "
            "poor risk/reward, oversized.\n"
            "Be neutral if: risk is acceptable but not compelling."
        )

    def _build_user_prompt(self, ticker: str, market_data: dict,
                           portfolio_context: dict) -> str:
        md = market_data or {}
        pc = portfolio_context or {}
        parts = [f"Ticker: {ticker}"]
        parts.append(f"Proposed Direction: {md.get('direction', 'N/A')}")
        parts.append(f"Proposed Size ($): {md.get('proposed_size', 'N/A')}")
        parts.append(f"Risk/Reward Ratio: {md.get('risk_reward', 'N/A')}")
        parts.append(f"Stop Loss Distance: {md.get('stop_loss_pct', 'N/A')}")
        # Portfolio context
        parts.append(f"Portfolio Equity: ${pc.get('equity', 'N/A')}")
        parts.append(f"Cash Available: ${pc.get('cash', 'N/A')}")
        parts.append(f"Current Drawdown: {pc.get('current_drawdown', 'N/A')}")
        parts.append(f"Number of Positions: {pc.get('num_positions', 'N/A')}")
        parts.append(f"Gross Exposure: {pc.get('gross_exposure', 'N/A')}")
        parts.append(f"Strategy Exposures: {json.dumps(pc.get('strategy_exposures', {}))}")
        parts.append(f"Sector of New Trade: {md.get('sector', 'N/A')}")
        parts.append(f"Same-Sector Positions: {pc.get('same_sector_count', 0)}")
        parts.append(f"Correlation Regime: {pc.get('correlation_regime', 'N/A')}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# AI Portfolio Manager
# ---------------------------------------------------------------------------

class AIPortfolioManager:
    """
    Orchestrates all five investor agents, aggregates their votes via
    confidence-weighted scoring, and produces a final approve/reject/reduce
    decision.

    Usage:
        manager = AIPortfolioManager(db_path="/path/to/alphadesk.db")
        decision = manager.evaluate_signal(trade_signal, market_data,
                                           portfolio_state, regime)
        if decision.approved:
            # proceed to execution
    """

    # Minimum trade value to warrant AI analysis (skip for tiny trades)
    MIN_TRADE_VALUE: float = 200.0

    # Approval threshold: final score must exceed this
    APPROVE_THRESHOLD: float = 0.2

    def __init__(
        self,
        db_path: str = "",
        daily_token_budget: int = 50_000,
        high_value_threshold: float = 1000.0,
    ):
        if not db_path:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alphadesk.db")

        self._db_path = db_path
        self._high_value = high_value_threshold
        self._llm = LLMClient(db_path=db_path, daily_token_budget=daily_token_budget)
        self._cache = AnalysisCache(ttl_seconds=3600)

        # Instantiate agents
        self.agents: List[BaseInvestorAgent] = [
            ValueAgent(self._llm, self._cache),
            MomentumAgent(self._llm, self._cache),
            ContrarianAgent(self._llm, self._cache),
            MacroAgent(self._llm, self._cache),
            RiskAgent(self._llm, self._cache),
        ]

        # Tracking for Telegram /ai command
        self.evaluation_history: List[dict] = []
        self.token_usage: dict = {"today": 0, "total": 0}
        self.cache_stats: dict = {"hits": 0, "misses": 0}

    # -- public API --

    def evaluate_signal(
        self,
        trade_signal,
        market_data: dict,
        portfolio_state: dict,
        regime: dict,
    ) -> AIDecision:
        """
        Run all agents on a trade signal and return an aggregated decision.

        Args:
            trade_signal: TradeSignal (or dict with symbol, entry_price,
                          stop_loss, take_profit, metadata, strategy_name,
                          confidence, direction).
            market_data: dict of technical/fundamental metrics for the ticker.
            portfolio_state: dict from PortfolioRiskManager.get_portfolio_summary().
            regime: dict from RegimeFingerprint.to_dict().

        Returns:
            AIDecision with approval status, confidence, per-agent results.
        """
        # --- Extract signal attributes (support both TradeSignal and dict) ---
        if hasattr(trade_signal, "symbol"):
            ticker = trade_signal.symbol
            entry = trade_signal.entry_price
            sl = trade_signal.stop_loss
            tp = trade_signal.take_profit
            meta = trade_signal.metadata or {}
            strategy = trade_signal.strategy_name
            sig_conf = trade_signal.confidence
            direction = trade_signal.direction
            rr = trade_signal.risk_reward_ratio
            size_pct = trade_signal.suggested_size_pct
        else:
            ticker = trade_signal.get("symbol", "")
            entry = trade_signal.get("entry_price", 0)
            sl = trade_signal.get("stop_loss", 0)
            tp = trade_signal.get("take_profit", 0)
            meta = trade_signal.get("metadata", {})
            strategy = trade_signal.get("strategy_name", "")
            sig_conf = trade_signal.get("confidence", 0.5)
            direction = trade_signal.get("direction", "Buy")
            rr = trade_signal.get("risk_reward_ratio", 0)
            size_pct = trade_signal.get("suggested_size_pct", 0.02)

        # --- Compute proposed dollar amount ---
        equity = portfolio_state.get("equity", 0) if portfolio_state else 0
        proposed_size = equity * size_pct if equity > 0 else 0

        # --- Skip AI for tiny trades ---
        if proposed_size < self.MIN_TRADE_VALUE and proposed_size > 0:
            logger.info("[AI] Skipping analysis for %s — trade size $%.0f < $%.0f threshold",
                        ticker, proposed_size, self.MIN_TRADE_VALUE)
            return AIDecision(
                approved=True, action="approve", confidence=sig_conf,
                final_score=0.5, agents=[], reasoning="Below AI analysis threshold",
                size_multiplier=1.0,
            )

        # --- Choose model based on trade value ---
        model = _SONNET_MODEL if proposed_size >= self._high_value else _HAIKU_MODEL

        # --- Enrich market_data with signal metadata + risk info ---
        enriched_md = {**meta, **(market_data or {})}
        enriched_md.update({
            "price": entry,
            "direction": direction,
            "strategy": strategy,
            "signal_confidence": sig_conf,
            "risk_reward": rr,
            "proposed_size": proposed_size,
            "stop_loss_pct": abs(entry - sl) / entry if entry > 0 else 0,
        })

        # Merge regime into market_data
        if regime:
            for k in ("volatility_regime", "trend_regime", "liquidity_regime",
                       "rate_regime", "correlation_regime", "hmm_regime"):
                enriched_md.setdefault(k, regime.get(k, "N/A"))

        # Build portfolio_context for agents — handle dataclass, dict, or None
        if portfolio_state is None:
            pc = {}
        elif isinstance(portfolio_state, dict):
            pc = dict(portfolio_state)
        else:
            from dataclasses import asdict, is_dataclass
            pc = asdict(portfolio_state) if is_dataclass(portfolio_state) else (
                vars(portfolio_state).copy() if hasattr(portfolio_state, "__dict__") else {}
            )
        if regime:
            pc["correlation_regime"] = regime.get("correlation_regime", "normal")
        pc.setdefault("regime", regime or {})

        # Count same-sector positions
        sector = meta.get("sector", "")
        if sector and pc.get("positions"):
            pc["same_sector_count"] = sum(
                1 for p in pc.get("positions", [])
                if p.get("sector", "") == sector
            )

        # --- Run all agents ---
        agent_results: List[Dict[str, Any]] = []
        total_tokens = 0

        for agent in self.agents:
            result = agent.analyze(
                ticker=ticker,
                market_data=enriched_md,
                portfolio_context=pc,
                model=model,
            )
            agent_results.append({
                "agent": agent.name,
                "signal": result.signal,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "weight": agent.weight,
            })

        # --- Confidence-weighted aggregation ---
        # final_score = sum(signal_value * confidence * weight) / sum(confidence * weight)
        numerator = 0.0
        denominator = 0.0
        bullish_count = 0
        total_agents = len(agent_results)

        for ar in agent_results:
            sig_val = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}.get(ar["signal"], 0.0)
            conf = ar["confidence"] / 100.0  # normalize to 0-1
            w = ar["weight"]

            numerator += sig_val * conf * w
            denominator += conf * w

            if ar["signal"] == "bullish":
                bullish_count += 1

        final_score = numerator / denominator if denominator > 0 else 0.0
        majority_bullish = bullish_count > total_agents / 2

        # --- Decision ---
        if final_score > self.APPROVE_THRESHOLD and majority_bullish:
            action = "approve"
            approved = True
            size_mult = 1.0
        elif final_score > 0 and majority_bullish:
            # Positive but below threshold — approve with reduced size
            action = "reduce_size"
            approved = True
            size_mult = 0.5 + (final_score / self.APPROVE_THRESHOLD) * 0.5
            size_mult = min(1.0, max(0.3, size_mult))
        else:
            action = "reject"
            approved = False
            size_mult = 0.0

        # Build reasoning summary
        bull_agents = [a["agent"] for a in agent_results if a["signal"] == "bullish"]
        bear_agents = [a["agent"] for a in agent_results if a["signal"] == "bearish"]
        reasoning = (
            f"Score={final_score:.2f} (threshold={self.APPROVE_THRESHOLD}). "
            f"Bullish: {bull_agents or 'none'}. "
            f"Bearish: {bear_agents or 'none'}. "
            f"Model: {model}."
        )

        decision = AIDecision(
            approved=approved,
            action=action,
            confidence=abs(final_score),
            final_score=final_score,
            agents=agent_results,
            reasoning=reasoning,
            size_multiplier=size_mult,
            tokens_used=total_tokens,
        )

        logger.info(
            "[AI] %s %s %s — score=%.2f, agents=%d bullish/%d bearish, action=%s",
            ticker, direction, strategy, final_score,
            bullish_count, len(bear_agents), action,
        )

        # Track for Telegram /ai command
        self.evaluation_history.append({
            "symbol": ticker,
            "ai_score": final_score,
            "approved": approved,
            "action": action,
            "agents": [{"name": a["agent"], "approved": a["signal"] == "bullish"} for a in agent_results],
            "reasoning": reasoning[:200],
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep only last 50
        if len(self.evaluation_history) > 50:
            self.evaluation_history = self.evaluation_history[-50:]
        self.token_usage["total"] += total_tokens
        self.token_usage["today"] += total_tokens
        self.cache_stats = {"hits": self._cache._hits if hasattr(self._cache, '_hits') else 0,
                            "misses": self._cache._misses if hasattr(self._cache, '_misses') else 0}

        return decision

    def get_daily_cost(self) -> Dict[str, Any]:
        """Return today's AI agent cost summary."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(total_tokens), 0), "
                    "COALESCE(SUM(cost_usd), 0) "
                    "FROM ai_agent_costs WHERE timestamp LIKE ?",
                    (f"{today}%",),
                ).fetchone()
                return {
                    "date": today,
                    "calls": row[0],
                    "tokens_used": row[1],
                    "cost_usd": round(row[2], 4),
                    "budget_remaining": self._llm.budget_remaining(),
                }
        except Exception:
            return {"date": today, "calls": 0, "tokens_used": 0, "cost_usd": 0, "budget_remaining": 0}


# ---------------------------------------------------------------------------
# __main__ test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick test: run all 5 agents on a mock signal.

    Requires ANTHROPIC_API_KEY in environment.
    Usage:
        cd /root/AlphaDesk && python -m core.ai_agents
    """
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the live test.")
        print("Running structure test only...\n")

        # Structure test — verify all classes instantiate
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alphadesk.db")
        manager = AIPortfolioManager(db_path=db_path, daily_token_budget=100_000)

        print(f"Agents: {[a.name for a in manager.agents]}")
        print(f"Weights: {[a.weight for a in manager.agents]}")

        # Test with mock (will return neutral due to no API key)
        mock_signal = {
            "symbol": "AAPL",
            "entry_price": 175.0,
            "stop_loss": 168.0,
            "take_profit": 190.0,
            "metadata": {"sector": "Technology", "rsi": 55, "pe_ratio": 28},
            "strategy_name": "momentum",
            "confidence": 0.75,
            "direction": "Buy",
            "risk_reward_ratio": 2.14,
            "suggested_size_pct": 0.05,
        }

        mock_portfolio = {
            "equity": 10000,
            "cash": 2000,
            "current_drawdown": 0.03,
            "num_positions": 3,
            "gross_exposure": 0.8,
            "strategy_exposures": {"momentum": 0.25, "factor_model": 0.25},
        }

        mock_regime = {
            "volatility_regime": "high",
            "trend_regime": "weak_up",
            "liquidity_regime": "normal",
            "rate_regime": "neutral",
            "correlation_regime": "normal",
            "hmm_regime": "bull",
        }

        decision = manager.evaluate_signal(
            mock_signal, mock_signal["metadata"], mock_portfolio, mock_regime
        )

        print(f"\nDecision: {decision.action}")
        print(f"Approved: {decision.approved}")
        print(f"Score: {decision.final_score:.2f}")
        print(f"Size multiplier: {decision.size_multiplier}")
        print(f"Reasoning: {decision.reasoning}")
        for a in decision.agents:
            print(f"  {a['agent']}: {a['signal']} ({a['confidence']}%) — {a['reasoning'][:80]}")

        print("\nStructure test passed.")
        sys.exit(0)

    # ---- Live test with real API ----
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alphadesk.db")
    manager = AIPortfolioManager(db_path=db_path, daily_token_budget=100_000)

    mock_signal = {
        "symbol": "AAPL",
        "entry_price": 175.0,
        "stop_loss": 168.0,
        "take_profit": 190.0,
        "metadata": {
            "sector": "Technology",
            "rsi": 55,
            "macd": 1.2,
            "pe_ratio": 28,
            "pb_ratio": 45,
            "roe": 0.147,
            "debt_equity": 1.8,
            "profit_margin": 0.25,
            "revenue_growth": 0.08,
            "volume_ratio": 1.1,
            "momentum_3m": 0.12,
            "bb_position": 0.65,
            "atr_pct": 0.018,
            "sma_50": 172,
            "sma_200": 165,
        },
        "strategy_name": "momentum",
        "confidence": 0.75,
        "direction": "Buy",
        "risk_reward_ratio": 2.14,
        "suggested_size_pct": 0.05,
    }

    mock_portfolio = {
        "equity": 10557,
        "cash": 9.76,
        "current_drawdown": 0.0,
        "num_positions": 3,
        "gross_exposure": 0.999,
        "strategy_exposures": {"momentum": 0.0, "factor_model": 0.0},
    }

    mock_regime = {
        "volatility_regime": "high",
        "trend_regime": "weak_up",
        "liquidity_regime": "normal",
        "rate_regime": "neutral",
        "correlation_regime": "normal",
        "hmm_regime": "bull",
    }

    print("Running live AI analysis for AAPL...\n")

    decision = manager.evaluate_signal(
        mock_signal,
        mock_signal["metadata"],
        mock_portfolio,
        mock_regime,
    )

    print(f"\n{'='*60}")
    print(f"DECISION: {decision.action.upper()}")
    print(f"Approved: {decision.approved}")
    print(f"Final Score: {decision.final_score:.3f}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"Size Multiplier: {decision.size_multiplier:.2f}")
    print(f"Tokens Used: {decision.tokens_used}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"{'='*60}")
    for a in decision.agents:
        print(f"  [{a['agent']}] {a['signal'].upper()} ({a['confidence']}%) "
              f"w={a['weight']} — {a['reasoning']}")

    # Show cost
    cost = manager.get_daily_cost()
    print(f"\nDaily cost: {cost}")
