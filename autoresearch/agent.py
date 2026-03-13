"""
AutoResearch Agent — Autonomous Strategy Optimizer (v2, Karpathy-style)

Karpathy-style autoresearch loop for trading strategies:
1. Create dedicated git branch for the session
2. Load baseline scores
3. Ask LLM to propose parameter changes (informed by program.md)
4. Git commit the change
5. Run backtest with proposed params
6. If score improves → keep commit (promote). If worse → git reset (reject).
7. Log to results.tsv
8. Repeat (forever in --infinite mode).

Usage:
    python autoresearch/agent.py --strategy momentum --rounds 20
    python autoresearch/agent.py --strategy all --infinite
    python autoresearch/agent.py --status
"""

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic

from autoresearch.backtest_runner import run_experiment, patch_strategies
from autoresearch.strategy_tuner import (
    get_params, MOMENTUM_PARAMS, MEAN_REVERSION_PARAMS,
    FACTOR_MODEL_PARAMS, FX_CARRY_PARAMS, BACKTEST_CONFIG,
)
from autoresearch.prepare_market import compute_score

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("autoresearch.agent")

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
HISTORY_FILE = os.path.join(BASE_DIR, "experiment_history.jsonl")
BEST_FILE = os.path.join(BASE_DIR, "best_params.json")
PROGRAM_FILE = os.path.join(BASE_DIR, "program.md")
RESULTS_TSV = os.path.join(BASE_DIR, "results.tsv")

# ── LLM Config ──
MODEL = "claude-sonnet-4-6"  # upgraded from Haiku for deeper reasoning
MAX_TOKENS = 2048

# ── Parameter Ranges (constraints for LLM) ──
PARAM_RANGES = {
    "momentum": {
        "breakout_period": (10, 40, "int"),
        "trend_sma": (20, 100, "int"),
        "long_trend_sma": (100, 252, "int"),
        "volume_threshold": (0.5, 3.0, "float"),
        "atr_multiplier": (1.0, 4.0, "float"),
        "min_momentum_3m": (0.0, 0.15, "float"),
        "rsi_overbought": (65, 85, "int"),
        "rsi_oversold": (20, 40, "int"),
        "max_positions": (3, 15, "int"),
        "min_confidence": (0.3, 0.8, "float"),
        "min_rr_ratio": (0.8, 3.0, "float"),
        "tp_atr_multiplier": (1.5, 8.0, "float"),
    },
    "mean_reversion": {
        "z_entry_long": (-3.0, -1.0, "float"),
        "z_entry_short": (1.0, 3.0, "float"),
        "z_exit": (0.0, 1.0, "float"),
        "z_stop": (2.5, 5.0, "float"),
        "min_lookback": (30, 120, "int"),
        "rsi_long_threshold": (20, 45, "int"),
        "rsi_short_threshold": (55, 80, "int"),
        "sl_atr_multiplier": (1.0, 3.0, "float"),
        "max_positions": (3, 12, "int"),
        "min_confidence": (0.3, 0.8, "float"),
        "min_rr_ratio": (0.5, 2.5, "float"),
    },
    "factor_model": {
        "value_weight": (0.1, 0.6, "float"),
        "quality_weight": (0.1, 0.6, "float"),
        "momentum_weight": (0.1, 0.6, "float"),
        "rebalance_days": (5, 63, "int"),
        "stop_loss_pct": (0.03, 0.15, "float"),
        "take_profit_pct": (0.05, 0.30, "float"),
        "max_positions": (5, 20, "int"),
        "min_data_days": (60, 252, "int"),
        "min_composite": (0.3, 0.7, "float"),
    },
    "fx_carry": {
        "min_carry_spread": (0.001, 0.03, "float"),
        "carry_weight": (0.2, 0.9, "float"),
        "momentum_weight": (0.1, 0.8, "float"),
        "trend_filter_sma": (20, 100, "int"),
        "atr_stop_multiplier": (1.0, 3.0, "float"),
        "min_composite_score": (0.01, 0.15, "float"),
        "max_positions": (2, 10, "int"),
        "min_confidence": (0.3, 0.8, "float"),
        "min_rr_ratio": (0.5, 2.0, "float"),
        "max_risk_per_pair": (0.005, 0.04, "float"),
    },
}


# ══════════════════════════════════════════════════════════════════
# Git Operations
# ══════════════════════════════════════════════════════════════════

def _git(cmd: str, check: bool = True) -> str:
    """Run a git command in the repo directory. Returns stdout."""
    result = subprocess.run(
        f"git {cmd}",
        shell=True,
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        logger.warning(f"git {cmd} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_setup_branch(tag: str) -> str:
    """Create or switch to autoresearch branch. Returns branch name."""
    branch = f"autoresearch/{tag}"
    existing = _git(f"branch --list {branch}", check=False)
    if existing:
        _git(f"checkout {branch}", check=False)
        logger.info(f"Switched to existing branch: {branch}")
    else:
        _git(f"checkout -b {branch}", check=False)
        logger.info(f"Created new branch: {branch}")
    return branch


def git_commit_experiment(strategy: str, exp_id: str, changes: dict, reasoning: str) -> str:
    """Commit strategy_tuner.py changes. Returns short commit hash."""
    _git("add autoresearch/strategy_tuner.py", check=False)
    changes_str = ", ".join(f"{k}={v}" for k, v in changes.items())
    msg = f"autoresearch({strategy}): {changes_str}\n\n{reasoning}\n\nExperiment: {exp_id}"
    # Use a temp file for commit message to avoid shell escaping issues
    msg_file = os.path.join(BASE_DIR, ".commit_msg.tmp")
    with open(msg_file, "w") as f:
        f.write(msg)
    _git(f"commit -F {msg_file}", check=False)
    os.remove(msg_file)
    return _git("rev-parse --short HEAD")


def git_reset_last() -> None:
    """Reset last commit (experiment rejected)."""
    _git("reset --hard HEAD~1", check=False)


def git_current_hash() -> str:
    """Get current short commit hash."""
    return _git("rev-parse --short HEAD")


# ══════════════════════════════════════════════════════════════════
# Results TSV (Karpathy-style logging)
# ══════════════════════════════════════════════════════════════════

def init_results_tsv():
    """Create results.tsv with header if it doesn't exist."""
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\tstrategy\tscore\tsharpe\treturn\ttrades\tmax_dd\tstatus\tdescription\n")


def append_results_tsv(commit: str, strategy: str, metrics: dict,
                        status: str, description: str):
    """Append one row to results.tsv."""
    init_results_tsv()
    score = metrics.get("score", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    ret = metrics.get("total_return", 0)
    trades = metrics.get("num_trades", 0)
    dd = metrics.get("max_drawdown", 0)
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{strategy}\t{score:.4f}\t{sharpe:.2f}\t{ret:+.4f}\t{trades}\t{dd:.4f}\t{status}\t{description}\n")


# ══════════════════════════════════════════════════════════════════
# Program.md loader
# ══════════════════════════════════════════════════════════════════

def load_program_md() -> str:
    """Load program.md instructions for the LLM."""
    if os.path.exists(PROGRAM_FILE):
        with open(PROGRAM_FILE) as f:
            return f.read()
    return ""


# ══════════════════════════════════════════════════════════════════
# History & Best Params (unchanged)
# ══════════════════════════════════════════════════════════════════

def load_history(strategy: str = None) -> list:
    """Load experiment history from JSONL file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    history = []
    with open(HISTORY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if strategy and entry.get("strategy") != strategy:
                continue
            history.append(entry)
    return history


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def append_history(entry: dict):
    """Append one experiment to history."""
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry, cls=_SafeEncoder) + "\n")


def load_best(strategy: str) -> dict:
    """Load best known params + score for a strategy."""
    if not os.path.exists(BEST_FILE):
        return {}
    with open(BEST_FILE, "r") as f:
        data = json.load(f)
    return data.get(strategy, {})


def save_best(strategy: str, params: dict, score: float, experiment_id: str):
    """Save new best params."""
    data = {}
    if os.path.exists(BEST_FILE):
        with open(BEST_FILE, "r") as f:
            data = json.load(f)
    data[strategy] = {
        "params": params,
        "score": score,
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
    }
    with open(BEST_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════════
# Program.md Self-Update (meta-learning)
# ══════════════════════════════════════════════════════════════════

PROGRAM_UPDATE_INTERVAL = 10  # update program.md every N rounds
MAX_PROGRAM_LINES = 150  # keep program.md concise

def update_program_md(client: Anthropic, strategy: str, history: list):
    """Summarize and replace program.md insights (not append-only).

    Every PROGRAM_UPDATE_INTERVAL rounds, the LLM reads ALL experiment
    history + current program.md, then writes a CONDENSED version that
    replaces the auto-discovered section. This prevents bloat and keeps
    the most actionable insights fresh.
    """
    if not history:
        return

    current_program = load_program_md()

    # Split program.md into base (manual) and auto-discovered sections
    base_section = current_program
    auto_marker = "## Auto-discovered Insights"
    if auto_marker in current_program:
        base_section = current_program[:current_program.index(auto_marker)].rstrip()

    # Build experiment summary per strategy
    strat_history = [h for h in history if h.get("strategy") == strategy]
    promoted = [h for h in strat_history if h.get("promoted")]
    rejected = [h for h in strat_history if not h.get("promoted")]

    promoted_text = ""
    for h in promoted[-15:]:
        m = h.get("metrics", {})
        promoted_text += (
            f"  - {h.get('changes_summary', '')} → score {m.get('score', 0):.4f} "
            f"(sharpe={m.get('sharpe_ratio', 0):.2f}, ret={m.get('total_return', 0):+.2%}, "
            f"trades={m.get('num_trades', 0)})\n"
        )

    rejected_text = ""
    for h in rejected[-20:]:
        m = h.get("metrics", {})
        rejected_text += (
            f"  - {h.get('changes_summary', '')} → score {m.get('score', 0):.4f} "
            f"({h.get('reasoning', '')[:80]})\n"
        )

    prompt = f"""You are the meta-learning module of an autonomous trading strategy optimizer.

Your job: REWRITE the auto-discovered insights section of program.md for the {strategy} strategy.
This replaces the old insights — be comprehensive but concise.

BASE PROGRAM (do NOT modify this):
{base_section[:2000]}

PROMOTED EXPERIMENTS (what worked) — last 15:
{promoted_text if promoted_text else "None yet."}

REJECTED EXPERIMENTS (what didn't work) — last 20:
{rejected_text if rejected_text else "None yet."}

TOTAL: {len(strat_history)} experiments, {len(promoted)} promoted, {len(rejected)} rejected

YOUR TASK:
Write a COMPLETE replacement for all auto-discovered insights for {strategy}.
Synthesize ALL experiments into the most actionable knowledge.

Include:
1. Parameter sensitivity map (which params have high/low impact)
2. Known dead ends (param values that always fail)
3. Best-performing parameter combinations
4. Promising unexplored directions
5. Parameter interaction effects

RULES:
- Write 8-15 bullet points maximum
- Be extremely specific: use actual parameter names, values, and ranges
- Deduplicate — one bullet per insight, no repetition
- Mark confidence level: [HIGH] [MEDIUM] [LOW]
- If a previous insight is outdated, drop it
- Max 80 lines total

Respond with ONLY the markdown (no code blocks, no preamble):"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        new_insights = response.content[0].text.strip()

        if "Not enough data" in new_insights:
            logger.info(f"[meta] Not enough data to update program.md for {strategy}")
            return

        # Also collect insights from other strategies (keep them)
        other_insights = ""
        for line in current_program.split("\n"):
            # Keep auto-discovered sections for OTHER strategies
            pass  # We rebuild from scratch below

        # Collect existing auto-sections for other strategies
        import re
        other_sections = []
        for match in re.finditer(
            r"(## Auto-discovered Insights — (\w+).*?)(?=## Auto-discovered|$)",
            current_program, re.DOTALL
        ):
            section_strategy = match.group(2)
            if section_strategy != strategy:
                other_sections.append(match.group(1).strip())

        # Rebuild program.md: base + other strategies + new insights for this strategy
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        rebuilt = base_section + "\n"
        for sec in other_sections:
            rebuilt += f"\n\n{sec}\n"
        rebuilt += (
            f"\n\n## Auto-discovered Insights — {strategy} ({timestamp})\n\n"
            f"{new_insights}\n"
        )

        with open(PROGRAM_FILE, "w") as f:
            f.write(rebuilt)

        # Check line count
        line_count = len(rebuilt.split("\n"))
        logger.info(f"[meta] Rewrote program.md for {strategy} ({line_count} lines)")

    except Exception as e:
        logger.warning(f"[meta] Failed to update program.md: {e}")


# ══════════════════════════════════════════════════════════════════
# LLM Proposal (enhanced with program.md + simplicity criterion)
# ══════════════════════════════════════════════════════════════════

def format_history_for_llm(history: list, last_n: int = 15) -> str:
    """Format recent experiment history for the LLM prompt."""
    if not history:
        return "No previous experiments."

    recent = history[-last_n:]
    lines = []
    for h in recent:
        m = h.get("metrics", {})
        promoted = "PROMOTED" if h.get("promoted") else "rejected"
        delta = h.get("score_delta", 0)
        lines.append(
            f"  {h['experiment_id']}: score={m.get('score', 0):.4f} "
            f"(delta={delta:+.4f}, {promoted}) "
            f"sharpe={m.get('sharpe_ratio', 0):.2f} "
            f"ret={m.get('total_return', 0):+.2%} "
            f"trades={m.get('num_trades', 0)} "
            f"dd={m.get('max_drawdown', 0):.2%} "
            f"changes={h.get('changes_summary', '')}"
        )
    return "\n".join(lines)


def build_proposal_prompt(strategy: str, current_params: dict,
                          baseline_score: float, history: list) -> str:
    """Build the LLM prompt to propose parameter changes."""
    ranges = PARAM_RANGES[strategy]
    history_text = format_history_for_llm(history)
    program_text = load_program_md()

    params_text = ""
    for k, v in current_params.items():
        r = ranges.get(k)
        if r:
            lo, hi, typ = r
            params_text += f"  {k}: {v}  (range: [{lo}, {hi}], type: {typ})\n"
        else:
            params_text += f"  {k}: {v}  (no range constraint)\n"

    # Count total experiments and consecutive rejections
    total_exps = len(history)
    consecutive_rejects = 0
    for h in reversed(history):
        if h.get("promoted"):
            break
        consecutive_rejects += 1

    stale_hint = ""
    if consecutive_rejects >= 5:
        stale_hint = (
            f"\nWARNING: {consecutive_rejects} consecutive rejections. "
            "Consider a more radical change, or try the opposite direction of recent attempts. "
            "Re-read the strategy-specific notes in program.md for new angles."
        )

    return f"""You are an autonomous trading strategy optimizer (autoresearch agent).

{program_text}

═══════════════════════════════════════════════════
CURRENT SESSION
═══════════════════════════════════════════════════

STRATEGY: {strategy}
BASELINE SCORE: {baseline_score:.4f}
TOTAL EXPERIMENTS: {total_exps}
CONSECUTIVE REJECTIONS: {consecutive_rejects}
{stale_hint}

CURRENT PARAMETERS:
{params_text}

EXPERIMENT HISTORY (recent, newest last):
{history_text}

YOUR TASK:
Analyze the experiment history. Based on what worked and what didn't, propose ONE set of parameter changes.

RULES:
- Change 1-3 parameters at a time (small, testable deltas)
- Stay within the valid ranges
- Look for patterns in what improved vs degraded the score
- Don't repeat exact changes that were already rejected
- Think about the interactions between parameters
- If no history exists, start with the most impactful parameter (highest sensitivity)
- SIMPLICITY: Prefer parameters near the middle of their range. Extreme values are fragile.
- ROBUSTNESS: Don't sacrifice trade count for marginal score gains. Min 50 trades is ideal.
- NEWLY WIRED: tp_atr_multiplier, min_confidence, min_rr_ratio, rsi_oversold (momentum), rsi_long_threshold, rsi_short_threshold, sl_atr_multiplier (mean_reversion) were previously dead params — they now affect the strategy. Explore these first!
- The scoring function now includes Calmar ratio (15%), uncapped returns (log-scale above 50%), and Sharpe up to 5.0.

Respond with ONLY a JSON object:
{{
  "changes": {{"param_name": new_value, ...}},
  "reasoning": "brief explanation of why these changes"
}}"""


def _algorithmic_mutation(strategy: str, current_params: dict,
                          history: list) -> dict:
    """Algorithmic mutation when LLM is stuck: random perturbation or
    crossover from historically promoted experiments."""
    import random

    ranges = PARAM_RANGES[strategy]
    promoted = [h for h in history if h.get("promoted") and h.get("strategy") == strategy]

    # Strategy 1: Crossover — blend current best with a random promoted experiment
    if promoted and random.random() < 0.4:
        donor = random.choice(promoted)
        donor_params = donor.get("params", {})
        # Pick 1-2 params from donor
        crossover_keys = random.sample(
            list(set(donor_params.keys()) & set(ranges.keys())),
            k=min(2, len(donor_params))
        )
        changes = {k: donor_params[k] for k in crossover_keys if k in donor_params}
        return {
            "changes": changes,
            "reasoning": f"[ALGORITHMIC] Crossover from {donor.get('experiment_id', '?')}: {list(changes.keys())}",
        }

    # Strategy 2: Random perturbation — pick 1-3 params, perturb ±5-20%
    param_keys = list(ranges.keys())
    n_changes = random.randint(1, 3)
    selected = random.sample(param_keys, k=min(n_changes, len(param_keys)))

    changes = {}
    for key in selected:
        lo, hi, typ = ranges[key]
        current_val = current_params.get(key, (lo + hi) / 2)
        span = hi - lo
        # Random perturbation: ±5% to ±20% of range
        delta = random.uniform(0.05, 0.20) * span * random.choice([-1, 1])
        new_val = current_val + delta
        new_val = max(lo, min(hi, new_val))
        if typ == "int":
            new_val = int(round(new_val))
        else:
            new_val = round(new_val, 4)
        changes[key] = new_val

    return {
        "changes": changes,
        "reasoning": f"[ALGORITHMIC] Random perturbation: {list(changes.keys())}",
    }


def propose_changes(client: Anthropic, strategy: str, current_params: dict,
                    baseline_score: float, history: list) -> dict:
    """Propose parameter changes: LLM-driven with algorithmic fallback."""

    # Count consecutive rejections for this strategy
    strat_history = [h for h in history if h.get("strategy") == strategy]
    consecutive_rejects = 0
    for h in reversed(strat_history):
        if h.get("promoted"):
            break
        consecutive_rejects += 1

    # Every 5th rejection, use algorithmic mutation instead of LLM
    if consecutive_rejects > 0 and consecutive_rejects % 5 == 0:
        logger.info(f"[{strategy}] {consecutive_rejects} consecutive rejections — using algorithmic mutation")
        return _algorithmic_mutation(strategy, current_params, strat_history)

    prompt = build_proposal_prompt(strategy, current_params, baseline_score, history)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Extract JSON from response (handle markdown code blocks)
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    proposal = json.loads(text)
    return proposal


# ══════════════════════════════════════════════════════════════════
# Param Validation
# ══════════════════════════════════════════════════════════════════

def apply_and_validate_changes(strategy: str, current_params: dict,
                                changes: dict) -> dict:
    """Apply proposed changes with range validation. Returns new params."""
    new_params = copy.deepcopy(current_params)
    ranges = PARAM_RANGES.get(strategy, {})

    for key, value in changes.items():
        if key not in current_params:
            logger.warning(f"Unknown param '{key}' — skipping")
            continue

        r = ranges.get(key)
        if r:
            lo, hi, typ = r
            if typ == "int":
                value = int(round(value))
            else:
                value = float(value)
            value = max(lo, min(hi, value))  # clamp to range

        new_params[key] = value

    # Special constraint: factor model weights must sum to ~1.0
    if strategy == "factor_model":
        w_sum = (new_params.get("value_weight", 0) +
                 new_params.get("quality_weight", 0) +
                 new_params.get("momentum_weight", 0))
        if w_sum > 0 and abs(w_sum - 1.0) > 0.01:
            new_params["value_weight"] /= w_sum
            new_params["quality_weight"] /= w_sum
            new_params["momentum_weight"] /= w_sum

    return new_params


def write_params_to_tuner(strategy: str, params: dict):
    """Write params back to strategy_tuner.py on disk (for git tracking)."""
    tuner_path = os.path.join(BASE_DIR, "strategy_tuner.py")
    with open(tuner_path, "r") as f:
        content = f.read()

    # Build the new dict text
    dict_name = {
        "momentum": "MOMENTUM_PARAMS",
        "mean_reversion": "MEAN_REVERSION_PARAMS",
        "factor_model": "FACTOR_MODEL_PARAMS",
        "fx_carry": "FX_CARRY_PARAMS",
    }[strategy]

    # Find the dict in file and replace it
    import re
    # Match from "DICT_NAME = {" to the closing "}"
    pattern = rf"({dict_name}\s*=\s*\{{)(.*?)(\}})"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        logger.warning(f"Could not find {dict_name} in strategy_tuner.py")
        return

    # Rebuild the dict content preserving comments from PARAM_RANGES
    ranges = PARAM_RANGES.get(strategy, {})
    lines = []
    for k, v in params.items():
        r = ranges.get(k)
        if isinstance(v, float):
            val_str = f"{v:.2f}" if abs(v) < 1 else f"{v}"
            # Clean up trailing zeros but keep at least one decimal
            if "." in val_str:
                val_str = val_str.rstrip("0").rstrip(".")
                if "." not in val_str:
                    val_str += ".0"
        else:
            val_str = str(v)

        if r:
            lo, hi, typ = r
            comment = f"# [{lo}, {hi}]"
        else:
            comment = ""
        lines.append(f'    "{k}": {val_str},'.ljust(42) + comment)

    new_dict = f"{dict_name} = {{\n" + "\n".join(lines) + "\n}"

    # Replace the old dict
    start = match.start()
    end = match.end()
    content = content[:start] + new_dict + content[end:]

    with open(tuner_path, "w") as f:
        f.write(content)


# ══════════════════════════════════════════════════════════════════
# Core Loop
# ══════════════════════════════════════════════════════════════════

def run_one_round(client: Anthropic, strategy: str, round_num: int,
                  current_params: dict, baseline_score: float,
                  history: list, use_git: bool = True) -> tuple:
    """Run one autoresearch round. Returns (new_params, new_score, promoted)."""

    exp_id = f"auto_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_r{round_num}"

    # 1. Propose changes
    logger.info(f"[Round {round_num}] Asking LLM for proposal...")
    proposal = propose_changes(client, strategy, current_params, baseline_score, history)
    changes = proposal.get("changes", {})
    reasoning = proposal.get("reasoning", "")

    if not changes:
        logger.warning(f"[Round {round_num}] LLM proposed no changes — skipping")
        return current_params, baseline_score, False

    logger.info(f"[Round {round_num}] Proposed: {changes}")
    logger.info(f"[Round {round_num}] Reasoning: {reasoning}")

    # 2. Apply changes with validation
    new_params = apply_and_validate_changes(strategy, current_params, changes)

    # 3. Write params to strategy_tuner.py on disk + git commit
    if use_git:
        write_params_to_tuner(strategy, new_params)
        commit_hash = git_commit_experiment(strategy, exp_id, changes, reasoning)
        logger.info(f"[Round {round_num}] Committed: {commit_hash}")
    else:
        commit_hash = "nogit"

    # 4. Inject new params into strategy_tuner module (runtime)
    import autoresearch.strategy_tuner as tuner
    param_dict_name = {
        "momentum": "MOMENTUM_PARAMS",
        "mean_reversion": "MEAN_REVERSION_PARAMS",
        "factor_model": "FACTOR_MODEL_PARAMS",
        "fx_carry": "FX_CARRY_PARAMS",
    }[strategy]
    original = getattr(tuner, param_dict_name).copy()
    getattr(tuner, param_dict_name).update(new_params)

    # 5. Run backtest
    logger.info(f"[Round {round_num}] Running backtest {exp_id}...")
    start_time = time.time()
    try:
        result = run_experiment(strategy, exp_id, timeout=300)
    except Exception as e:
        logger.error(f"[Round {round_num}] Backtest CRASHED: {e}")
        getattr(tuner, param_dict_name).update(original)
        if use_git:
            git_reset_last()
            # Restore tuner file on disk
            write_params_to_tuner(strategy, current_params)
        append_results_tsv(commit_hash, strategy, {}, "crash", str(e)[:80])
        return current_params, baseline_score, False
    elapsed = time.time() - start_time

    if result is None:
        logger.error(f"[Round {round_num}] No result — reverting")
        getattr(tuner, param_dict_name).update(original)
        if use_git:
            git_reset_last()
            write_params_to_tuner(strategy, current_params)
        append_results_tsv(commit_hash, strategy, {}, "crash", "no result")
        return current_params, baseline_score, False

    new_score = result["metrics"]["score"]
    score_delta = new_score - baseline_score

    # 6. Simplicity check: penalize if trades dropped too low
    num_trades = result["metrics"].get("num_trades", 0)
    profit_factor = result["metrics"].get("profit_factor", 0)
    MIN_PROMOTION_DELTA = 0.002  # Avoid noise-driven promotions
    promoted = (new_score - baseline_score) >= MIN_PROMOTION_DELTA

    # Robustness guard: reject even if score improved but trades < 20 or profit_factor < 0.8
    if promoted and (num_trades < 20 or profit_factor < 0.8):
        logger.warning(
            f"[Round {round_num}] Score improved but FRAGILE "
            f"(trades={num_trades}, pf={profit_factor:.2f}) — rejecting"
        )
        promoted = False

    # 6b. STAGED VALIDATION: out-of-sample backtest on 2020-2022
    oos_score = None
    if promoted:
        logger.info(f"[Round {round_num}] In-sample passed — running OOS validation (2020-2022)...")
        try:
            oos_result = run_experiment(
                strategy, f"{exp_id}_oos", timeout=300,
                start_date="2020-01-01", end_date="2022-12-31",
            )
            if oos_result:
                oos_score = oos_result["metrics"]["score"]
                oos_trades = oos_result["metrics"].get("num_trades", 0)
                # OOS must have at least 50% of in-sample score and >= 10 trades
                oos_threshold = new_score * 0.5
                if oos_score < oos_threshold or oos_trades < 10:
                    logger.warning(
                        f"[Round {round_num}] OOS FAILED: score {oos_score:.4f} "
                        f"(need >= {oos_threshold:.4f}), trades={oos_trades} — OVERFITTING detected"
                    )
                    promoted = False
                else:
                    logger.info(
                        f"[Round {round_num}] OOS PASSED: score {oos_score:.4f}, "
                        f"trades={oos_trades} — robust improvement confirmed"
                    )
            else:
                logger.warning(f"[Round {round_num}] OOS backtest returned no result — rejecting")
                promoted = False
        except Exception as e:
            logger.warning(f"[Round {round_num}] OOS backtest crashed: {e} — rejecting")
            promoted = False

    # 7. Record history (JSONL)
    changes_summary = ", ".join(f"{k}:{current_params.get(k)}->{v}" for k, v in changes.items())
    entry = {
        "experiment_id": exp_id,
        "strategy": strategy,
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "commit": commit_hash,
        "baseline_score": baseline_score,
        "new_score": new_score,
        "score_delta": round(score_delta, 4),
        "promoted": promoted,
        "changes": changes,
        "changes_summary": changes_summary,
        "reasoning": reasoning,
        "params": new_params,
        "metrics": result["metrics"],
        "oos_score": oos_score,
        "elapsed_seconds": round(elapsed, 1),
    }
    append_history(entry)

    # 8. Record in results.tsv (Karpathy-style)
    status = "keep" if promoted else "discard"
    append_results_tsv(commit_hash, strategy, result["metrics"], status, changes_summary[:100])

    # 9. Promote or revert
    if promoted:
        logger.info(
            f"[Round {round_num}] PROMOTED! score {baseline_score:.4f} -> {new_score:.4f} "
            f"(+{score_delta:.4f}) [{elapsed:.0f}s] | {changes_summary}"
        )
        save_best(strategy, new_params, new_score, exp_id)
        return new_params, new_score, True
    else:
        logger.info(
            f"[Round {round_num}] Rejected. score {baseline_score:.4f} -> {new_score:.4f} "
            f"({score_delta:+.4f}) [{elapsed:.0f}s] | {changes_summary}"
        )
        # Revert runtime params
        getattr(tuner, param_dict_name).update(original)
        # Git reset to previous commit
        if use_git:
            git_reset_last()
            write_params_to_tuner(strategy, current_params)
        return current_params, baseline_score, False


def run_autoresearch(strategy: str, num_rounds: int = 20,
                     infinite: bool = False, use_git: bool = True):
    """Main autoresearch loop for one strategy."""
    client = Anthropic()
    patch_strategies()
    init_results_tsv()

    mode_label = "INFINITE" if infinite else f"{num_rounds} rounds"
    logger.info(f"{'='*60}")
    logger.info(f"  AutoResearch Agent v2 (Karpathy-style) — {strategy}")
    logger.info(f"  Mode: {mode_label}")
    logger.info(f"  Git branching: {'ON' if use_git else 'OFF'}")
    logger.info(f"{'='*60}")

    # Setup git branch
    if use_git:
        tag = datetime.now().strftime("%b%d").lower()
        branch = git_setup_branch(f"{tag}-{strategy}")
        logger.info(f"  Branch: {branch}")

    # Load baseline
    best = load_best(strategy)
    if best and best.get("params"):
        current_params = best["params"]
        baseline_score = best["score"]
        logger.info(f"Resuming from best: score={baseline_score:.4f} ({best['experiment_id']})")
    else:
        current_params = get_params(strategy)
        # Run baseline backtest
        logger.info("Running baseline backtest...")
        base_id = f"auto_{strategy}_baseline_{datetime.now().strftime('%Y%m%d')}"
        result = run_experiment(strategy, base_id, timeout=300)
        if result is None:
            logger.error("Baseline backtest failed!")
            return
        baseline_score = result["metrics"]["score"]
        save_best(strategy, current_params, baseline_score, base_id)
        append_results_tsv(
            git_current_hash() if use_git else "base",
            strategy, result["metrics"], "keep", "baseline"
        )
        logger.info(f"Baseline score: {baseline_score:.4f}")

    history = load_history(strategy)
    promoted_count = 0
    total_improvement = 0.0
    start_score = baseline_score
    round_num = 0

    # ── THE LOOP (runs forever in infinite mode) ──
    while True:
        round_num += 1

        # Check round limit (unless infinite)
        if not infinite and round_num > num_rounds:
            break

        logger.info(f"\n{'─'*50}")
        if infinite:
            logger.info(f"Round {round_num} (infinite) | Current best: {baseline_score:.4f}")
        else:
            logger.info(f"Round {round_num}/{num_rounds} | Current best: {baseline_score:.4f}")
        logger.info(f"{'─'*50}")

        try:
            current_params, baseline_score, promoted = run_one_round(
                client, strategy, round_num, current_params,
                baseline_score, history, use_git=use_git
            )
            if promoted:
                promoted_count += 1
                total_improvement = baseline_score - start_score

            # Reload history for next round
            history = load_history(strategy)

            # Meta-learning: update program.md every N rounds
            if round_num % PROGRAM_UPDATE_INTERVAL == 0:
                logger.info(f"[meta] Round {round_num} — updating program.md with learned insights...")
                try:
                    update_program_md(client, strategy, history)
                except Exception as e:
                    logger.warning(f"[meta] program.md update failed: {e}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user — stopping.")
            break
        except Exception as e:
            logger.error(f"Round {round_num} error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"  AutoResearch Complete — {strategy}")
    logger.info(f"  Rounds: {round_num - 1 if not infinite else round_num}")
    logger.info(f"  Promoted: {promoted_count}/{round_num}")
    logger.info(f"  Score: {start_score:.4f} -> {baseline_score:.4f} ({total_improvement:+.4f})")
    logger.info(f"{'='*60}")

    # Switch back to main branch
    if use_git:
        _git("checkout main", check=False)
        _git("checkout master", check=False)

    return {
        "strategy": strategy,
        "rounds": round_num,
        "promoted": promoted_count,
        "start_score": start_score,
        "final_score": baseline_score,
        "improvement": total_improvement,
        "best_params": current_params,
    }


# ══════════════════════════════════════════════════════════════════
# Status & Telegram
# ══════════════════════════════════════════════════════════════════

def print_status():
    """Print summary of all experiments."""
    history = load_history()
    if not history:
        print("No experiments yet.")
        return

    by_strategy = {}
    for h in history:
        s = h["strategy"]
        by_strategy.setdefault(s, []).append(h)

    for strategy, exps in sorted(by_strategy.items()):
        promoted = [e for e in exps if e.get("promoted")]
        rejected = [e for e in exps if not e.get("promoted")]
        best = load_best(strategy)

        print(f"\n{'='*60}")
        print(f"  {strategy.upper()}")
        print(f"  Experiments: {len(exps)} ({len(promoted)} promoted, {len(rejected)} rejected)")
        if best:
            print(f"  Best score: {best['score']:.4f} ({best['experiment_id']})")
        print(f"{'='*60}")

        for e in exps[-10:]:
            m = e.get("metrics", {})
            tag = "+" if e.get("promoted") else "-"
            commit = e.get("commit", "?")[:7]
            print(
                f"  [{tag}] {commit} {e['experiment_id']}: "
                f"score={m.get('score', 0):.4f} ({e.get('score_delta', 0):+.4f}) "
                f"sharpe={m.get('sharpe_ratio', 0):.2f} "
                f"ret={m.get('total_return', 0):+.2%} "
                f"| {e.get('changes_summary', '')}"
            )

    # Also show results.tsv if it exists
    if os.path.exists(RESULTS_TSV):
        print(f"\n{'='*60}")
        print("  RESULTS.TSV (last 20)")
        print(f"{'='*60}")
        with open(RESULTS_TSV) as f:
            lines = f.readlines()
        for line in lines[-21:]:  # header + last 20
            print(f"  {line.rstrip()}")


def send_telegram_summary(results: list):
    """Send autoresearch summary via Telegram."""
    try:
        env_path = os.path.join(os.path.dirname(BASE_DIR), ".env")
        token = chat_id = ""
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if k == "TELEGRAM_BOT_TOKEN":
                            token = v
                        elif k == "TELEGRAM_CHAT_ID":
                            chat_id = v

        if not token or not chat_id:
            return

        import httpx
        msg = "<b>AUTORESEARCH v2 COMPLETE</b>\n"
        for r in results:
            icon = "+" if r["improvement"] > 0 else "="
            msg += (
                f"\n[{icon}] <b>{r['strategy']}</b>\n"
                f"  Score: {r['start_score']:.4f} -> {r['final_score']:.4f} "
                f"({r['improvement']:+.4f})\n"
                f"  Promoted: {r['promoted']}/{r['rounds']}\n"
            )

        httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
        )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch Agent v2 (Karpathy-style)")
    parser.add_argument("--strategy", help="momentum|mean_reversion|factor_model|fx_carry|all")
    parser.add_argument("--rounds", type=int, default=20, help="Experiments per strategy")
    parser.add_argument("--infinite", action="store_true", help="Run forever (NEVER STOP)")
    parser.add_argument("--no-git", action="store_true", help="Disable git branching")
    parser.add_argument("--status", action="store_true", help="Show experiment status")

    args = parser.parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    if not args.strategy:
        parser.error("--strategy required (or use --status)")

    strategies = (
        ["momentum", "mean_reversion", "factor_model", "fx_carry"]
        if args.strategy == "all"
        else [args.strategy]
    )

    use_git = not args.no_git

    results = []
    for strat in strategies:
        r = run_autoresearch(
            strat, args.rounds,
            infinite=args.infinite,
            use_git=use_git,
        )
        if r:
            results.append(r)

    if results:
        send_telegram_summary(results)
