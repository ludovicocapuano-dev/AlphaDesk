# AlphaDesk — AutoResearch Program

This is the instruction file for the autonomous strategy optimization agent.
The agent modifies `strategy_tuner.py`, the human modifies this `program.md`.

## Setup

The repo has three files that matter:

- **`strategy_tuner.py`** — the only file the agent modifies. Contains parameters for all 4 strategies (Momentum, Mean Reversion, Factor Model, FX Carry). Each parameter has a valid range in comments.
- **`backtest_runner.py`** — fixed. Runs backtests, computes metrics. Do not modify.
- **`prepare_market.py`** — fixed. Data loading, universe selection, scoring function. Do not modify.

## Scoring (v2 — updated)

The objective is to **maximize the composite score** (higher = better):

```
score = 0.35 * sharpe_norm + 0.25 * return_norm + 0.15 * drawdown_norm + 0.10 * trade_activity + 0.15 * calmar_norm
```

Where:
- `sharpe_norm = (sharpe + 3) / 8` — Sharpe clamped to [-3, 5], normalized to [0, 1]
- `return_norm` — uncapped, log-compressed above +50% return
- `drawdown_norm = 1 - max_drawdown` — lower drawdown = better
- `trade_activity = min(1, num_trades / 80)` — ramp to 80 trades
- `calmar_norm = min(1, calmar_ratio / 5)` — bonus for high return/drawdown ratio

**Minimum 10 trades required** — score = 0 otherwise.
**Minimum promotion delta: 0.002** — noise-level improvements are rejected.

## Experimentation Rules

**What you CAN do:**
- Change 1-3 parameters at a time in `strategy_tuner.py`
- Any parameter within its documented range is fair game
- Combine insights from multiple experiments

**What you CANNOT do:**
- Modify `backtest_runner.py` or `prepare_market.py`
- Change the scoring function or evaluation harness
- Add new parameters that don't exist in the strategy code
- Change BACKTEST_CONFIG (capital, dates, commission)

**The goal is simple: get the highest composite score.**

## Simplicity Criterion

- Parameters near the edges of their valid ranges are suspicious. Prefer mid-range values unless there's strong evidence.
- If two parameter sets produce similar scores, prefer the one closer to "default" values.
- A parameter change that improves score but reduces num_trades below 50 is risky.

## Strategy-Specific Notes

### Momentum
- `breakout_period` and `trend_sma` interact strongly — change together carefully
- `atr_multiplier` controls stop tightness: too tight = stopped out on noise, too loose = large losses
- `volume_threshold` range is [0.5, 3.0] — values below 1.0 disable the filter
- **NEW**: `tp_atr_multiplier`, `min_confidence`, `min_rr_ratio`, `rsi_oversold` are now WIRED into the strategy (previously dead). Explore these!
- `trend_sma` and `long_trend_sma` now compute dynamic SMAs (no longer tied to hardcoded sma_50/sma_200 columns)

### Mean Reversion
- `z_entry_long` and `z_entry_short` should be symmetric-ish (e.g., -2.5 / +2.5)
- Tighter Z thresholds = more trades but potentially lower quality
- **NEW**: `rsi_long_threshold`, `rsi_short_threshold`, `sl_atr_multiplier` are now WIRED (previously dead). Explore these!
- **NEW**: `min_confidence` and `min_rr_ratio` are now passed to filter_signals (previously ignored)

### Factor Model
- **Weights must sum to 1.0** — the agent auto-normalizes, but propose sensible ratios
- `min_data_days` was previously 252 (caused 0 trades) — don't go above 120
- `rebalance_days` controls turnover — shorter = more trades but higher costs
- Note: `stop_loss_pct`, `take_profit_pct`, `min_composite`, `min_data_days` are read via monkey-patched backtest method

### FX Carry
- `carry_weight + momentum_weight` should sum to ~1.0
- `min_carry_spread` is the minimum interest rate differential — too high = no signals
- This strategy is most sensitive to `atr_stop_multiplier` and `min_composite_score`

## The Experiment Loop

Each experiment runs on a dedicated git branch (`autoresearch/<tag>`).

1. Review current params and experiment history
2. Propose 1-3 parameter changes with reasoning
3. Apply changes to `strategy_tuner.py`
4. Git commit the change
5. Run in-sample backtest (2023-2025, ~2-5 min)
6. Evaluate: did score improve by >= 0.002?
7. If improved → run **out-of-sample backtest** (2020-2022)
8. OOS must score >= 50% of in-sample score with >= 10 trades
9. If both pass → PROMOTE. If OOS fails → REJECT (overfitting detected)
10. Log result to `results.tsv`
11. Repeat

**IMPORTANT**: Parameters that only work on 2023-2025 but fail on 2020-2022
are overfitted. Prefer robust parameters that work across both periods.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human.
If you run out of ideas:
- Try the newly-wired parameters first (tp_atr_multiplier, rsi_oversold, rsi_long_threshold, etc.)
- Try combining previous near-misses
- Try more radical changes (bigger deltas)
- Try the opposite of what you've been doing
- Focus on the strategy with the lowest current score


## Auto-discovered Insights — momentum (2026-03-13 18:37)

## Auto-Discovered Insights — Momentum Strategy (120 experiments)

- [HIGH] **Current baseline is hard-locked at score=0.8420** with breakout_period=20, atr_multiplier=3.25, min_confidence=0.49, min_momentum_3m=0.032, tp_atr_multiplier=5.5. Any micro-adjustment to these core parameters returns ≤0.8420 and fails the 0.002 promotion delta. This plateau has survived 20+ consecutive rejection attempts.

- [HIGH] **min_momentum_3m is the highest-impact single parameter found.** Raising from 0.02→0.03 gained +0.0134 score; 0.03→0.032 gained +0.0014 more. Values below 0.025 degrade performance; values above 0.038 reduce trades too aggressively and hurt trade_activity scoring. Optimal: 0.032.

- [HIGH] **breakout_period=20 is the sweet spot.** Values of 18 scored 0.8222–0.8273; values of 22 scored 0.8121. Rolling back to 18 or pushing to 22 both hurt. Do not move this parameter.

- [HIGH] **atr_multiplier in range [3.15–3.30] is saturated.** All values tested within this band return 0.8392–0.8420. Moving below 3.0 (score=0.7952) or above 3.35 (score=0.8121) degrades sharply. No headroom exists here.

- [HIGH] **tp_atr_multiplier in range [5.2–5.8] is exhausted.** Tested values: 5.2→0.8420, 5.3→0.8398, 5.4→0.8398, 5.5→0.8420, 5.8→0.8392. Pushing to 6.5 returned identical 0.8420. Values above 6.0 combined with low min_momentum_3m (0.02) hurt to 0.8285. Dead zone.

- [HIGH] **min_confidence in [0.45–0.52] is a dead zone.** Values below 0.45 degrade sharpe; 0.51–0.52 with other changes scored 0.8392–0.8420. The 0.49 baseline is near-optimal; no combination tested here has escaped the plateau.

- [MEDIUM] **long_trend_sma is inert across [180–232].** Values of 180, 210, 220, 232 all returned identical 0.8420. This parameter appears decoupled from performance in the current regime — deprioritize.

- [MEDIUM] **volume_threshold and max_positions show weak negative sensitivity.** volume_threshold 1.0→0.9 was rejected; 1.0→1.3 with min_momentum_3m=0.038 scored 0.8328. max_positions 12→10 hurt (0.8324); 12→14 returned 0.8420. These parameters exert minimal positive leverage.

- [MEDIUM] **rsi_oversold changes consistently hurt.** Both 25→30 (score=0.8310) and 25→35 (score=0.8305) when combined with looser min_momentum_3m degraded performance. RSI filter at 25 appears load-bearing; do not loosen it.

- [MEDIUM] **min_rr_ratio is neutral to slightly negative.** Values 1.0→0.9, 1.0→1.2, 1.0→1.5 all returned 0.8420 or below. No gain is achievable here; leave at 1.0.

- [LOW] **Unexplored parameters to attempt as escape from plateau:** trend_sma (only tested 40→50, returned 0.8420), rsi_overbought (untested in promoted experiments), and position_size_pct if it exists in valid range. A 3-parameter combination touching entirely different axes (e.g., trend_sma + rsi_overbought + volume_threshold) has not been attempted simultaneously.

- [LOW] **Combinatorial resets may unlock new baselines.** The plateau may be a local maximum of the current parameter topology. A deliberate 3-parameter shift moving breakout_period to 22, atr_multiplier to 2.9, and min_confidence to 0.46 simultaneously (not individually) has not been cleanly tested and represents the most distinct untried direction.


## Auto-discovered Insights — mean_reversion (2026-03-13 19:07)

## Auto-Discovered Insights — Mean Reversion (last updated: 130 experiments)

- [HIGH] **Current best score: 0.7082** at z_entry_long=-2.15, z_entry_short=2.15, z_exit=0.08, min_confidence=0.6, min_rr_ratio=1.0, z_stop=3.5. This is a robust local optimum — 88+ consecutive rejections confirm it is extremely hard to escape.

- [HIGH] **z_entry thresholds are the highest-impact parameters.** The optimal band is narrow: ±2.1–2.15 scored 0.7082, while ±2.0 drops to 0.7023, ±2.25 drops to 0.7008, ±2.3 drops to 0.6814, and ±2.5 drops to 0.6749. Tighter entries (±1.8) score 0.6738. Do not deviate beyond ±0.1 from ±2.15.

- [HIGH] **z_exit is effectively inert above 0.08.** Values of 0.15, 0.25, 0.3, 0.5 all return 0.7082 or worse when combined with other changes. The current value of 0.08 should be held fixed.

- [HIGH] **min_confidence optimum is 0.6.** Raising to 0.65 or lowering to 0.5 both fail to improve score. This parameter is saturated.

- [HIGH] **min_rr_ratio=1.0 is optimal.** Raising to 1.5 or 2.0 yields identical or worse scores. Do not increase.

- [HIGH] **z_stop, sl_atr_multiplier, min_lookback, max_positions are all inert at current baseline.** z_stop=3.5 vs 4.5, sl_atr_multiplier=1.5–2.8, min_lookback=30–120, max_positions=6–12 all return exactly 0.7082 or worse. These parameters do not offer escape from the local optimum.

- [HIGH] **RSI thresholds (rsi_long_threshold, rsi_short_threshold) are fully inert.** Values of 22/78, 25/75, 28/72, 30/70, 35/65 all return exactly 0.7082. RSI parameters are either not wired to the scoring-relevant path or the current regime doesn't activate them.

- [MEDIUM] **Trade count sweet spot is ~175–197.** Scores above 0.70 cluster in this range. The trade_activity component (ramp to 80) is fully satisfied; further increasing trades does not help scoring but reducing below ~80 risks penalties.

- [MEDIUM] **Sharpe and return are the binding constraints.** Best promoted configs show sharpe ~0.44–0.45 and return ~+3.85%. The score formula weights sharpe (0.35) most heavily — any escape from the local optimum must improve sharpe, not just return.

- [MEDIUM] **Multi-parameter changes that shift z_entry away from ±2.15 consistently degrade performance**, even when combined with compensating adjustments to sl_atr_multiplier, z_exit, or min_lookback. Compound experiments around z_entry are exhausted.

- [LOW] **Promising unexplored directions:** (a) Very fine micro-steps on z_entry (e.g., ±2.12 or ±2.17) have not been explicitly tested — the gap between ±2.10 and ±2.15 may hide marginal gain. (b) Interaction between min_lookback values below 60 and z_entry=±2.15 is underexplored. (c) If the strategy supports any position-sizing or volatility-scaling parameter not yet tested, that would be the highest-priority next direction.

- [LOW] **No parameter combination tested has beaten 0.7082 in 88+ consecutive attempts.** The strategy may be at a global optimum for this parameter space and scoring function. Future effort should focus on identifying any untested parameters or verifying that all parameters are correctly wired in backtest_runner.py.


## Auto-discovered Insights — factor_model (2026-03-13 19:10)

## Auto-Discovered Insights — Factor Model (109 experiments, 14 promoted)

- [HIGH] **Best confirmed configuration**: value_weight=0.33, quality_weight=0.33, momentum_weight=0.34, min_composite=0.465, rebalance_days=21, stop_loss_pct=0.035, take_profit_pct=0.3, min_data_days=60, max_positions=15 → score 0.8898 (current peak).

- [HIGH] **Factor weights: balanced near-equal weighting wins**. The 0.33/0.33/0.34 split (score 0.8898) outperforms skewed configurations. Value-heavy (0.35) and momentum-heavy (0.35) combos score ~0.8885. Quality-heavy (0.4) is a confirmed dead end (score 0.8676, sharpe drops to 1.52). Do not push quality_weight above 0.37.

- [HIGH] **min_composite is high-sensitivity**: Range 0.44–0.47 is the productive zone. Below 0.44 generates too many low-quality trades; above 0.47 cuts trade count too aggressively (trades drop below 300, hurting trade_activity score). Current optimum near 0.465.

- [HIGH] **stop_loss_pct sweet spot is 0.033–0.036**. Values below 0.032 (e.g., 0.0315) and above 0.037 are confirmed losers. The 0.035 value combined with min_composite=0.46 produced a clean promotion. Micro-adjustments below ±0.001 produce noise-level changes.

- [HIGH] **take_profit_pct: 0.30–0.32 is the viable range**. Values of 0.28–0.29 consistently underperform (score drops ~0.01). The 0.31 value with rebalance_days=19 produced the +0.0074 delta promotion. Values above 0.32 are unexplored but likely over-restrict exits.

- [HIGH] **rebalance_days 19–22 is the confirmed sweet spot**. Values of 15 and 18 reduce score (~0.8766–0.8819). Values of 21–22 are near-equivalent; 19 with tight stop_loss (0.0325) achieved 0.8841. Avoid extremes below 18 or above 23.

- [MEDIUM] **max_positions=15 is optimal**. Reducing to 12 consistently scores below baseline (0.8823–0.8858) across 4+ experiments with various combinations. Do not reduce below 15.

- [MEDIUM] **min_data_days: 60 is sufficient**. Increasing to 75, 100, or 120 produced no promotions across 4 experiments (scores 0.8823–0.8858, all rejected). The data quality benefit is not worth the universe reduction.

- [MEDIUM] **value_weight=0.25/quality_weight=0.40 is a confirmed dead end**. This combination appears repeatedly in rejected experiments (5+ times, consistently scoring 0.8846 — below promotion delta of 0.002 from 0.8898 peak). Do not revisit this weight split.

- [MEDIUM] **Parameter interaction: stop_loss + take_profit must move together**. Changing only one while holding the other produces rejections. The 0.033/0.31 and 0.035/0.30 pairs are validated; mismatched pairs (0.034/0.28) reliably fail.

- [LOW] **Unexplored: take_profit_pct above 0.32** (valid range likely extends to 0.35+). Given that 0.31 promoted and 0.32 is untested, there may be marginal gain with a simultaneous stop_loss_pct increase to 0.036.

- [LOW] **Unexplored: max_positions above 15** (if valid range allows 18–20). Current trade counts of 300–390 suggest capacity exists; more positions could push trade_activity closer to the 80-trade ramp ceiling without sacrificing quality.

- [LOW] **Strategy is near a local optimum**. The last 20 experiments show diminishing returns with delta < 0.002 on most attempts. Meaningful progress likely requires simultaneous shifts in 2–3 interacting parameters (e.g., take_profit_pct + stop_loss_pct + rebalance_days) rather than single-parameter micro-tuning.


## Auto-discovered Insights — fx_carry (2026-04-12 07:26)

## Auto-Discovered Insights — FX Carry Strategy

*Last updated: 647 experiments, 65 promoted, 582 rejected*

---

- [HIGH] **Optimal baseline locked at score 0.6619–0.6620**: params `min_carry_spread=0.0145–0.0146`, `atr_stop_multiplier=1.72`, `carry_weight=0.6`, `momentum_weight=0.5`, `min_composite_score=0.095`, `min_confidence=0.45`, `max_positions=9–10`, `min_rr_ratio=1.0`, `max_risk_per_pair=0.012–0.018`. This configuration is highly stable and reproduced across 10+ experiments.

- [HIGH] **`min_carry_spread` is the single highest-impact parameter**: values below 0.013 severely degrade score (0.60–0.63 range). The sweet spot is tightly bounded at 0.0145–0.0146. Values above 0.0148 also hurt (score drops to 0.6156–0.6570). Do not deviate more than ±0.0003 from 0.0146.

- [HIGH] **`atr_stop_multiplier=1.72` is a sharp optimum**: 1.70 and 1.74 both score ~0.658–0.662, while 1.75–1.76 drops to 0.638–0.645. Values above 1.73 or below 1.71 are consistently rejected. Treat 1.72 as fixed.

- [HIGH] **`carry_weight=0.6` + `momentum_weight=0.5` is the winning blend**: carry_weight above 0.62 or momentum_weight below 0.48 degrades score. The 0.6/0.5 combination recurs in all top-scoring experiments. Note: these technically sum to 1.1 — verify normalization behavior in strategy code before pushing further.

- [HIGH] **`min_rr_ratio=1.0` strongly outperforms default 1.5**: every promoted experiment uses 1.0. Raising it back toward 1.5 cuts trade count below the 80-trade activity threshold, reducing `trade_activity` score component.

- [HIGH] **`min_confidence=0.45` is the effective lower bound**: dropping to 0.40 yields identical score (0.6619, rejected for no delta). Values above 0.50 reduce trade count and hurt `trade_activity`. Do not lower further — no evidence of gain below 0.45.

- [HIGH] **`max_positions=9–10` beats default 6**: 9 and 10 both produce ~0.6619. No benefit seen beyond 10 (tested at 10, no improvement). Treat as settled at 9–10.

- [MEDIUM] **`max_risk_per_pair` has low sensitivity in range 0.012–0.028**: scores are nearly identical (0.6616–0.6620) across this range. Prefer 0.018 as a mid-risk default; values outside 0.010–0.025 are untested or degrading.

- [MEDIUM] **`min_composite_score=0.095` is optimal; narrow tolerance**: 0.092 produces 81–86 trades but Sharpe drops to negative (-0.04). 0.097–0.098 slightly reduces trade count. Stay at 0.095.

- [MEDIUM] **Score plateau is real — incremental tuning near baseline yields < 0.002 delta**: over 20 consecutive experiments near the optimum have been rejected for sub-threshold improvement. The scoring function's `sharpe_norm` component (currently 0.24) appears to be the binding constraint requiring a structural shift to break through.

- [MEDIUM] **Promising unexplored directions**: (a) `trend_filter_sma` has only one test (20→20, rejected at wrong spread), try values 15–25 with baseline params; (b) investigate whether `carry_weight + momentum_weight > 1.0` is valid or causes renormalization that caps gains; (c) no experiments have isolated `lookback_period` or `volatility_scaling` if they exist in strategy code.

- [LOW] **Lowering `min_carry_spread` below 0.013 is a confirmed dead end**: three independent experiments (→0.013, →0.0130, →0.0089) all score 0.63 or below. The strategy appears to trade noise at lower thresholds, increasing drawdown and reducing Sharpe.

- [LOW] **Algorithmic random perturbations (crossover/random) have 0% promotion rate**: all 3 logged algorithmic experiments were rejected. Manual targeted changes near known-good values are more effective than stochastic search in this regime.
