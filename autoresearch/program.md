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


## Auto-discovered Insights — fx_carry (2026-03-18 07:26)

## Auto-Discovered Insights — FX Carry (last updated: 187 experiments)

- [HIGH] **Best confirmed configuration**: min_carry_spread=0.0145, atr_stop_multiplier=1.72 → score=0.6603 (sharpe=0.21, ret=+0.14%, trades=81). This is the global peak and the primary target to beat.

- [HIGH] **min_carry_spread is the highest-impact parameter**: Tightening from 0.018→0.0145 produced monotonic score gains (+0.05 total). However, dropping below ~0.010 flips returns negative (sharpe drops to -0.3 to -0.7 range), destroying return_norm and calmar_norm despite trade count gains.

- [HIGH] **atr_stop_multiplier sweet spot is 1.70–1.75**: Values of 1.7–1.72 consistently score well. Going below 1.65 hurts performance; 1.3 and 1.55 were both rejected. Do not push below 1.65 or above 1.85 without strong justification.

- [HIGH] **Trade count vs. quality tradeoff**: Scores above 0.65 require 80–90 trades (hitting trade_activity cap). But achieving high trade count via very low min_carry_spread (<0.010) produces negative Sharpe, netting lower composite scores than ~82 trades with positive returns.

- [HIGH] **Dead end — min_carry_spread below 0.009**: Values of 0.008 and 0.0085 both produce negative returns (ret=-0.53%, -0.26%) and negative Sharpe (-0.67, -0.34). The carry signal quality degrades below this threshold.

- [HIGH] **Dead end — trend_filter_sma changes**: Values of 30, 40, 50, and 75 were all rejected vs. baseline of 20. trend_filter_sma=20 is optimal; do not modify this parameter.

- [MEDIUM] **carry_weight=0.7, momentum_weight=0.3** underperforms vs. carry_weight=0.6/momentum_weight=0.5 when min_carry_spread is low (<0.010). At low spreads, more momentum weight makes negative-return trades worse. At mid-range spreads (~0.0145), default weights (0.6/0.5) appear superior.

- [MEDIUM] **min_rr_ratio has low marginal impact**: Tested values of 0.5, 0.8, 1.0, 1.2, 1.5. Reducing from 1.5→1.0 with min_composite_score=0.095 gave score=0.6534 (trades=102, but sharpe=-0.30). Lower min_rr_ratio increases trades but at cost of signal quality when combined with low min_carry_spread.

- [MEDIUM] **min_composite_score has narrow useful range**: Values 0.095–0.105 show marginal differences. Below 0.095 or above 0.12 both rejected. Not worth tuning independently; pair only with min_carry_spread adjustments.

- [MEDIUM] **Promising unexplored direction**: Try min_carry_spread=0.013–0.015, atr_stop_multiplier=1.73–1.78, with momentum_weight=0.5 and carry_weight=0.6. This targets the positive-return regime (sharpe>0) while pushing toward 80 trades. No experiment has combined mid-range spread with slightly higher ATR multiplier and default weights simultaneously.

- [LOW] **max_positions has negligible effect**: Tested 5, 6, 10 — no meaningful score difference at current parameter settings. Leave at default (6).

- [LOW] **max_risk_per_pair=0.028 rejected**: Higher risk sizing (0.028 vs. 0.015) did not improve score; position sizing is not the binding constraint at current regime.

- [LOW] **min_confidence=0.4–0.6735 range explored — no uplift**: Changing min_confidence from 0.6 has not helped in any combination tested. Treat as fixed at 0.6.
