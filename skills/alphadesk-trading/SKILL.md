---
name: alphadesk-trading
description: |
  WHAT: Manage the AlphaDesk quant trading system — check portfolio, run signal scans,
  monitor risk, rebalance strategies, inspect ML ensemble, and query trade history.
  WHEN: User says "check portfolio", "run signals", "scan markets", "risk check",
  "rebalance", "show trades", "strategy status", "ML status", "regime check",
  "daily summary", or any AlphaDesk trading operation.
version: 1.0.0
author: ludovicocapuano-dev
---

# AlphaDesk Trading System Skill

You are managing AlphaDesk, a multi-strategy quant trading system on eToro.

## System Overview

- **Entry point**: `/root/AlphaDesk/main.py` (single cycle), `/root/AlphaDesk/scheduler.py` (H24)
- **Database**: `/root/AlphaDesk/alphadesk.db` (SQLite)
- **Venv**: `/root/AlphaDesk/.venv/`
- **Logs**: `/root/AlphaDesk/logs/`
- **Config**: `/root/AlphaDesk/.env` + `/root/AlphaDesk/config/settings.py`

## Strategies (4+1)

| Strategy | File | Allocation | Description |
|----------|------|-----------|-------------|
| Momentum | `strategies/momentum.py` | 25% | 12-1 vol-adjusted, trend following |
| Mean Reversion | `strategies/mean_reversion.py` | 30% | Z-score + pairs + PCA stat-arb |
| Factor Model | `strategies/factor_model.py` | 25% | Value/Quality/Momentum scoring |
| FX Carry | `strategies/fx_carry.py` | 20% | Interest rate differential |
| PEAD | `strategies/pead.py` | 10% | Post-earnings drift (optional) |

## Available Operations

### 1. Portfolio Check
Run `scripts/portfolio_check.py` to get current portfolio state including:
- Equity, cash, invested amounts
- Open positions with P&L
- Current drawdown level
- Regime fingerprint

### 2. Signal Scan
Run `scripts/signal_scan.py` to execute one signal cycle:
- Detects market regime (HMM + volatility + trend)
- Generates signals from all qualified strategies
- ML ensemble filters (veto low-probability trades)
- Meta-labeler gate (per-strategy false positive filter)
- Reports signals with confidence scores

### 3. Risk Monitor
Run `scripts/risk_check.py` to assess risk:
- Portfolio drawdown vs graduated circuit breaker levels
- Strategy qualification status (50 trades + Sharpe > 0.3)
- Regime extremity check
- Position concentration

### 4. Trade History
Query the SQLite database directly for trade history:
```sql
-- Recent trades
SELECT * FROM trades ORDER BY opened_at DESC LIMIT 20;

-- Strategy performance
SELECT strategy, COUNT(*) as trades, AVG(pnl_pct) as avg_pnl,
       SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
FROM trades WHERE closed_at IS NOT NULL
GROUP BY strategy;

-- Signal log with ML decisions
SELECT * FROM signals ORDER BY created_at DESC LIMIT 20;
```

### 5. ML Ensemble Status
Check ML model status, version, and recent prediction accuracy.

### 6. Regime Check
Run regime detection standalone to assess current market conditions.

## Execution Rules

1. **Always activate venv**: `source /root/AlphaDesk/.venv/bin/activate`
2. **Working directory**: `cd /root/AlphaDesk`
3. **Never execute real trades** without explicit user confirmation
4. **Read-only by default**: Portfolio checks and risk monitoring are safe
5. **Signal scans**: Show signals but require confirmation before execution
6. **Database queries**: Always use read-only queries unless updating labels

## Risk Thresholds (Graduated Drawdown)

| Level | Drawdown | Action |
|-------|----------|--------|
| 0 | < 5% | Normal trading |
| 1 | 5-10% | Reduce size 50%, MR+Factor only |
| 2 | 10-15% | Reduce size 75%, MR only |
| 3 | 15-20% | Reduce size 90%, MR only |
| 4 | 20-25% | Halt new trades |
| 5 | > 25% | Close all positions |

## Key ML Components

- **ML Ensemble** (`core/ml_ensemble.py`): PyTorch meta-model, predicts P(profit)
- **Meta-Labeler** (`core/meta_labeler.py`): Per-strategy GBM filter (purged K-fold CV)
- **Regime Detector** (`core/regime_detector.py`): HMM 2-state + vol/trend regime
- **Outcome Labeler** (`core/outcome_labeler.py`): Triple-barrier labels at multiple horizons
- **News Sentiment** (`core/news_sentiment.py`): VADER + RSS macro/stock sentiment

## Response Format

When reporting results, use this structure:

```
## AlphaDesk Status [timestamp]

**Portfolio**: $X equity | $Y cash | Z positions
**Regime**: [bull/bear] | Vol: [low/medium/high/extreme] | VIX: X
**Drawdown**: X% (Level N)

### Signals (if applicable)
| Symbol | Strategy | Direction | Confidence | R:R |
|--------|----------|-----------|------------|-----|

### Alerts (if any)
- [risk alerts, regime changes, etc.]
```
