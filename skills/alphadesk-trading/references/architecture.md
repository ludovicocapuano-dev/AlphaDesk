# AlphaDesk Architecture Reference

## Signal Flow

```
Market Data (eToro API)
    │
    ▼
DataEngine (OHLCV + indicators + FFD + CUSUM)
    │
    ▼
RegimeDetector (HMM + vol/trend + SADF bubbles)
    │
    ├─► Allocation adjustment (regime-based + IC-weighted)
    │
    ▼
Strategies (Momentum, MR, Factor, FX, PEAD)
    │
    ├─► Strategy qualification gate (50 trades + Sharpe > 0.3)
    │
    ▼
ML Ensemble (PyTorch, P(profit) prediction)
    │
    ├─► Meta-Labeler (per-strategy GBM, purged K-fold)
    │
    ▼
Risk Manager (drawdown circuit breakers, position limits)
    │
    ├─► Spread check (skip if > 2x median)
    │
    ▼
Position Sizer (Kelly shrinkage + EWMA ATR)
    │
    ▼
eToro Execution (with ±240s jitter)
    │
    ▼
Outcome Labeler (triple-barrier, multi-horizon)
    │
    ▼
ML Retrain (daily at 03:15 UTC)
```

## Key Files

| Component | File | Purpose |
|-----------|------|---------|
| Orchestrator | `main.py` | Single-cycle entry point |
| Scheduler | `scheduler.py` | H24 APScheduler (15min signals, 5min risk) |
| Data Engine | `core/data_engine.py` | OHLCV, indicators, FFD, CUSUM, covariance denoising, SADF |
| Regime | `core/regime_detector.py` | HMM + volatility + trend classification |
| ML Ensemble | `core/ml_ensemble.py` | PyTorch meta-model |
| Meta-Labeler | `core/meta_labeler.py` | Per-strategy false positive filter |
| Outcome Labeler | `core/outcome_labeler.py` | Triple-barrier labeling |
| News Sentiment | `core/news_sentiment.py` | VADER + RSS feeds |
| Risk Manager | `risk/portfolio_risk.py` | Drawdown circuit breakers |
| Position Sizer | `risk/position_sizer.py` | Kelly + EWMA ATR sizing |
| eToro Client | `core/etoro_client.py` | API wrapper + spread monitoring |
| Database | `utils/db.py` | SQLite trade/signal logging |
| Config | `config/settings.py` | All parameters |
| Instruments | `config/instruments.py` | eToro instrument IDs |

## Database Schema (key tables)

- `trades`: symbol, strategy, direction, amount, entry_price, pnl_pct, pnl_dollars, opened_at, closed_at
- `signals`: symbol, strategy_name, signal, confidence, executed, regime_fingerprint, feature_vector
- `daily_snapshots`: equity, cash, n_positions, drawdown
- `outcome_labels`: signal_id, horizon, label, return_pct (triple-barrier)

## Strategy Parameters

### Momentum
- 12-1 month returns (skip last month), vol-adjusted
- Entry: top 20% ranked, trend confirmation
- Stop: 2x ATR, Target: 3x ATR

### Mean Reversion
- Z-score entry: < -2 (long) or > +2 (short)
- RSI confirmation: < 35 (long) or > 65 (short)
- Pairs: Kalman hedge ratio, cointegration test
- PCA stat-arb: SVD residual z-score > 2

### Factor Model
- Value (35%) + Quality (30%) + Momentum (35%)
- Quality tilt to 50% when VIX > 25
- Denoised covariance for inverse-variance weighting

### FX Carry
- Interest rate differential + trend filter
- G10 pairs, momentum overlay

### PEAD
- Earnings surprise > 3% via yfinance
- Hold 60 days, 8% stop loss
