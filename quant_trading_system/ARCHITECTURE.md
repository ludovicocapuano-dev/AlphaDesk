# Quant Trading System — Architecture Document
## Codename: "AlphaDesk"

---

## 1. System Overview

Multi-strategy quantitative trading system connecting to eToro's public API.
Designed for aggressive risk profile on Equities (US/EU) and Forex.

```
┌─────────────────────────────────────────────────────────┐
│                    SCHEDULER (cron/APScheduler)          │
│                    Runs H24 on VPS                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  DATA     │  │  SIGNAL      │  │  RISK            │  │
│  │  ENGINE   │→ │  GENERATOR   │→ │  MANAGER         │  │
│  │           │  │              │  │                  │  │
│  │ • eToro   │  │ • Momentum   │  │ • Position size  │  │
│  │   WebSocket│ │ • MeanRev    │  │ • Max drawdown   │  │
│  │ • News API│  │ • Factor     │  │ • Correlation    │  │
│  │ • Macro   │  │ • FX Carry   │  │ • VaR/CVaR      │  │
│  └──────────┘  └──────────────┘  └────────┬─────────┘  │
│                                            │            │
│                                   ┌────────▼─────────┐  │
│                                   │  EXECUTION       │  │
│                                   │  ENGINE          │  │
│                                   │                  │  │
│                                   │ • Order mgmt     │  │
│                                   │ • Slippage ctrl  │  │
│                                   │ • eToro API      │  │
│                                   └────────┬─────────┘  │
│                                            │            │
│                                   ┌────────▼─────────┐  │
│                                   │  MONITOR &       │  │
│                                   │  ALERTING        │  │
│                                   │                  │  │
│                                   │ • P&L tracking   │  │
│                                   │ • Telegram bot   │  │
│                                   │ • Daily reports  │  │
│                                   └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 2. Strategy Mix

### 2.1 Equities — Momentum/Trend Following (30% allocation)
- Dual momentum: absolute + relative
- Timeframes: daily bars, with weekly confirmation
- Entry: breakout above 20-day high with volume confirmation
- Exit: trailing stop at 2x ATR, or momentum reversal signal
- Universe: S&P 500 + STOXX 600 liquid names

### 2.2 Equities — Mean Reversion (20% allocation)
- Z-score based: buy when Z < -2, sell when Z > +2
- Pairs trading on correlated stocks (cointegration test)
- Bollinger Band squeeze + RSI divergence
- Holding period: 1-5 days

### 2.3 Equities — Multi-Factor (20% allocation)
- Factors: Value (P/E, P/B), Quality (ROE, debt/equity), Momentum (12-1 month)
- Monthly rebalance
- Long top quintile, avoid bottom quintile
- Fama-French style factor scoring

### 2.4 Forex — Carry Trade + Momentum (30% allocation)
- Carry: long high-yield, short low-yield currencies
- Momentum overlay: 1-month and 3-month returns
- Pairs: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, EUR/GBP
- Risk: hard stop at 1.5% of portfolio per pair

## 3. Risk Management Framework

### Position Sizing
- Kelly Criterion (half-Kelly for safety)
- Max 5% portfolio risk per single trade
- Max 15% in correlated positions
- Max 25% gross exposure per strategy

### Portfolio-Level Controls
- Max drawdown trigger: -15% → reduce all positions 50%
- Max drawdown halt: -25% → close all, pause 48h
- Daily VaR limit: 3% of portfolio (95% confidence)
- Correlation matrix updated daily

### Per-Trade Controls
- Mandatory stop-loss on every trade
- Max holding period per strategy type
- Slippage budget: 0.1% equities, 0.05% FX

## 4. Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| eToro WebSocket | Real-time prices, spreads | Streaming |
| eToro REST API | Portfolio, positions, history | On-demand |
| Yahoo Finance (yfinance) | Historical OHLCV, fundamentals | Daily |
| FRED API | Macro indicators (rates, CPI, GDP) | Weekly |
| News API / RSS | Market news sentiment | Hourly |

## 5. Tech Stack

- **Language**: Python 3.11+
- **Scheduling**: APScheduler + systemd on VPS
- **Data**: pandas, numpy, scipy
- **ML/Stats**: scikit-learn, statsmodels
- **API Client**: httpx (async), websockets
- **Storage**: SQLite (positions, trades log), CSV (signals)
- **Alerting**: Telegram Bot API
- **Deployment**: systemd service on VPS

## 6. File Structure

```
quant_trading_system/
├── config/
│   ├── settings.py          # API keys, parameters
│   └── instruments.py       # Tracked instruments
├── core/
│   ├── etoro_client.py      # eToro API wrapper
│   ├── data_engine.py       # Market data aggregation
│   └── execution_engine.py  # Order management
├── strategies/
│   ├── base_strategy.py     # Abstract strategy class
│   ├── momentum.py          # Trend following
│   ├── mean_reversion.py    # Mean reversion
│   ├── factor_model.py      # Multi-factor
│   └── fx_carry.py          # FX carry + momentum
├── risk/
│   ├── position_sizer.py    # Kelly criterion, sizing
│   ├── portfolio_risk.py    # VaR, drawdown, limits
│   └── risk_monitor.py      # Real-time risk tracking
├── utils/
│   ├── logger.py            # Structured logging
│   ├── telegram_bot.py      # Alert notifications
│   └── db.py                # SQLite trade journal
├── main.py                  # Entry point / orchestrator
├── scheduler.py             # H24 scheduling
└── requirements.txt
```
