#!/usr/bin/env python3
"""AlphaDesk — Signal Scan (dry-run: generates signals without executing trades)."""

import asyncio
import json
import os
import sys

sys.path.insert(0, "/root/AlphaDesk")
os.chdir("/root/AlphaDesk")

from config.settings import config
from core.etoro_client import EtoroClient
from core.data_engine import DataEngine
from core.regime_detector import RegimeDetector
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.factor_model import FactorModelStrategy
from strategies.fx_carry import FXCarryStrategy

try:
    from strategies.pead import PEADStrategy
    HAS_PEAD = True
except ImportError:
    HAS_PEAD = False

try:
    from core.news_sentiment import NewsSentiment
    HAS_NEWS = True
except ImportError:
    HAS_NEWS = False

from utils.db import TradeDB


async def signal_scan():
    etoro = EtoroClient(
        user_key=config.etoro.user_key,
        api_key=config.etoro.api_key,
        base_url=config.etoro.base_url,
        environment=config.etoro.environment,
        timeout=config.etoro.request_timeout,
        max_retries=config.etoro.max_retries,
    )
    data_engine = DataEngine(etoro)
    regime_detector = RegimeDetector()
    db = TradeDB(config.db_path)

    strategies = [
        MomentumStrategy(config.allocation.momentum),
        MeanReversionStrategy(config.allocation.mean_reversion),
        FactorModelStrategy(config.allocation.factor_model),
        FXCarryStrategy(config.allocation.fx_carry),
    ]
    if HAS_PEAD:
        strategies.append(PEADStrategy(allocation_pct=0.10))

    results = {
        "status": "ok",
        "regime": None,
        "sentiment": None,
        "signals": [],
        "strategy_status": [],
    }

    try:
        # 1. Detect regime
        from config.instruments import US_EQUITIES
        market_data = {}
        for symbol, meta in list(US_EQUITIES.items())[:10]:
            inst_id = meta.get("etoro_id")
            if inst_id:
                df = await data_engine.get_ohlcv(inst_id, symbol, "OneDay", 60)
                if not df.empty:
                    df = data_engine.compute_indicators(df)
                    market_data[symbol] = df

        if market_data:
            regime = regime_detector.detect(market_data)
            results["regime"] = regime.to_dict() if regime else None

            # Pass vol regime for factor model quality tilt
            if regime:
                data_engine._last_vol_regime = regime.data.get("volatility_regime")

        # 2. News sentiment
        if HAS_NEWS:
            try:
                ns = NewsSentiment()
                macro = ns.get_macro_sentiment()
                results["sentiment"] = {
                    "macro_score": macro.get("score", 0),
                    "n_articles": macro.get("n_articles", 0),
                    "label": macro.get("label", "neutral"),
                }
            except Exception:
                pass

        # 3. Generate signals (dry-run)
        for strategy in strategies:
            # Strategy qualification
            perf = db.get_strategy_performance(strategy.name, days=180)
            qualified = True
            if perf["trades"] > 0 and (perf["trades"] < 50 or perf.get("sharpe", 0) < 0.3):
                qualified = False

            results["strategy_status"].append({
                "name": strategy.name,
                "allocation": strategy.allocation_pct,
                "qualified": qualified,
                "trades_180d": perf["trades"],
                "sharpe_180d": perf.get("sharpe", 0),
            })

            try:
                signals = await strategy.generate_signals(data_engine, [])
                for s in signals:
                    results["signals"].append({
                        "symbol": s.symbol,
                        "strategy": s.strategy_name,
                        "direction": s.signal.name if hasattr(s.signal, "name") else str(s.signal),
                        "confidence": round(s.confidence, 3),
                        "entry": round(s.entry_price, 2),
                        "stop_loss": round(s.stop_loss, 2),
                        "take_profit": round(s.take_profit, 2),
                        "rr_ratio": round(s.risk_reward_ratio, 2) if s.risk_reward_ratio else 0,
                        "size_pct": round(s.suggested_size_pct, 4),
                        "metadata": {k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in (s.metadata or {}).items()
                                     if k != "df"},
                    })
            except Exception as e:
                results["strategy_status"][-1]["error"] = str(e)

        print(json.dumps(results, indent=2, default=str))

    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
    finally:
        await etoro.close()


if __name__ == "__main__":
    asyncio.run(signal_scan())
