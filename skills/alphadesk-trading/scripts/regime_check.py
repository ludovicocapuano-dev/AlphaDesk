#!/usr/bin/env python3
"""AlphaDesk — Regime Detection (standalone, read-only)."""

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

try:
    from core.news_sentiment import NewsSentiment
    HAS_NEWS = True
except ImportError:
    HAS_NEWS = False


async def regime_check():
    etoro = EtoroClient(
        user_key=config.etoro.user_key,
        api_key=config.etoro.api_key,
        base_url=config.etoro.base_url,
        environment=config.etoro.environment,
        timeout=config.etoro.request_timeout,
        max_retries=config.etoro.max_retries,
    )
    data_engine = DataEngine(etoro)
    detector = RegimeDetector()

    try:
        from config.instruments import US_EQUITIES
        market_data = {}
        for symbol, meta in list(US_EQUITIES.items())[:15]:
            inst_id = meta.get("etoro_id")
            if inst_id:
                df = await data_engine.get_ohlcv(inst_id, symbol, "OneDay", 120)
                if not df.empty:
                    df = data_engine.compute_indicators(df)
                    market_data[symbol] = df

        result = {"status": "ok", "regime": None, "sentiment": None, "bubbles": []}

        if market_data:
            regime = detector.detect(market_data)
            if regime:
                result["regime"] = regime.to_dict()

        # News sentiment
        if HAS_NEWS:
            try:
                ns = NewsSentiment()
                result["sentiment"] = {
                    "macro": ns.get_macro_sentiment(),
                    "fed": ns.get_fed_sentiment(),
                }
            except Exception:
                pass

        # Bubble detection (SADF)
        for symbol, df in list(market_data.items())[:10]:
            try:
                bubble_df = data_engine.detect_bubbles(df["close"])
                if bubble_df is not None and not bubble_df.empty:
                    latest = bubble_df.iloc[-1]
                    if latest.get("is_bubble", False):
                        result["bubbles"].append({
                            "symbol": symbol,
                            "sadf_stat": float(latest["sadf_stat"]),
                        })
            except Exception:
                pass

        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
    finally:
        await etoro.close()


if __name__ == "__main__":
    asyncio.run(regime_check())
