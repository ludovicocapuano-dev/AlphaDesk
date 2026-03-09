"""
AlphaDesk -- ML Bootstrap
Generates synthetic labeled training data from historical backtests,
inserts it into the signals table, and trains the ML ensemble.

This moves the system from COLD START to ACTIVE.

Usage:
    python -m core.ml_bootstrap
"""

import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoresearch.prepare_market import (
    EQUITY_SYMBOLS, FX_SYMBOLS, FX_META,
    load_equity_data, load_fx_data,
)
from autoresearch.strategy_tuner import get_params, apply_params, BACKTEST_CONFIG
from autoresearch.backtest_runner import patch_strategies, create_strategy
from backtester.engine import BacktestConfig, BacktestEngine
from core.data_engine import DataEngine
from core.ml_ensemble import MLEnsemble, FeaturePipeline
from strategies.base_strategy import Signal

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("alphadesk.ml_bootstrap")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "alphadesk.db")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "models")


# =====================================================================
#  Regime synthesis from price data
# =====================================================================

def classify_volatility(vol_20d: float) -> str:
    """Map annualized 20-day vol to regime bucket."""
    if vol_20d < 0.12:
        return "low"
    elif vol_20d < 0.22:
        return "medium"
    elif vol_20d < 0.35:
        return "high"
    return "extreme"


def classify_trend(sma_20: float, sma_50: float, sma_200: float,
                   momentum_3m: float) -> str:
    """Classify trend regime from moving averages and momentum."""
    if sma_20 > sma_50 > sma_200 and momentum_3m > 0.10:
        return "strong_up"
    elif sma_20 > sma_200 and momentum_3m > 0.02:
        return "weak_up"
    elif sma_20 < sma_50 < sma_200 and momentum_3m < -0.10:
        return "strong_down"
    elif sma_20 < sma_200 and momentum_3m < -0.02:
        return "weak_down"
    return "ranging"


def classify_liquidity(volume_ratio: float) -> str:
    """Classify liquidity from volume ratio."""
    if volume_ratio < 0.6:
        return "low"
    elif volume_ratio > 1.5:
        return "high"
    return "normal"


def build_regime(row: pd.Series) -> dict:
    """Build a regime fingerprint dict from an OHLCV row with indicators."""
    vol = row.get("volatility_20d", 0.18)
    if pd.isna(vol):
        vol = 0.18

    sma_20 = row.get("sma_20", row["close"])
    sma_50 = row.get("sma_50", row["close"])
    sma_200 = row.get("sma_200", row["close"])
    mom_3m = row.get("momentum_3m", 0.0)
    vol_ratio = row.get("volume_ratio", 1.0)

    for v in [sma_20, sma_50, sma_200, mom_3m, vol_ratio]:
        if pd.isna(v):
            v = 0.0

    return {
        "volatility_regime": classify_volatility(float(vol)),
        "trend_regime": classify_trend(
            float(sma_20 if not pd.isna(sma_20) else row["close"]),
            float(sma_50 if not pd.isna(sma_50) else row["close"]),
            float(sma_200 if not pd.isna(sma_200) else row["close"]),
            float(mom_3m if not pd.isna(mom_3m) else 0),
        ),
        "liquidity_regime": classify_liquidity(
            float(vol_ratio if not pd.isna(vol_ratio) else 1.0)
        ),
        "rate_regime": "neutral",
        "correlation_regime": "normal",
    }


def build_signal_data(row: pd.Series, strategy_name: str,
                      direction: str, confidence: float) -> dict:
    """Build the signal_data dict matching FeaturePipeline.extract_features."""
    rsi = row.get("rsi", 50.0)
    macd = row.get("macd", 0.0)
    mom_3m = row.get("momentum_3m", 0.0)
    bb_mid = row.get("bb_mid", row["close"])
    bb_upper = row.get("bb_upper", row["close"] * 1.02)
    bb_lower = row.get("bb_lower", row["close"] * 0.98)
    atr = row.get("atr", row["close"] * 0.02)
    zscore = row.get("zscore", 0.0)
    vol_ratio = row.get("volume_ratio", 1.0)
    sma_20 = row.get("sma_20", row["close"])
    sma_50 = row.get("sma_50", row["close"])

    # Sanitize NaNs
    for name in ["rsi", "macd", "mom_3m", "bb_mid", "bb_upper", "bb_lower",
                 "atr", "zscore", "vol_ratio", "sma_20", "sma_50"]:
        val = locals()[name]
        if pd.isna(val):
            locals()[name]  # just to keep linter happy
            if name == "rsi":
                rsi = 50.0
            elif name == "macd":
                macd = 0.0
            elif name == "mom_3m":
                mom_3m = 0.0
            elif name == "atr":
                atr = row["close"] * 0.02
            elif name == "zscore":
                zscore = 0.0
            elif name == "vol_ratio":
                vol_ratio = 1.0
            elif name == "sma_20":
                sma_20 = row["close"]
            elif name == "sma_50":
                sma_50 = row["close"]
            elif name == "bb_mid":
                bb_mid = row["close"]
            elif name == "bb_upper":
                bb_upper = row["close"] * 1.02
            elif name == "bb_lower":
                bb_lower = row["close"] * 0.98

    # bb_position: where price sits within bands [0,1]
    bb_range = float(bb_upper) - float(bb_lower)
    if bb_range > 0:
        bb_position = (row["close"] - float(bb_lower)) / bb_range
    else:
        bb_position = 0.5

    # Strategy-specific score slots
    momentum_score = 0.0
    mr_zscore = 0.0
    factor_score = 0.0
    fx_carry_score = 0.0

    if strategy_name == "momentum":
        momentum_score = float(mom_3m) if not pd.isna(mom_3m) else 0.0
    elif strategy_name == "mean_reversion":
        mr_zscore = float(zscore) if not pd.isna(zscore) else 0.0
    elif strategy_name == "factor_model":
        factor_score = confidence
    elif strategy_name == "fx_carry":
        fx_carry_score = confidence

    atr_pct = float(atr) / row["close"] if row["close"] > 0 else 0.02

    # Determine hour from index if available
    hour = 14  # Default market hours
    if hasattr(row, "name") and hasattr(row.name, "hour"):
        hour = row.name.hour

    sma_cross = bool(float(sma_20) > float(sma_50)) if not pd.isna(sma_20) and not pd.isna(sma_50) else False

    return {
        "momentum_score": momentum_score,
        "mr_zscore": mr_zscore,
        "factor_score": factor_score,
        "fx_carry_score": fx_carry_score,
        "confidence": confidence,
        "risk_reward": 2.0,
        "atr_pct": atr_pct,
        "volume_ratio": float(vol_ratio) if not pd.isna(vol_ratio) else 1.0,
        "rsi": float(rsi) if not pd.isna(rsi) else 50.0,
        "macd": float(macd) if not pd.isna(macd) else 0.0,
        "bb_position": min(max(bb_position, 0), 1),
        "momentum_3m": float(mom_3m) if not pd.isna(mom_3m) else 0.0,
        "sma_cross": sma_cross,
        "hour": hour,
    }


# =====================================================================
#  Generate synthetic labeled data from backtests
# =====================================================================

def generate_labeled_data(strategy_name: str, price_data: dict) -> list:
    """
    Run a backtest and generate labeled training rows from completed trades.

    Each trade produces a signal-like row with:
    - Features extracted at entry time
    - Outcome labels computed from actual trade PnL at various horizons
    """
    logger.info(f"Generating labeled data for {strategy_name}...")

    patch_strategies()
    strategy = create_strategy(strategy_name)

    bc = BACKTEST_CONFIG
    config = BacktestConfig(
        initial_capital=bc["initial_capital"],
        commission_pct=bc["commission_pct"],
        slippage_pct=bc["slippage_pct"],
        max_positions=bc["max_positions"],
        risk_per_trade=bc["risk_per_trade"],
        start_date=bc["start_date"],
        end_date=bc["end_date"],
    )

    engine = BacktestEngine(config)
    result = engine.run(strategy, price_data)

    logger.info(
        f"  {strategy_name}: {result.num_trades} trades, "
        f"return={result.total_return:+.2%}, sharpe={result.sharpe_ratio:.2f}"
    )

    if result.num_trades == 0:
        logger.warning(f"  {strategy_name} produced 0 trades -- skipping")
        return []

    # Convert trades to labeled signal rows
    rows = []
    pipeline = FeaturePipeline()

    for trade in result.trades:
        symbol = trade.symbol
        if symbol not in price_data:
            continue

        df = price_data[symbol]
        entry_date = trade.entry_date

        # Find the row at entry date
        if entry_date not in df.index:
            # Find nearest
            idx = df.index.get_indexer([entry_date], method="nearest")[0]
            if idx < 0 or idx >= len(df):
                continue
            entry_date = df.index[idx]

        row = df.loc[entry_date]

        direction = "BUY" if trade.direction == "Buy" else "SELL"
        confidence = 0.6  # Default backtest confidence

        # Build feature inputs
        regime = build_regime(row)
        signal_data = build_signal_data(row, strategy_name, direction, confidence)

        # Extract feature vector
        features = pipeline.extract_features(signal_data, regime)

        # Compute outcome at multiple horizons using actual price data
        outcomes = compute_forward_returns(
            df, entry_date, direction, trade.pnl_pct
        )

        rows.append({
            "timestamp": entry_date.isoformat() if hasattr(entry_date, "isoformat") else str(entry_date),
            "symbol": symbol,
            "strategy": strategy_name,
            "signal_type": direction,
            "confidence": confidence,
            "entry_price": trade.entry_price,
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "risk_reward": abs(trade.take_profit - trade.entry_price) / max(abs(trade.entry_price - trade.stop_loss), 0.01),
            "metadata": json.dumps({"source": "bootstrap", "trade_pnl_pct": round(trade.pnl_pct, 6)}),
            "regime_fingerprint": json.dumps(regime),
            "feature_vector": json.dumps(signal_data),
            "outcome_15m_pnl": outcomes.get("15m", 0.0),
            "outcome_1h_pnl": outcomes.get("1h", 0.0),
            "outcome_4h_pnl": outcomes.get("4h", 0.0),
            "outcome_24h_pnl": outcomes.get("24h", trade.pnl_pct),
            "outcome_15m_correct": 1 if outcomes.get("15m", 0) > 0.001 else 0,
            "outcome_1h_correct": 1 if outcomes.get("1h", 0) > 0.001 else 0,
            "outcome_4h_correct": 1 if outcomes.get("4h", 0) > 0.001 else 0,
            "outcome_24h_correct": 1 if outcomes.get("24h", trade.pnl_pct) > 0.001 else 0,
            "outcome_labeled": 1,
            "ml_training_ready": 1,
        })

    logger.info(f"  Generated {len(rows)} labeled samples for {strategy_name}")
    return rows


def compute_forward_returns(df: pd.DataFrame, entry_date,
                            direction: str, final_pnl: float) -> dict:
    """
    Compute forward returns at 15m, 1h, 4h, 24h horizons from daily data.

    Since we only have daily data, we approximate:
    - 15m ~ intraday fraction of day 0 return (0.25 * day 0)
    - 1h  ~ day 0 return (single day move)
    - 4h  ~ average of day 0 and day 1
    - 24h ~ day 1 close vs entry
    """
    idx = df.index.get_loc(entry_date)
    entry_price = df.iloc[idx]["close"]

    results = {}
    horizons = {"15m": 0, "1h": 0, "4h": 1, "24h": 1}

    for horizon, days_ahead in horizons.items():
        target_idx = min(idx + max(days_ahead, 1), len(df) - 1)
        future_price = df.iloc[target_idx]["close"]

        if direction == "BUY":
            pnl = (future_price - entry_price) / entry_price
        else:
            pnl = (entry_price - future_price) / entry_price

        # For sub-day horizons, scale by fraction
        if horizon == "15m":
            pnl *= 0.25
        elif horizon == "1h" and days_ahead == 0:
            # Use high/low range as proxy
            day_range = (df.iloc[idx]["high"] - df.iloc[idx]["low"]) / entry_price
            if direction == "BUY":
                pnl = day_range * 0.3  # Approximate 1h of favorable move
            else:
                pnl = day_range * 0.3

            # Flip sign if trade was a loser overall
            if final_pnl < 0:
                pnl = -abs(pnl)

        results[horizon] = round(pnl, 6)

    return results


# =====================================================================
#  Database operations
# =====================================================================

def insert_labeled_rows(rows: list):
    """Insert labeled signal rows into the database."""
    if not rows:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    columns = [
        "timestamp", "symbol", "strategy", "signal_type", "confidence",
        "entry_price", "stop_loss", "take_profit", "risk_reward",
        "metadata", "regime_fingerprint", "feature_vector",
        "outcome_15m_pnl", "outcome_1h_pnl", "outcome_4h_pnl", "outcome_24h_pnl",
        "outcome_15m_correct", "outcome_1h_correct", "outcome_4h_correct", "outcome_24h_correct",
        "outcome_labeled", "ml_training_ready",
    ]
    placeholders = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)

    inserted = 0
    for row in rows:
        values = [row.get(c) for c in columns]
        try:
            cursor.execute(f"INSERT INTO signals ({col_str}) VALUES ({placeholders})", values)
            inserted += 1
        except Exception as e:
            logger.error(f"Insert error: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Inserted {inserted} rows into signals table")
    return inserted


def get_training_data_from_db() -> pd.DataFrame:
    """Pull labeled data from the database in the format MLEnsemble.train expects."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT symbol, strategy, signal_type, confidence,
               entry_price, stop_loss, take_profit,
               risk_reward AS risk_reward_ratio,
               metadata, regime_fingerprint, feature_vector,
               outcome_15m_pnl, outcome_1h_pnl,
               outcome_4h_pnl, outcome_24h_pnl,
               outcome_1h_correct AS label
        FROM signals
        WHERE ml_training_ready = 1
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return df

    # Expand JSON columns into flat features (matching OutcomeLabeler._expand_features)
    expanded_rows = []
    for _, row in df.iterrows():
        flat = dict(row)

        try:
            meta = json.loads(row.get("metadata", "{}") or "{}")
            for k, v in meta.items():
                if isinstance(v, (int, float)):
                    flat[f"meta_{k}"] = v
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            regime = json.loads(row.get("regime_fingerprint", "{}") or "{}")
            for k, v in regime.items():
                if k not in ("timestamp", "hash"):
                    flat[f"regime_{k}"] = v
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            features = json.loads(row.get("feature_vector", "{}") or "{}")
            for k, v in features.items():
                if isinstance(v, (int, float, bool)):
                    flat[f"feat_{k}"] = float(v)
        except (json.JSONDecodeError, TypeError):
            pass

        expanded_rows.append(flat)

    return pd.DataFrame(expanded_rows)


# =====================================================================
#  Main bootstrap
# =====================================================================

def bootstrap():
    """Run the full bootstrap pipeline."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("ML BOOTSTRAP -- Starting")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Model dir: {MODEL_DIR}")
    logger.info("=" * 60)

    # Check if we already have enough data
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE ml_training_ready = 1"
    ).fetchone()[0]
    conn.close()

    if existing >= 200:
        logger.info(f"Already have {existing} labeled samples -- skipping data generation")
        logger.info("Proceeding directly to training...")
    else:
        logger.info(f"Only {existing} labeled samples -- generating from backtests")

        # Load price data (uses cache from autoresearch/)
        logger.info("Loading equity data...")
        equity_data = load_equity_data()
        logger.info(f"Loaded {len(equity_data)} equity instruments")

        logger.info("Loading FX data...")
        fx_data = load_fx_data()
        logger.info(f"Loaded {len(fx_data)} FX pairs")

        # Run backtests for each strategy and collect labeled rows
        all_rows = []

        # Momentum (uses equities)
        rows = generate_labeled_data("momentum", equity_data)
        all_rows.extend(rows)

        # Mean Reversion (uses equities)
        rows = generate_labeled_data("mean_reversion", equity_data)
        all_rows.extend(rows)

        # Factor Model (uses equities)
        rows = generate_labeled_data("factor_model", equity_data)
        all_rows.extend(rows)

        # FX Carry (uses FX pairs)
        rows = generate_labeled_data("fx_carry", fx_data)
        all_rows.extend(rows)

        logger.info(f"Total labeled samples generated: {len(all_rows)}")

        if not all_rows:
            logger.error("No samples generated from any strategy -- cannot train")
            return

        # Insert into database
        inserted = insert_labeled_rows(all_rows)
        logger.info(f"Inserted {inserted} samples into database")

    # Pull training data
    training_df = get_training_data_from_db()
    logger.info(f"Training data: {len(training_df)} rows, {training_df.shape[1]} columns")

    if len(training_df) < 50:
        logger.error(f"Only {len(training_df)} samples -- insufficient for training")
        return

    # Train the ML ensemble
    logger.info("Initializing ML ensemble...")
    ensemble = MLEnsemble(model_dir=MODEL_DIR)

    # Force training even below 200 samples if we have at least 50
    force = len(training_df) < 200
    if force:
        logger.info(f"Using force=True (have {len(training_df)} < 200 min)")

    result = ensemble.train(training_df, force=force)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("ML BOOTSTRAP -- Complete")
    logger.info(f"Status: {result.get('status', 'unknown')}")
    logger.info(f"Model version: {result.get('version', '?')}")
    logger.info(f"Validation accuracy: {result.get('val_accuracy', 0):.1%}")
    logger.info(f"Training samples: {result.get('samples', 0)}")
    logger.info(f"Epochs: {result.get('epochs', 0)}")
    logger.info(f"Val loss: {result.get('val_loss', 0):.4f}")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info(f"ML ensemble is now: {'ACTIVE' if ensemble.is_active else 'COLD START'}")
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    bootstrap()
