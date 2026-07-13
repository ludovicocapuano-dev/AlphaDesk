"""
AlphaDesk — Outcome Labeler
Inspired by Chris Turner's closed-loop learning approach.

Every 15 minutes, sweeps through logged decisions and backfills price outcomes
at 15m, 1h, and 24h horizons. Labels each decision as correct/incorrect.
This creates training-ready data for the ML ensemble.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("alphadesk.outcome_labeler")


class OutcomeLabeler:
    """
    Backfills outcomes for logged trading decisions.

    For each decision:
    - Fetches the price at entry_time + 15m, +1h, +24h
    - Computes PnL at each horizon
    - Labels whether the decision was 'correct'
    - Marks the row as training-ready

    This is the critical link that turns raw signal logs into supervised
    learning datasets for the PyTorch ensemble.
    """

    HORIZONS = {
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "24h": timedelta(hours=24),
    }

    # Default triple-barrier parameters
    TB_MAX_HOLDING = 20          # bars
    TB_ATR_MULTIPLIER = 2.0      # volatility multiplier for barrier width
    TB_LOOKBACK = 20             # bars for rolling volatility

    # A 'correct' buy is one where the price went up by more than slippage
    CORRECTNESS_THRESHOLD = 0.001  # 0.1% (above spread/slippage)

    def __init__(self, db_path: str = "data/alphadesk.db"):
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self):
        """Add outcome columns to the signals table if not present."""
        with sqlite3.connect(self.db_path) as conn:
            # Check existing columns
            cursor = conn.execute("PRAGMA table_info(signals)")
            columns = {row[1] for row in cursor.fetchall()}

            migrations = []
            needed_columns = {
                "outcome_15m_pnl": "REAL",
                "outcome_1h_pnl": "REAL",
                "outcome_4h_pnl": "REAL",
                "outcome_24h_pnl": "REAL",
                "outcome_15m_correct": "INTEGER",
                "outcome_1h_correct": "INTEGER",
                "outcome_4h_correct": "INTEGER",
                "outcome_24h_correct": "INTEGER",
                "outcome_labeled": "INTEGER DEFAULT 0",
                "regime_fingerprint": "TEXT",
                "feature_vector": "TEXT",
                "ml_training_ready": "INTEGER DEFAULT 0",
                # Triple-barrier labeling columns (AFML Ch. 3)
                "tb_label": "INTEGER",
                "tb_exit_type": "TEXT",
                "tb_holding_bars": "INTEGER",
                "tb_return": "REAL",
            }

            for col_name, col_type in needed_columns.items():
                if col_name not in columns:
                    migrations.append(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}")

            for sql in migrations:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass  # Column already exists

            conn.commit()

    @staticmethod
    def compute_dynamic_barriers(close: pd.Series, entry_idx: int,
                                  atr_multiplier: float = 2.0,
                                  lookback: int = 20) -> tuple:
        """
        Compute dynamic upper/lower barriers based on recent volatility.

        Uses rolling standard deviation of log-returns as a proxy for ATR.
        Upper = atr_multiplier * rolling_std
        Lower = -atr_multiplier * rolling_std

        Args:
            close: Full price series.
            entry_idx: Integer index of entry point in the series.
            atr_multiplier: Multiplier applied to volatility.
            lookback: Number of bars for rolling volatility window.

        Returns:
            (upper_pct, lower_pct) as positive and negative fractions.
        """
        # Use up to entry_idx for lookback (avoid look-ahead bias)
        start = max(0, entry_idx - lookback)
        window = close.iloc[start:entry_idx + 1]

        if len(window) < 2:
            # Fallback: 2% barriers when insufficient history
            return (0.02, -0.02)

        log_returns = np.log(window / window.shift(1)).dropna()
        vol = float(log_returns.std())

        if vol <= 0 or np.isnan(vol):
            return (0.02, -0.02)

        upper = atr_multiplier * vol
        lower = -atr_multiplier * vol
        return (upper, lower)

    @staticmethod
    def triple_barrier_label(close: pd.Series, entry_idx: int,
                             upper: float, lower: float,
                             max_holding: int = 20) -> dict:
        """
        Triple-barrier labeling (López de Prado, AFML Ch. 3).

        Three barriers determine the label:
        1. Upper barrier (take-profit): touched first → label = +1
        2. Lower barrier (stop-loss): touched first → label = -1
        3. Vertical barrier (max holding period): hit → label = sign(return)

        Barriers are set dynamically based on volatility (ATR or rolling std).

        Args:
            close: Full price series.
            entry_idx: Integer index of entry point in the series.
            upper: Upper barrier as fraction (e.g., 0.02 = +2%).
            lower: Lower barrier as fraction (e.g., -0.02 = -2%).
            max_holding: Maximum holding period in bars.

        Returns:
            dict with: label (+1, -1, 0), exit_idx, exit_type ('upper'/'lower'/'vertical'),
                       return_pct, holding_bars
        """
        entry_price = close.iloc[entry_idx]
        end_idx = min(entry_idx + max_holding, len(close) - 1)

        # Walk forward bar by bar
        for i in range(entry_idx + 1, end_idx + 1):
            ret = (close.iloc[i] - entry_price) / entry_price

            # Check upper barrier first (take-profit)
            if ret >= upper:
                return {
                    "label": 1,
                    "exit_idx": i,
                    "exit_type": "upper",
                    "return_pct": round(ret, 6),
                    "holding_bars": i - entry_idx,
                }

            # Check lower barrier (stop-loss)
            if ret <= lower:
                return {
                    "label": -1,
                    "exit_idx": i,
                    "exit_type": "lower",
                    "return_pct": round(ret, 6),
                    "holding_bars": i - entry_idx,
                }

        # Vertical barrier: max holding period reached
        final_ret = (close.iloc[end_idx] - entry_price) / entry_price
        label = int(np.sign(final_ret)) if final_ret != 0 else 0

        return {
            "label": label,
            "exit_idx": end_idx,
            "exit_type": "vertical",
            "return_pct": round(final_ret, 6),
            "holding_bars": end_idx - entry_idx,
        }

    async def label_outcomes(self, data_engine) -> int:
        """
        Sweep through unlabeled decisions and backfill outcomes.
        Called every 15 minutes by the scheduler.

        Returns number of decisions labeled.
        """
        labeled_count = 0
        # Writes are deferred to a short batch transaction at the end.
        # Holding a write txn open across the network awaits below kept the
        # SQLite writer lock for minutes per cycle and starved every other
        # writer ("database is locked" on trade execution, AI cost logging).
        pending_outcome: list = []   # (set_clauses_sql, values, signal_id)
        pending_tb: list = []        # (label, exit_type, holding_bars, return_pct, id)

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row

            # Get unlabeled signals that are old enough for at least 15m outcome
            cutoff_15m = (datetime.utcnow() - self.HORIZONS["15m"]).isoformat()
            cutoff_24h = (datetime.utcnow() - self.HORIZONS["24h"]).isoformat()

            rows = conn.execute("""
                SELECT id, timestamp, symbol, signal_type, entry_price,
                       outcome_labeled, metadata
                FROM signals
                WHERE outcome_labeled = 0
                  AND timestamp < ?
                ORDER BY timestamp ASC
                LIMIT 500
            """, (cutoff_15m,)).fetchall()

            for row in rows:
                try:
                    signal_id = row["id"]
                    signal_time = datetime.fromisoformat(row["timestamp"])
                    symbol = row["symbol"]
                    entry_price = row["entry_price"]
                    direction = row["signal_type"]

                    if entry_price is None or entry_price <= 0:
                        continue

                    # Check which horizons we can fill
                    updates = {}
                    all_horizons_filled = True

                    for horizon_name, horizon_delta in self.HORIZONS.items():
                        target_time = signal_time + horizon_delta

                        # Only label if enough time has passed
                        if datetime.utcnow() < target_time:
                            all_horizons_filled = False
                            continue

                        # Get price at target time
                        future_price = await self._get_price_at_time(
                            data_engine, symbol, target_time
                        )

                        if future_price is None:
                            all_horizons_filled = False
                            continue

                        # Compute PnL
                        if direction in ("BUY", "STRONG_BUY"):
                            pnl_pct = (future_price - entry_price) / entry_price
                        else:
                            pnl_pct = (entry_price - future_price) / entry_price

                        # Label correctness
                        is_correct = 1 if pnl_pct > self.CORRECTNESS_THRESHOLD else 0

                        updates[f"outcome_{horizon_name}_pnl"] = round(pnl_pct, 6)
                        updates[f"outcome_{horizon_name}_correct"] = is_correct

                    if updates:
                        # Build UPDATE statement (deferred — no write lock here)
                        set_clauses = ", ".join(f"{k} = ?" for k in updates.keys())
                        values = list(updates.values())

                        if all_horizons_filled:
                            set_clauses += ", outcome_labeled = 1, ml_training_ready = 1"

                        pending_outcome.append((set_clauses, values, signal_id))
                        labeled_count += 1

                except Exception as e:
                    logger.error(f"Labeling error for signal {row['id']}: {e}")

            # --- Triple-barrier labeling pass ---
            tb_rows = conn.execute("""
                SELECT id, timestamp, symbol, signal_type, entry_price
                FROM signals
                WHERE tb_label IS NULL
                  AND entry_price IS NOT NULL
                  AND entry_price > 0
                  AND timestamp < ?
                ORDER BY timestamp ASC
                LIMIT 500
            """, (cutoff_15m,)).fetchall()

            for row in tb_rows:
                try:
                    signal_id = row["id"]
                    signal_time = datetime.fromisoformat(row["timestamp"])
                    symbol = row["symbol"]
                    entry_price = row["entry_price"]

                    # Fetch enough bars for lookback + max_holding
                    bars_needed = self.TB_LOOKBACK + self.TB_MAX_HOLDING + 5
                    from config.instruments import US_EQUITIES, EU_EQUITIES, FX_PAIRS
                    all_instruments = {**US_EQUITIES, **EU_EQUITIES, **FX_PAIRS}
                    instrument_id = all_instruments.get(symbol, {}).get("etoro_id")

                    if not instrument_id:
                        continue

                    df = await data_engine.get_ohlcv(
                        instrument_id, symbol, "OneHour", bars_needed
                    )
                    if df is None or df.empty:
                        continue

                    # Normalize close column (data_engine may return 'close'/'Close')
                    if "close" not in df.columns:
                        if "Close" in df.columns:
                            df = df.rename(columns={"Close": "close"})
                        else:
                            continue

                    # Normalize index to DatetimeIndex (sometimes get_ohlcv returns
                    # a RangeIndex with timestamps in a column)
                    if not isinstance(df.index, pd.DatetimeIndex):
                        ts_col = next(
                            (c for c in ("timestamp", "datetime", "date", "Date",
                                         "fromDate", "FromDate")
                             if c in df.columns),
                            None,
                        )
                        if ts_col is None:
                            continue
                        df = df.copy()
                        df.index = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                        df = df[df.index.notna()]
                        if df.empty:
                            continue

                    close = df["close"].reset_index(drop=True)
                    if len(close) < self.TB_LOOKBACK + self.TB_MAX_HOLDING:
                        continue  # not enough data yet

                    # Align target timestamp tz/precision with df.index
                    target_ts = pd.Timestamp(signal_time)
                    if df.index.tz is not None and target_ts.tz is None:
                        target_ts = target_ts.tz_localize("UTC").tz_convert(df.index.tz)
                    elif df.index.tz is None and target_ts.tz is not None:
                        target_ts = target_ts.tz_localize(None)

                    idx = df.index.get_indexer([target_ts], method="nearest")[0]
                    if idx < 0 or idx >= len(close):
                        continue

                    # Need at least max_holding bars after entry
                    if idx + self.TB_MAX_HOLDING >= len(close):
                        continue

                    # Compute dynamic barriers from volatility
                    upper, lower = self.compute_dynamic_barriers(
                        close, idx,
                        atr_multiplier=self.TB_ATR_MULTIPLIER,
                        lookback=self.TB_LOOKBACK,
                    )

                    # Run triple-barrier labeling
                    tb = self.triple_barrier_label(
                        close, idx, upper, lower, self.TB_MAX_HOLDING
                    )

                    # For SELL/SHORT signals, flip the label
                    direction = row["signal_type"]
                    if direction not in ("BUY", "STRONG_BUY"):
                        tb["label"] = -tb["label"]
                        tb["return_pct"] = -tb["return_pct"]

                    pending_tb.append((tb["label"], tb["exit_type"],
                                       tb["holding_bars"], tb["return_pct"],
                                       signal_id))

                except Exception as e:
                    logger.error(f"Triple-barrier labeling error for signal {row['id']}: {e}")

        # ── Batch write: one short transaction, milliseconds of writer lock ──
        if pending_outcome or pending_tb:
            with sqlite3.connect(self.db_path, timeout=30) as wconn:
                for set_clauses, values, signal_id in pending_outcome:
                    wconn.execute(
                        f"UPDATE signals SET {set_clauses} WHERE id = ?",
                        values + [signal_id],
                    )
                if pending_tb:
                    wconn.executemany(
                        """UPDATE signals
                           SET tb_label = ?, tb_exit_type = ?,
                               tb_holding_bars = ?, tb_return = ?
                           WHERE id = ?""",
                        pending_tb,
                    )
                wconn.commit()

        if labeled_count > 0:
            logger.info(f"Labeled outcomes for {labeled_count} decisions")

        return labeled_count

    async def _get_price_at_time(self, data_engine, symbol: str,
                                   target_time: datetime) -> Optional[float]:
        """Get the closing price closest to target_time.

        Auto-extends fetch window for historical targets, with tz alignment
        between target_time and df.index. Returns None if target falls outside
        the available data range by more than tolerance.
        """
        try:
            # Resolve canonical instrument_id across equities, FX, ETFs, commodities
            from config.instruments import US_EQUITIES, EU_EQUITIES, FX_PAIRS
            all_instruments: dict = {**US_EQUITIES, **EU_EQUITIES, **FX_PAIRS}
            try:
                from config.instruments import ETFS
                all_instruments.update(ETFS)
            except (ImportError, AttributeError):
                pass
            try:
                from config.instruments import COMMODITIES
                all_instruments.update(COMMODITIES)
            except (ImportError, AttributeError):
                pass

            instrument_id = all_instruments.get(symbol, {}).get("etoro_id")
            if not instrument_id:
                return None

            # Choose fetch size based on how far back target_time is
            now_utc = datetime.utcnow()
            target_age = (now_utc - target_time).total_seconds() / 3600  # hours

            # Pick (period, count, tolerance_seconds) by horizon age
            if target_age <= 48:
                period, count, tol_sec = "OneHour", 96, 3600 * 2
            elif target_age <= 24 * 30:
                period, count, tol_sec = "OneHour", int(min(target_age, 720)) + 48, 3600 * 2
            else:
                # Old signal: switch to daily, fetch enough back
                period, count, tol_sec = "OneDay", min(int(target_age // 24) + 5, 365), 86400 * 2

            df = await data_engine.get_ohlcv(instrument_id, symbol, period, count)
            if df is None or df.empty or "close" not in df.columns:
                return None

            # Align tz between target and df.index
            target_ts = pd.Timestamp(target_time)
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None and target_ts.tz is None:
                    target_ts = target_ts.tz_localize("UTC").tz_convert(df.index.tz)
                elif df.index.tz is None and target_ts.tz is not None:
                    target_ts = target_ts.tz_localize(None)
            else:
                return None  # cannot align without DatetimeIndex

            idx_arr = df.index.get_indexer([target_ts], method="nearest")
            if len(idx_arr) == 0:
                return None
            idx = idx_arr[0]
            if idx < 0 or idx >= len(df):
                return None

            # Reject if "nearest" is too far from target (data doesn't cover this range)
            actual_ts = df.index[idx]
            diff_sec = abs((actual_ts - target_ts).total_seconds())
            if diff_sec > tol_sec:
                return None

            return float(df.iloc[idx]["close"])

        except Exception as e:
            logger.debug(f"Price lookup failed for {symbol} at {target_time}: {e}")

        return None

    def get_training_data(self, strategy: str = None,
                           min_age_hours: int = 24,
                           limit: int = 10000) -> pd.DataFrame:
        """
        Extract training-ready data for the ML ensemble.

        Returns DataFrame with features, regime fingerprint, and outcome labels.
        """
        with sqlite3.connect(self.db_path) as conn:
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
            """
            params = []

            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            df = pd.read_sql_query(query, conn, params=params)

        # Parse metadata and regime into features
        if not df.empty:
            df = self._expand_features(df)

        return df

    def _expand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand JSON metadata and regime into flat features."""
        expanded_rows = []

        for _, row in df.iterrows():
            flat = dict(row)

            # Parse metadata
            try:
                meta = json.loads(row.get("metadata", "{}") or "{}")
                for k, v in meta.items():
                    if isinstance(v, (int, float)):
                        flat[f"meta_{k}"] = v
            except (json.JSONDecodeError, TypeError):
                pass

            # Parse regime fingerprint
            try:
                regime = json.loads(row.get("regime_fingerprint", "{}") or "{}")
                for k, v in regime.items():
                    if k != "timestamp" and k != "hash":
                        flat[f"regime_{k}"] = v
            except (json.JSONDecodeError, TypeError):
                pass

            # Parse feature vector. Two on-disk formats coexist:
            #   - list  (main.py path): ml_result["feature_vector"] is a 20/23-dim
            #     array → index as feat_0, feat_1, …
            #   - dict  (ml_bootstrap path): named features → feat_<name>
            try:
                features = json.loads(row.get("feature_vector", "{}") or "{}")
                if isinstance(features, dict):
                    items = features.items()
                elif isinstance(features, (list, tuple)):
                    items = enumerate(features)
                else:
                    items = []
                for k, v in items:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        flat[f"feat_{k}"] = v
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

            expanded_rows.append(flat)

        return pd.DataFrame(expanded_rows)

    def get_labeling_stats(self) -> dict:
        """Get statistics about outcome labeling progress."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            labeled = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE outcome_labeled = 1"
            ).fetchone()[0]
            training_ready = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE ml_training_ready = 1"
            ).fetchone()[0]

            # Win rate by horizon
            stats = {"total": total, "labeled": labeled, "training_ready": training_ready}

            for horizon in ["15m", "1h", "4h", "24h"]:
                correct = conn.execute(
                    f"SELECT AVG(outcome_{horizon}_correct) FROM signals "
                    f"WHERE outcome_{horizon}_correct IS NOT NULL"
                ).fetchone()[0]
                stats[f"accuracy_{horizon}"] = round(correct, 3) if correct else None

        return stats
