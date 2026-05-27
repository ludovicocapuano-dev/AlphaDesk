"""
Backfill outcome labels for signals accumulated while the labeler was broken
(2026-03 → 2026-05-22, ~8k signals with tb_label IS NULL).

Run after applying the outcome_labeler dtype/Column fix. Calls the existing
label_outcomes() method in a loop (500 rows per pass) until no more pending.

Usage:
    cd /home/alphadesk/ibkr-paper
    ./venv/bin/python scripts/backfill_outcome_labels.py [--dry-run] [--max-passes N]
"""
import argparse
import asyncio
import logging
import sqlite3
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import config
from core.data_engine import DataEngine
from core.outcome_labeler import OutcomeLabeler

# Use the broker the service is configured with (eToro or IBKR)
import os
BROKER = os.getenv("BROKER", "etoro").lower()


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("backfill")


def count_pending(db_path: str) -> dict:
    with sqlite3.connect(db_path) as conn:
        tb_null = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE tb_label IS NULL AND entry_price > 0"
        ).fetchone()[0]
        outcome_pending = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome_labeled = 0 AND entry_price > 0"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    return {"tb_null": tb_null, "outcome_pending": outcome_pending, "total": total}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Count only, no DB writes")
    parser.add_argument("--max-passes", type=int, default=50,
                        help="Max iterations through label_outcomes (each does up to 500 rows)")
    parser.add_argument("--sleep-between", type=float, default=2.0,
                        help="Seconds to sleep between passes (rate-limit data API)")
    args = parser.parse_args()

    log = setup_logging()
    log.info(f"DB: {config.db_path}")
    log.info(f"Broker for OHLCV: {BROKER}")

    pre = count_pending(config.db_path)
    log.info(f"Pending: tb_label NULL={pre['tb_null']}, outcome_labeled=0 {pre['outcome_pending']}, total signals={pre['total']}")

    if args.dry_run:
        log.info("--dry-run: exiting without writes")
        return

    if pre["tb_null"] == 0 and pre["outcome_pending"] == 0:
        log.info("Nothing to backfill. Exiting.")
        return

    # Build broker client (same logic as main.py)
    if BROKER.startswith("ibkr"):
        from core.ibkr_client import IBKRClient
        port = 4002 if BROKER == "ibkr_paper" else 4001
        broker = IBKRClient(host=os.getenv("IBKR_HOST", "127.0.0.1"), port=port,
                           client_id=int(os.getenv("IBKR_CLIENT_ID", "47")))
    else:
        from core.etoro_client import EtoroClient
        broker = EtoroClient(
            user_key=os.getenv("ETORO_USER_KEY", ""),
            api_key=os.getenv("ETORO_API_KEY", ""),
            env=os.getenv("ETORO_ENV", "Demo"),
        )

    data_engine = DataEngine(broker)
    labeler = OutcomeLabeler(db_path=config.db_path)

    total_labeled = 0
    pass_num = 0
    prev_tb_null = pre["tb_null"]
    started = time.time()
    while pass_num < args.max_passes:
        pass_num += 1
        try:
            count = await labeler.label_outcomes(data_engine)
        except Exception as e:
            log.error(f"Pass {pass_num} failed: {e}")
            await asyncio.sleep(args.sleep_between * 2)
            continue

        total_labeled += count
        pending = count_pending(config.db_path)
        tb_progress = prev_tb_null - pending["tb_null"]
        elapsed = time.time() - started
        log.info(
            f"Pass {pass_num}: outcome={count} tb_labeled_delta={tb_progress} | "
            f"remaining tb_null={pending['tb_null']} outcome={pending['outcome_pending']} | "
            f"elapsed={elapsed:.0f}s"
        )

        # Terminate on no progress (both passes stuck) or all done
        if pending["tb_null"] == 0 and count == 0:
            log.info("All signals labeled — done.")
            break
        if count == 0 and tb_progress == 0:
            log.info("No progress this pass on either outcome or tb_label — stopping.")
            break

        prev_tb_null = pending["tb_null"]
        await asyncio.sleep(args.sleep_between)

    final = count_pending(config.db_path)
    log.info("=" * 60)
    log.info(f"BACKFILL COMPLETE in {time.time() - started:.0f}s")
    log.info(f"Total labeled: {total_labeled} (across {pass_num} passes)")
    log.info(f"Remaining tb_null: {final['tb_null']}, outcome_pending: {final['outcome_pending']}")
    log.info("=" * 60)

    # Cleanup
    if hasattr(broker, "close"):
        try:
            await broker.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
