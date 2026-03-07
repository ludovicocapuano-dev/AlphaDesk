"""
AlphaDesk — Daily Retrain Script
Scheduled at 03:15 UTC (when all markets are closed).

Workflow:
1. Pull latest labeled outcomes from SQLite
2. Check drift on recent predictions
3. Train shadow model
4. Promote if accuracy beats production by 2%+
5. Log results and notify via Telegram
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config
from core.outcome_labeler import OutcomeLabeler
from core.ml_ensemble import MLEnsemble
from utils.telegram_bot import TelegramNotifier
from utils.logger import setup_logging

logger = logging.getLogger("alphadesk.retrain")


async def daily_retrain():
    """Execute the daily retrain pipeline."""
    setup_logging(config.log_path)
    logger.info("=" * 60)
    logger.info("DAILY RETRAIN — Starting")
    logger.info(f"Time: {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)

    notifier = TelegramNotifier(
        bot_token=config.telegram.bot_token,
        chat_id=config.telegram.chat_id,
        enabled=config.telegram.enabled,
    )

    # 1. Get training data
    labeler = OutcomeLabeler(config.db_path)
    stats = labeler.get_labeling_stats()
    logger.info(f"Labeling stats: {json.dumps(stats, indent=2)}")

    if stats["training_ready"] < 200:
        msg = (
            f"Retrain skipped: only {stats['training_ready']} labeled samples "
            f"(need 200+)"
        )
        logger.info(msg)
        await notifier.send(f"🔄 <b>Daily Retrain</b>\n{msg}")
        return {"status": "skipped", "reason": msg}

    training_df = labeler.get_training_data(limit=50000)
    logger.info(f"Training data: {len(training_df)} rows, {training_df.shape[1]} columns")

    # 2. Check drift
    ensemble = MLEnsemble(model_dir=os.path.join(os.path.dirname(config.db_path), "models"))
    drift = ensemble.check_drift()
    logger.info(f"Drift check: PSI={drift['psi']:.4f}, warning={drift['warning']}")

    # 3. Train
    result = ensemble.train(training_df)
    logger.info(f"Training result: {json.dumps(result, indent=2)}")

    # 4. Report
    status_emoji = {
        "promoted": "✅",
        "shadow_kept": "📊",
        "skipped": "⏭️",
        "error": "❌",
    }

    emoji = status_emoji.get(result.get("status", ""), "❓")
    report = (
        f"🔄 <b>Daily Retrain Report</b>\n\n"
        f"{emoji} Status: {result.get('status', 'unknown')}\n"
        f"📈 Model v{result.get('version', '?')}\n"
        f"🎯 Accuracy: {result.get('val_accuracy', 0):.1%}\n"
        f"📊 Samples: {result.get('samples', 0)}\n"
        f"📉 Val Loss: {result.get('val_loss', 0):.4f}\n"
        f"🔍 Drift PSI: {drift['psi']:.4f}\n"
    )

    if result.get("status") == "promoted":
        report += f"\n🏆 Shadow promoted to production!"
    elif result.get("status") == "shadow_kept":
        report += (
            f"\n⚖️ Shadow ({result.get('shadow_accuracy', 0):.1%}) "
            f"didn't beat production ({result.get('production_accuracy', 0):.1%})"
        )

    # Add labeling stats
    report += (
        f"\n\n📋 <b>Labeling Stats</b>\n"
        f"Total signals: {stats['total']}\n"
        f"Labeled: {stats['labeled']}\n"
        f"Training-ready: {stats['training_ready']}\n"
    )

    for horizon in ["15m", "1h", "4h", "24h"]:
        acc = stats.get(f"accuracy_{horizon}")
        if acc is not None:
            report += f"Win rate {horizon}: {acc:.1%}\n"

    await notifier.send(report)
    logger.info("Retrain report sent via Telegram")

    return result


async def label_and_retrain(data_engine=None):
    """
    Combined: label new outcomes, then retrain.
    Called from scheduler or standalone.
    """
    labeler = OutcomeLabeler(config.db_path)

    # Label outcomes (needs data_engine for price lookups)
    if data_engine:
        labeled = await labeler.label_outcomes(data_engine)
        logger.info(f"Labeled {labeled} new outcomes")

    # Retrain
    return await daily_retrain()


if __name__ == "__main__":
    asyncio.run(daily_retrain())
