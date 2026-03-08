"""
AlphaDesk — H24 Scheduler (v2 — Self-Learning Ensemble)
APScheduler-based continuous operation for VPS deployment.

v2 additions:
- Outcome labeling every 15 min (after signal scan)
- Daily retrain at 03:15 UTC
- Drift monitoring every 4 hours
- ML status in weekend report
"""

import asyncio
import logging
import random
import signal
import sys

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import config
from main import AlphaDesk
from utils.logger import setup_logging

logger = logging.getLogger("alphadesk.scheduler")


class AlphaDeskScheduler:
    """
    H24 Scheduler running on VPS (v2 with ML ensemble).

    Schedule:
    - Signal scan: every 15 minutes
    - Outcome labeling: every 15 minutes (offset by 2 min from scan)
    - Risk check: every 5 minutes
    - Daily summary: 21:00 UTC (after US close)
    - Daily retrain: 03:15 UTC (all markets closed)
    - Drift monitoring: every 4 hours
    - Data refresh: every 4 hours
    - Weekend maintenance: Saturday 06:00 UTC
    """

    def __init__(self):
        self.desk = AlphaDesk()
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self._running = True

    def setup_schedules(self):
        """Configure all scheduled jobs."""

        # ── Signal Scan (every 15 min) ──
        self.scheduler.add_job(
            self._safe_run(self.desk.run_signal_scan, jitter_seconds=240),
            IntervalTrigger(minutes=config.signal_scan_interval_minutes),
            id="signal_scan",
            name="Signal Scan (v2 + ML)",
            max_instances=1,
            misfire_grace_time=300,
        )

        # ── Outcome Labeling (every 15 min, offset by 2 min) ──
        self.scheduler.add_job(
            self._safe_run(self.desk.run_outcome_labeling),
            CronTrigger(minute="2,17,32,47"),
            id="outcome_labeling",
            name="Outcome Labeler",
            max_instances=1,
            misfire_grace_time=300,
        )

        # ── Risk Monitor (every 5 min) ──
        self.scheduler.add_job(
            self._safe_run(self.desk.run_risk_check),
            IntervalTrigger(minutes=config.risk_check_interval_minutes),
            id="risk_check",
            name="Risk Monitor",
            max_instances=1,
        )

        # ── Daily Summary (21:00 UTC) ──
        self.scheduler.add_job(
            self._safe_run(self.desk.run_daily_summary),
            CronTrigger(hour=21, minute=0),
            id="daily_summary",
            name="Daily Summary",
        )

        # ── Daily Retrain (03:15 UTC — all markets closed) ──
        self.scheduler.add_job(
            self._safe_run(self.desk.run_daily_retrain),
            CronTrigger(hour=3, minute=15),
            id="daily_retrain",
            name="ML Daily Retrain",
            misfire_grace_time=1800,
        )

        # ── Drift Monitor (every 4 hours) ──
        self.scheduler.add_job(
            self._safe_run(self._check_drift),
            IntervalTrigger(hours=4),
            id="drift_monitor",
            name="ML Drift Monitor",
        )

        # ── Data Cache Refresh (every 4 hours) ──
        self.scheduler.add_job(
            self._safe_run(self._refresh_data),
            IntervalTrigger(hours=4),
            id="data_refresh",
            name="Data Refresh",
        )

        # ── Weekend Maintenance (Saturday 06:00 UTC) ──
        self.scheduler.add_job(
            self._safe_run(self._weekend_maintenance),
            CronTrigger(day_of_week="sat", hour=6, minute=0),
            id="weekend_maintenance",
            name="Weekend Maintenance",
        )

        logger.info("All schedules configured:")
        for job in self.scheduler.get_jobs():
            logger.info(f"  • {job.name} — next run: {job.next_run_time}")

    async def _refresh_data(self):
        """Refresh cached data."""
        self.desk.data_engine.clear_cache()
        logger.info("Data cache refreshed")

    async def _check_drift(self):
        """Check ML feature drift."""
        drift = self.desk.ml_ensemble.check_drift()
        if drift["warning"]:
            await self.desk.notifier.notify_risk_alert(
                "ML Drift Warning",
                f"PSI: {drift['psi']:.4f}\n"
                f"Needs retrain: {drift['needs_retrain']}\n"
                f"Samples monitored: {drift['samples_monitored']}"
            )
        if drift["needs_retrain"]:
            logger.warning("Major drift detected — triggering emergency retrain")
            await self.desk.run_daily_retrain()

    async def _weekend_maintenance(self):
        """Weekend maintenance: analytics, cleanup, ML report."""
        logger.info("Running weekend maintenance...")

        # Performance analytics per strategy
        report_lines = []
        for strategy in self.desk.strategies:
            perf = self.desk.db.get_strategy_performance(strategy.name, days=30)
            line = (
                f"[{strategy.name}] 30d: {perf['trades']} trades, "
                f"WR={perf['win_rate']:.0%}, Sharpe={perf.get('sharpe', 0):.2f}"
            )
            logger.info(line)
            report_lines.append(line)

        # ML ensemble status
        ml_status = self.desk.ml_ensemble.get_status()
        ml_line = (
            f"ML Ensemble: v{ml_status['model_version']}, "
            f"{'ACTIVE' if ml_status['active'] else 'COLD START'}, "
            f"accuracy={ml_status['accuracy']:.1%}, "
            f"drift PSI={ml_status['drift_psi']:.4f}"
        )
        report_lines.append(ml_line)

        # Labeling stats
        label_stats = self.desk.outcome_labeler.get_labeling_stats()
        label_line = (
            f"Labeling: {label_stats['labeled']}/{label_stats['total']} labeled, "
            f"{label_stats['training_ready']} training-ready"
        )
        report_lines.append(label_line)

        # Send summary
        report = "\n".join(report_lines)
        await self.desk.notifier.send(
            f"📊 <b>WEEKEND REPORT</b>\n<pre>{report}</pre>\n"
            "System will resume Monday."
        )

    def _safe_run(self, coro_func, jitter_seconds: int = 0):
        """Wrap async functions with error handling and optional jitter."""
        async def wrapper():
            try:
                if jitter_seconds > 0:
                    delay = random.randint(0, jitter_seconds)
                    logger.debug(f"Applying {delay}s jitter before {coro_func.__name__}")
                    await asyncio.sleep(delay)
                await coro_func()
            except Exception as e:
                logger.error(f"Scheduled job failed: {e}", exc_info=True)
                try:
                    await self.desk.notifier.notify_risk_alert(
                        "System Error", f"Job failed: {str(e)[:200]}"
                    )
                except:
                    pass
        return wrapper

    async def start(self):
        """Start the scheduler."""
        setup_logging(config.log_path)

        logger.info("╔══════════════════════════════════════════╗")
        logger.info("║  AlphaDesk H24 Scheduler v2              ║")
        logger.info("║  Self-Learning Ensemble Edition           ║")
        logger.info("╚══════════════════════════════════════════╝")

        # Initialize
        await self.desk.initialize()

        # Setup schedules
        self.setup_schedules()

        # Start
        self.scheduler.start()
        logger.info("Scheduler started. Running H24.")

        ml_status = self.desk.ml_ensemble.get_status()
        await self.desk.notifier.send(
            "🟢 <b>AlphaDesk v2 ONLINE</b>\n"
            f"Environment: {config.etoro.environment}\n"
            f"Strategies: {len(self.desk.strategies)}\n"
            f"ML Ensemble: v{ml_status['model_version']} "
            f"({'Active' if ml_status['active'] else 'Cold Start'})\n"
            "Scheduler running H24."
        )

        # Keep running
        try:
            while self._running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            await self.stop()

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping scheduler...")
        self._running = False
        self.scheduler.shutdown(wait=True)
        await self.desk.shutdown()
        await self.desk.notifier.send("🔴 <b>AlphaDesk v2 OFFLINE</b>")
        logger.info("Scheduler stopped.")


def main():
    """Entry point."""
    scheduler = AlphaDeskScheduler()

    # Handle SIGTERM (systemd stop)
    def handle_signal(signum, frame):
        scheduler._running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    asyncio.run(scheduler.start())


if __name__ == "__main__":
    main()
