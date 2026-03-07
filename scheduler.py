"""
AlphaDesk — H24 Scheduler
APScheduler-based continuous operation for VPS deployment.
Run this as a systemd service for true 24/7 operation.
"""

import asyncio
import logging
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
    H24 Scheduler running on VPS.

    Schedule:
    - Signal scan: every 15 minutes during market hours
    - Risk check: every 5 minutes
    - Daily summary: 21:00 UTC (after US close)
    - Data refresh: every 4 hours
    - FX scan: every 30 minutes (forex is 24/5)
    """

    def __init__(self):
        self.desk = AlphaDesk()
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self._running = True

    def setup_schedules(self):
        """Configure all scheduled jobs."""

        # ── Equity Signal Scan ──
        # US market: 14:30-21:00 UTC (9:30-16:00 ET)
        # EU market: 08:00-16:30 UTC
        # Scan every 15 min during combined hours (08:00-21:00 UTC)
        self.scheduler.add_job(
            self._safe_run(self.desk.run_signal_scan),
            IntervalTrigger(minutes=config.signal_scan_interval_minutes),
            id="signal_scan",
            name="Signal Scan (Equities + FX)",
            max_instances=1,
            misfire_grace_time=300,
        )

        # ── Risk Monitor ──
        # Every 5 minutes, always active
        self.scheduler.add_job(
            self._safe_run(self.desk.run_risk_check),
            IntervalTrigger(minutes=config.risk_check_interval_minutes),
            id="risk_check",
            name="Risk Monitor",
            max_instances=1,
        )

        # ── Daily Summary ──
        # At 21:00 UTC (after US market close)
        self.scheduler.add_job(
            self._safe_run(self.desk.run_daily_summary),
            CronTrigger(hour=21, minute=0),
            id="daily_summary",
            name="Daily Summary",
        )

        # ── Data Cache Refresh ──
        # Every 4 hours, clear stale data
        self.scheduler.add_job(
            self._safe_run(self._refresh_data),
            IntervalTrigger(hours=4),
            id="data_refresh",
            name="Data Refresh",
        )

        # ── Weekend Maintenance ──
        # Saturday 06:00 UTC — cleanup and analytics
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

    async def _weekend_maintenance(self):
        """Weekend maintenance: analytics, cleanup."""
        logger.info("Running weekend maintenance...")
        # Performance analytics per strategy
        for strategy in self.desk.strategies:
            perf = self.desk.db.get_strategy_performance(strategy.name, days=30)
            logger.info(
                f"[{strategy.name}] 30d: {perf['trades']} trades, "
                f"WR={perf['win_rate']:.0%}, Sharpe={perf.get('sharpe', 0):.2f}"
            )

        # Send summary
        await self.desk.notifier.send(
            "📊 <b>WEEKEND REPORT</b>\nMaintenance complete. "
            "System will resume Monday."
        )

    def _safe_run(self, coro_func):
        """Wrap async functions with error handling."""
        async def wrapper():
            try:
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

        logger.info("╔══════════════════════════════════════╗")
        logger.info("║     AlphaDesk H24 Scheduler          ║")
        logger.info("╚══════════════════════════════════════╝")

        # Initialize
        await self.desk.initialize()

        # Setup schedules
        self.setup_schedules()

        # Start
        self.scheduler.start()
        logger.info("Scheduler started. Running H24.")

        await self.desk.notifier.send(
            "🟢 <b>AlphaDesk ONLINE</b>\n"
            f"Environment: {config.etoro.environment}\n"
            f"Strategies: {len(self.desk.strategies)}\n"
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
        await self.desk.notifier.send("🔴 <b>AlphaDesk OFFLINE</b>")
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
