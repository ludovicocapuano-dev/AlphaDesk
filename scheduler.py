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
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import pandas as pd
import exchange_calendars as xcals

from config.settings import config
from main import AlphaDesk
from utils.logger import setup_logging
from utils.telegram_bot import TelegramBot

logger = logging.getLogger("alphadesk.scheduler")

# Market calendars
NYSE = xcals.get_calendar("XNYS")  # US equities & ETFs

def is_market_open() -> bool:
    """Check if any relevant market is open (equities or FX)."""
    now = pd.Timestamp.now(tz="UTC")
    weekday = now.weekday()  # 0=Mon, 6=Sun

    # FX: 24/5 (Sunday 22:00 UTC → Friday 22:00 UTC)
    if weekday == 6 and now.hour >= 22:
        return True  # Sunday evening FX open
    if weekday == 5 and now.hour >= 22:
        return False  # Saturday — all closed
    if weekday == 5:
        return False  # Saturday before 22:00
    if weekday == 6:
        return False  # Sunday before 22:00

    # Signal scan only during NYSE hours (all active strategies are equity-based)
    # NYSE: 14:30-21:00 UTC (EDT) or 14:30-21:00 UTC
    try:
        return NYSE.is_trading_minute(now)
    except Exception:
        # Fallback: approximate NYSE hours
        return 14 <= now.hour < 21


class AlphaDeskScheduler:
    """
    H24 Scheduler running on VPS (v2 with ML ensemble + macro events).

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
        self.bot = TelegramBot(self.desk)
        self._running = True
        self._macro_trader = None
        self._news_radar = None

    def setup_schedules(self):
        """Configure all scheduled jobs."""

        # ── Signal Scan (every 15 min) ──
        self.scheduler.add_job(
            self._safe_run(self.desk.run_signal_scan, jitter_seconds=240,
                          require_market_open=True),
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
            self._safe_run(self.desk.run_risk_check,
                          require_market_open=True),
            IntervalTrigger(minutes=config.risk_check_interval_minutes),
            id="risk_check",
            name="Risk Monitor",
            max_instances=1,
        )

        # ── News Radar (every 30 min, 24/7) ──
        self.scheduler.add_job(
            self._safe_run(self._run_news_radar),
            IntervalTrigger(minutes=30),
            id="news_radar",
            name="News Radar",
            max_instances=1,
            misfire_grace_time=600,
        )

        # ── Correlation Monitor (every 4h, 24/7) ──
        self.scheduler.add_job(
            self._safe_run(self.desk.run_correlation_check),
            IntervalTrigger(hours=4),
            id="correlation_monitor",
            name="Correlation Monitor",
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

        # ── Weekly Rebalance Check (Sunday 18:00 UTC — before Monday open) ──
        # TODO: Add weekly rebalance check here:
        #   self.scheduler.add_job(
        #       self._safe_run(self._weekly_rebalance_check),
        #       CronTrigger(day_of_week="sun", hour=18, minute=0),
        #       id="weekly_rebalance",
        #       name="Weekly Rebalance Check",
        #   )
        # Implement _weekly_rebalance_check() to call:
        #   from risk.portfolio_rebalancer import PortfolioRebalancer
        #   rebalancer = PortfolioRebalancer()
        #   report = rebalancer.analyze(equity, cash, positions)
        #   msg = PortfolioRebalancer.format_telegram(report)
        #   await self.desk.notifier.send(msg)

        # ── Weekend Maintenance (Saturday 06:00 UTC) ──
        self.scheduler.add_job(
            self._safe_run(self._weekend_maintenance),
            CronTrigger(day_of_week="sat", hour=6, minute=0),
            id="weekend_maintenance",
            name="Weekend Maintenance",
        )

        # ── Macro Event Scheduler (daily 06:00 UTC, Mon-Fri) ──
        # Checks BLS calendar for today's releases, sets up event handlers
        self.scheduler.add_job(
            self._safe_run(self._schedule_macro_events),
            CronTrigger(hour=6, minute=0, day_of_week="mon-fri"),
            id="macro_event_scheduler",
            name="Macro Event Scheduler",
        )

        logger.info("All schedules configured:")
        for job in self.scheduler.get_jobs():
            nrt = getattr(job, 'next_run_time', None)
            logger.info(f"  • {job.name} — next run: {nrt}")

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

    async def _run_news_radar(self):
        """Scan global news feeds for market-moving events."""
        try:
            if self._news_radar is None:
                from core.news_radar import NewsRadar
                self._news_radar = NewsRadar(
                    notifier=self.desk.notifier,
                    db_path=config.db_path,
                )
            report = await self._news_radar.scan()
            if report.get("critical", 0) > 0:
                logger.info(
                    f"📡 News Radar: {report['critical']} CRITICAL events, "
                    f"{report['alerted']} alerted"
                )
            # Invalidate regime cache if any critical event fired
            # (forces regime recomputation on next signal scan)
            if report.get("critical", 0) > 0 and hasattr(self.desk, "_current_regime"):
                self.desk._current_regime = None
        except Exception as e:
            logger.warning(f"News Radar error: {e}")

    async def _schedule_macro_events(self):
        """Check BLS calendar for today's releases and schedule handlers."""
        from core.macro_events import MacroEventTrader

        if not self._macro_trader:
            self._macro_trader = MacroEventTrader(
                self.desk.etoro, self.desk.notifier
            )

        today = datetime.utcnow().date()
        events = self._macro_trader.get_upcoming_events(days_ahead=1)

        for ev in events:
            ev_date = ev["datetime"].date()
            if ev_date != today:
                continue

            # Schedule handler 1 minute before release
            trigger_time = ev["datetime"] - timedelta(minutes=1)
            if trigger_time < datetime.utcnow():
                # Already past — run immediately if within 5 min window
                if (datetime.utcnow() - ev["datetime"]).total_seconds() < 300:
                    logger.info(f"[MACRO] {ev['name']} releasing NOW — handling immediately")
                    await self._handle_macro_event(ev["event"])
                continue

            job_id = f"macro_{ev['event']}_{ev_date}"
            # Remove existing job if re-scheduled
            try:
                self.scheduler.remove_job(job_id)
            except Exception:
                pass

            self.scheduler.add_job(
                self._safe_run(
                    lambda event_key=ev["event"]: self._handle_macro_event(event_key)
                ),
                "date",
                run_date=trigger_time,
                id=job_id,
                name=f"Macro: {ev['name']}",
            )
            logger.info(
                f"[MACRO] Scheduled {ev['name']} handler at {trigger_time.strftime('%H:%M')} UTC"
            )
            await self.desk.notifier.send_message(
                f"🏛️ *MACRO ALERT*: {ev['name']} releasing today at "
                f"{ev['datetime'].strftime('%H:%M')} UTC"
            )

    async def _handle_macro_event(self, event_key: str):
        """Handle a macro data release — poll, compare, trade."""
        if not self._macro_trader:
            return

        # Get portfolio value for sizing
        portfolio_value = 15000  # Default
        try:
            state = self.desk.risk_manager.state
            if state and state.equity > 0:
                portfolio_value = state.equity
        except Exception:
            pass

        result = await self._macro_trader.handle_event(event_key, portfolio_value)
        logger.info(f"[MACRO] {event_key} result: {result.get('status')}")

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

    def _safe_run(self, coro_func, jitter_seconds: int = 0,
                  require_market_open: bool = False):
        """Wrap async functions with error handling and optional jitter.

        Args:
            require_market_open: If True, skip execution when markets are closed.
        """
        async def wrapper():
            try:
                if require_market_open and not is_market_open():
                    logger.debug(f"Markets closed — skipping {coro_func.__name__}")
                    return

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

        # Start scheduler + interactive bot
        self.scheduler.start()
        await self.bot.start()
        logger.info("Scheduler + Telegram bot started. Running H24.")

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
        await self.bot.stop()
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
