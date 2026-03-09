"""
AlphaDesk — Telegram Bot (Interactive + Notifications)

Commands:
  /status      — Portfolio snapshot (equity, cash, positions, drawdown)
  /positions   — Open positions with P&L
  /signals     — Run dry-run signal scan
  /risk        — Risk assessment + drawdown levels
  /regime      — Market regime + sentiment + bubbles
  /trades      — Recent trade history
  /performance — Strategy performance (30d)
  /ml          — ML ensemble status
  /rebalance   — Portfolio rebalancing analysis
  /help        — Show available commands
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger("alphadesk.telegram")


# ─────────────────── Notifier (passive alerts, used by main.py) ───────────────────

class TelegramNotifier:
    """Send formatted alerts to Telegram."""

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token) and bool(chat_id)

    async def send(self, message: str, parse_mode: str = "HTML"):
        """Send a message to Telegram."""
        if not self.enabled:
            return

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    self.API_URL.format(token=self.bot_token),
                    json={
                        "chat_id": self.chat_id,
                        "text": message,
                        "parse_mode": parse_mode,
                    },
                    timeout=10,
                )
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def notify_signal(self, signal):
        """Send a trade signal notification."""
        emoji = "🟢" if signal.signal.value > 0 else "🔴"
        msg = (
            f"{emoji} <b>SIGNAL: {signal.signal.name}</b>\n"
            f"📊 {signal.symbol} | {signal.strategy_name}\n"
            f"💰 Entry: {signal.entry_price:.4f}\n"
            f"🛑 Stop: {signal.stop_loss:.4f}\n"
            f"🎯 Target: {signal.take_profit:.4f}\n"
            f"📈 R:R = {signal.risk_reward_ratio:.2f}\n"
            f"🔒 Confidence: {signal.confidence:.0%}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        await self.send(msg)

    async def notify_trade_executed(self, trade_result: dict):
        """Notify when a trade is actually executed."""
        msg = (
            f"✅ <b>TRADE EXECUTED</b>\n"
            f"📊 {trade_result.get('symbol', '?')}\n"
            f"Direction: {trade_result.get('direction', '?')}\n"
            f"Amount: ${trade_result.get('amount', 0):,.2f}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        await self.send(msg)

    async def notify_risk_alert(self, alert_type: str, details: str):
        """Send a risk management alert."""
        msg = (
            f"⚠️ <b>RISK ALERT: {alert_type}</b>\n"
            f"{details}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        await self.send(msg)

    async def notify_daily_summary(self, summary: dict):
        """Send end-of-day portfolio summary."""
        msg = (
            f"📋 <b>DAILY SUMMARY</b>\n"
            f"{'─' * 25}\n"
            f"💼 Equity: ${summary.get('equity', 0):,.2f}\n"
            f"💵 Cash: ${summary.get('cash', 0):,.2f}\n"
            f"📊 Positions: {summary.get('num_positions', 0)}\n"
            f"📉 Drawdown: {summary.get('current_drawdown', 0):.1%}\n"
            f"📈 Daily VaR: {summary.get('daily_var_95', 0):.2%}\n"
            f"{'─' * 25}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        await self.send(msg)


# ─────────────────── Interactive Bot (command handler) ───────────────────

class TelegramBot:
    """
    Interactive Telegram bot for AlphaDesk.
    Runs alongside the scheduler, handles user commands.
    """

    def __init__(self, desk):
        """
        Args:
            desk: AlphaDesk instance (shared with scheduler)
        """
        self.desk = desk
        self.bot_token = desk.notifier.bot_token
        self.chat_id = desk.notifier.chat_id
        self._app = None

    def _is_authorized(self, chat_id: int) -> bool:
        """Only respond to the authorized chat."""
        return str(chat_id) == str(self.chat_id)

    async def start(self):
        """Start the bot polling loop."""
        from telegram import Update
        from telegram.ext import (
            Application,
            CommandHandler,
            ContextTypes,
        )

        self._app = Application.builder().token(self.bot_token).build()

        # Register commands
        self._app.add_handler(CommandHandler("start", self._cmd_help))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("signals", self._cmd_signals))
        self._app.add_handler(CommandHandler("risk", self._cmd_risk))
        self._app.add_handler(CommandHandler("regime", self._cmd_regime))
        self._app.add_handler(CommandHandler("trades", self._cmd_trades))
        self._app.add_handler(CommandHandler("performance", self._cmd_performance))
        self._app.add_handler(CommandHandler("ml", self._cmd_ml))
        self._app.add_handler(CommandHandler("rebalance", self._cmd_rebalance))
        self._app.add_handler(CommandHandler("set_eur", self._cmd_set_eur))

        logger.info("Telegram bot starting polling...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot polling active")

    async def stop(self):
        """Stop the bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram bot stopped")

    # ─── Command Handlers ───

    async def _cmd_help(self, update, context):
        if not self._is_authorized(update.effective_chat.id):
            return
        msg = (
            "🤖 <b>AlphaDesk v2 — Commands</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "/status — Portfolio snapshot\n"
            "/positions — Open positions + P&L\n"
            "/signals — Dry-run signal scan\n"
            "/risk — Risk & drawdown check\n"
            "/regime — Market regime + sentiment\n"
            "/trades — Recent trade history\n"
            "/performance — Strategy stats (30d)\n"
            "/ml — ML ensemble status\n"
            "/rebalance — Rebalancing analysis\n"
            "/set_eur &lt;amount&gt; — Update EUR wallet balance\n"
            "/help — This message"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_status(self, update, context):
        """Portfolio snapshot."""
        if not self._is_authorized(update.effective_chat.id):
            return
        await update.message.reply_text("⏳ Fetching portfolio...")

        try:
            await self.desk.update_portfolio_state()
            state = self.desk.risk_manager.state
            summary = self.desk.risk_manager.get_portfolio_summary()
            dd_action = self.desk.risk_manager.get_drawdown_action()

            dd_pct = summary.get("current_drawdown", 0)
            dd_level = dd_action.level if dd_action else 0
            dd_emoji = "🟢" if dd_level == 0 else ("🟡" if dd_level <= 2 else "🔴")

            msg = (
                f"💼 <b>PORTFOLIO STATUS</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 Equity: <b>${state.equity:,.2f}</b>\n"
                f"💵 Cash: ${state.cash:,.2f}\n"
                f"📊 Invested: ${state.invested:,.2f}\n"
                f"📈 Positions: {len(state.positions)}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"{dd_emoji} Drawdown: {dd_pct:.1%} (Level {dd_level})\n"
                f"🛡️ VaR 95%: {summary.get('daily_var_95', 0):.2%}\n"
                f"{'🚫 TRADING HALTED' if state.is_halted else '✅ Trading active'}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
            )
            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_positions(self, update, context):
        """List open positions with P&L."""
        if not self._is_authorized(update.effective_chat.id):
            return

        try:
            await self.desk.update_portfolio_state()
            positions = self.desk.risk_manager.state.positions

            if not positions:
                await update.message.reply_text("📭 No open positions.")
                return

            lines = ["📊 <b>OPEN POSITIONS</b>\n━━━━━━━━━━━━━━━━━━━━"]

            total_pnl = 0
            for p in positions:
                sym = p.get("symbol", "?")
                amt = p.get("initialAmountInDollars", 0)
                pnl = p.get("netProfit", 0)
                total_pnl += pnl
                pnl_pct = (pnl / amt * 100) if amt > 0 else 0
                emoji = "🟢" if pnl >= 0 else "🔴"
                strategy = p.get("strategy_tag", "")

                lines.append(
                    f"{emoji} <b>{sym}</b> ${amt:,.0f}\n"
                    f"   P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                    f"{f'  [{strategy}]' if strategy else ''}"
                )

            lines.append(f"\n━━━━━━━━━━━━━━━━━━━━")
            pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
            lines.append(f"{pnl_emoji} Total P&L: <b>${total_pnl:+,.2f}</b>")

            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_signals(self, update, context):
        """Dry-run signal scan."""
        if not self._is_authorized(update.effective_chat.id):
            return
        await update.message.reply_text("⏳ Running signal scan (dry-run)...")

        try:
            # Detect regime
            await self.desk.detect_regime()
            if self.desk._current_regime:
                self.desk.data_engine._last_vol_regime = (
                    self.desk._current_regime.data.get("volatility_regime")
                )
            self.desk._apply_regime_allocations()
            self.desk._apply_ic_weighting()

            # Generate signals (no execution)
            all_signals = []
            for strategy in self.desk.strategies:
                try:
                    signals = await strategy.generate_signals(self.desk.data_engine, [])
                    all_signals.extend(signals)
                except Exception as e:
                    logger.error(f"Signal scan error [{strategy.name}]: {e}")

            if not all_signals:
                regime_info = ""
                if self.desk._current_regime:
                    rd = self.desk._current_regime.data
                    regime_info = (
                        f"\n\nRegime: {rd.get('trend_regime', '?')} | "
                        f"Vol: {rd.get('volatility_regime', '?')}"
                    )
                await update.message.reply_text(
                    f"📭 No signals this cycle.{regime_info}"
                )
                return

            # Format
            lines = [f"📡 <b>SIGNALS ({len(all_signals)})</b>\n━━━━━━━━━━━━━━━━━━━━"]
            for s in all_signals[:15]:  # Cap at 15 for readability
                direction = s.signal.name if hasattr(s.signal, "name") else str(s.signal)
                emoji = "🟢" if "BUY" in direction else "🔴"
                rr = s.risk_reward_ratio if s.risk_reward_ratio else 0

                lines.append(
                    f"{emoji} <b>{s.symbol}</b> [{s.strategy_name}]\n"
                    f"   {direction} | Conf: {s.confidence:.0%} | R:R: {rr:.1f}\n"
                    f"   Entry: {s.entry_price:.2f} | SL: {s.stop_loss:.2f} | TP: {s.take_profit:.2f}"
                )

            if len(all_signals) > 15:
                lines.append(f"\n... +{len(all_signals) - 15} more")

            # Regime footer
            if self.desk._current_regime:
                rd = self.desk._current_regime.data
                lines.append(
                    f"\n━━━━━━━━━━━━━━━━━━━━\n"
                    f"Regime: {rd.get('trend_regime', '?')} | "
                    f"Vol: {rd.get('volatility_regime', '?')}"
                )

            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_risk(self, update, context):
        """Risk assessment."""
        if not self._is_authorized(update.effective_chat.id):
            return

        try:
            await self.desk.update_portfolio_state()
            summary = self.desk.risk_manager.get_portfolio_summary()
            dd_action = self.desk.risk_manager.get_drawdown_action()
            should_reduce, reduction = self.desk.risk_manager.should_reduce_all()

            dd_pct = summary.get("current_drawdown", 0)
            dd_level = dd_action.level if dd_action else 0

            # Strategy health
            strat_lines = []
            for strategy in self.desk.strategies:
                perf = self.desk.db.get_strategy_performance(strategy.name, days=180)
                qualified = perf["trades"] >= 50 and perf.get("sharpe", 0) > 0.3
                q_emoji = "✅" if qualified else ("⚠️" if perf["trades"] > 0 else "❓")
                strat_lines.append(
                    f"  {q_emoji} {strategy.name}: {perf['trades']} trades, "
                    f"Sharpe {perf.get('sharpe', 0):.2f}"
                )

            level_bar = "🟢" * max(0, 5 - dd_level) + "🔴" * dd_level

            msg = (
                f"🛡️ <b>RISK ASSESSMENT</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📉 Drawdown: {dd_pct:.1%}\n"
                f"Level: {level_bar} ({dd_level}/5)\n"
            )

            if dd_action and dd_level > 0:
                msg += (
                    f"Size mult: {dd_action.size_multiplier:.0%}\n"
                    f"Allowed: {dd_action.allowed_strategies or 'all'}\n"
                )

            if should_reduce:
                msg += f"⚠️ <b>REDUCE ALL by {reduction:.0%}</b>\n"

            msg += (
                f"\n<b>Strategy Health (180d):</b>\n"
                + "\n".join(strat_lines)
                + f"\n\n⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
            )

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_regime(self, update, context):
        """Market regime detection."""
        if not self._is_authorized(update.effective_chat.id):
            return
        await update.message.reply_text("⏳ Detecting regime...")

        try:
            await self.desk.detect_regime()
            regime = self.desk._current_regime

            if not regime:
                await update.message.reply_text("❌ Could not detect regime.")
                return

            rd = regime.data
            trend = rd.get("trend_regime", "?")
            vol = rd.get("volatility_regime", "?")
            hmm = rd.get("hmm_regime", "?")

            trend_emoji = {"strong_up": "🚀", "weak_up": "📈", "neutral": "➡️",
                          "weak_down": "📉", "strong_down": "💥"}.get(trend, "❓")
            vol_emoji = {"low": "😴", "medium": "😐", "high": "😰",
                        "extreme": "🔥"}.get(vol, "❓")

            msg = (
                f"🌍 <b>MARKET REGIME</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"{trend_emoji} Trend: <b>{trend}</b>\n"
                f"{vol_emoji} Volatility: <b>{vol}</b>\n"
                f"🧠 HMM: <b>{hmm}</b>\n"
                f"{'⚠️ EXTREME regime' if regime.is_extreme else '✅ Normal regime'}\n"
            )

            # News sentiment
            news_score = rd.get("news_sentiment")
            fed_score = rd.get("fed_sentiment")
            if news_score is not None:
                s_emoji = "🟢" if news_score > 0.1 else ("🔴" if news_score < -0.1 else "⚪")
                msg += f"\n{s_emoji} News sentiment: {news_score:+.2f}"
            if fed_score is not None:
                msg += f"\n🏛️ Fed sentiment: {fed_score:+.2f}"

            # Bubble detection
            bubbles = []
            from config.instruments import US_EQUITIES
            for symbol, meta in list(US_EQUITIES.items())[:10]:
                inst_id = meta.get("etoro_id")
                if not inst_id:
                    continue
                try:
                    df = await self.desk.data_engine.get_ohlcv(inst_id, symbol, "OneDay", 120)
                    if not df.empty:
                        bubble_df = self.desk.data_engine.detect_bubbles(df["close"])
                        if bubble_df is not None and not bubble_df.empty:
                            latest = bubble_df.iloc[-1]
                            if latest.get("is_bubble", False):
                                bubbles.append(f"🫧 {symbol} (SADF: {latest['sadf_stat']:.2f})")
                except Exception:
                    pass

            if bubbles:
                msg += "\n\n<b>Bubble alerts:</b>\n" + "\n".join(bubbles)

            msg += f"\n\n⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_trades(self, update, context):
        """Recent trade history."""
        if not self._is_authorized(update.effective_chat.id):
            return

        try:
            db_path = self.desk.db.db_path
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                "SELECT * FROM trades ORDER BY open_time DESC LIMIT 10"
            ).fetchall()
            conn.close()

            if not rows:
                await update.message.reply_text("📭 No trades in history.")
                return

            lines = ["📜 <b>RECENT TRADES</b>\n━━━━━━━━━━━━━━━━━━━━"]
            for r in rows:
                r = dict(r)
                sym = r.get("symbol", "?")
                direction = r.get("direction", "?")
                pnl = r.get("pnl") or 0
                pnl_pct = r.get("pnl_pct") or 0
                status = r.get("status", "?")
                strategy = r.get("strategy", "?")
                opened = r.get("open_time", "?")[:10]

                if status == "closed":
                    emoji = "🟢" if pnl >= 0 else "🔴"
                    lines.append(
                        f"{emoji} {sym} {direction} [{strategy}]\n"
                        f"   P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%) | {opened}"
                    )
                else:
                    lines.append(
                        f"⏳ {sym} {direction} [{strategy}]\n"
                        f"   Open since {opened}"
                    )

            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_performance(self, update, context):
        """Strategy performance summary."""
        if not self._is_authorized(update.effective_chat.id):
            return

        try:
            lines = ["📊 <b>STRATEGY PERFORMANCE (30d)</b>\n━━━━━━━━━━━━━━━━━━━━"]

            for strategy in self.desk.strategies:
                perf = self.desk.db.get_strategy_performance(strategy.name, days=30)
                trades = perf["trades"]
                if trades == 0:
                    lines.append(f"❓ <b>{strategy.name}</b>: no trades")
                    continue

                wr = perf.get("win_rate", 0)
                sharpe = perf.get("sharpe", 0)
                total_ret = perf.get("total_return", 0)
                avg_win = perf.get("avg_win", 0)
                avg_loss = perf.get("avg_loss", 0)

                emoji = "🟢" if total_ret > 0 else "🔴"
                lines.append(
                    f"{emoji} <b>{strategy.name}</b>\n"
                    f"   Trades: {trades} | WR: {wr:.0%} | Sharpe: {sharpe:.2f}\n"
                    f"   Return: {total_ret:+.2%} | Avg W: {avg_win:+.2%} / L: {avg_loss:+.2%}"
                )

            lines.append(f"\nAllocation: {', '.join(f'{s.name}={s.allocation_pct:.0%}' for s in self.desk.strategies)}")
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_ml(self, update, context):
        """ML ensemble status."""
        if not self._is_authorized(update.effective_chat.id):
            return

        try:
            ml_status = self.desk.ml_ensemble.get_status()
            drift = self.desk.ml_ensemble.check_drift()

            active_emoji = "🟢" if ml_status["active"] else "🟡"
            drift_emoji = "🟢" if not drift["warning"] else "🔴"

            msg = (
                f"🧠 <b>ML ENSEMBLE STATUS</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"{active_emoji} Status: {'ACTIVE' if ml_status['active'] else 'COLD START'}\n"
                f"📦 Model version: v{ml_status['model_version']}\n"
                f"🎯 Accuracy: {ml_status['accuracy']:.1%}\n"
                f"{drift_emoji} Drift PSI: {drift['psi']:.4f}\n"
                f"📊 Samples monitored: {drift['samples_monitored']}\n"
            )

            if drift["needs_retrain"]:
                msg += "⚠️ <b>RETRAIN NEEDED</b> (major drift)\n"

            # Meta-labeler status
            if self.desk.meta_labeler is not None:
                msg += "\n<b>Meta-Labeler:</b> Active ✅"
                for s in self.desk.strategies:
                    has_model = hasattr(self.desk.meta_labeler, '_models') and s.name in self.desk.meta_labeler._models
                    msg += f"\n  {'✅' if has_model else '❌'} {s.name}"
            else:
                msg += "\n<b>Meta-Labeler:</b> Not loaded"

            # Labeling stats
            label_stats = self.desk.outcome_labeler.get_labeling_stats()
            msg += (
                f"\n\n<b>Outcome Labeling:</b>\n"
                f"  Labeled: {label_stats.get('labeled', 0)}/{label_stats.get('total', 0)}\n"
                f"  Training-ready: {label_stats.get('training_ready', 0)}"
            )

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_rebalance(self, update, context):
        """Portfolio rebalancing analysis."""
        if not self._is_authorized(update.effective_chat.id):
            return
        await update.message.reply_text("⏳ Running rebalance analysis...")

        try:
            from risk.portfolio_rebalancer import PortfolioRebalancer

            await self.desk.update_portfolio_state()
            state = self.desk.risk_manager.state

            rebalancer = PortfolioRebalancer()
            report = rebalancer.analyze(
                equity=state.equity,
                cash=state.cash,
                positions=state.positions,
            )

            msg = PortfolioRebalancer.format_telegram(report)

            # Telegram messages have a 4096-char limit; split if needed
            if len(msg) <= 4096:
                await update.message.reply_text(msg, parse_mode="HTML")
            else:
                # Send in chunks at line boundaries
                chunks = []
                current = ""
                for line in msg.split("\n"):
                    if len(current) + len(line) + 1 > 4000:
                        chunks.append(current)
                        current = line
                    else:
                        current = current + "\n" + line if current else line
                if current:
                    chunks.append(current)
                for chunk in chunks:
                    await update.message.reply_text(chunk, parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_set_eur(self, update, context):
        """Update EUR wallet balance. Usage: /set_eur 5000"""
        if not self._is_authorized(update.effective_chat.id):
            return

        if not context.args:
            await update.message.reply_text("Usage: /set_eur <amount>\nExample: /set_eur 5000")
            return

        try:
            amount = float(context.args[0].replace(",", "."))
        except ValueError:
            await update.message.reply_text("❌ Invalid amount. Use: /set_eur 5000")
            return

        # Update .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            lines = []
            found = False
            with open(env_path) as f:
                for line in f:
                    if line.startswith("ETORO_EUR_BALANCE="):
                        lines.append(f"ETORO_EUR_BALANCE={amount}\n")
                        found = True
                    else:
                        lines.append(line)
            if not found:
                lines.append(f"ETORO_EUR_BALANCE={amount}\n")
            with open(env_path, "w") as f:
                f.writelines(lines)

        # Update runtime environment
        os.environ["ETORO_EUR_BALANCE"] = str(amount)

        eur_usd = float(os.getenv("ETORO_EUR_USD_RATE", "1.085"))
        usd_equiv = amount * eur_usd

        await update.message.reply_text(
            f"✅ EUR balance updated: €{amount:,.2f}\n"
            f"USD equivalent: ${usd_equiv:,.2f} (rate {eur_usd})\n"
            f"Use /status to see updated portfolio."
        )
        logger.info(f"EUR balance updated to {amount} via Telegram")
