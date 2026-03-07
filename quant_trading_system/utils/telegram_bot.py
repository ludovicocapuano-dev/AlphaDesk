"""
AlphaDesk — Telegram Alert Bot
Sends trade signals, risk alerts, and daily summaries via Telegram.
"""

import logging
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger("alphadesk.telegram")


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
            f"⏰ {datetime.utcnow().strftime('%H:%M UTC')}"
        )
        await self.send(msg)

    async def notify_trade_executed(self, trade_result: dict):
        """Notify when a trade is actually executed."""
        msg = (
            f"✅ <b>TRADE EXECUTED</b>\n"
            f"📊 {trade_result.get('symbol', '?')}\n"
            f"Direction: {trade_result.get('direction', '?')}\n"
            f"Amount: ${trade_result.get('amount', 0):,.2f}\n"
            f"⏰ {datetime.utcnow().strftime('%H:%M UTC')}"
        )
        await self.send(msg)

    async def notify_risk_alert(self, alert_type: str, details: str):
        """Send a risk management alert."""
        msg = (
            f"⚠️ <b>RISK ALERT: {alert_type}</b>\n"
            f"{details}\n"
            f"⏰ {datetime.utcnow().strftime('%H:%M UTC')}"
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
            f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        await self.send(msg)
