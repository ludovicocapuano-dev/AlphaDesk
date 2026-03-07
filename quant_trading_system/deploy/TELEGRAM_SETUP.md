# Telegram Bot Setup — Guida Completa

## 1. Creare il Bot

1. Apri Telegram e cerca **@BotFather**
2. Invia `/newbot`
3. Scegli un nome per il bot (es. "AlphaDesk Trader")
4. Scegli un username (es. `alphadesk_trader_bot`) — deve finire con `bot`
5. BotFather ti restituirà un **token** tipo:
   ```
   7123456789:AAH1234567890abcdefghijklmnopqrst
   ```
   Questo è il tuo `TELEGRAM_BOT_TOKEN`.

## 2. Ottenere il Chat ID

1. Avvia una conversazione con il tuo bot (cerca il suo username e premi Start)
2. Invia un messaggio qualsiasi al bot (es. "ciao")
3. Apri nel browser:
   ```
   https://api.telegram.org/bot<IL_TUO_TOKEN>/getUpdates
   ```
4. Nella risposta JSON, trova `"chat":{"id": 123456789}` — questo è il tuo `TELEGRAM_CHAT_ID`

## 3. Configurare il .env

Aggiungi al file `.env` sul VPS:

```bash
TELEGRAM_BOT_TOKEN=7123456789:AAH1234567890abcdefghijklmnopqrst
TELEGRAM_CHAT_ID=123456789
```

## 4. Cosa Riceverai

Il bot ti invierà automaticamente:

### Segnali di Trading (ogni 15 minuti se presenti)
```
🟢 SIGNAL: STRONG_BUY
📊 AAPL | momentum
💰 Entry: 195.4200
🛑 Stop: 188.3100
🎯 Target: 216.7500
📈 R:R = 3.00
🔒 Confidence: 82%
⏰ 14:30 UTC
```

### Esecuzione Trade
```
✅ TRADE EXECUTED
📊 AAPL
Direction: Buy
Amount: $4,250.00
⏰ 14:31 UTC
```

### Alert di Rischio (in tempo reale)
```
⚠️ RISK ALERT: Drawdown Protection
Drawdown: 16.2%
Action: Reducing all positions by 50%
⏰ 15:45 UTC
```

### Summary Giornaliero (21:00 UTC)
```
📋 DAILY SUMMARY
─────────────────────────
💼 Equity: $104,250.00
💵 Cash: $52,100.00
📊 Positions: 8
📉 Drawdown: 3.2%
📈 Daily VaR: 1.8%
─────────────────────────
⏰ 2026-03-06 21:00 UTC
```

### Status di Sistema
```
🟢 AlphaDesk ONLINE
Environment: Demo
Strategies: 4
Scheduler running H24.
```

## 5. Comandi Opzionali (avanzato)

Per aggiungere comandi interattivi al bot (es. /status, /positions),
puoi estendere `utils/telegram_bot.py` con un handler di comandi:

```python
# Aggiungere a telegram_bot.py
async def start_command_handler(self):
    """Listen for commands from Telegram."""
    import httpx
    offset = 0
    while True:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"https://api.telegram.org/bot{self.bot_token}/getUpdates",
                    params={"offset": offset, "timeout": 30}
                )
                data = resp.json()
                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message", {}).get("text", "")
                    if msg == "/status":
                        # Send portfolio status
                        pass
                    elif msg == "/positions":
                        # Send open positions
                        pass
                    elif msg == "/stop":
                        # Emergency stop
                        pass
        except Exception:
            await asyncio.sleep(5)
```

## 6. Test

Per verificare che funzioni, esegui:

```bash
cd ~/alphadesk
source venv/bin/activate
python -c "
import asyncio
from utils.telegram_bot import TelegramNotifier
import os

async def test():
    bot = TelegramNotifier(
        os.getenv('TELEGRAM_BOT_TOKEN'),
        os.getenv('TELEGRAM_CHAT_ID')
    )
    await bot.send('🧪 Test: AlphaDesk Telegram integration working!')
    print('Message sent!')

asyncio.run(test())
"
```

Se ricevi il messaggio su Telegram, tutto funziona.
