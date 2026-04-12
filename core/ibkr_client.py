"""
AlphaDesk — IBKR API Client
Async client for Interactive Brokers via ib_insync (TWS/IB Gateway).

Drop-in replacement for EtoroClient — same method names, same return shapes
where possible. Uses a virtual instrument_id system that maps to IBKR
contracts (Stock/Forex/ETF) so the rest of AlphaDesk doesn't need changes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ib_insync import (
    IB, Stock, Forex, Contract, MarketOrder, Order,
    BarDataList, Position, PortfolioItem, AccountValue,
)

logger = logging.getLogger("alphadesk.ibkr")


# ── Instrument mapping ──
# IBKR doesn't use eToro-style integer IDs — it uses Contract objects.
# We keep a dict mapping symbol → (secType, exchange, currency) and
# cache the resolved conId for fast lookups.

INSTRUMENT_SPECS: Dict[str, Dict[str, Any]] = {
    # US Equities (NYSE/NASDAQ)
    "AAPL":  {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "MSFT":  {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "GOOGL": {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "AMZN":  {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "NVDA":  {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "META":  {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "TSLA":  {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "JPM":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "V":     {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "JNJ":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "WMT":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "PG":    {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "XOM":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "UNH":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "HD":    {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "BAC":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "DIS":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "KO":    {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "PFE":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "NFLX":  {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "SLB":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    # ETFs
    "XLE":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "OIH":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "GLD":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "SLV":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "GDX":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "TLT":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "SPY":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    "QQQ":   {"type": "STK", "exchange": "SMART", "currency": "USD"},
    # Forex
    "EURUSD": {"type": "CASH", "exchange": "IDEALPRO", "currency": "USD", "base": "EUR"},
    "GBPUSD": {"type": "CASH", "exchange": "IDEALPRO", "currency": "USD", "base": "GBP"},
    "USDJPY": {"type": "CASH", "exchange": "IDEALPRO", "currency": "JPY", "base": "USD"},
    "AUDUSD": {"type": "CASH", "exchange": "IDEALPRO", "currency": "USD", "base": "AUD"},
    "USDCHF": {"type": "CASH", "exchange": "IDEALPRO", "currency": "CHF", "base": "USD"},
    "EURGBP": {"type": "CASH", "exchange": "IDEALPRO", "currency": "GBP", "base": "EUR"},
}

# Virtual instrument IDs for compatibility with eToro-shaped portfolio data
# (so strategies don't need to change). Mapped deterministically from symbol.
_SYMBOL_TO_VIRTUAL_ID: Dict[str, int] = {}
_VIRTUAL_ID_TO_SYMBOL: Dict[int, str] = {}


def _virtual_id_for(symbol: str) -> int:
    """Get or create a stable virtual instrument ID for a symbol."""
    if symbol in _SYMBOL_TO_VIRTUAL_ID:
        return _SYMBOL_TO_VIRTUAL_ID[symbol]
    # Use a hash-based ID in the 900000+ range to avoid collision
    # with eToro IDs which are typically < 10000
    vid = 900000 + (hash(symbol) % 100000)
    while vid in _VIRTUAL_ID_TO_SYMBOL:
        vid += 1
    _SYMBOL_TO_VIRTUAL_ID[symbol] = vid
    _VIRTUAL_ID_TO_SYMBOL[vid] = symbol
    return vid


def _symbol_for(instrument_id: int) -> Optional[str]:
    """Reverse lookup: virtual ID → symbol."""
    return _VIRTUAL_ID_TO_SYMBOL.get(instrument_id)


# Pre-populate the mapping so it's deterministic at startup
for _sym in INSTRUMENT_SPECS.keys():
    _virtual_id_for(_sym)


class IBKRClient:
    """Async client for Interactive Brokers via IB Gateway.

    Implements the same interface as EtoroClient so the rest of AlphaDesk
    works without modification. Uses ib_insync which wraps the raw TWS API.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4002,
                 client_id: int = 1, timeout: int = 30,
                 max_retries: int = 3):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.max_retries = max_retries
        self._ib: Optional[IB] = None
        self._contract_cache: Dict[str, Contract] = {}
        self._lock = asyncio.Lock()

    async def _ensure_connected(self) -> IB:
        """Ensure we have a live connection to IB Gateway."""
        async with self._lock:
            if self._ib is None or not self._ib.isConnected():
                self._ib = IB()
                try:
                    await self._ib.connectAsync(
                        self.host, self.port,
                        clientId=self.client_id,
                        timeout=self.timeout,
                    )
                    accounts = self._ib.managedAccounts()
                    logger.info(f"Connected to IBKR: {accounts}")
                except Exception as e:
                    logger.error(f"Failed to connect to IB Gateway at {self.host}:{self.port}: {e}")
                    raise
            return self._ib

    # ────────────────────── Contracts ──────────────────────

    async def _resolve_contract(self, symbol: str) -> Optional[Contract]:
        """Build + qualify an IBKR Contract for a symbol."""
        if symbol in self._contract_cache:
            return self._contract_cache[symbol]

        spec = INSTRUMENT_SPECS.get(symbol)
        if not spec:
            logger.warning(f"No IBKR spec for symbol {symbol}")
            return None

        if spec["type"] == "STK":
            contract = Stock(symbol, spec["exchange"], spec["currency"])
        elif spec["type"] == "CASH":
            contract = Forex(f"{spec['base']}{spec['currency']}")
        else:
            logger.warning(f"Unsupported contract type: {spec['type']}")
            return None

        try:
            ib = await self._ensure_connected()
            qualified = await ib.qualifyContractsAsync(contract)
            if qualified:
                self._contract_cache[symbol] = qualified[0]
                return qualified[0]
        except Exception as e:
            logger.error(f"Contract qualification failed for {symbol}: {e}")
        return None

    async def search_instruments(self, query: str) -> dict:
        """Search instruments by symbol. Returns eToro-shaped dict."""
        symbol = query.upper()
        contract = await self._resolve_contract(symbol)
        if contract:
            return {
                "items": [{
                    "internalSymbolFull": symbol,
                    "instrumentId": _virtual_id_for(symbol),
                    "internalInstrumentDisplayName": symbol,
                }]
            }
        return {"items": []}

    async def get_instruments(self) -> List[dict]:
        """Return all pre-mapped instruments."""
        return [
            {
                "internalSymbolFull": sym,
                "instrumentId": _virtual_id_for(sym),
                "internalInstrumentDisplayName": sym,
            }
            for sym in INSTRUMENT_SPECS.keys()
        ]

    # ────────────────────── Market Data ──────────────────────

    async def get_candles(self, instrument_id: int, direction: str = "desc",
                          interval: str = "OneDay", count: int = 100) -> dict:
        """Get OHLCV candles. Returns eToro-shaped response.

        eToro format: {"interval": "OneDay", "candles": [{"instrumentId": N, "candles": [...]}]}
        """
        symbol = _symbol_for(instrument_id)
        if not symbol:
            logger.warning(f"Unknown instrument_id {instrument_id} for get_candles")
            return {"interval": interval, "candles": []}

        contract = await self._resolve_contract(symbol)
        if not contract:
            return {"interval": interval, "candles": []}

        # Map eToro intervals to IBKR barSize
        bar_size_map = {
            "OneMinute":   "1 min",
            "FiveMinutes": "5 mins",
            "OneHour":     "1 hour",
            "OneDay":      "1 day",
            "OneWeek":     "1 week",
            "OneMonth":    "1 month",
        }
        bar_size = bar_size_map.get(interval, "1 day")

        # Duration: estimate based on count + bar_size
        if "day" in bar_size:
            duration = f"{min(count + 10, 365)} D"
        elif "week" in bar_size:
            duration = f"{min((count + 5) * 7, 730)} D"
        elif "hour" in bar_size:
            duration = f"{min((count + 10) // 6 + 1, 30)} D"
        elif "min" in bar_size:
            duration = f"{min((count + 10), 1000)} S"
        else:
            duration = "1 M"

        try:
            ib = await self._ensure_connected()
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES" if contract.secType == "STK" else "MIDPOINT",
                useRTH=True,
                formatDate=1,
            )

            # Convert to eToro-shaped candles
            candles = []
            for b in bars[-count:]:
                candles.append({
                    "fromDate": b.date.isoformat() if hasattr(b.date, "isoformat") else str(b.date),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume) if b.volume else 0.0,
                })

            if direction == "asc":
                pass  # Already ascending from IBKR
            else:
                candles.reverse()

            return {
                "interval": interval,
                "candles": [{
                    "instrumentId": instrument_id,
                    "candles": candles,
                }]
            }
        except Exception as e:
            logger.error(f"get_candles failed for {symbol}: {e}")
            return {"interval": interval, "candles": []}

    async def get_rates(self, instrument_ids: List[int]) -> dict:
        """Get current bid/ask for instruments."""
        rates = []
        for iid in instrument_ids:
            symbol = _symbol_for(iid)
            if not symbol:
                continue
            contract = await self._resolve_contract(symbol)
            if not contract:
                continue
            try:
                ib = await self._ensure_connected()
                ticker = ib.reqMktData(contract, "", False, False)
                await asyncio.sleep(2)  # Wait for data
                rates.append({
                    "instrumentId": iid,
                    "bid": ticker.bid or ticker.close or 0,
                    "ask": ticker.ask or ticker.close or 0,
                    "last": ticker.last or ticker.close or 0,
                })
                ib.cancelMktData(contract)
            except Exception as e:
                logger.debug(f"get_rates failed for {symbol}: {e}")
        return {"rates": rates}

    async def check_spread(self, instrument_id: int, median_spread: float = None) -> dict:
        """Check current spread for an instrument. eToro compat stub."""
        rates = await self.get_rates([instrument_id])
        if rates.get("rates"):
            r = rates["rates"][0]
            bid, ask = r["bid"], r["ask"]
            if bid > 0 and ask > 0:
                spread = (ask - bid) / ((ask + bid) / 2)
                return {"current_spread": spread, "ok": True}
        return {"current_spread": 0, "ok": False}

    # ────────────────────── Account / Portfolio ──────────────────────

    async def get_portfolio(self) -> dict:
        """Return portfolio in eToro-compatible shape."""
        ib = await self._ensure_connected()
        await ib.reqPositionsAsync()
        positions = ib.positions()
        portfolio_items = ib.portfolio()

        # Index portfolio items by conId for PnL lookup
        items_by_conid = {p.contract.conId: p for p in portfolio_items}

        positions_out = []
        for pos in positions:
            symbol = pos.contract.symbol
            vid = _virtual_id_for(symbol)
            item = items_by_conid.get(pos.contract.conId)

            # IBKR position: avgCost is cost basis per share
            avg_cost = float(pos.avgCost) if pos.avgCost else 0.0
            units = float(pos.position)
            amount = abs(units) * avg_cost  # Dollar amount invested
            is_buy = units > 0

            market_value = float(item.marketValue) if item else amount
            unrealized_pnl = float(item.unrealizedPNL) if item else 0.0
            current_rate = float(item.marketPrice) if item else avg_cost

            positions_out.append({
                "positionID": int(pos.contract.conId),
                "instrumentID": vid,
                "symbol": symbol,
                "isBuy": is_buy,
                "amount": amount,
                "openRate": avg_cost,
                "units": abs(units),
                "unrealizedPnL": {
                    "pnL": unrealized_pnl,
                    "currentRate": current_rate,
                    "marginInAccountCurrency": amount,
                    "exposureInAccountCurrency": market_value,
                    "instrumentID": vid,
                },
            })

        return {
            "clientPortfolio": {
                "positions": positions_out,
                "orders": [],
                "ordersForOpen": [],
                "stockOrders": [],
                "entryOrders": [],
                "exitOrders": [],
                "mirrors": [],
                "credit": 0.0,
                "bonusCredit": 0.0,
            }
        }

    async def get_positions(self) -> List[dict]:
        """Get all open positions (flat list)."""
        data = await self.get_portfolio()
        return data.get("clientPortfolio", {}).get("positions", [])

    async def get_pnl(self) -> dict:
        """Get real account PnL — same shape as get_portfolio but with unrealizedPnL top-level."""
        portfolio = await self.get_portfolio()
        total_pnl = sum(
            p["unrealizedPnL"]["pnL"]
            for p in portfolio["clientPortfolio"]["positions"]
        )
        portfolio["clientPortfolio"]["unrealizedPnL"] = total_pnl
        portfolio["clientPortfolio"]["accountCurrencyId"] = 1
        return portfolio

    async def get_trade_history(self, min_date: str = "2024-01-01",
                                 page: int = 1, page_size: int = 100) -> list:
        """Get closed trade history via IBKR executions."""
        try:
            ib = await self._ensure_connected()
            await ib.reqAllOpenOrdersAsync()
            trades = ib.trades()
            closed = []
            for t in trades:
                if t.orderStatus.status in ("Filled", "Cancelled"):
                    closed.append({
                        "symbol": t.contract.symbol,
                        "instrumentID": _virtual_id_for(t.contract.symbol),
                        "orderID": t.order.orderId,
                        "action": t.order.action,
                        "amount": float(t.order.totalQuantity),
                        "status": t.orderStatus.status,
                    })
            return closed
        except Exception as e:
            logger.error(f"get_trade_history failed: {e}")
            return []

    async def get_account_summary(self) -> dict:
        """Get account summary (balance, cash, etc)."""
        ib = await self._ensure_connected()
        summary = await ib.accountSummaryAsync()
        result = {}
        for s in summary:
            if s.tag in ("NetLiquidation", "TotalCashValue", "AvailableFunds",
                          "BuyingPower", "AccountCode", "Currency"):
                result[s.tag] = s.value
        return result

    # ────────────────────── Trading ──────────────────────

    async def open_position(self, instrument_id: int, is_buy: bool,
                             amount: float, stop_loss: Optional[float] = None,
                             take_profit: Optional[float] = None,
                             leverage: int = 1) -> dict:
        """Open a new position via market order.

        `amount` is the dollar amount — we compute the share quantity
        from the current price.
        """
        symbol = _symbol_for(instrument_id)
        if not symbol:
            raise ValueError(f"Unknown instrument_id: {instrument_id}")

        contract = await self._resolve_contract(symbol)
        if not contract:
            raise ValueError(f"Could not resolve contract for {symbol}")

        ib = await self._ensure_connected()

        # Get current price to compute shares from dollar amount
        ticker = ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(2)
        price = ticker.last or ticker.close or 0
        if not price or price != price:  # NaN check
            price = ticker.marketPrice()
        if not price or price != price:  # Still NaN
            # Fallback: use last historical close
            bars = await ib.reqHistoricalDataAsync(
                contract, endDateTime="", durationStr="2 D",
                barSizeSetting="1 day",
                whatToShow="TRADES" if contract.secType == "STK" else "MIDPOINT",
                useRTH=True, formatDate=1,
            )
            if bars:
                price = float(bars[-1].close)
        ib.cancelMktData(contract)

        if not price or price <= 0 or price != price:
            raise RuntimeError(f"Could not get price for {symbol}")

        shares = int(amount / price)
        if shares < 1:
            shares = 1

        action = "BUY" if is_buy else "SELL"
        order = MarketOrder(action, shares)

        # Attach SL/TP as child orders if provided
        # For now, we use plain market orders; TP/SL handled software-side
        if stop_loss is not None or take_profit is not None:
            logger.info(f"SL/TP passed but using software-side monitoring "
                        f"(SL={stop_loss}, TP={take_profit})")

        logger.info(f"Opening {action} {shares} {symbol} @ ~${price:.2f} "
                    f"(target ${amount})")
        trade = ib.placeOrder(contract, order)

        # Wait a moment for the order to be acknowledged
        await asyncio.sleep(3)

        order_id = trade.order.orderId
        status = trade.orderStatus.status

        # Return in eToro-shaped response
        return {
            "orderForOpen": {
                "instrumentID": instrument_id,
                "orderID": order_id,
                "statusID": 1 if status == "Submitted" else (2 if status == "Filled" else 0),
                "amount": amount,
                "isBuy": is_buy,
                "filled": trade.orderStatus.filled,
                "avgFillPrice": trade.orderStatus.avgFillPrice,
                "status": status,
            },
            "order": {"orderID": order_id},
            "positionId": order_id,  # Use orderId as pseudo position ID
        }

    async def close_position(self, position_id: int, instrument_id: int,
                              units_to_deduct: Optional[float] = None) -> dict:
        """Close an existing position.

        For IBKR: position_id is the conId (from our get_portfolio shim).
        We find the position and place an opposite market order.
        """
        ib = await self._ensure_connected()
        positions = ib.positions()

        # Find position by conId (we stored it as positionID)
        target = None
        for pos in positions:
            if pos.contract.conId == position_id:
                target = pos
                break

        if not target:
            logger.warning(f"Position {position_id} not found for close")
            return {"error": "position_not_found"}

        units = abs(float(target.position))
        if units_to_deduct is not None and units_to_deduct < units:
            units = units_to_deduct

        is_long = float(target.position) > 0
        action = "SELL" if is_long else "BUY"
        order = MarketOrder(action, int(units))

        logger.info(f"Closing position: {target.contract.symbol} {action} {units} shares")
        trade = ib.placeOrder(target.contract, order)
        await asyncio.sleep(2)

        return {
            "orderForClose": {
                "positionID": position_id,
                "instrumentID": instrument_id,
                "orderID": trade.order.orderId,
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
            }
        }

    # ────────────────────── Watchlist (stubs) ──────────────────────

    async def get_watchlist(self) -> List[dict]:
        """IBKR doesn't have a native watchlist concept — return empty."""
        return []

    async def add_to_watchlist(self, instrument_id: int) -> dict:
        """No-op stub for compat."""
        return {"ok": True}

    # ────────────────────── Lifecycle ──────────────────────

    async def close(self):
        """Disconnect from IB Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR")

    async def __aenter__(self):
        await self._ensure_connected()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
