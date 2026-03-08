"""
AlphaDesk — eToro API Client
Async HTTP + WebSocket client for eToro's public API.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger("alphadesk.etoro")


class EtoroClient:
    """Async client for eToro REST + WebSocket API."""

    def __init__(self, user_key: str, api_key: str, base_url: str,
                 environment: str = "Demo", timeout: int = 30, max_retries: int = 3):
        self.user_key = user_key
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.environment = environment
        self.timeout = timeout
        self.max_retries = max_retries
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "x-api-key": self.api_key,
                    "x-user-key": self.user_key,
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._http_client

    def _auth_headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "x-user-key": self.user_key,
            "x-request-id": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an authenticated API request with retries."""
        client = await self._get_client()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                headers = {"x-request-id": str(uuid.uuid4())}
                response = await client.request(method, path, headers=headers, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    # Rate limited — back off exponentially
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                elif e.response.status_code >= 500:
                    logger.warning(f"Server error {e.response.status_code}, retrying...")
                    await asyncio.sleep(1)
                else:
                    raise
            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"Request error: {e}, retrying...")
                await asyncio.sleep(1)

        raise last_error

    # ────────────────────── Market Data ──────────────────────

    async def get_instruments(self) -> List[dict]:
        """Fetch all available instruments and their metadata."""
        data = await self._request("GET", "/market-data/instruments")
        return data.get("instruments", data) if isinstance(data, dict) else data

    async def search_instruments(self, query: str) -> dict:
        """Search for instruments by name or symbol."""
        return await self._request("GET", "/market-data/search", params={"q": query})

    async def get_rates(self, instrument_ids: List[int]) -> dict:
        """Get current rates for instruments."""
        ids_str = ",".join(str(i) for i in instrument_ids)
        return await self._request("GET", "/market-data/instruments/rates", params={"instrumentIds": ids_str})

    async def get_candles(self, instrument_id: int, direction: str = "desc",
                          interval: str = "OneDay", count: int = 100) -> dict:
        """Get historical OHLCV candles.
        interval: OneMinute, FiveMinutes, OneHour, OneDay, OneWeek, OneMonth
        direction: asc or desc
        """
        return await self._request(
            "GET",
            f"/market-data/instruments/{instrument_id}/history/candles/{direction}/{interval}/{count}",
        )

    async def check_spread(self, instrument_id: int, median_spread: float = None) -> dict:
        """Check bid-ask spread for an instrument.
        Returns spread info and whether it's safe to trade (spread < 2x median).
        """
        try:
            data = await self.get_rates([instrument_id])
            rates = data if isinstance(data, list) else data.get("rates", [data])
            if not rates:
                return {"ok": True, "spread": 0, "reason": "No rate data"}

            rate = rates[0] if isinstance(rates, list) else rates
            ask = rate.get("askPrice", rate.get("ask", 0))
            bid = rate.get("bidPrice", rate.get("bid", 0))
            mid = (ask + bid) / 2 if (ask + bid) > 0 else 1
            spread = (ask - bid) / mid if mid > 0 else 0

            if median_spread and spread > 2 * median_spread:
                return {
                    "ok": False,
                    "spread": spread,
                    "median": median_spread,
                    "reason": f"Spread {spread:.4%} > 2x median {median_spread:.4%}",
                }
            return {"ok": True, "spread": spread}
        except Exception as e:
            logger.warning(f"Spread check failed for {instrument_id}: {e}")
            return {"ok": True, "spread": 0, "reason": f"Check failed: {e}"}

    # ────────────────────── Portfolio ──────────────────────

    async def get_portfolio(self) -> dict:
        """Get current portfolio with all open positions."""
        return await self._request("GET", "/trading/info/portfolio")

    async def get_positions(self) -> List[dict]:
        """Get all open positions."""
        data = await self.get_portfolio()
        return data.get("clientPortfolio", {}).get("positions", [])

    async def get_pnl(self) -> dict:
        """Get real account PnL and portfolio details."""
        return await self._request("GET", "/trading/info/real/pnl")

    async def get_trade_history(self, min_date: str = "2024-01-01", page: int = 1, page_size: int = 100) -> list:
        """Get closed trade history."""
        return await self._request("GET", "/trading/info/trade/history", params={
            "minDate": min_date,
            "page": page,
            "pageSize": page_size,
        })

    # ────────────────────── Trading ──────────────────────

    async def open_position(self, instrument_id: int, is_buy: bool,
                            amount: float, stop_loss: float = None,
                            take_profit: float = None,
                            leverage: int = 1) -> dict:
        """
        Open a new position via market order.

        Args:
            instrument_id: eToro instrument ID
            is_buy: True for long, False for short
            amount: Dollar amount to invest
            stop_loss: Stop loss rate (price level)
            take_profit: Take profit rate (price level)
            leverage: Leverage multiplier (1 = no leverage)
        """
        payload = {
            "InstrumentID": instrument_id,
            "IsBuy": is_buy,
            "Amount": amount,
            "Leverage": leverage,
            "IsTslEnabled": False,
            "IsNoStopLoss": stop_loss is None,
            "IsNoTakeProfit": take_profit is None,
        }
        if stop_loss is not None:
            payload["StopLossRate"] = stop_loss
        if take_profit is not None:
            payload["TakeProfitRate"] = take_profit

        direction = "BUY" if is_buy else "SELL"
        logger.info(f"Opening {direction} position: {instrument_id}, ${amount}, "
                     f"SL={stop_loss}, TP={take_profit}, leverage={leverage}x")
        return await self._request("POST", "/trading/execution/market-open-orders/by-amount", json=payload)

    async def close_position(self, position_id: int, instrument_id: int,
                              units_to_deduct: float = None) -> dict:
        """Close an existing position (full or partial)."""
        payload = {"InstrumentId": instrument_id}
        if units_to_deduct is not None:
            payload["UnitsToDeduct"] = units_to_deduct
        logger.info(f"Closing position: {position_id}")
        return await self._request("POST", f"/trading/execution/market-close-orders/positions/{position_id}", json=payload)

    # ────────────────────── Social / Watchlist ──────────────────────

    async def get_watchlist(self) -> List[dict]:
        """Get user's watchlist."""
        return await self._request("GET", "/watchlist")

    async def add_to_watchlist(self, instrument_id: int) -> dict:
        """Add instrument to watchlist."""
        return await self._request("POST", "/watchlist", json={"instrumentId": instrument_id})

    # ────────────────────── Lifecycle ──────────────────────

    async def close(self):
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class EtoroWebSocket:
    """WebSocket client for real-time price streaming."""

    def __init__(self, ws_url: str, user_key: str, api_key: str):
        self.ws_url = ws_url
        self.user_key = user_key
        self.api_key = api_key
        self._ws = None
        self._callbacks: Dict[str, List[Callable]] = {}

    def on(self, event: str, callback: Callable):
        """Register a callback for a specific event type."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    async def connect(self, instrument_ids: List[int]):
        """Connect to WebSocket and subscribe to instrument rates."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. pip install websockets")
            return

        auth_params = f"?userKey={self.user_key}&apiKey={self.api_key}"
        url = f"{self.ws_url}{auth_params}"

        async with websockets.connect(url) as ws:
            self._ws = ws
            # Subscribe to instruments
            subscribe_msg = json.dumps({
                "action": "subscribe",
                "channels": [{"name": "rates", "instrumentIds": instrument_ids}]
            })
            await ws.send(subscribe_msg)
            logger.info(f"WebSocket connected, subscribed to {len(instrument_ids)} instruments")

            async for message in ws:
                try:
                    data = json.loads(message)
                    event_type = data.get("type", "unknown")
                    for cb in self._callbacks.get(event_type, []):
                        await cb(data) if asyncio.iscoroutinefunction(cb) else cb(data)
                    for cb in self._callbacks.get("*", []):
                        await cb(data) if asyncio.iscoroutinefunction(cb) else cb(data)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON WebSocket message: {message[:100]}")
                except Exception as e:
                    logger.error(f"WebSocket handler error: {e}")
