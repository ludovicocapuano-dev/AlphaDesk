"""
AlphaDesk — eToro API Client
Async HTTP + WebSocket client for eToro's public API.
"""

import asyncio
import json
import logging
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
                headers=self._auth_headers(),
                timeout=self.timeout,
            )
        return self._http_client

    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.user_key}",
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/json",
            "X-Environment": self.environment,
        }

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an authenticated API request with retries."""
        client = await self._get_client()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await client.request(method, path, **kwargs)
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
        data = await self._request("GET", "/metadata/instruments")
        return data.get("instruments", data) if isinstance(data, dict) else data

    async def get_instrument_rate(self, instrument_id: int) -> dict:
        """Get current rate for a specific instrument."""
        return await self._request("GET", f"/instruments/{instrument_id}/rate")

    async def get_rates(self, instrument_ids: List[int]) -> List[dict]:
        """Get rates for multiple instruments."""
        ids_str = ",".join(str(i) for i in instrument_ids)
        return await self._request("GET", f"/instruments/rates", params={"instrumentIds": ids_str})

    async def get_candles(self, instrument_id: int, period: str = "OneDay",
                          count: int = 100) -> List[dict]:
        """Get historical OHLCV candles.
        period: OneMinute, FiveMinutes, OneHour, OneDay, OneWeek, OneMonth
        """
        return await self._request("GET", f"/instruments/{instrument_id}/candles", params={
            "period": period,
            "count": count,
        })

    # ────────────────────── Portfolio ──────────────────────

    async def get_portfolio(self) -> dict:
        """Get current portfolio with all open positions."""
        return await self._request("GET", "/portfolio")

    async def get_positions(self) -> List[dict]:
        """Get all open positions."""
        data = await self.get_portfolio()
        return data.get("positions", [])

    async def get_account_balance(self) -> dict:
        """Get account balance, equity, and available cash."""
        return await self._request("GET", "/account/balance")

    async def get_trade_history(self, start_date: str = None, end_date: str = None) -> List[dict]:
        """Get closed trade history."""
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
        return await self._request("GET", "/trades/history", params=params)

    # ────────────────────── Trading ──────────────────────

    async def open_position(self, instrument_id: int, direction: str,
                            amount: float, stop_loss: float = None,
                            take_profit: float = None,
                            leverage: int = 1) -> dict:
        """
        Open a new position.

        Args:
            instrument_id: eToro instrument ID
            direction: "Buy" or "Sell"
            amount: Dollar amount to invest
            stop_loss: Stop loss rate (price level)
            take_profit: Take profit rate (price level)
            leverage: Leverage multiplier (1 = no leverage)
        """
        payload = {
            "instrumentId": instrument_id,
            "direction": direction,
            "amount": amount,
            "leverage": leverage,
        }
        if stop_loss is not None:
            payload["stopLossRate"] = stop_loss
        if take_profit is not None:
            payload["takeProfitRate"] = take_profit

        logger.info(f"Opening {direction} position: {instrument_id}, ${amount}, "
                     f"SL={stop_loss}, TP={take_profit}, leverage={leverage}x")
        return await self._request("POST", "/trades", json=payload)

    async def close_position(self, position_id: int) -> dict:
        """Close an existing position."""
        logger.info(f"Closing position: {position_id}")
        return await self._request("DELETE", f"/trades/{position_id}")

    async def update_position(self, position_id: int,
                               stop_loss: float = None,
                               take_profit: float = None) -> dict:
        """Update stop loss or take profit on an existing position."""
        payload = {}
        if stop_loss is not None:
            payload["stopLossRate"] = stop_loss
        if take_profit is not None:
            payload["takeProfitRate"] = take_profit

        logger.info(f"Updating position {position_id}: SL={stop_loss}, TP={take_profit}")
        return await self._request("PATCH", f"/trades/{position_id}", json=payload)

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
