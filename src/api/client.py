"""
NinjaTrader API Client

REST API client for NinjaTrader's trading infrastructure.
Supports both the official NinjaTrader API and the Tradovate API (same parent company).
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import AsyncIterator
import httpx
import websockets
from websockets.client import WebSocketClientProtocol

from .models import Quote, Position, Order, Account, Bar, OrderRequest, HistoricalDataRequest


class NinjaTraderClient:
    """
    Async client for NinjaTrader REST API.

    Supports:
    - Authentication
    - Real-time quotes via WebSocket
    - Historical data retrieval
    - Order management
    - Account/Position queries
    """

    # API endpoints - using Tradovate API (NinjaTrader's parent platform)
    BASE_URL = "https://api.tradovate.com/v1"
    WS_URL = "wss://md.tradovate.com/v1/websocket"
    AUTH_URL = "https://api.tradovate.com/v1/auth/accesstokenrequest"

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        demo: bool = True
    ):
        """
        Initialize API client.

        Args:
            username: NinjaTrader/Tradovate username
            password: Account password
            api_key: API key (alternative to user/pass)
            api_secret: API secret
            demo: Use demo environment
        """
        self.username = username
        self.password = password
        self.api_key = api_key
        self.api_secret = api_secret
        self.demo = demo

        self._access_token: str | None = None
        self._token_expiry: datetime | None = None
        self._client: httpx.AsyncClient | None = None
        self._ws: WebSocketClientProtocol | None = None

        if demo:
            self.BASE_URL = "https://demo.tradovate.com/v1"
            self.WS_URL = "wss://md-demo.tradovate.com/v1/websocket"
            self.AUTH_URL = "https://demo.tradovate.com/v1/auth/accesstokenrequest"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self):
        """Establish connection and authenticate."""
        self._client = httpx.AsyncClient(timeout=30.0)
        await self._authenticate()

    async def disconnect(self):
        """Close all connections."""
        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._client:
            await self._client.aclose()
            self._client = None

    async def _authenticate(self):
        """Authenticate and get access token."""
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")

        auth_data = {
            "name": self.username,
            "password": self.password,
            "appId": "NinjaTraderBot",
            "appVersion": "1.0.0"
        }

        if self.api_key:
            auth_data["apiKey"] = self.api_key

        response = await self._client.post(self.AUTH_URL, json=auth_data)
        response.raise_for_status()

        data = response.json()
        self._access_token = data.get("accessToken")
        expires_in = data.get("expirationTime", 3600)
        self._token_expiry = datetime.now() + timedelta(seconds=expires_in)

    async def _ensure_authenticated(self):
        """Ensure valid authentication."""
        if not self._access_token or (
            self._token_expiry and datetime.now() >= self._token_expiry
        ):
            await self._authenticate()

    def _headers(self) -> dict:
        """Get request headers with auth."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }

    # Account Methods

    async def get_accounts(self) -> list[Account]:
        """Get all accounts."""
        await self._ensure_authenticated()

        response = await self._client.get(
            f"{self.BASE_URL}/account/list",
            headers=self._headers()
        )
        response.raise_for_status()

        accounts = []
        for acc in response.json():
            accounts.append(Account(
                account_id=str(acc["id"]),
                name=acc.get("name", ""),
                cash_balance=acc.get("cashBalance", 0),
                realized_pnl=acc.get("realizedPnL", 0),
                unrealized_pnl=acc.get("openPnL", 0),
                margin_used=acc.get("marginUsed", 0)
            ))

        return accounts

    async def get_positions(self, account_id: str | None = None) -> list[Position]:
        """Get open positions."""
        await self._ensure_authenticated()

        url = f"{self.BASE_URL}/position/list"
        if account_id:
            url = f"{self.BASE_URL}/position/ldeps?masterid={account_id}"

        response = await self._client.get(url, headers=self._headers())
        response.raise_for_status()

        positions = []
        for pos in response.json():
            if pos.get("netPos", 0) != 0:
                positions.append(Position(
                    symbol=pos.get("contractId", ""),
                    quantity=pos.get("netPos", 0),
                    average_price=pos.get("netPrice", 0),
                    unrealized_pnl=pos.get("openPnL", 0),
                    realized_pnl=pos.get("realizedPnL", 0)
                ))

        return positions

    # Order Methods

    async def place_order(self, order: OrderRequest, account_id: str) -> Order:
        """Place a new order."""
        await self._ensure_authenticated()

        # Map order type to API format
        action = "Buy" if order.side == "buy" else "Sell"
        order_type_map = {
            "market": "Market",
            "limit": "Limit",
            "stop": "Stop",
            "stop_limit": "StopLimit"
        }

        order_data = {
            "accountId": int(account_id),
            "symbol": order.symbol,
            "action": action,
            "orderQty": order.quantity,
            "orderType": order_type_map[order.order_type],
        }

        if order.price:
            order_data["price"] = order.price
        if order.stop_price:
            order_data["stopPrice"] = order.stop_price

        response = await self._client.post(
            f"{self.BASE_URL}/order/placeorder",
            headers=self._headers(),
            json=order_data
        )
        response.raise_for_status()

        data = response.json()
        return Order(
            order_id=str(data.get("orderId", "")),
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            status="working",
            timestamp=datetime.now()
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        await self._ensure_authenticated()

        response = await self._client.post(
            f"{self.BASE_URL}/order/cancelorder",
            headers=self._headers(),
            json={"orderId": int(order_id)}
        )

        return response.status_code == 200

    async def get_orders(self, account_id: str | None = None) -> list[Order]:
        """Get open orders."""
        await self._ensure_authenticated()

        response = await self._client.get(
            f"{self.BASE_URL}/order/list",
            headers=self._headers()
        )
        response.raise_for_status()

        orders = []
        for ord in response.json():
            orders.append(Order(
                order_id=str(ord["id"]),
                symbol=str(ord.get("contractId", "")),
                side="buy" if ord.get("action") == "Buy" else "sell",
                order_type=ord.get("orderType", "market").lower(),
                quantity=ord.get("orderQty", 0),
                price=ord.get("price"),
                stop_price=ord.get("stopPrice"),
                status=ord.get("ordStatus", "pending").lower(),
                filled_quantity=ord.get("filledQty", 0),
                average_fill_price=ord.get("avgFillPrice"),
                timestamp=datetime.fromisoformat(ord.get("timestamp", datetime.now().isoformat()))
            ))

        return orders

    # Market Data Methods

    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol."""
        await self._ensure_authenticated()

        response = await self._client.get(
            f"{self.BASE_URL}/md/getQuote",
            headers=self._headers(),
            params={"symbol": symbol}
        )
        response.raise_for_status()

        data = response.json()
        return Quote(
            symbol=symbol,
            bid=data.get("bid", 0),
            ask=data.get("ask", 0),
            last=data.get("last", 0),
            bid_size=data.get("bidSize", 0),
            ask_size=data.get("askSize", 0),
            volume=data.get("volume", 0),
            timestamp=datetime.now()
        )

    async def get_historical_bars(
        self,
        request: HistoricalDataRequest
    ) -> list[Bar]:
        """Get historical bar data."""
        await self._ensure_authenticated()

        # Map timeframe to API format
        tf_map = {
            "1min": {"unit": "Min", "value": 1},
            "5min": {"unit": "Min", "value": 5},
            "15min": {"unit": "Min", "value": 15},
            "1hour": {"unit": "Hour", "value": 1},
            "1day": {"unit": "Day", "value": 1}
        }

        tf = tf_map.get(request.timeframe, {"unit": "Min", "value": 1})

        params = {
            "symbol": request.symbol,
            "chartDescription": {
                "underlyingType": "Tick",
                "elementSize": tf["value"],
                "elementSizeUnit": tf["unit"]
            },
            "timeRange": {
                "asFarAsTimestamp": request.start_date.isoformat(),
            }
        }

        if request.end_date:
            params["timeRange"]["closestTimestamp"] = request.end_date.isoformat()

        response = await self._client.post(
            f"{self.BASE_URL}/md/getChart",
            headers=self._headers(),
            json=params
        )
        response.raise_for_status()

        data = response.json()
        bars = []

        for bar_data in data.get("bars", []):
            bars.append(Bar(
                symbol=request.symbol,
                timestamp=datetime.fromisoformat(bar_data["timestamp"]),
                open=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=bar_data.get("volume", 0),
                timeframe=request.timeframe
            ))

        return bars

    # WebSocket Streaming

    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        """
        Stream real-time quotes via WebSocket.

        Args:
            symbols: List of symbols to stream

        Yields:
            Quote objects as they arrive
        """
        await self._ensure_authenticated()

        async with websockets.connect(
            self.WS_URL,
            extra_headers={"Authorization": f"Bearer {self._access_token}"}
        ) as ws:
            self._ws = ws

            # Subscribe to symbols
            for symbol in symbols:
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "symbol": symbol
                }))

            # Stream quotes
            async for message in ws:
                data = json.loads(message)

                if data.get("type") == "quote":
                    yield Quote(
                        symbol=data["symbol"],
                        bid=data.get("bid", 0),
                        ask=data.get("ask", 0),
                        last=data.get("last", 0),
                        bid_size=data.get("bidSize", 0),
                        ask_size=data.get("askSize", 0),
                        volume=data.get("volume", 0),
                        timestamp=datetime.now()
                    )


# Convenience functions

async def get_client(
    username: str,
    password: str,
    demo: bool = True
) -> NinjaTraderClient:
    """Create and connect a client."""
    client = NinjaTraderClient(username=username, password=password, demo=demo)
    await client.connect()
    return client
