"""
API Data Models

Pydantic models for NinjaTrader API responses.
"""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class Quote(BaseModel):
    """Real-time quote data."""
    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: int = 0
    ask_size: int = 0
    volume: int = 0
    timestamp: datetime


class Bar(BaseModel):
    """OHLCV bar data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "1min"


class Position(BaseModel):
    """Open position."""
    symbol: str
    quantity: int
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    market_value: float = 0.0


class Order(BaseModel):
    """Order details."""
    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop", "stop_limit"]
    quantity: int
    price: float | None = None
    stop_price: float | None = None
    status: Literal["pending", "working", "filled", "cancelled", "rejected"]
    filled_quantity: int = 0
    average_fill_price: float | None = None
    timestamp: datetime


class Account(BaseModel):
    """Account information."""
    account_id: str
    name: str
    cash_balance: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    buying_power: float = 0.0
    margin_used: float = 0.0


class OrderRequest(BaseModel):
    """Order submission request."""
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop", "stop_limit"]
    quantity: int
    price: float | None = None
    stop_price: float | None = None
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = "day"


class HistoricalDataRequest(BaseModel):
    """Request for historical data."""
    symbol: str
    timeframe: str = "1min"  # 1min, 5min, 15min, 1hour, 1day
    start_date: datetime
    end_date: datetime | None = None
    include_extended_hours: bool = False
