"""
Trading Bot Configuration

All settings for live trading with $1,000 account.
"""

from dataclasses import dataclass
from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TradingConfig:
    """Configuration for live trading."""

    # Account settings
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.02  # 2% risk per trade = $20
    max_daily_loss: float = 0.05  # 5% max daily loss = $50
    max_contracts: int = 5  # Cap position size

    # Symbol settings (MES Micro E-mini S&P 500)
    symbol: str = "MESZ4"  # Current front-month contract
    contract_value: float = 5.0  # $5 per point
    tick_size: float = 0.25
    tick_value: float = 1.25
    intraday_margin: float = 50.0  # $50 per contract intraday

    # Commission and slippage
    commission_per_contract: float = 2.50  # Round-trip
    slippage_ticks: int = 1

    # Strategy parameters (Williams Fractals - validated)
    timeframe: str = "5min"
    fractal_period: int = 3
    ma_fast: int = 20
    ma_medium: int = 50
    ma_slow: int = 100
    atr_period: int = 14
    atr_multiplier: float = 1.5
    target_rr: float = 1.5

    # Trading hours (Central Time)
    # CME Globex: Sunday 5pm - Friday 4pm CT
    trading_start_hour: int = 8   # 8 AM CT
    trading_end_hour: int = 15    # 3 PM CT (flatten by 3:30)
    flatten_hour: int = 15
    flatten_minute: int = 30

    # Session windows (optimal liquidity)
    # Morning session: 8:30 AM - 11:30 AM CT
    # Afternoon session: 1:00 PM - 3:00 PM CT
    morning_start: int = 8
    morning_end: int = 11
    afternoon_start: int = 13
    afternoon_end: int = 15

    # API settings
    demo_mode: bool = True  # Start in demo mode

    @property
    def max_risk_amount(self) -> float:
        """Maximum dollar risk per trade."""
        return self.initial_capital * self.risk_per_trade

    @property
    def daily_loss_limit(self) -> float:
        """Daily loss limit in dollars."""
        return self.initial_capital * self.max_daily_loss


@dataclass
class APIConfig:
    """NinjaTrader/Tradovate API configuration."""

    username: str = os.getenv("TRADOVATE_USERNAME", "")
    password: str = os.getenv("TRADOVATE_PASSWORD", "")
    api_key: str = os.getenv("TRADOVATE_API_KEY", "")
    api_secret: str = os.getenv("TRADOVATE_API_SECRET", "")
    demo: bool = os.getenv("TRADOVATE_DEMO", "true").lower() == "true"

    # Endpoints
    base_url: str = "https://demo.tradovate.com/v1" if demo else "https://api.tradovate.com/v1"
    ws_url: str = "wss://md-demo.tradovate.com/v1/websocket" if demo else "wss://md.tradovate.com/v1/websocket"


# Default instances
TRADING_CONFIG = TradingConfig()
API_CONFIG = APIConfig()


def get_position_size(
    capital: float,
    entry: float,
    stop: float,
    risk_pct: float = 0.02,
    contract_value: float = 5.0,
    max_contracts: int = 5
) -> int:
    """
    Calculate position size based on risk.

    For $1,000 account with 2% risk:
    - Risk amount = $20
    - With 5-point stop (typical), risk per contract = $25
    - Position size = 1 contract

    Args:
        capital: Current account balance
        entry: Entry price
        stop: Stop loss price
        risk_pct: Fraction of capital to risk
        contract_value: Dollar value per point
        max_contracts: Maximum allowed contracts

    Returns:
        Number of contracts to trade
    """
    if entry == 0 or stop == 0:
        return 1

    risk_amount = capital * risk_pct
    risk_per_contract = abs(entry - stop) * contract_value

    if risk_per_contract == 0:
        return 1

    size = int(risk_amount / risk_per_contract)
    return max(1, min(size, max_contracts))
