# NinjaTrader Trading Playbook

## Prime Directive: Maximize Profit Within All Rules

This playbook defines all constraints and rules for the ML trading system operating on NinjaTrader.
The bot must never violate these rules while pursuing maximum profitability.

---

## Table of Contents
1. [Account Configuration](#1-account-configuration)
2. [Trading Hours](#2-trading-hours)
3. [Margin Rules](#3-margin-rules)
4. [Fee Structure](#4-fee-structure)
5. [Order Types & Execution](#5-order-types--execution)
6. [Position Limits](#6-position-limits)
7. [API Constraints](#7-api-constraints)
8. [Risk Management Rules](#8-risk-management-rules)
9. [Contract Specifications](#9-contract-specifications)
10. [Prohibited Actions](#10-prohibited-actions)

---

## 1. Account Configuration

### Account Details
```python
ACCOUNT_CONFIG = {
    "initial_capital": 1000.00,  # USD deposited
    "plan": "free",  # free | monthly | lifetime
    "primary_instrument": "MES",  # Micro E-mini S&P 500
    "broker": "ninjatrader",
}
```

### What You Can Trade With $1,000
| Mode | MES Contracts | ES Contracts |
|------|---------------|--------------|
| **Intraday** | 20 contracts ($50 margin each) | 2 contracts ($500 margin each) |
| **Overnight** | 0 contracts ($1,166 margin each) | 0 contracts ($11,660 margin each) |

**RULE: With $1,000, you MUST day-trade only. Close all positions before 4:45 PM ET.**

---

## 2. Trading Hours

### Session Times (All times in ET)
```python
TRADING_HOURS = {
    # Session boundaries
    "week_start": "Sunday 6:00 PM ET",
    "week_end": "Friday 5:00 PM ET",

    # Daily cycle
    "session_open": "6:00 PM ET (prior day)",
    "session_close": "5:00 PM ET",

    # Critical times
    "intraday_margin_ends": "4:45 PM ET",  # MUST close positions by this time
    "maintenance_halt_start": "4:15 PM ET",
    "maintenance_halt_end": "4:30 PM ET",
    "new_session_start": "6:00 PM ET",

    # Regular trading hours (highest liquidity)
    "rth_open": "9:30 AM ET",
    "rth_close": "4:00 PM ET",
}
```

### Optimal Trading Windows
```python
OPTIMAL_WINDOWS = {
    "prime": ("9:30 AM ET", "11:30 AM ET"),  # Best liquidity, tightest spreads
    "secondary": ("3:45 PM ET", "4:15 PM ET"),  # Closing activity
    "avoid": [
        ("4:15 PM ET", "4:30 PM ET"),  # Maintenance halt
        ("12:00 PM ET", "2:00 PM ET"),  # Lunch lull, wide spreads
        ("6:00 PM ET", "9:30 AM ET"),  # Overnight - low volume
    ]
}
```

### Critical Deadlines
| Time (ET) | Action Required |
|-----------|-----------------|
| **4:45 PM** | CLOSE ALL INTRADAY POSITIONS (or have overnight margin) |
| **4:15 PM** | No order execution until 4:30 PM |
| **15 min before news** | Margin increases 4x - reduce position size |

### Holiday Schedule 2025
```python
MARKET_CLOSED = [
    "2025-01-01",  # New Year's Day
    "2025-12-25",  # Christmas
]

EARLY_CLOSE_1PM_ET = [
    "2025-11-27",  # Thanksgiving
    "2025-12-24",  # Christmas Eve
    # Close positions by 12:45 PM ET on these days
]
```

---

## 3. Margin Rules

### Intraday Margin (Day Trading)
```python
INTRADAY_MARGIN = {
    "MES": 50,      # Micro E-mini S&P 500
    "MNQ": 100,     # Micro E-mini Nasdaq-100
    "MYM": 50,      # Micro E-mini Dow
    "M2K": 50,      # Micro E-mini Russell 2000
    "ES": 500,      # E-mini S&P 500
    "NQ": 1000,     # E-mini Nasdaq-100
}
```

### Overnight/Exchange Margin (Holding Positions)
```python
OVERNIGHT_MARGIN = {
    "MES": 1166,    # Initial: $1,166, Maintenance: $1,060
    "MNQ": 1738,    # Initial: $1,738, Maintenance: $1,580
    "ES": 11660,    # Initial: $11,660, Maintenance: $10,600
    "NQ": 17380,    # Initial: $17,380, Maintenance: $15,800
}
```

### Margin Multiplier Events
```python
MARGIN_4X_EVENTS = [
    "Employment Report",      # First Friday of month, 8:30 AM ET
    "FOMC Announcement",      # Variable dates
    "CPI Release",            # Monthly
    "PPI Release",            # Monthly
    "GDP Release",            # Quarterly
]

# Rule: 15 minutes BEFORE these events, margin becomes 4x normal
# Duration: Until ~5 minutes after event ends
```

### Margin Violation Consequences
```python
MARGIN_VIOLATIONS = {
    "margin_call_fee": 50,          # USD
    "liquidation_fee_first": 25,    # USD
    "liquidation_fee_repeat": 50,   # USD
    "response_time": "24 hours",    # To wire funds
    "auto_liquidation": "4:45 PM ET",  # If insufficient margin
}
```

---

## 4. Fee Structure

### Commission per Contract (Round-Turn = 2x)
```python
# Free Plan (current)
COMMISSIONS_FREE_PLAN = {
    "MES": {"commission": 0.39, "exchange": 0.37, "nfa": 0.02, "clearing": 0.19},  # Total: $0.97/side
    "ES": {"commission": 1.29, "exchange": 1.40, "nfa": 0.02, "clearing": 0.19},   # Total: $2.90/side
}

# All-in costs per ROUND TURN (entry + exit)
ROUND_TURN_COST = {
    "MES": 1.94,  # $0.97 x 2
    "ES": 5.80,   # $2.90 x 2
}
```

### Other Fees
```python
ACCOUNT_FEES = {
    "inactivity_fee": 35,           # USD/month if no trades for 30 days
    "withdrawal_ach": 0,            # Free
    "withdrawal_wire_domestic": 30, # USD
    "withdrawal_wire_intl": 30,     # USD equivalent
    "data_feed_level1": 12,         # USD/month (CME bundle)
    "data_feed_level2": 41,         # USD/month (CME bundle)
}
```

### Minimum Profit Threshold
```python
# To be profitable on a trade, must exceed round-turn costs
MIN_PROFIT_TICKS = {
    "MES": 2,  # $1.94 cost / $1.25 per tick = 1.55 ticks → round up to 2
    "ES": 2,   # $5.80 cost / $12.50 per tick = 0.46 ticks → round up to 1, but 2 for safety
}
```

---

## 5. Order Types & Execution

### Supported Order Types
```python
ORDER_TYPES = {
    "market": True,       # Immediate execution at best price
    "limit": True,        # Exact price, may not fill
    "stop": True,         # Becomes market when triggered
    "stop_limit": True,   # Becomes limit when triggered
    "mit": True,          # Market If Touched
    "oco": True,          # One Cancels Other
    "bracket": True,      # Entry + Stop + Target (via OCO)
}

# NOT SUPPORTED
UNSUPPORTED_ORDERS = ["fok", "ioc"]  # Fill or Kill, Immediate or Cancel
```

### Order Placement Rules
```python
ORDER_RULES = {
    # Stop order placement
    "buy_stop": "must be ABOVE current ask",
    "sell_stop": "must be BELOW current bid",

    # MIT order placement
    "buy_mit": "must be BELOW current bid",
    "sell_mit": "must be ABOVE current ask",

    # Time in force
    "default_tif": "DAY",
    "gtc_max_days": 60,
}
```

### Order Rejection Prevention
```python
def validate_stop_order(side: str, stop_price: float, current_bid: float, current_ask: float) -> bool:
    """Validate stop order price before submission."""
    if side == "buy":
        return stop_price > current_ask
    else:  # sell
        return stop_price < current_bid
```

---

## 6. Position Limits

### Dynamic Position Limits (Scale with Account Balance)
```python
# Margin constants (set by CME/NinjaTrader)
MARGIN_REQUIREMENTS = {
    "MES_intraday": 50,      # $50 per contract during market hours
    "MES_overnight": 1166,   # $1,166 per contract to hold overnight
    "ES_intraday": 500,      # $500 per contract
    "ES_overnight": 11660,   # $11,660 per contract
}

def calculate_position_limits(account_balance: float) -> dict:
    """Calculate dynamic position limits based on current account balance."""
    return {
        # Intraday limits
        "max_mes_intraday": int(account_balance / MARGIN_REQUIREMENTS["MES_intraday"]),
        "max_es_intraday": int(account_balance / MARGIN_REQUIREMENTS["ES_intraday"]),

        # Overnight limits
        "max_mes_overnight": int(account_balance / MARGIN_REQUIREMENTS["MES_overnight"]),
        "max_es_overnight": int(account_balance / MARGIN_REQUIREMENTS["ES_overnight"]),

        # Risk limits (percentage-based, scale with balance)
        "max_daily_loss_usd": account_balance * 0.10,    # 10% daily stop
        "max_drawdown_usd": account_balance * 0.20,      # 20% max drawdown
        "risk_per_trade_usd": account_balance * 0.02,    # 2% per trade

        # Conservative position sizing
        "max_contracts_per_trade": max(1, min(int(account_balance / 500), 10)),
        "max_open_positions": 1,
        "max_daily_trades": 50,
    }

# Example scaling at different account sizes:
# $1,000:  20 MES intraday, 0 overnight, $100 daily loss limit
# $2,500:  50 MES intraday, 2 overnight, $250 daily loss limit
# $5,000:  100 MES intraday, 4 overnight, $500 daily loss limit
# $10,000: 200 MES intraday, 8 overnight, $1,000 daily loss limit
# $25,000: 500 MES intraday, 21 overnight, $2,500 daily loss limit
```

### Overnight Holding Thresholds
```python
def can_hold_overnight(account_balance: float, contracts: int, symbol: str = "MES") -> bool:
    """Check if account has sufficient margin to hold position overnight."""
    required_margin = contracts * MARGIN_REQUIREMENTS[f"{symbol}_overnight"]
    return account_balance >= required_margin

# Examples:
# $1,000 account: can_hold_overnight(1000, 1, "MES") = False (need $1,166)
# $2,500 account: can_hold_overnight(2500, 2, "MES") = True (have $2,500, need $2,332)
# $5,000 account: can_hold_overnight(5000, 4, "MES") = True (have $5,000, need $4,664)
```

### Risk Per Trade
```python
RISK_MANAGEMENT = {
    "risk_per_trade_pct": 2.0,          # 2% of account per trade
    "max_position_size_by_atr": True,   # Size based on ATR
    "default_stop_atr_mult": 2.0,       # Stop at 2x ATR
    "default_target_atr_mult": 3.0,     # Target at 3x ATR
}

def calculate_risk_per_trade(account_balance: float) -> float:
    """Calculate max risk per trade in USD."""
    return account_balance * (RISK_MANAGEMENT["risk_per_trade_pct"] / 100)
```

---

## 7. API Constraints

### Rate Limits
```python
API_LIMITS = {
    "requests_per_hour": 5000,          # Rolling 60-minute window
    "requests_per_minute": 80,          # Approximate safe limit
    "websocket_connections": 1,         # Max concurrent connections
    "max_contracts_per_order": 100,     # Broker limit
}
```

### API Access Requirements
```python
API_REQUIREMENTS = {
    "min_account_balance": 1000,        # USD
    "api_subscription_cost": 25,        # USD/month (if needed)
    "cme_data_license": 390,            # USD/month (if real-time data via API)
}
```

### Rate Limit Handling
```python
def handle_rate_limit():
    """If rate limit exceeded, wait before retrying."""
    # HTTP 429 = Too Many Requests
    # Wait 60 seconds minimum
    # P-ticket status = wait 1+ hour
    pass
```

---

## 8. Risk Management Rules

### Daily Loss Limits
```python
DAILY_LIMITS = {
    "max_loss_usd": 100,                # Stop trading after $100 loss
    "max_loss_pct": 10,                 # 10% of account
    "max_consecutive_losses": 3,        # Stop after 3 losses in a row
    "cooldown_after_max_loss": 3600,    # Wait 1 hour after hitting limit
}
```

### Position Sizing Algorithm
```python
def calculate_position_size(
    account_balance: float,
    risk_per_trade_pct: float,
    stop_distance_ticks: int,
    tick_value: float
) -> int:
    """Calculate optimal position size based on risk."""
    risk_amount = account_balance * (risk_per_trade_pct / 100)
    risk_per_contract = stop_distance_ticks * tick_value
    position_size = int(risk_amount / risk_per_contract)
    return max(1, min(position_size, POSITION_LIMITS["max_contracts_per_trade"]))
```

### Stop Loss Rules
```python
STOP_LOSS_RULES = {
    "always_use_stops": True,           # NEVER trade without a stop
    "stop_type": "stop_market",         # Use stop-market, not stop-limit
    "min_stop_distance_ticks": 4,       # Minimum 4 ticks ($5 for MES)
    "max_stop_distance_ticks": 40,      # Maximum 40 ticks ($50 for MES)
    "trailing_stop_enabled": False,     # Start with fixed stops
}
```

---

## 9. Contract Specifications

### MES (Micro E-mini S&P 500) - Primary Instrument
```python
MES_SPECS = {
    "symbol": "MES",
    "exchange": "CME",
    "tick_size": 0.25,                  # Minimum price movement
    "tick_value": 1.25,                 # USD per tick
    "point_value": 5.00,                # USD per point (4 ticks)
    "contract_months": ["H", "M", "U", "Z"],  # Mar, Jun, Sep, Dec
    "trading_hours": "Nearly 24/5",
    "settlement": "Cash settled",
}

# 2025 Contract Schedule
MES_CONTRACTS_2025 = {
    "MESH5": {"expiry": "2025-03-21", "roll_date": "2025-03-10"},
    "MESM5": {"expiry": "2025-06-20", "roll_date": "2025-06-09"},
    "MESU5": {"expiry": "2025-09-19", "roll_date": "2025-09-08"},
    "MESZ5": {"expiry": "2025-12-19", "roll_date": "2025-12-08"},
}
```

### Contract Rollover Rules
```python
ROLLOVER_RULES = {
    "roll_before_expiry_days": 5,       # Roll 5 days before expiration
    "roll_on_volume_shift": True,       # Roll when new contract has more volume
    "never_hold_past_expiry": True,     # CRITICAL: Will be liquidated with fees
}
```

---

## 10. Prohibited Actions

### Never Do These
```python
PROHIBITED_ACTIONS = [
    "hold_overnight_without_margin",     # Will be liquidated
    "trade_during_maintenance_halt",     # 4:15-4:30 PM ET
    "exceed_daily_loss_limit",           # Stop trading at -$100
    "trade_without_stop_loss",           # Always use stops
    "trade_during_4x_margin_events",     # Unless position size reduced 4x
    "hold_past_contract_expiry",         # Will be liquidated with fees
    "exceed_api_rate_limits",            # 5000/hour
    "trade_more_than_intraday_margin_allows",  # Max 20 MES with $1000
]
```

### Automatic Safeguards
```python
SAFEGUARDS = {
    "auto_close_at_445pm": True,        # Close all positions at 4:45 PM ET
    "auto_stop_at_daily_limit": True,   # Stop trading at max daily loss
    "reduce_size_before_news": True,    # Cut position size before major news
    "check_margin_before_entry": True,  # Verify margin available
    "validate_stop_price": True,        # Ensure stops are valid
}
```

---

## Backtesting Configuration

### Use These Settings for Realistic Backtests
```python
def create_backtest_config(initial_capital: float = 1000) -> dict:
    """Create backtest config with dynamic limits based on capital."""
    limits = calculate_position_limits(initial_capital)

    return {
        # Costs (fixed by broker)
        "commission_per_contract": 0.97,    # One side
        "slippage_ticks": 1,                # Assume 1 tick slippage

        # Position sizing (scales with capital)
        "initial_capital": initial_capital,
        "risk_per_trade_pct": 2.0,
        "max_position_size": limits["max_contracts_per_trade"],

        # Time filters
        "trade_only_rth": True,             # Only trade 9:30 AM - 4:00 PM ET
        "avoid_first_5_min": True,          # Skip opening volatility
        "close_before_445pm": True,         # Exit all by 4:45 PM ET

        # MES specifications (fixed)
        "tick_size": 0.25,
        "tick_value": 1.25,

        # Risk limits (scales with capital)
        "max_daily_loss": limits["max_daily_loss_usd"],
        "max_drawdown": limits["max_drawdown_usd"],
        "max_mes_overnight": limits["max_mes_overnight"],
        "always_use_stops": True,
    }

# Example configs at different account sizes:
# create_backtest_config(1000)   -> max_daily_loss=$100, max_drawdown=$200, overnight=0
# create_backtest_config(5000)   -> max_daily_loss=$500, max_drawdown=$1000, overnight=4
# create_backtest_config(25000)  -> max_daily_loss=$2500, max_drawdown=$5000, overnight=21
```

### Overnight Strategy Unlocks
```python
def get_strategy_options(account_balance: float) -> dict:
    """Determine which strategies are available based on account size."""
    can_overnight = account_balance >= MARGIN_REQUIREMENTS["MES_overnight"]

    return {
        "day_trading": True,                    # Always available
        "scalping": True,                       # Always available
        "swing_trading": can_overnight,         # Requires overnight margin
        "position_trading": can_overnight,      # Requires overnight margin
        "overnight_gap_plays": can_overnight,   # Requires overnight margin

        # Account thresholds for advanced strategies
        "min_for_overnight": 1166,              # Hold 1 MES overnight
        "min_for_swing": 2332,                  # Hold 2+ MES for swing trades
        "min_for_diversification": 5000,        # Trade multiple contracts
    }

# At $1,000: Day trading and scalping only
# At $2,500: Can hold 2 contracts overnight (swing trading unlocked)
# At $5,000: Can diversify with 4 contracts overnight
```

---

## ML Strategy Optimization Goals

### Primary Objective
```
MAXIMIZE: Total Net Profit
SUBJECT TO: All constraints in this playbook
```

### Secondary Objectives (in order)
1. Maximize Sharpe Ratio (risk-adjusted returns)
2. Minimize Maximum Drawdown
3. Maximize Win Rate (target > 50%)
4. Minimize Average Loss / Maximize Average Win

### Fitness Function for ML
```python
def calculate_fitness(backtest_result, account_balance: float) -> float:
    """
    Fitness function for ML strategy optimization.
    Uses dynamic limits based on account balance.
    Higher is better.
    """
    limits = calculate_position_limits(account_balance)

    # Primary: Net profit (as percentage of account for fair comparison)
    profit_pct = (backtest_result.total_return / account_balance) * 100
    profit_score = profit_pct * 10  # Scale for fitness

    # Penalty for violations (dynamic based on account size)
    violation_penalty = 0

    # Drawdown penalty (scales with account)
    max_allowed_drawdown = limits["max_drawdown_usd"]
    if backtest_result.max_drawdown > max_allowed_drawdown:
        excess = (backtest_result.max_drawdown - max_allowed_drawdown) / max_allowed_drawdown
        violation_penalty -= 1000 * (1 + excess)  # Progressive penalty

    # Overnight without margin = disqualify
    if backtest_result.held_overnight_without_margin:
        violation_penalty -= 10000

    # Daily loss limit violation
    if backtest_result.max_daily_loss > limits["max_daily_loss_usd"]:
        violation_penalty -= 500

    # Risk adjustment
    sharpe_bonus = backtest_result.sharpe_ratio * 100

    # Win rate bonus
    winrate_bonus = (backtest_result.win_rate - 0.5) * 200

    # Profit factor bonus (reward consistent profitability)
    if backtest_result.profit_factor > 1.5:
        pf_bonus = (backtest_result.profit_factor - 1) * 50
    else:
        pf_bonus = 0

    return profit_score + sharpe_bonus + winrate_bonus + pf_bonus + violation_penalty


def calculate_fitness_growth_aware(backtest_result, starting_balance: float) -> float:
    """
    Enhanced fitness function that rewards strategies that scale well.
    As the account grows during backtest, limits dynamically adjust.
    """
    # Track how well strategy respects growing limits
    growth_bonus = 0

    # If account grew, check that strategy adapted to new limits
    ending_balance = starting_balance + backtest_result.total_return
    if ending_balance > starting_balance:
        growth_rate = (ending_balance / starting_balance) - 1
        growth_bonus = growth_rate * 500  # Reward sustainable growth

        # Extra bonus if strategy used overnight holding after crossing threshold
        if starting_balance < 1166 and ending_balance >= 1166:
            if backtest_result.used_overnight_after_threshold:
                growth_bonus += 100  # Bonus for adapting to new capabilities

    base_fitness = calculate_fitness(backtest_result, starting_balance)
    return base_fitness + growth_bonus
```

---

## Sources

- [NinjaTrader Margin Policy](https://ninjatrader.com/pricing/margins-position-management/)
- [NinjaTrader Account Fees](https://ninjatrader.com/pricing/account-fees/)
- [NinjaTrader Trading Hours](https://support.ninjatrader.com/s/article/Futures-Trading-Hours)
- [CME Group Trading Hours](https://www.cmegroup.com/trading-hours.html)
- [CME Equity Index Roll Dates](https://www.cmegroup.com/trading/equity-index/rolldates.html)
- [Tradovate API Documentation](https://api.tradovate.com/)
- [NinjaTrader Developer Portal](https://developer.ninjatrader.com/products/api)

---

*Last Updated: December 2025*
*Version: 1.0*
