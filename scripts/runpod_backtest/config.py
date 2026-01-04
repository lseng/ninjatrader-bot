"""
RunPod Distributed Backtesting Configuration

Controls all parameters for the comprehensive strategy search.
"""

# Data Configuration
DATA_PATH = "/workspace/data/MES_1s_2years.parquet"  # Path on RunPod

# Timeframes to test
TIMEFRAMES = [1, 5]  # 1-minute and 5-minute

# Contract specifications (MES)
CONTRACT_VALUE = 5.0  # $5 per point
COMMISSION_PER_TRADE = 2.50  # Round-trip commission
SLIPPAGE_TICKS = 1  # 1 tick slippage = $1.25

# Trading session (for EOD flatten)
TRADING_START_HOUR = 8  # 8am CT
TRADING_END_HOUR = 15  # 3pm CT (flatten by 3pm, market closes 4pm)

# Initial capital
INITIAL_CAPITAL = 1000

# Risk parameters
MAX_POSITION_SIZE = 5  # Max contracts
DAILY_LOSS_LIMIT = 200  # Stop trading if down $200

# Strategy parameters to sweep
STRATEGY_PARAMS = {
    # Williams Fractals (Strategy 5) - specifically mentioned for 1m scalping
    "fractals": {
        "period": [2, 3],
        "ma_fast": [10, 20],
        "ma_medium": [30, 50],
        "ma_slow": [80, 100],
    },

    # MACD + 200 MA (Strategy 6) - 86% win rate claimed
    "macd": {
        "fast": [8, 12],
        "slow": [21, 26],
        "signal": [7, 9],
        "ma_period": [100, 200],
    },

    # Triple SuperTrend (Strategy 8)
    "triple_supertrend": {
        "atr1": [10, 12],
        "mult1": [2.5, 3.0],
        "atr2": [8, 10],
        "mult2": [1.0, 1.5],
        "atr3": [9, 11],
        "mult3": [1.5, 2.0],
    },

    # Liquidity Sweep (Strategy 2)
    "liquidity_sweep": {
        "lookback": [10, 20, 30],
        "confirmation_bars": [1, 2, 3],
    },

    # Supply/Demand Zones (Strategy 4)
    "supply_demand": {
        "impulse_threshold": [0.002, 0.003, 0.005],
        "zone_buffer": [0.001, 0.002],
        "min_rr": [2.0, 2.5, 3.0],
    },

    # ATR-based stops (Strategy 11)
    "atr_stop": {
        "period": [10, 14, 20],
        "multiplier": [1.5, 2.0, 2.5, 3.0],
    },

    # Heikin Ashi (Strategy 7)
    "heikin_ashi": {
        "confirmation_bars": [1, 2, 3],
        "doji_threshold": [0.2, 0.3],
    },

    # Momentum candles (Strategy 1)
    "momentum": {
        "lookback": [3, 5],
        "body_multiplier": [1.5, 2.0],
    },
}

# Strategy combinations to test
STRATEGY_COMBINATIONS = [
    # Single strategies
    ["fractals"],
    ["macd"],
    ["triple_supertrend"],
    ["liquidity_sweep"],
    ["supply_demand"],

    # High-priority combinations (based on video insights)
    ["fractals", "macd"],  # Fractals entry + MACD filter
    ["liquidity_sweep", "supply_demand"],  # Smart money concepts
    ["triple_supertrend", "macd"],  # Trend confirmation
    ["fractals", "liquidity_sweep"],  # Scalping + liquidity
    ["momentum", "supply_demand"],  # Breakout + zones

    # Full combinations
    ["fractals", "macd", "atr_stop"],
    ["liquidity_sweep", "supply_demand", "heikin_ashi"],
    ["triple_supertrend", "fractals", "atr_stop"],
]

# Trading modes
TRADING_MODES = ["long_only", "short_only", "both"]

# Exit strategies
EXIT_STRATEGIES = [
    {"type": "fixed_rr", "target_rr": 1.5},
    {"type": "fixed_rr", "target_rr": 2.0},
    {"type": "trailing_atr", "atr_mult": 2.0},
    {"type": "supertrend_exit"},
    {"type": "time_based", "max_bars": 30},
]

# Label generation
LABEL_PARAMS = [
    {"lookahead": 3, "threshold": 0.001},
    {"lookahead": 5, "threshold": 0.002},
    {"lookahead": 10, "threshold": 0.003},
]

# Parallel execution
NUM_WORKERS_PER_CPU = 4  # Workers per CPU core
