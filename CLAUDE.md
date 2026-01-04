# NinjaTrader Trading Bot - Claude Code Instructions

## Project Overview

This is an ML-based trading bot for NinjaTrader/Tradovate with backtesting capabilities AND live trading assistance using Smart Money Concepts (SMC) methodology.

## Directory Structure

```
ninjatrader-bot/
├── src/
│   ├── api/          # NinjaTrader/Tradovate API client
│   ├── backtesting/  # Backtesting engine and metrics
│   ├── data/         # Data loading and processing
│   ├── ml/           # Machine learning models
│   └── strategies/   # Trading strategies
├── scripts/          # CLI scripts
├── tests/            # Test suite
├── data/             # Historical data storage
├── models/           # Saved ML models
├── output/           # Backtest results
├── knowledge/        # Trading knowledge base (SMC strategies)
│   ├── strategies/   # Trading strategies (SMC reversals, etc.)
│   ├── setups/       # Setup checklists for live trading
│   └── reference/    # Key concepts and quick reference
└── agents/           # Agent configurations for delegation
```

---

## LIVE TRADING MODE

When the user requests live trading assistance, follow these protocols:

### 1. Spawn Live Trading Agent
For live trading sessions, delegate to a background agent to save context:

```
Use Task tool with subagent_type="general-purpose" and prompt:
"You are a live trading assistant. Read the knowledge base at:
- /Users/leoneng/Downloads/ninjatrader-bot/knowledge/strategies/smc_reversals.md
- /Users/leoneng/Downloads/ninjatrader-bot/knowledge/setups/live_trading_checklist.md
- /Users/leoneng/Downloads/ninjatrader-bot/knowledge/reference/key_concepts.md

Monitor price action from /tmp/fractal_bot.log
Provide trading signals based on SMC methodology.
[Include specific user preferences like aggressive/conservative style]"
```

### 2. Price Data Source
- **Primary**: `/tmp/fractal_bot.log`
- Contains: Real-time price, bar OHLC, MA20/MA50/MA200, ATR
- Format: `HH:MM:SS | INFO | SYSTEM | BAR | O:xxxx H:xxxx L:xxxx C:xxxx | MA:xxx/xxx/xxx | ATR:x.xx`

### 3. Contract Specifications
- **ES (E-mini S&P 500)**: $50/point, $12.50/tick
- **MES (Micro E-mini)**: $5/point, $1.25/tick

### 4. Trading Strategy: SMC Reversals
When analyzing setups, apply this framework:
1. **Liquidity Raid** - Price sweeps highs/lows (stop hunts)
2. **Displacement** - Strong candles confirm shift (NOT entry)
3. **Break of Structure (BOS)** - Candle CLOSES through swing
4. **Retracement** - Wait for pullback to FVG + Order Block
5. **Target** - Opposing liquidity pool

See: `knowledge/strategies/smc_reversals.md` for full details.

### 5. Signal Output Format
When providing trade signals:
```
=== SIGNAL: [LONG/SHORT/WAIT] ===
Price: [current]
Entry: [level]
Stop: [level] ([X] pts = $[Y])
Target: [level] ([X] pts = $[Y])
R:R: [ratio]

Reasoning:
- [SMC criteria met/not met]

Checklist:
[x] Liquidity raid
[x] Displacement
[x] BOS confirmed
[ ] Retracement to FVG (waiting)
```

### 6. Position Management
When user has an open position:
- Monitor P&L continuously
- Alert at key levels (stop approach, target approach)
- Suggest: Hold / Take profit / Move stop to BE

---

## Knowledge Base

### Trading Strategies (`knowledge/strategies/`)
- `smc_reversals.md` - Smart Money Concepts reversal framework
- Add new strategies here as markdown files

### Setup Checklists (`knowledge/setups/`)
- `live_trading_checklist.md` - Pre-trade, entry, and management checklists

### Reference (`knowledge/reference/`)
- `key_concepts.md` - Liquidity, structure, FVG, OB definitions
- Contract specs, session times, risk management

### Adding New Knowledge
When user provides new trading concepts/strategies:
1. Create markdown file in appropriate `knowledge/` subdirectory
2. Structure with clear headings and checklists
3. Include quick-reference summaries

---

## Quick Start (Backtesting)

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Generate sample data
python scripts/generate_sample_data.py --start 2024-01-01 --end 2024-12-31

# 3. Run a backtest
python scripts/run_backtest.py --data data/historical/MES_1min_2024-01-01_2024-12-31.csv --strategy sma
```

## Key Components

### Data Loading
- `NinjaTraderDataLoader`: Parses NT8 exported semicolon-separated files
- `DataProcessor`: Adds technical indicators and creates ML features

### Backtesting
- `BacktestEngine`: Event-driven backtest with commission/slippage
- `PerformanceMetrics`: Sharpe, Sortino, drawdown, win rate, etc.

### ML Models
- `RandomForestModel`: Random forest classifier for signals
- `GradientBoostingModel`: Gradient boosting classifier
- `StrategyModel`: Ensemble of multiple models
- `StrategyOptimizer`: Optuna-based hyperparameter tuning

### Strategies
- `SMAStrategy`: SMA crossover
- `RSIStrategy`: RSI mean reversion
- `MACDStrategy`: MACD trend following
- `MLStrategy`: ML model-based signals

## Available Skills

Use these Claude Code skills for common operations:

- `/backtest` - Run strategy backtests
- `/train-model` - Train ML models
- `/analyze-data` - Analyze historical data
- `/generate-data` - Generate synthetic data
- `/live-trade` - Start live trading assistant (reads knowledge base)

## API Integration

The bot supports:
1. **NinjaTrader/Tradovate API** - Direct REST + WebSocket API
2. **Manual Data Export** - Load NT8 exported .txt files

For API access, create a `.env` file:

```env
NINJATRADER_USERNAME=your_username
NINJATRADER_PASSWORD=your_password
NINJATRADER_DEMO=true
```

## Development Guidelines

1. **Time-Series Aware**: Always use `TimeSeriesSplit` for CV, never random splits
2. **No Look-Ahead Bias**: Features must only use past data
3. **Realistic Costs**: Include commission ($2.50/contract) and slippage (1 tick)
4. **Risk Management**: Default 2% risk per trade, use stops

## Common Tasks

### Add a New Strategy

1. Create file in `src/strategies/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signal()` method
4. Register in `src/strategies/__init__.py`

### Add a New ML Model

1. Create class inheriting from `BaseModel` in `src/ml/models.py`
2. Implement `fit()` and `predict()` methods
3. Add to `StrategyModel` ensemble if needed

### Add Trading Knowledge

1. Create markdown file in `knowledge/strategies/` or `knowledge/setups/`
2. Use clear structure with checklists
3. Include one-sentence summaries for quick reference

### Optimize Strategy Parameters

```python
from src.ml.optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(n_trials=100)
result = optimizer.optimize_random_forest(X, y)
print(f"Best params: {result.best_params}")
```

---

## Session Tracking

During live trading sessions, track:
- Session P&L (running total)
- Individual trade log (entry, exit, P&L)
- Win rate and R:R statistics

Output session summary on request or at session end.
