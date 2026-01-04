# Trading Strategies Extracted from YouTube Videos

## Source Videos
1. **"How to Avoid False Breakouts"** - TradingLab (3.4M views)
2. **"The Only Liquidity Guide You'll EVER NEED"** - TradingLab
3. **"Candlestick Patterns Don't Work (Unless You Do This)"** - TradingLab

---

## Strategy 1: Breakout Confirmation with Momentum Candles

### Core Concept
Don't enter immediately on a breakout. Wait for **momentum candle confirmation**.

### Rules
1. **Identify consolidation** - Multiple touches of support AND resistance
2. **Wait for breakout** - Price breaks through key level
3. **Confirm with momentum candles**:
   - Option A: **1 large candle** with majority of body beyond the level
   - Option B: **3 consecutive candles** in same direction beyond level
4. **Entry**: Close of momentum candle confirmation
5. **Stop Loss**: Just below the broken resistance (for longs) or above broken support (for shorts)

### Exit Strategy: 2-Step Take Profit System
1. Set initial take profit at **1.5R** (1.5x risk)
2. At 1.5R, **sell half position** to lock in profit
3. Move stop loss to **breakeven** (original take profit)
4. Use **Chandelier Exit** (ATR multiplier = 2) for remaining position
5. Exit remaining when Chandelier changes color

### Implementation as Features
```python
# Momentum candle detection
momentum_candle = (candle_body > threshold) & (body_beyond_level > 0.5)
three_candle_momentum = (close > close.shift(1)) & (close.shift(1) > close.shift(2)) & (close.shift(2) > close.shift(3))
```

---

## Strategy 2: Liquidity Hunting (Smart Money Concepts)

### Core Concept
Big players (algorithms) hunt **stop losses** at obvious levels. Trade WITH them, not against them.

### Key Insight
> "In order to be successful at trading, you need to be entering where most people are exiting."

### Liquidity Zones
- **Equal Highs/Lows**: Areas where multiple swing points align = obvious stop loss clusters
- **Consolidation Boundaries**: Top and bottom of ranges have heavy stop losses
- **False Breakouts**: Algorithms sweep liquidity before reversing

### The Liquidity Sweep Pattern
1. Price consolidates, creating obvious highs/lows
2. Price breaks level (triggering stops) - this is the **liquidity sweep**
3. Price quickly reverses back into range
4. **Entry**: On the reversal candle back into range
5. **Target**: Opposite side of consolidation (where OTHER stops are)

### Implementation as Features
```python
# Equal highs/lows detection
equal_highs = (high - high.shift(n)).abs() < tolerance
equal_lows = (low - low.shift(n)).abs() < tolerance

# Liquidity sweep detection
sweep_high = (high > recent_high) & (close < recent_high)  # Wick above, close below
sweep_low = (low < recent_low) & (close > recent_low)  # Wick below, close above
```

---

## Strategy 3: Candlestick Context Analysis

### Core Concept
Candlestick patterns only work **in context**. Must consider:
1. **Candle type** (strength, control shift, indecision)
2. **Location** (at key levels vs middle of range)
3. **Size relative to recent candles**

### Three Candle Types

#### 1. Strength Candles
- Large body, small wicks
- Shows one side in complete control
- **Strongest**: Large body with NO wick on one side
- Larger = stronger signal

#### 2. Control Shift Candles (Reversal)
- Long wicks, small body
- Shows rejected price action
- Long upper wick = sellers taking over
- Long lower wick = buyers taking over
- Body color doesn't matter, **wick tells the story**

#### 3. Indecision Candles
- Small body, wicks on both sides
- Doji-like patterns
- Shows equilibrium
- **Don't trade these alone** - wait for confirmation

### The "Story" Approach
Instead of pattern matching, read the **narrative**:
1. Who was in control? (look at recent candles)
2. Is control shifting? (reversal candle appearing)
3. Is there confirmation? (follow-through candle)

### Implementation as Features
```python
# Candle type classification
body_size = abs(close - open)
upper_wick = high - max(close, open)
lower_wick = min(close, open) - low
total_range = high - low

# Strength candle
strength_candle = (body_size / total_range) > 0.7

# Control shift candle (reversal)
bullish_reversal = (lower_wick / total_range) > 0.6
bearish_reversal = (upper_wick / total_range) > 0.6

# Indecision candle
indecision = (body_size / total_range) < 0.3
```

---

## Unified Strategy: Combining All Concepts

### Long Entry Conditions
1. **Consolidation identified** (range established)
2. **Liquidity sweep of lows** (stop hunt below range)
3. **Bullish control shift candle** (long lower wick)
4. **Momentum confirmation** (strength candle or 3 green candles)
5. **Entry** at close of confirmation candle
6. **Stop** below the liquidity sweep low
7. **Target** opposite liquidity zone (highs)

### Short Entry Conditions
1. **Consolidation identified** (range established)
2. **Liquidity sweep of highs** (stop hunt above range)
3. **Bearish control shift candle** (long upper wick)
4. **Momentum confirmation** (strength candle or 3 red candles)
5. **Entry** at close of confirmation candle
6. **Stop** above the liquidity sweep high
7. **Target** opposite liquidity zone (lows)

---

## Feature Engineering for ML Model

Based on these strategies, implement these features:

### Breakout Features
- `consolidation_range`: (highest_high - lowest_low) over N bars
- `bars_in_range`: Count of bars within range
- `breakout_strength`: Body size of breakout candle / ATR
- `momentum_confirmation`: 3 consecutive candles in direction

### Liquidity Features
- `equal_highs_nearby`: Boolean, equal highs within N bars
- `equal_lows_nearby`: Boolean, equal lows within N bars
- `sweep_detected`: Wick beyond level but close inside
- `volume_on_sweep`: Volume spike on sweep candle

### Candle Type Features
- `body_ratio`: body_size / total_range
- `upper_wick_ratio`: upper_wick / total_range
- `lower_wick_ratio`: lower_wick / total_range
- `candle_type`: 0=indecision, 1=strength_bull, -1=strength_bear, 2=reversal_bull, -2=reversal_bear

### Context Features
- `relative_body_size`: body_size / avg_body_size(20)
- `at_resistance`: close near recent highs
- `at_support`: close near recent lows
- `trend_context`: Higher highs/lows or lower highs/lows

---

## Key Takeaways

1. **Never enter on first breakout** - Wait for momentum confirmation
2. **Trade liquidity sweeps** - Enter where others are stopped out
3. **Read candle stories** - Type + location + size = context
4. **Use 2-step exits** - Lock in profit, let winners run
5. **Chandelier Exit** - ATR-based trailing stop for trend following

---

## Strategy 4: 3-Step Supply & Demand Formula (Video 4)

**Source:** "The Only Trading Strategy You'll Ever Need" - TradingLab (858K views)

### The 3-Step Formula

#### Step 1: Market Structure (Valid Highs/Lows)

**Critical Concept: Not all highs/lows are valid!**

A low is only **validated** if it breaks the previous high.
A high is only **validated** if it breaks the previous low.

**Uptrend Rules:**
- Price makes higher highs AND higher lows
- A low is only valid if it created a new higher high
- Stay bullish until the **valid low** is broken
- Ignore any low that didn't break the previous high

**Downtrend Rules:**
- Price makes lower lows AND lower highs  
- A high is only valid if it created a new lower low
- Stay bearish until the **valid high** is broken
- Ignore any high that didn't break the previous low

**Key Insight:**
> "A lot of traders see a broken low and think it's a reversal. But if that low never broke the previous high, it's not a valid low - we're still in an uptrend!"

#### Step 2: Supply & Demand Zones

**Demand Zone (for uptrends):**
- Find consolidation BEFORE a strong upward move
- Mark the candle RIGHT BEFORE the impulse move
- Draw zone from that candle's low to high
- Entry: When price returns to this zone
- Stop: Below the demand zone
- Target: Recent highs

**Supply Zone (for downtrends):**
- Find consolidation BEFORE a strong downward move
- Mark the candle RIGHT BEFORE the impulse move
- Draw zone from that candle's high to low
- Entry: When price returns to this zone
- Stop: Above the supply zone
- Target: Recent lows

**Rules:**
- In uptrend → ONLY look for demand zones (longs)
- In downtrend → ONLY look for supply zones (shorts)
- Never counter-trend trade

#### Step 3: Risk to Reward Filter

**Minimum R:R = 2.5:1**

- For every $100 risked, expect $250+ reward
- If R:R < 2.5, **skip the trade** even if steps 1 & 2 are valid
- This single rule massively improves win rate

### Implementation as Features

```python
# Valid swing detection
def find_valid_lows(highs, lows):
    valid_lows = []
    last_high = highs[0]
    for i in range(1, len(highs)):
        if highs[i] > last_high:  # New higher high
            # The low before this high is now validated
            valid_lows.append(lows[i-1:i].min())
            last_high = highs[i]
    return valid_lows

# Trend determination
def get_trend(price, valid_low, valid_high):
    if price > valid_low:
        return "UPTREND"
    elif price < valid_high:
        return "DOWNTREND"
    return "NEUTRAL"

# Supply/Demand zone detection
def find_demand_zone(df, impulse_threshold=0.003):
    """Find consolidation followed by strong up move"""
    returns = df['close'].pct_change()
    impulse_bars = returns > impulse_threshold
    
    zones = []
    for i in range(20, len(df)):
        if impulse_bars[i]:
            # Mark candle before impulse
            zone_high = df['high'].iloc[i-1]
            zone_low = df['low'].iloc[i-1]
            zones.append((zone_low, zone_high, i))
    return zones

# Risk:Reward calculation
def check_rr(entry, stop, target, min_rr=2.5):
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = reward / risk if risk > 0 else 0
    return rr >= min_rr, rr
```

### Trading Rules Summary

| Condition | Action |
|-----------|--------|
| Uptrend + Price at demand zone + R:R > 2.5 | **LONG** |
| Downtrend + Price at supply zone + R:R > 2.5 | **SHORT** |
| Counter-trend setup | **NO TRADE** |
| R:R < 2.5 | **NO TRADE** |

### Key Takeaways

1. **Valid structure matters** - Not every swing is tradeable
2. **Trade with trend only** - Longs in uptrend, shorts in downtrend
3. **Zone entry** - Wait for price to return to supply/demand
4. **R:R filter** - 2.5:1 minimum or skip
5. **Repeat forever** - Same process, consistent results


---

## Strategy 5: Williams Fractals Scalping (Video 5)

**Source:** "EASY Scalping Strategy For Day Trading" - TradingLab (335K views)

### Indicators Required
1. **Williams Fractals** (period = 2)
2. **Three Moving Averages**: 20, 50, 100 period

### Strategy Rules

#### For LONG Trades

**Setup Conditions:**
1. MA alignment: 20 > 50 > 100 (green > yellow > red)
2. MAs must NOT be crossing/tangled
3. Price pulls back UNDER the 20 MA (or 50 MA)

**Entry Signal:**
- Williams Fractal shows **green arrow** (bullish fractal)

**Stop Loss:**
- If entry near 20 MA → Stop below 50 MA
- If entry near 50 MA → Stop below 100 MA

**Take Profit:**
- 1.5:1 Risk/Reward ratio

**Critical Rule:**
> If price closes BELOW the 100 MA, **DO NOT** take the next green arrow signal. The probability of further decline is much higher.

#### For SHORT Trades

**Setup Conditions:**
1. MA alignment: 100 > 50 > 20 (inverted - bearish order)
2. MAs must NOT be crossing/tangled
3. Price pulls back ABOVE the 20 MA (or 50 MA)

**Entry Signal:**
- Williams Fractal shows **red arrow** (bearish fractal)

**Stop Loss:**
- If entry near 20 MA → Stop above 50 MA
- If entry near 50 MA → Stop above 100 MA

**Take Profit:**
- 1.5:1 Risk/Reward ratio

**Critical Rule:**
> If price closes ABOVE the 100 MA (in downtrend), **DO NOT** take the next red arrow signal.

### Williams Fractals Explained

A **fractal** is a swing point:
- **Bullish fractal** (green arrow): Bar with lowest low surrounded by higher lows
- **Bearish fractal** (red arrow): Bar with highest high surrounded by lower highs

With period=2, it checks 2 bars on each side (5 bars total pattern).

### Implementation as Features

```python
import pandas as pd
import numpy as np

def williams_fractals(df, period=2):
    """
    Detect Williams Fractals.
    
    Returns:
        bullish_fractal: True where bullish fractal detected
        bearish_fractal: True where bearish fractal detected
    """
    highs = df['high']
    lows = df['low']
    
    bullish = pd.Series(False, index=df.index)
    bearish = pd.Series(False, index=df.index)
    
    for i in range(period, len(df) - period):
        # Bullish fractal: low is lowest of surrounding bars
        if lows.iloc[i] == lows.iloc[i-period:i+period+1].min():
            bullish.iloc[i] = True
        
        # Bearish fractal: high is highest of surrounding bars
        if highs.iloc[i] == highs.iloc[i-period:i+period+1].max():
            bearish.iloc[i] = True
    
    return bullish, bearish


def ma_alignment(df, fast=20, medium=50, slow=100):
    """
    Check moving average alignment.
    
    Returns:
        bullish_aligned: 20 > 50 > 100
        bearish_aligned: 100 > 50 > 20
        tangled: MAs are crossing
    """
    ma_fast = df['close'].rolling(fast).mean()
    ma_medium = df['close'].rolling(medium).mean()
    ma_slow = df['close'].rolling(slow).mean()
    
    bullish_aligned = (ma_fast > ma_medium) & (ma_medium > ma_slow)
    bearish_aligned = (ma_slow > ma_medium) & (ma_medium > ma_fast)
    
    # Tangled = not clearly aligned
    tangled = ~bullish_aligned & ~bearish_aligned
    
    return bullish_aligned, bearish_aligned, tangled, ma_fast, ma_medium, ma_slow


def fractal_scalp_signals(df):
    """
    Generate scalping signals using Williams Fractals + MAs.
    """
    bullish_frac, bearish_frac = williams_fractals(df)
    bull_align, bear_align, tangled, ma20, ma50, ma100 = ma_alignment(df)
    
    close = df['close']
    
    # Long signal conditions
    long_signal = (
        bullish_frac &                    # Bullish fractal
        bull_align &                      # MAs aligned bullish
        ~tangled &                        # Not tangled
        (close < ma20) &                  # Price pulled back below 20 MA
        (close > ma100)                   # NOT below 100 MA (critical rule)
    )
    
    # Short signal conditions
    short_signal = (
        bearish_frac &                    # Bearish fractal
        bear_align &                      # MAs aligned bearish
        ~tangled &                        # Not tangled
        (close > ma20) &                  # Price pulled back above 20 MA
        (close < ma100)                   # NOT above 100 MA (critical rule)
    )
    
    return long_signal, short_signal, ma20, ma50, ma100
```

### Key Insights

1. **Trend filter**: MA alignment prevents counter-trend trades
2. **Pullback requirement**: Don't chase - wait for retracement
3. **Fractal confirmation**: Swing point validation before entry
4. **100 MA rule**: Critical filter to avoid failed trades
5. **Scalp-friendly**: Works on 1-minute timeframe

### Warning for 1-Minute Trading

> "You will get a TON of signals on 1-minute. Make sure fees are low, otherwise fees will ruin gains."

Consider:
- Using on 5m or 15m for fewer but cleaner signals
- Or aggregate 1-second data to 1-minute for this strategy


---

## Strategy 6: MACD + 200 MA (86% Win Rate)

**Source:** "BEST MACD Trading Strategy" - TradingLab

### MACD Components
1. **MACD Line** (blue): 12-day EMA
2. **Signal Line** (orange): 26-day EMA  
3. **Histogram**: Difference between MACD and Signal
4. **Zero Line**: Center reference

### The Strategy

**Long Entry:**
1. Price must be ABOVE 200 MA (uptrend filter)
2. MACD lines cross upward
3. Crossover must happen BELOW zero line

**Short Entry:**
1. Price must be BELOW 200 MA (downtrend filter)
2. MACD lines cross downward
3. Crossover must happen ABOVE zero line

**Why 200 MA matters:** MACD alone gives false signals in ranging markets. The 200 MA ensures you only trade WITH the trend.

```python
def macd_signals(df, fast=12, slow=26, signal=9, ma_period=200):
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    ma_200 = df['close'].rolling(ma_period).mean()
    
    # Long: MACD crosses above signal, below zero, price > 200 MA
    long_signal = (
        (macd_line > signal_line) & 
        (macd_line.shift(1) <= signal_line.shift(1)) &
        (macd_line < 0) &
        (df['close'] > ma_200)
    )
    
    # Short: MACD crosses below signal, above zero, price < 200 MA
    short_signal = (
        (macd_line < signal_line) & 
        (macd_line.shift(1) >= signal_line.shift(1)) &
        (macd_line > 0) &
        (df['close'] < ma_200)
    )
    
    return long_signal, short_signal
```

---

## Strategy 7: Heikin Ashi Reversal

**Source:** "The Heikin Ashi Trading Strategy" - TradingLab

### Heikin Ashi Basics
- Shows AVERAGE price, not real price
- Smooths out noise
- Makes trends clearer

### Trend Identification

**Strong Uptrend:**
- Green candles
- Large bodies
- NO lower wicks

**Strong Downtrend:**
- Red candles
- Large bodies
- NO upper wicks

### Reversal Signal: Doji Candle

A **Doji** = small body + wicks on BOTH sides

**Long Entry:**
1. Downtrend in place
2. Doji candle appears (opposite color)
3. Wait for 2 green candles with wicks ONLY on top
4. Enter on 2nd confirmation candle

**Short Entry:**
1. Uptrend in place
2. Doji candle appears (opposite color)
3. Wait for 2 red candles with wicks ONLY on bottom
4. Enter on 2nd confirmation candle

```python
def heikin_ashi(df):
    ha = pd.DataFrame()
    ha['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha['open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    ha['high'] = df[['high', 'open', 'close']].max(axis=1)
    ha['low'] = df[['low', 'open', 'close']].min(axis=1)
    return ha

def ha_doji(ha_df, body_threshold=0.3):
    body = abs(ha_df['close'] - ha_df['open'])
    range_ = ha_df['high'] - ha_df['low']
    return body / range_ < body_threshold
```

---

## Strategy 8: Triple SuperTrend

**Source:** "How To Identify Trends in Markets" - TradingLab

### Setup
Add 3 SuperTrend indicators with settings:
1. ATR Period=12, Multiplier=3
2. ATR Period=10, Multiplier=1
3. ATR Period=11, Multiplier=2

### Rules

**Long Entry:** ALL 3 lines are GREEN
**Short Entry:** ALL 3 lines are RED
**No Trade:** Lines are mixed colors (ranging market)

**Exit Options:**
1. Fixed 1.5R take profit
2. When ANY line changes color
3. Combine with 200 MA for trend filter

```python
def supertrend(df, period, multiplier):
    hl2 = (df['high'] + df['low']) / 2
    atr = df['high'].rolling(period).max() - df['low'].rolling(period).min()
    
    upper = hl2 + (multiplier * atr)
    lower = hl2 - (multiplier * atr)
    
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    
    return trend  # 1=bullish, -1=bearish
```

---

## Strategy 9: DEMA + SuperTrend (130% in 2 months)

**Source:** "Highly Profitable DEMA + SuperTrend" - TradingLab

### Setup
1. **DEMA** (Double EMA): Period = 200
2. **SuperTrend**: ATR Period=12, Multiplier=3

### Rules

**Long Entry:**
1. Price ABOVE 200 DEMA
2. SuperTrend flips to BUY signal
3. Wait for signal candle to CLOSE
4. Stop loss at SuperTrend line

**Short Entry:**
1. Price BELOW 200 DEMA
2. SuperTrend flips to SELL signal
3. Wait for signal candle to CLOSE
4. Stop loss at SuperTrend line

**Exit:** When SuperTrend flips opposite direction (unlimited profit potential)

### Why DEMA?
- Faster than regular EMA
- Less lag = earlier entries
- Created in 1994 specifically to reduce lag

---

## Strategy 10: Fibonacci Retracement Levels

**Source:** "The ULTIMATE Fibonacci Retracement Tool Guide" - TradingLab

### Key Levels
- 23.6%
- 38.2%
- 50.0%
- 61.8% (Golden Ratio - most important)
- 78.6%

### How To Use

1. Find clear swing high and swing low
2. Draw Fib from swing LOW to swing HIGH (for uptrend)
3. Draw from WICK to WICK (not body!)
4. Wait for pullback to hit a Fib level
5. Look for reversal confirmation at that level

### Most Reliable Levels
- **61.8%** - Golden ratio, highest probability
- **50.0%** - Psychological level
- **38.2%** - Shallow pullback in strong trends

```python
def fib_levels(swing_low, swing_high):
    diff = swing_high - swing_low
    return {
        '0.0': swing_high,
        '23.6': swing_high - 0.236 * diff,
        '38.2': swing_high - 0.382 * diff,
        '50.0': swing_high - 0.500 * diff,
        '61.8': swing_high - 0.618 * diff,
        '78.6': swing_high - 0.786 * diff,
        '100.0': swing_low
    }
```

---

## Strategy 11: ATR-Based Stop Loss

**Source:** "How To Know Where to Set Your Stop Loss" - TradingLab

### The Problem
Setting stops at obvious levels (swing lows) = getting stopped out before the move.

### The Solution: ATR

ATR = Average True Range of last 14 candles

**Stop Loss Formula:**
```
Stop Loss = Entry Price - (ATR × Multiplier)
```

Typical multipliers:
- Conservative: 1.5 × ATR
- Standard: 2.0 × ATR
- Wide: 3.0 × ATR

### Why ATR Works
- Adapts to current volatility
- Different for each asset/timeframe
- Prevents arbitrary fixed-dollar stops

```python
def atr_stop_loss(df, entry_price, direction='long', multiplier=2.0, period=14):
    tr = pd.DataFrame()
    tr['hl'] = df['high'] - df['low']
    tr['hc'] = abs(df['high'] - df['close'].shift(1))
    tr['lc'] = abs(df['low'] - df['close'].shift(1))
    tr['tr'] = tr.max(axis=1)
    
    atr = tr['tr'].rolling(period).mean().iloc[-1]
    
    if direction == 'long':
        return entry_price - (atr * multiplier)
    else:
        return entry_price + (atr * multiplier)
```

---

## Strategy 12: Doji Candle Rules

**Source:** "The Common MISTAKE Traders Make With Doji Candles" - TradingLab

### The Mistake
Seeing a Doji and immediately expecting reversal.

### The Truth
Doji = **INDECISION**, not reversal

### Correct Usage
1. Doji appears after a trend
2. **Wait for confirmation** - next candle must confirm direction
3. If next candle is opposite color with strong body = reversal likely
4. If next candle continues trend = Doji was just a pause

### Key Rule
> "A Doji by itself means NOTHING. Always wait for the next candle to confirm."

---

## Summary: All 12 Strategies

| # | Strategy | Entry Trigger | Best For |
|---|----------|---------------|----------|
| 1 | Breakout + Momentum | 3 candles beyond level | Trend continuation |
| 2 | Liquidity Sweep | Sweep + reversal candle | Counter-trend entries |
| 3 | Candlestick Context | Strength/reversal candles | Signal filtering |
| 4 | Supply/Demand Zones | Return to zone | Trend pullbacks |
| 5 | Williams Fractals | Fractal + MA alignment | Scalping |
| 6 | MACD + 200 MA | MACD cross + trend filter | Trend trading |
| 7 | Heikin Ashi | Doji + 2 confirmation | Reversal trading |
| 8 | Triple SuperTrend | All 3 same color | Trend confirmation |
| 9 | DEMA + SuperTrend | Price > DEMA + signal | Swing trading |
| 10 | Fibonacci | Pullback to key level | Entry optimization |
| 11 | ATR Stop Loss | ATR × multiplier | Risk management |
| 12 | Doji Rules | Doji + confirmation | Avoiding false signals |

