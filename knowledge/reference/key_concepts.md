# Trading Key Concepts Reference

## Liquidity

### Buy-Side Liquidity (BSL)
- Located ABOVE price (highs)
- Where buy stops and sell limit orders rest
- Targets: Equal highs, session highs, swing highs, resistance levels
- When raided: Short opportunity may follow

### Sell-Side Liquidity (SSL)
- Located BELOW price (lows)
- Where sell stops and buy limit orders rest
- Targets: Equal lows, session lows, swing lows, support levels
- When raided: Long opportunity may follow

---

## Market Structure

### Break of Structure (BOS)
- Candle CLOSES through a swing point
- Bullish BOS: Close above swing high
- Bearish BOS: Close below swing low
- Confirms trend continuation or reversal

### Change of Character (CHoCH)
- First BOS against the prevailing trend
- Signals potential reversal
- More significant than regular BOS

### Swing Points
- Swing High: High with lower highs on both sides
- Swing Low: Low with higher lows on both sides
- Key levels for BOS confirmation

---

## Price Action Patterns

### Displacement
- Strong, aggressive directional move
- Large-bodied candles with small wicks
- Creates imbalances (FVGs)
- Confirms institutional intent
- NOT an entry - it's confirmation

### Fair Value Gap (FVG) / Imbalance
- Gap in price action between 3 candles
- Bullish FVG: Gap between candle 1 high and candle 3 low
- Bearish FVG: Gap between candle 1 low and candle 3 high
- Price tends to return to fill these gaps
- Entry zone for retracements

### Order Block (OB)
- Last opposing candle before displacement
- Bullish OB: Last red candle before up move
- Bearish OB: Last green candle before down move
- Represents institutional order placement
- Strong when combined with FVG

### Inversion FVG
- A FVG that **flips** from resistance to support (or vice versa)
- Occurs when price closes THROUGH the FVG
- Former resistance → support (bullish)
- Former support → resistance (bearish)
- High probability when nested at key DRT levels

### Consequent Encroachment (CE)
- The **50% level** of any FVG
- Used to gauge if a level is being defended
- If bodies fail to close through CE = level is holding
- Key confirmation for support/resistance

### CE Break = FVG Fill Prediction
**Critical rule for anticipating FVG fills:**

- If candle **CLOSES ABOVE** the CE of a **bearish FVG** → FVG will likely FILL
- If candle **CLOSES BELOW** the CE of a **bullish FVG** → FVG will likely FILL

```
BEARISH FVG example:
    ┌─────────────┐ FVG TOP (target)
    │             │
    │ ─ ─ CE ─ ─  │ ← 50% level
    │      ▲      │
    │  CLOSE HERE │ ← If candle closes above CE...
    └─────────────┘ FVG BOTTOM
           │
           ▼
    Next 1-2 candles will likely fill to FVG TOP
```

**Trading implication:**
- CE break = momentum confirmation
- Can enter expecting full FVG fill
- Stop below the CE level
- Target: opposite end of FVG

### Balanced Price Range (BPR)
- Two overlapping FVGs (bullish + bearish)
- Buy-side delivery followed by sell-side delivery
- Creates a "balanced" zone where price has been efficiently delivered both ways
- **Key property**: When price returns to BPR, it acts as strong S/R
- If price moved UP from BPR → BPR is support
- If price moved DOWN from BPR → BPR is resistance
- High probability reversal zone for entries

### Turtle Soup Entry
Liquidity raid setup where price sweeps a high/low then reverses into a FVG.

**Bearish Turtle Soup (HIGH PROBABILITY):**
- Old high exists (buy-side liquidity above it)
- Bearish FVG rests ABOVE that old high
- Price sweeps above the old high (raids BSL, traps longs)
- Price reverses and enters the bearish FVG
- **Entry**: SHORT in the bearish FVG
- **Stop**: Above the FVG
- **Target**: Sell-side liquidity below

**Bullish Turtle Soup (inverse):**
- Old low exists (sell-side liquidity below it)
- Bullish FVG rests BELOW that old low
- Price sweeps below the old low (raids SSL, traps shorts)
- Price reverses and enters the bullish FVG
- **Entry**: LONG in the bullish FVG
- **Stop**: Below the FVG
- **Target**: Buy-side liquidity above

---

## Dealing Range (DRT)

### What is a Dealing Range?
- The range between a significant high and low
- Used to identify institutional price levels
- Grade from high to low (or low to high)

### DRT Levels
```
HIGH ──────── 0% (top of range)
25 DRT ────── 25% (discount for longs)
50 DRT ────── 50% (equilibrium)
75 DRT ────── 75% (premium for shorts)
LOW ───────── 100% (bottom of range)
```

### How to Use DRT
- **Longs**: Look for entries at 25 DRT or below (discount)
- **Shorts**: Look for entries at 75 DRT or above (premium)
- **Confirmation**: FVGs should exist at DRT levels
- **High Probability**: Inversion FVG nested at 25 DRT (longs) or 75 DRT (shorts)

---

## Moving Averages (This Bot)

### MA20 (Fast)
- Short-term trend indicator
- Price above = short-term bullish
- Price below = short-term bearish
- First support/resistance level

### MA50 (Medium)
- Key support/resistance
- Commonly watched by institutions
- Strong bounces often occur here

### MA200 (Slow)
- Major trend indicator
- Price above = bullish bias
- Price below = bearish bias
- Major support/resistance level

---

## Contract Specifications

### ES (E-mini S&P 500)
- Tick Size: 0.25 points
- Tick Value: $12.50
- Point Value: $50.00
- Example: 5 point move = $250

### MES (Micro E-mini S&P 500)
- Tick Size: 0.25 points
- Tick Value: $1.25
- Point Value: $5.00
- Example: 5 point move = $25

---

## Session Times (Eastern)

### Session Windows
| Session | Time (ET) | Characteristics |
|---------|-----------|-----------------|
| **Asia** | 6:00 PM - 3:00 AM | Range-bound, sets initial levels |
| **London** | 3:00 AM - 9:30 AM | Often creates "Judas swing" (fake move) |
| **NY AM** | 9:30 AM - 12:00 PM | Highest volume, real move often here |
| **NY PM** | 12:00 PM - 4:00 PM | Continuation or reversal of AM move |

### Power of Three (PO3) - Daily Open

**The daily open at 12:00 AM (midnight) New York ET is the KEY level.**

```
BULLISH BIAS:
    Daily Open (midnight) ─────────────
                │
                ▼
    Price goes BELOW open (manipulation)
    Sweeps sell-side liquidity
    Break structure with displacement
    Enter LONG on FVG retracement
                │
                ▼
    Price EXPANDS above daily open (distribution)


BEARISH BIAS:
    Price EXPANDS below daily open (distribution)
                ▲
                │
    Enter SHORT on FVG retracement
    Break structure with displacement
    Sweeps buy-side liquidity
    Price goes ABOVE open (manipulation)
                ▲
                │
    Daily Open (midnight) ─────────────
```

**The Rule:**
- **Bullish day**: Look to BUY below the daily open
- **Bearish day**: Look to SELL above the daily open

**PO3 + 2022 Model (Bullish Example):**
1. Mark daily open (12:00 AM ET)
2. Wait for price to trade BELOW open
3. Take sell-side liquidity (sweep lows below open)
4. Break structure with displacement
5. Enter LONG on retracement into FVG
6. Let price expand ABOVE daily open
7. Close for profit

**Why this works:**
- Midnight open = equilibrium for the day
- Smart money pushes price to one side to grab liquidity
- Then drives it to the other side for the real move
- Aligns with AMD: Accumulation near open → Manipulation away → Distribution back through

---

### AMD Cycle (Accumulation → Manipulation → Distribution)

The key question: **What did Asia do?**

**Scenario 1: Asia ACCUMULATED (ranged, tight range)**
```
Asia: Accumulated (range) → builds liquidity
London: MANIPULATES → raids Asia H or L (Judas swing)
NY: DISTRIBUTES → real expansion move
```

**Scenario 2: Asia EXPANDED (trended, wide range)**
```
Asia: Expanded (trend) → already moved
London: ACCUMULATES → ranges, builds new liquidity
NY: MANIPULATES → raids London liquidity, then moves
```

**How to identify:**
- Asia ACCUMULATED: Range < 10-15 pts, price oscillates
- Asia EXPANDED: Range > 20+ pts, clear directional move

### Session Liquidity Targets
- **Asia High/Low**: Key liquidity targets for London/NY
- **London High/Low**: Key targets for NY session
- **Previous Day High/Low (PDH/PDL)**: Major liquidity pools

### Judas Swing (London Open)
- London often makes a FALSE move first (Judas swing)
- Sweeps Asia session liquidity (high or low)
- Then reverses for the real move
- Look for: Asia high/low sweep → reversal → target opposite side
- **Only expect Judas swing if Asia ACCUMULATED**

### NY Session Behavior
- Often targets London high/low
- Or continues London's real move after Judas swing
- 9:30-10:00 AM: Initial volatility (wait for direction)
- 10:00-10:30 AM: Opening range established
- Best setups often after 10:00 AM

### Key Times to Watch
- 9:30 AM: Market open (high volatility, wait)
- 10:00 AM: Opening range forming
- 12:00-1:00 PM: Lunch (avoid, choppy)
- 2:00 PM: Afternoon session begins
- 3:00-4:00 PM: Power hour
- 3:50-4:00 PM: Avoid (closing volatility)

---

## Risk Management

### Position Sizing
- Risk max 1-2% of account per trade
- Calculate: Risk $ / (Stop distance in points × Point value)

### Stop Placement
- Beyond the liquidity raid level
- Or 1-1.5x ATR from entry
- Never move stop further from entry

### Take Profit
- Target opposing liquidity pool
- Or key support/resistance level
- Consider partials at 1:1 R:R

### R:R Guidelines
- Minimum: 1:1.5
- Target: 1:2 or better
- Don't take trades below 1:1
