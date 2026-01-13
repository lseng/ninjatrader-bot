# Order Blocks - Complete Trading Guide

## Overview
Order blocks are key price zones where institutional traders have placed significant orders. Unlike support/resistance, order blocks are one-time use zones that must meet specific validation criteria.

---

## Order Blocks vs Support/Resistance

| Aspect | Order Blocks | Support/Resistance |
|--------|--------------|-------------------|
| Formation | Significant price move + inefficiency | Multiple rejections at level |
| Drawing | Thick zones (full candle H-L) | Thin lines or narrow zones |
| Retests | One-time use (mitigated after touch) | Can be retested multiple times |
| Validation | Requires BOS/CHoCH confirmation | Based on historical rejections |

---

## The 3 Rules for Valid Order Blocks

An order block is ONLY valid if ALL three rules are met:

### Rule 1: Must Create Inefficiency (Gap/FVG)
- The move from the OB must create a Fair Value Gap
- Gap = space between candle 1's wick and candle 3's wick
- If wicks overlap with candle 2's body = NO gap = INVALID OB

```
VALID (has gap):          INVALID (no gap):
    |                         |
   [2]                       [2]
    |                        |||
         <- GAP ->          [1][3]
   [1]  [3]                   |
    |    |
```

### Rule 2: Must Be Unmitigated
- Order blocks are ONE-TIME USE
- If price (even just a wick) touches the OB zone, it's MITIGATED
- Mitigated OBs cannot be used for entries
- Exception: Breaker blocks (see strategy 3)

### Rule 3: Must Lead to BOS or CHoCH
- The move from the OB must break market structure
- BOS = Break of Structure (trend continuation)
- CHoCH = Change of Character (trend reversal)
- If no structure break occurs = OB is NOT valid

**Validation Flow:**
```
Significant move + Gap detected
            |
            v
    Did it break structure?
     /              \
   YES               NO
    |                 |
 VALID OB      NOT a valid OB
```

---

## Drawing Order Blocks

Once validated, draw the OB on the **first candle before the inefficiency**:
- **Bullish OB**: Last RED candle before bullish displacement
- **Bearish OB**: Last GREEN candle before bearish displacement
- Draw rectangle from candle HIGH to candle LOW (full range)

---

## Strategy 1: Multi-Timeframe Confirmation

**Concept:** Find OB on higher timeframe, confirm entry on lower timeframe.

### Timeframe Correlation
| HTF (Order Block) | LTF (Entry Confirmation) |
|-------------------|--------------------------|
| Weekly | 4-Hour |
| Daily | 1-Hour |
| 4-Hour | 15-Minute |
| 1-Hour | 5-Minute |

### Process
1. **HTF**: Identify valid order block (all 3 rules met)
2. **Wait**: For price to retrace to the OB zone
3. **LTF**: Switch to lower timeframe when price enters OB
4. **Confirm**: Look for reversal signal (engulfing, BOS, etc.)
5. **Enter**: After confirmation pattern completes

### Entry Confirmation Tools (LTF)
- Bearish/Bullish Engulfing pattern
- Change of character on LTF
- Displacement candles showing reversal
- Break of LTF structure

### Trade Setup
- **Entry**: After LTF confirmation in HTF OB zone
- **Stop Loss**: Just beyond the OB zone
- **Take Profit**: 2x stop loss (2R minimum)

---

## Strategy 2: Inducement Traps

**Concept:** Minor key level forms near major OB = trap setup for stop hunts.

### Pattern Structure
```
         Minor Key Level (Inducement)
    -------- multiple rejections --------
              |
              v
    Price breaks through (traps traders)
              |
              v
    ========= MAJOR ORDER BLOCK =========
              |
              v
         Price reverses from OB
```

### Why It Works
1. Institutions enter at OB, forming the zone
2. Price creates minor S/R level above/below OB
3. Retail traders see "obvious" level, place stops beyond it
4. Smart money pushes through minor level to trigger stops
5. This selling/buying pressure lets institutions re-enter at OB
6. Price reverses from OB, trapping breakout traders

### Trade Setup
- **Entry**: Limit order in middle of major OB
- **Stop Loss**: Just beyond the OB (not the inducement level)
- **Take Profit**: 2-3x risk, or back to inducement zone
- **Key**: Order only triggers if price reaches OB

### Identification Checklist
- [ ] Major OB identified (valid by 3 rules)
- [ ] Minor key level forms near OB (2-4 rejections)
- [ ] Gap between minor level and OB
- [ ] Price breaks minor level toward OB
- [ ] Enter at OB, target back through minor level

---

## Strategy 3: Breaker Blocks

**Concept:** A broken (mitigated) order block flips to act as the opposite zone.

### How Breaker Blocks Form
1. Valid bullish OB exists (support zone)
2. Price breaks DOWN through the OB (mitigates it)
3. This break creates a CHoCH (character change)
4. Former support OB now acts as RESISTANCE
5. Price retraces back up to the broken OB
6. Enter SHORT at the breaker block

### Breaker Block Logic
```
BULLISH OB (support)
        |
        v
    Price breaks down through OB
        |
        v
    OB is now MITIGATED
        |
        v
    But it becomes a BREAKER BLOCK (resistance)
        |
        v
    SHORT when price retraces to it
```

### Key Rules
- Breaker blocks are also ONE-TIME use
- After price retests the breaker, it's done
- Must have clear break through the original OB
- Enter on retracement, not the initial break

### Trade Setup
- **Entry**: When price retraces to breaker block zone
- **Stop Loss**: Just beyond the breaker block
- **Take Profit**: 2x risk, or next liquidity target

---

## Quick Reference Checklist

### Valid Bullish OB Entry:
- [ ] Significant upward move from the zone
- [ ] Gap/inefficiency created (FVG above)
- [ ] BOS or CHoCH confirmed (higher high made)
- [ ] OB is unmitigated (price hasn't touched it)
- [ ] Wait for price to retrace TO the OB
- [ ] LTF confirmation (if using multi-TF)
- [ ] Enter LONG, stop below OB, target 2R+

### Valid Bearish OB Entry:
- [ ] Significant downward move from the zone
- [ ] Gap/inefficiency created (FVG below)
- [ ] BOS or CHoCH confirmed (lower low made)
- [ ] OB is unmitigated (price hasn't touched it)
- [ ] Wait for price to retrace TO the OB
- [ ] LTF confirmation (if using multi-TF)
- [ ] Enter SHORT, stop above OB, target 2R+

### Breaker Block Entry:
- [ ] Previously valid OB has been broken through
- [ ] Break created CHoCH (trend reversal)
- [ ] Price retracing back to the broken OB
- [ ] Enter OPPOSITE direction of original OB
- [ ] Stop beyond the breaker zone
- [ ] Target 2R or next liquidity pool

---

## One-Sentence Summary

> Order blocks require gap + unmitigated + structure break; use multi-TF confirmation for entries, watch for inducement traps near OBs, and trade breaker blocks when OBs get broken.

---

## Common Mistakes

1. **Trading any candle as OB** - Must have inefficiency + BOS/CHoCH
2. **Using mitigated OBs** - Once touched, they're done
3. **No LTF confirmation** - Reduces win rate significantly
4. **Ignoring inducement** - Minor levels near OBs = trap potential
5. **Fighting breakers** - Broken OBs flip, don't expect them to hold
6. **Wrong candle selection** - OB is the LAST opposing candle before move
