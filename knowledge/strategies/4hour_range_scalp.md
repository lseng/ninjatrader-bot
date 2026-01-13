# 4-Hour Range Scalping Strategy

## Overview
A simple, rule-based scalping strategy using the first 4-hour candle of the day as a range. Trade false breakouts when price closes outside the range then re-enters. No indicators required.

**Core Concept:** False breakouts of the opening range tend to reverse. When price breaks out and comes back inside, trade in the direction of the reentry.

---

## The 3-Step Checklist

### Step 1: Mark the 4-Hour Range
- Use the **first 4-hour candle** of the trading day (New York time)
- Draw horizontal lines at the HIGH and LOW of that candle
- Extend lines to end of day
- **CRITICAL: Wait for the 4H candle to fully CLOSE before marking**

**Setup on TradingView:**
1. Set timeframe to 4H
2. Go to timezone settings (bottom right) → Select "New York"
3. Find first 4H candle of current date
4. Mark high and low, extend to end of day

### Step 2: Wait for Breakout + Re-entry (5-min chart)
- Switch to 5-minute timeframe
- Wait for a 5-min candle to **CLOSE outside** the range
- Then wait for price to **CLOSE back inside** the range
- **CRITICAL: Wicks don't count - candle body must CLOSE outside/inside**
- All must happen within the same trading day

### Step 3: Entry, Stop Loss, Take Profit

| Scenario | Entry Direction |
|----------|-----------------|
| Price breaks ABOVE range high → re-enters | **SHORT** |
| Price breaks BELOW range low → re-enters | **LONG** |

**Stop Loss:** At the exact high/low of the breakout move
**Take Profit:** 2x stop loss size (2R target)

---

## Quick Reference Checklist

### Long Setup (Break Below → Re-entry):
- [ ] 4H range marked (first candle of day, NY time)
- [ ] 5-min candle CLOSES below range low
- [ ] 5-min candle CLOSES back inside range
- [ ] Enter LONG on re-entry close
- [ ] Stop: Low of breakout move
- [ ] Target: 2R (2x stop loss distance)

### Short Setup (Break Above → Re-entry):
- [ ] 4H range marked (first candle of day, NY time)
- [ ] 5-min candle CLOSES above range high
- [ ] 5-min candle CLOSES back inside range
- [ ] Enter SHORT on re-entry close
- [ ] Stop: High of breakout move
- [ ] Target: 2R (2x stop loss distance)

---

## Large Breakout Adjustment

When the breakout move is very large (stop loss would be too wide):
- Instead of using the exact high/low of the breakout
- Find the **nearest key level** (order block, S/R, etc.)
- Place stop loss at that level instead
- This keeps risk manageable while still protecting the trade

---

## Key Rules

1. **Closes Only** - Wicks breaking the range don't count; need candle CLOSE
2. **Same Day Only** - All steps must occur within the same trading day as the 4H range
3. **Wait for Full Close** - 4H candle must be fully closed before marking range
4. **Multiple Setups OK** - Can take multiple valid setups within the same day
5. **No Re-entry = No Trade** - If price breaks out but doesn't re-enter by end of day, no trade

---

## Session Timing (New York Time)

| Session | 4H Candle Times |
|---------|-----------------|
| Candle 1 | 00:00 - 04:00 (mark this one) |
| Candle 2 | 04:00 - 08:00 |
| Candle 3 | 08:00 - 12:00 |
| Candle 4 | 12:00 - 16:00 |
| Candle 5 | 16:00 - 20:00 |
| Candle 6 | 20:00 - 00:00 |

---

## Works Across Markets

Backtested performance (from examples):
- **Crypto (BTC):** 5W / 2L = 72% win rate, +8R
- **Forex (EUR/USD):** 5W / 1L = 83% win rate, +9R
- **Gold (XAU):** 6W / 4L = 60% win rate, +8R

---

## Combining with Other Systems

This strategy can be combined with:
- **SMC Concepts:** Only take setups that align with HTF liquidity/structure
- **Trend Analysis:** Only trade in direction of higher timeframe trend
- **Price Action:** Look for rejection patterns at range levels
- **Session Bias:** Focus on setups during London/NY overlap

---

## One-Sentence Summary
> Mark first 4H candle range → Wait for 5-min close outside then back inside → Enter opposite direction of breakout with 2R target.

---

## Common Mistakes

1. **Marking range too early** - Wait for 4H candle to fully close
2. **Counting wicks** - Only candle CLOSES matter, not wicks
3. **Trading outside the day** - Range only valid for that trading day
4. **Stop too tight on large breakouts** - Use key levels instead
5. **Chasing breakouts** - This is a FADE strategy, not a breakout strategy
