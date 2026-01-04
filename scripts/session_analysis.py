#!/usr/bin/env python3
"""
Session Analysis - AMD Cycle Detection
Fetches historical data to analyze Asia/London/NY sessions

Usage:
    python scripts/session_analysis.py

Requires: TopStepX API credentials in environment or .env
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Add topstep-trading-bot to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "topstep-trading-bot"))

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')


def get_session_times(date: datetime = None) -> Dict[str, tuple]:
    """Get session start/end times for a given date (in ET)."""
    if date is None:
        date = datetime.now(ET)

    # Normalize to date only
    d = date.date()

    # Session times (Eastern Time)
    # Asia: 6PM previous day to 3AM current day
    asia_start = datetime(d.year, d.month, d.day, 18, 0, tzinfo=ET) - timedelta(days=1)
    asia_end = datetime(d.year, d.month, d.day, 3, 0, tzinfo=ET)

    # London: 3AM to 9:30AM
    london_start = datetime(d.year, d.month, d.day, 3, 0, tzinfo=ET)
    london_end = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)

    # NY AM: 9:30AM to 12PM
    ny_am_start = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)
    ny_am_end = datetime(d.year, d.month, d.day, 12, 0, tzinfo=ET)

    # NY PM: 12PM to 4PM
    ny_pm_start = datetime(d.year, d.month, d.day, 12, 0, tzinfo=ET)
    ny_pm_end = datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET)

    return {
        'ASIA': (asia_start, asia_end),
        'LONDON': (london_start, london_end),
        'NY_AM': (ny_am_start, ny_am_end),
        'NY_PM': (ny_pm_start, ny_pm_end),
    }


def get_current_session() -> str:
    """Determine current trading session."""
    now = datetime.now(ET)
    hour = now.hour
    minute = now.minute
    time_val = hour * 60 + minute

    if time_val >= 18 * 60 or time_val < 3 * 60:
        return 'ASIA'
    elif time_val < 9 * 60 + 30:
        return 'LONDON'
    elif time_val < 12 * 60:
        return 'NY_AM'
    elif time_val < 16 * 60:
        return 'NY_PM'
    else:
        return 'CLOSED'


def analyze_session_behavior(high: float, low: float, open_price: float, close_price: float) -> Dict:
    """Analyze if session accumulated or expanded."""
    range_pts = high - low
    body = abs(close_price - open_price)

    # Thresholds for ES/MES
    ACCUMULATION_THRESHOLD = 12  # Less than 12 pts = accumulation
    EXPANSION_THRESHOLD = 18     # More than 18 pts = expansion

    if range_pts < ACCUMULATION_THRESHOLD:
        behavior = 'ACCUMULATED'
        description = f'Tight range ({range_pts:.2f} pts) - liquidity building'
    elif range_pts > EXPANSION_THRESHOLD:
        behavior = 'EXPANDED'
        direction = 'BULLISH' if close_price > open_price else 'BEARISH'
        description = f'Wide range ({range_pts:.2f} pts) - {direction} expansion'
    else:
        behavior = 'MIXED'
        description = f'Moderate range ({range_pts:.2f} pts) - unclear'

    return {
        'behavior': behavior,
        'description': description,
        'range': range_pts,
        'high': high,
        'low': low,
        'open': open_price,
        'close': close_price,
    }


def predict_next_session(asia_behavior: str, london_behavior: str = None) -> Dict:
    """Predict next session behavior based on AMD cycle."""
    predictions = {}

    if asia_behavior == 'ACCUMULATED':
        predictions['LONDON'] = {
            'expected': 'MANIPULATE',
            'action': 'Expect Judas swing - raid Asia H or L then reverse',
            'watch': 'Look for sweep of Asia high/low, then reversal',
        }
        predictions['NY'] = {
            'expected': 'DISTRIBUTE',
            'action': 'Real expansion move after London manipulation',
            'watch': 'Enter on London reversal, target opposite liquidity',
        }
    elif asia_behavior == 'EXPANDED':
        predictions['LONDON'] = {
            'expected': 'ACCUMULATE',
            'action': 'Expect ranging/consolidation - no Judas swing',
            'watch': 'Wait for London to build liquidity',
        }
        predictions['NY'] = {
            'expected': 'MANIPULATE',
            'action': 'NY will raid London liquidity, then move',
            'watch': 'Wait for NY to sweep London H/L before entering',
        }
    else:
        predictions['LONDON'] = {
            'expected': 'UNCLEAR',
            'action': 'Asia behavior mixed - wait for clarity',
            'watch': 'Monitor London to see if it ranges or trends',
        }
        predictions['NY'] = {
            'expected': 'UNCLEAR',
            'action': 'Wait for London to complete before predicting',
            'watch': 'Analyze London behavior first',
        }

    return predictions


async def fetch_session_data():
    """Fetch session data from TopStepX API."""
    try:
        # Add topstep-trading-bot to path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "topstep-trading-bot"))

        from src.api.client import TopstepXClient

        async with TopstepXClient() as client:
            # Get current MES contract
            contracts = await client.get_contracts("MES")
            if not contracts:
                print("No MES contracts found")
                return None

            contract_id = contracts[0].id
            print(f"  Using contract: {contract_id}")

            # Get today's date in ET
            now = datetime.now(ET)
            sessions = get_session_times(now)

            session_data = {}

            for session_name, (start, end) in sessions.items():
                # Skip future sessions
                if start > now:
                    print(f"  {session_name}: Not started yet")
                    continue

                # Adjust end time if session is ongoing
                actual_end = min(end, now)

                # Convert to UTC for API
                start_utc = start.astimezone(ZoneInfo('UTC')).replace(tzinfo=None)
                end_utc = actual_end.astimezone(ZoneInfo('UTC')).replace(tzinfo=None)

                print(f"  {session_name}: Fetching {start.strftime('%I:%M %p')} - {actual_end.strftime('%I:%M %p')} ET...")

                # Fetch bars
                bars = await client.get_historical_bars(
                    contract_id=contract_id,
                    start_time=start_utc,
                    end_time=end_utc,
                    unit=2,  # Minute
                    unit_number=5,  # 5-minute bars
                    limit=500,
                )

                if bars:
                    high = max(b.high for b in bars)
                    low = min(b.low for b in bars)
                    open_price = bars[0].open
                    close_price = bars[-1].close

                    session_data[session_name] = analyze_session_behavior(
                        high, low, open_price, close_price
                    )
                    print(f"    Got {len(bars)} bars: {low:.2f} - {high:.2f}")
                else:
                    print(f"    No bars returned")

            return session_data

    except ImportError as e:
        print(f"Could not import TopStepX client: {e}")
        print("Make sure you're running from the correct directory")
        return None
    except Exception as e:
        print(f"Could not fetch from API: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_session_analysis(session_data: Dict = None):
    """Print session analysis and AMD predictions."""
    current = get_current_session()
    now = datetime.now(ET)

    print("=" * 60)
    print("           SESSION & AMD CYCLE ANALYSIS")
    print("=" * 60)
    print(f"\n  Current Time: {now.strftime('%I:%M %p ET')} ({now.strftime('%A')})")
    print(f"  Current Session: {current}")

    if session_data:
        print("\n  SESSION DATA:")
        print("  " + "-" * 40)

        asia_behavior = None
        london_behavior = None

        for session, data in session_data.items():
            print(f"\n  {session}:")
            print(f"    Behavior: {data['behavior']}")
            print(f"    {data['description']}")
            print(f"    Range: {data['low']:.2f} - {data['high']:.2f} ({data['range']:.2f} pts)")

            if session == 'ASIA':
                asia_behavior = data['behavior']
            elif session == 'LONDON':
                london_behavior = data['behavior']

        # AMD Predictions
        if asia_behavior:
            print("\n  " + "=" * 40)
            print("  AMD CYCLE PREDICTION")
            print("  " + "=" * 40)

            predictions = predict_next_session(asia_behavior, london_behavior)

            for session, pred in predictions.items():
                if session == 'LONDON' and 'LONDON' in session_data:
                    continue  # Skip if London already happened
                if session == 'NY' and current in ['NY_AM', 'NY_PM', 'CLOSED']:
                    continue  # Skip if NY already happened

                print(f"\n  {session} Session:")
                print(f"    Expected: {pred['expected']}")
                print(f"    Action: {pred['action']}")
                print(f"    Watch: {pred['watch']}")
    else:
        print("\n  No session data available.")
        print("  Use --asia-high/--asia-low to input manually, or")
        print("  ensure TopStepX API credentials are configured.")

    # Manual input reminder
    print("\n  " + "-" * 40)
    print("  To input session levels manually:")
    print("    python scripts/smc_analysis.py --asia-high 6920 --asia-low 6900")
    print("=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Session & AMD Cycle Analysis')
    parser.add_argument('--asia-high', type=float, help='Asia session high')
    parser.add_argument('--asia-low', type=float, help='Asia session low')
    parser.add_argument('--asia-open', type=float, help='Asia session open')
    parser.add_argument('--asia-close', type=float, help='Asia session close')
    parser.add_argument('--london-high', type=float, help='London session high')
    parser.add_argument('--london-low', type=float, help='London session low')
    parser.add_argument('--fetch', action='store_true', help='Fetch from TopStepX API')
    args = parser.parse_args()

    session_data = None

    # Try to fetch from API
    if args.fetch:
        session_data = asyncio.run(fetch_session_data())

    # Or use manual input
    if not session_data and args.asia_high and args.asia_low:
        asia_open = args.asia_open or args.asia_low
        asia_close = args.asia_close or args.asia_high

        session_data = {
            'ASIA': analyze_session_behavior(
                args.asia_high, args.asia_low, asia_open, asia_close
            )
        }

        if args.london_high and args.london_low:
            session_data['LONDON'] = analyze_session_behavior(
                args.london_high, args.london_low,
                args.london_low, args.london_high  # Approximate
            )

    print_session_analysis(session_data)


if __name__ == '__main__':
    main()
