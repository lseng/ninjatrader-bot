#!/usr/bin/env python3
"""
NinjaTrader Bot Runner

Main entry point for running the trading bot.

Usage:
    # Paper trading (simulation)
    python scripts/run_bot.py paper --capital 1000

    # Live trading (requires API credentials)
    python scripts/run_bot.py live --capital 1000

    # Backtest
    python scripts/run_bot.py backtest --data path/to/data.parquet
"""

import sys
from pathlib import Path
import argparse
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_paper(args):
    """Run paper trading simulation."""
    from paper_trade import PaperTrader, load_sample_data

    print("=" * 60)
    print("PAPER TRADING MODE")
    print("=" * 60)

    df = load_sample_data(args.data)

    trader = PaperTrader(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        max_contracts=args.max_contracts,
        speed=args.speed
    )

    trader.run(df)


def run_live(args):
    """Run live trading bot."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    # Check for credentials
    username = os.getenv("TRADOVATE_USERNAME")
    password = os.getenv("TRADOVATE_PASSWORD")

    if not username or not password:
        print("ERROR: Missing API credentials!")
        print("Please set TRADOVATE_USERNAME and TRADOVATE_PASSWORD in .env file")
        print()
        print("Example .env file:")
        print("  TRADOVATE_USERNAME=your_email@example.com")
        print("  TRADOVATE_PASSWORD=your_password")
        print("  TRADOVATE_DEMO=true")
        sys.exit(1)

    from config import TradingConfig, APIConfig
    from bot import TradingBot

    # Create config
    config = TradingConfig(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        max_contracts=args.max_contracts,
        symbol=args.symbol,
        demo_mode=args.demo
    )

    api_config = APIConfig()

    print("=" * 60)
    print("LIVE TRADING MODE")
    print(f"Demo: {args.demo}")
    print(f"Capital: ${args.capital:,.2f}")
    print(f"Symbol: {args.symbol}")
    print("=" * 60)

    if not args.demo:
        confirm = input("\nWARNING: Live trading with REAL money! Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)

    bot = TradingBot(config=config, api_config=api_config)
    asyncio.run(bot.run())


def run_backtest(args):
    """Run strategy backtest."""
    import pandas as pd
    from strategies.williams_fractals_strategy import WilliamsFractalsStrategy, backtest_strategy

    print("=" * 60)
    print("BACKTEST MODE")
    print("=" * 60)

    # Load data
    if args.data:
        df = pd.read_parquet(args.data)
    else:
        # Try default locations
        paths = [
            Path("/Users/leoneng/Downloads/ninjatrader-bot/data/historical/MES_5min.parquet"),
            Path("/Users/leoneng/Downloads/topstep-trading-bot/data/historical/MES/MES_1s_2years.parquet"),
        ]
        for p in paths:
            if p.exists():
                df = pd.read_parquet(p)
                break
        else:
            print("ERROR: No data file found. Please provide --data path")
            sys.exit(1)

    # Resample if needed
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    if len(df) > 500000:
        df = df.resample('5min').agg({
            'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last',
            'volume': 'sum'
        }).dropna()
    df = df.reset_index()

    print(f"Data: {len(df):,} bars")

    # Run backtest
    results = backtest_strategy(df)

    print()
    print("RESULTS:")
    print(f"  Total Return: {results['total_return']:.1f}%")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['win_rate']:.1f}%")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    print(f"  Long P&L: ${results['long_pnl']:,.2f}")
    print(f"  Short P&L: ${results['short_pnl']:,.2f}")
    print(f"  Final Capital: ${results['final_capital']:,.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="NinjaTrader Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading simulation
  python scripts/run_bot.py paper --capital 1000 --speed 100

  # Live demo trading
  python scripts/run_bot.py live --capital 1000 --demo

  # Live real trading (CAUTION!)
  python scripts/run_bot.py live --capital 1000 --no-demo

  # Backtest on historical data
  python scripts/run_bot.py backtest --data data/MES_5min.parquet
"""
    )

    subparsers = parser.add_subparsers(dest="mode", help="Trading mode")

    # Paper trading
    paper = subparsers.add_parser("paper", help="Paper trading simulation")
    paper.add_argument("--capital", type=float, default=1000, help="Starting capital")
    paper.add_argument("--risk", type=float, default=0.02, help="Risk per trade")
    paper.add_argument("--max-contracts", type=int, default=5, help="Max contracts")
    paper.add_argument("--speed", type=float, default=100, help="Simulation speed")
    paper.add_argument("--data", type=str, help="Path to data file")

    # Live trading
    live = subparsers.add_parser("live", help="Live trading")
    live.add_argument("--capital", type=float, default=1000, help="Starting capital")
    live.add_argument("--risk", type=float, default=0.02, help="Risk per trade")
    live.add_argument("--max-contracts", type=int, default=5, help="Max contracts")
    live.add_argument("--symbol", type=str, default="MESZ4", help="Symbol to trade")
    live.add_argument("--demo", action="store_true", default=True, help="Demo mode (default)")
    live.add_argument("--no-demo", dest="demo", action="store_false", help="Real money trading")

    # Backtest
    backtest = subparsers.add_parser("backtest", help="Strategy backtest")
    backtest.add_argument("--data", type=str, help="Path to data file")

    args = parser.parse_args()

    if args.mode == "paper":
        run_paper(args)
    elif args.mode == "live":
        run_live(args)
    elif args.mode == "backtest":
        run_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
