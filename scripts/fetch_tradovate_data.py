#!/usr/bin/env python3
"""
Fetch historical data from Tradovate API.

Works on Mac - no NinjaTrader desktop needed!

Usage:
    python scripts/fetch_tradovate_data.py --symbol MES --days 30
    python scripts/fetch_tradovate_data.py --symbol ES --days 90 --timeframe 5min
"""

import asyncio
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import httpx


class TradovateDataFetcher:
    """Fetch historical data from Tradovate API."""

    # Demo endpoints (use live for real data)
    DEMO_URL = "https://demo.tradovate.com/v1"
    LIVE_URL = "https://live.tradovate.com/v1"
    MD_URL = "https://md.tradovate.com/v1"

    def __init__(self, username: str, password: str, demo: bool = True):
        self.username = username
        self.password = password
        self.base_url = self.DEMO_URL if demo else self.LIVE_URL
        self.access_token = None
        self.client = None

    async def connect(self):
        """Authenticate with Tradovate."""
        self.client = httpx.AsyncClient(timeout=30.0)

        auth_url = f"{self.base_url}/auth/accesstokenrequest"
        response = await self.client.post(auth_url, json={
            "name": self.username,
            "password": self.password,
            "appId": "NinjaTraderBot",
            "appVersion": "1.0.0",
            "cid": 0,
            "sec": ""
        })

        if response.status_code != 200:
            raise Exception(f"Auth failed: {response.text}")

        data = response.json()
        self.access_token = data.get("accessToken")
        print(f"Authenticated successfully")
        return self

    async def disconnect(self):
        """Close connection."""
        if self.client:
            await self.client.aclose()

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    async def get_contract_id(self, symbol: str) -> int:
        """Get contract ID for a symbol."""
        response = await self.client.get(
            f"{self.base_url}/contract/find",
            params={"name": symbol},
            headers=self._headers()
        )

        if response.status_code != 200:
            raise Exception(f"Contract lookup failed: {response.text}")

        data = response.json()
        return data.get("id")

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1min",
        start_date: datetime = None,
        end_date: datetime = None
    ) -> list[dict]:
        """
        Fetch historical bars.

        Note: Tradovate limits historical data based on your subscription.
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        # Map timeframe
        tf_map = {
            "1min": {"elementSize": 1, "elementSizeUnit": "Minute"},
            "5min": {"elementSize": 5, "elementSizeUnit": "Minute"},
            "15min": {"elementSize": 15, "elementSizeUnit": "Minute"},
            "1hour": {"elementSize": 1, "elementSizeUnit": "Hour"},
            "1day": {"elementSize": 1, "elementSizeUnit": "Day"},
        }

        tf = tf_map.get(timeframe, tf_map["1min"])

        # Get contract ID
        contract_id = await self.get_contract_id(symbol)
        print(f"Contract ID for {symbol}: {contract_id}")

        # Request chart data
        response = await self.client.post(
            f"{self.MD_URL}/getChart",
            headers=self._headers(),
            json={
                "symbol": symbol,
                "chartDescription": {
                    "underlyingType": "MinuteBar",
                    "elementSize": tf["elementSize"],
                    "elementSizeUnit": tf["elementSizeUnit"],
                    "withHistogram": False
                },
                "timeRange": {
                    "asFarAsTimestamp": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "closestTimestamp": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
            }
        )

        if response.status_code != 200:
            print(f"Chart request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return []

        data = response.json()
        bars = data.get("bars", [])
        print(f"Fetched {len(bars)} bars")

        return bars

    async def fetch_and_save(
        self,
        symbol: str,
        days: int,
        timeframe: str,
        output_dir: Path
    ):
        """Fetch data and save to file."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        print(f"Fetching {symbol} {timeframe} data for last {days} days...")

        bars = await self.get_historical_bars(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if not bars:
            print("No data returned. Check your subscription level.")
            return

        # Convert to CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            f.write("timestamp,open,high,low,close,volume\n")
            for bar in bars:
                ts = bar.get("timestamp", "")
                o = bar.get("open", 0)
                h = bar.get("high", 0)
                l = bar.get("low", 0)
                c = bar.get("close", 0)
                v = bar.get("upVolume", 0) + bar.get("downVolume", 0)
                f.write(f"{ts},{o},{h},{l},{c},{v}\n")

        print(f"Saved to {filepath}")
        return filepath


async def main():
    parser = argparse.ArgumentParser(description="Fetch Tradovate historical data")
    parser.add_argument("--symbol", default="MESZ4", help="Contract symbol (e.g., MESZ4, ESZ4)")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--timeframe", default="1min", choices=["1min", "5min", "15min", "1hour", "1day"])
    parser.add_argument("--username", help="Tradovate username (or set TRADOVATE_USERNAME env)")
    parser.add_argument("--password", help="Tradovate password (or set TRADOVATE_PASSWORD env)")
    parser.add_argument("--demo", action="store_true", help="Use demo environment")
    parser.add_argument("--output", default="data/historical", help="Output directory")

    args = parser.parse_args()

    # Get credentials
    import os
    username = args.username or os.environ.get("TRADOVATE_USERNAME")
    password = args.password or os.environ.get("TRADOVATE_PASSWORD")

    if not username or not password:
        print("Error: Provide --username and --password or set environment variables")
        print("  TRADOVATE_USERNAME=your_email")
        print("  TRADOVATE_PASSWORD=your_password")
        return

    fetcher = TradovateDataFetcher(username, password, demo=args.demo)

    try:
        await fetcher.connect()
        await fetcher.fetch_and_save(
            symbol=args.symbol,
            days=args.days,
            timeframe=args.timeframe,
            output_dir=Path(args.output)
        )
    finally:
        await fetcher.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
