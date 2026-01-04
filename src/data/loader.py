"""
NinjaTrader Data Loader

Handles loading historical data exported from NinjaTrader 8.
NinjaTrader exports data in semicolon-separated format with specific column structures.

Supported formats:
- Minute bars: Date;Time;Open;High;Low;Close;Volume
- Tick data: Date;Time;Price;Volume
- Daily bars: Date;Open;High;Low;Close;Volume
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Literal
from pydantic import BaseModel
import pyarrow.parquet as pq


class BarData(BaseModel):
    """OHLCV bar data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class NinjaTraderDataLoader:
    """
    Load and parse NinjaTrader exported data files.

    NinjaTrader exports historical data in semicolon-separated .txt files.
    This loader handles the various formats NT8 can export.
    """

    # NinjaTrader uses semicolon as delimiter
    DELIMITER = ";"

    # Column mappings for different data types
    MINUTE_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]
    TICK_COLUMNS = ["date", "time", "price", "volume"]
    DAILY_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

    def __init__(self, data_path: str | Path):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the directory containing NT8 exported files
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

    def load_minute_bars(
        self,
        filename: str,
        symbol: str | None = None
    ) -> pd.DataFrame:
        """
        Load minute bar data from NinjaTrader export.

        Args:
            filename: Name of the exported file
            symbol: Optional symbol name to add as column

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        filepath = self.data_path / filename

        df = pd.read_csv(
            filepath,
            sep=self.DELIMITER,
            header=None,
            names=self.MINUTE_COLUMNS,
            parse_dates={"timestamp": ["date", "time"]},
            dtype={
                "open": np.float64,
                "high": np.float64,
                "low": np.float64,
                "close": np.float64,
                "volume": np.int64,
            }
        )

        df = df.sort_values("timestamp").reset_index(drop=True)

        if symbol:
            df["symbol"] = symbol

        return df

    def load_tick_data(
        self,
        filename: str,
        symbol: str | None = None
    ) -> pd.DataFrame:
        """
        Load tick data from NinjaTrader export.

        Args:
            filename: Name of the exported file
            symbol: Optional symbol name to add as column

        Returns:
            DataFrame with columns: timestamp, price, volume
        """
        filepath = self.data_path / filename

        df = pd.read_csv(
            filepath,
            sep=self.DELIMITER,
            header=None,
            names=self.TICK_COLUMNS,
            parse_dates={"timestamp": ["date", "time"]},
            dtype={
                "price": np.float64,
                "volume": np.int64,
            }
        )

        df = df.sort_values("timestamp").reset_index(drop=True)

        if symbol:
            df["symbol"] = symbol

        return df

    def load_daily_bars(
        self,
        filename: str,
        symbol: str | None = None
    ) -> pd.DataFrame:
        """
        Load daily bar data from NinjaTrader export.

        Args:
            filename: Name of the exported file
            symbol: Optional symbol name to add as column

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        filepath = self.data_path / filename

        df = pd.read_csv(
            filepath,
            sep=self.DELIMITER,
            header=None,
            names=self.DAILY_COLUMNS,
            parse_dates=["date"],
            dtype={
                "open": np.float64,
                "high": np.float64,
                "low": np.float64,
                "close": np.float64,
                "volume": np.int64,
            }
        )

        df = df.rename(columns={"date": "timestamp"})
        df = df.sort_values("timestamp").reset_index(drop=True)

        if symbol:
            df["symbol"] = symbol

        return df

    def load_auto_detect(
        self,
        filename: str,
        symbol: str | None = None
    ) -> pd.DataFrame:
        """
        Auto-detect file format and load data.

        Args:
            filename: Name of the exported file
            symbol: Optional symbol name to add as column

        Returns:
            DataFrame with parsed data
        """
        filepath = self.data_path / filename

        # Read first line to detect format
        with open(filepath, "r") as f:
            first_line = f.readline().strip()

        parts = first_line.split(self.DELIMITER)
        num_columns = len(parts)

        if num_columns == 7:
            return self.load_minute_bars(filename, symbol)
        elif num_columns == 6:
            return self.load_daily_bars(filename, symbol)
        elif num_columns == 4:
            return self.load_tick_data(filename, symbol)
        else:
            raise ValueError(f"Unknown file format with {num_columns} columns")

    def resample_bars(
        self,
        df: pd.DataFrame,
        timeframe: str = "5min"
    ) -> pd.DataFrame:
        """
        Resample bar data to a different timeframe.

        Args:
            df: DataFrame with OHLCV data
            timeframe: Target timeframe (e.g., '5min', '15min', '1H', '4H', '1D')

        Returns:
            Resampled DataFrame
        """
        df = df.set_index("timestamp")

        resampled = df.resample(timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        return resampled.reset_index()

    def load_parquet(
        self,
        filename: str,
        symbol: str | None = None
    ) -> pd.DataFrame:
        """
        Load data from parquet file.

        Args:
            filename: Name of the parquet file
            symbol: Optional symbol name to add as column

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        filepath = self.data_path / filename

        df = pd.read_parquet(filepath)

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]

        # Ensure timestamp is timezone-naive for consistency
        if "timestamp" in df.columns:
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

        df = df.sort_values("timestamp").reset_index(drop=True)

        if symbol:
            df["symbol"] = symbol

        return df

    def list_available_files(self) -> list[str]:
        """List all data files in the data directory."""
        txt_files = [f.name for f in self.data_path.glob("*.txt")]
        csv_files = [f.name for f in self.data_path.glob("*.csv")]
        parquet_files = [f.name for f in self.data_path.glob("*.parquet")]
        return txt_files + csv_files + parquet_files


def download_sample_data():
    """
    Download sample futures data for testing.
    Uses free data sources when available.
    """
    # This would integrate with free data sources
    # For now, we'll create synthetic data for testing
    print("Note: For real data, export from NinjaTrader 8:")
    print("  1. Open NinjaTrader 8")
    print("  2. Go to Tools > Historical Data")
    print("  3. Select your instrument and timeframe")
    print("  4. Click Export and save to ./data/historical/")
    print()
    print("Alternative: Use Tickstory for free tick data")
    print("  https://tickstory.com/")
