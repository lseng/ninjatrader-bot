"""
Data Processor

Handles data cleaning, feature engineering, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from typing import Literal
import ta


class DataProcessor:
    """
    Process OHLCV data for ML model training.

    Features include:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Price action features
    - Volume analysis
    - Time-based features
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize processor with OHLCV data.

        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self.df = df.copy()
        self._validate_columns()

    def _validate_columns(self):
        """Validate required columns exist."""
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def add_technical_indicators(self) -> "DataProcessor":
        """Add common technical indicators."""
        df = self.df

        # Trend indicators
        df["sma_10"] = ta.trend.sma_indicator(df["close"], window=10)
        df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
        df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
        df["ema_10"] = ta.trend.ema_indicator(df["close"], window=10)
        df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)

        # MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # RSI
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        df["rsi_6"] = ta.momentum.rsi(df["close"], window=6)

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"], window=20)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()

        # ATR
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=14
        )

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"]
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # ADX
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        # CCI
        df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"])

        # Williams %R
        df["williams_r"] = ta.momentum.williams_r(
            df["high"], df["low"], df["close"]
        )

        self.df = df
        return self

    def add_price_action_features(self) -> "DataProcessor":
        """Add price action derived features."""
        df = self.df

        # Candle body and wick sizes
        df["body_size"] = abs(df["close"] - df["open"])
        df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
        df["candle_range"] = df["high"] - df["low"]

        # Body to range ratio
        df["body_range_ratio"] = df["body_size"] / df["candle_range"].replace(0, np.nan)

        # Bullish/Bearish
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)

        # Gap analysis
        df["gap_up"] = (df["open"] > df["high"].shift(1)).astype(int)
        df["gap_down"] = (df["open"] < df["low"].shift(1)).astype(int)
        df["gap_size"] = df["open"] - df["close"].shift(1)

        # Price position within range
        df["close_position"] = (df["close"] - df["low"]) / df["candle_range"].replace(0, np.nan)

        # Returns
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)
        df["return_20"] = df["close"].pct_change(20)

        # Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Volatility
        df["volatility_10"] = df["return_1"].rolling(10).std()
        df["volatility_20"] = df["return_1"].rolling(20).std()

        self.df = df
        return self

    def add_volume_features(self) -> "DataProcessor":
        """Add volume-based features."""
        df = self.df

        # Volume moving averages
        df["volume_sma_10"] = df["volume"].rolling(10).mean()
        df["volume_sma_20"] = df["volume"].rolling(20).mean()

        # Relative volume
        df["relative_volume"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

        # Volume trend
        df["volume_change"] = df["volume"].pct_change()

        # Price-volume correlation
        df["pv_correlation"] = df["close"].rolling(20).corr(df["volume"])

        # On-Balance Volume
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

        # Volume-Weighted Average Price (VWAP) - session based
        df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

        # Money Flow Index
        df["mfi"] = ta.volume.money_flow_index(
            df["high"], df["low"], df["close"], df["volume"]
        )

        self.df = df
        return self

    def add_time_features(self) -> "DataProcessor":
        """Add time-based features."""
        df = self.df

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Time components
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month

        # Session flags (US market hours in ET)
        df["is_market_open"] = (
            (df["hour"] >= 9) & (df["hour"] < 16) |
            ((df["hour"] == 9) & (df["minute"] >= 30))
        ).astype(int)

        # Pre/post market
        df["is_premarket"] = (
            (df["hour"] >= 4) & (df["hour"] < 9) |
            ((df["hour"] == 9) & (df["minute"] < 30))
        ).astype(int)

        df["is_afterhours"] = (
            (df["hour"] >= 16) & (df["hour"] < 20)
        ).astype(int)

        # Cyclical encoding for hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day of week cyclical
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        self.df = df
        return self

    def add_all_features(self) -> "DataProcessor":
        """Add all feature groups."""
        return (
            self.add_technical_indicators()
            .add_price_action_features()
            .add_volume_features()
            .add_time_features()
        )

    def create_labels(
        self,
        lookahead: int = 5,
        threshold: float = 0.001,
        label_type: Literal["binary", "ternary", "regression"] = "ternary"
    ) -> "DataProcessor":
        """
        Create target labels for ML training.

        Args:
            lookahead: Number of bars to look ahead
            threshold: Minimum price change for classification
            label_type: Type of labels to create
        """
        df = self.df

        # Future return
        df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1

        if label_type == "binary":
            # 1 = up, 0 = down
            df["label"] = (df["future_return"] > 0).astype(int)

        elif label_type == "ternary":
            # 1 = up, 0 = neutral, -1 = down
            df["label"] = np.where(
                df["future_return"] > threshold, 1,
                np.where(df["future_return"] < -threshold, -1, 0)
            )

        elif label_type == "regression":
            # Direct return prediction
            df["label"] = df["future_return"]

        self.df = df
        return self

    def get_features_and_labels(
        self,
        feature_columns: list[str] | None = None,
        dropna: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Get feature matrix and labels for ML training.

        Args:
            feature_columns: Specific columns to use as features
            dropna: Whether to drop rows with NaN values

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        df = self.df.copy()

        if dropna:
            df = df.dropna()

        if feature_columns is None:
            # Exclude non-feature columns
            exclude = ["timestamp", "label", "future_return", "symbol"]
            feature_columns = [c for c in df.columns if c not in exclude]

        X = df[feature_columns]
        y = df["label"] if "label" in df.columns else None

        return X, y

    def get_dataframe(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        return self.df
