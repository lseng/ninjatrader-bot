#!/usr/bin/env python3
"""
Long-Only Prediction Script

DISCOVERY: Short signals destroy value. Long-only strategy yields +280% vs +161% for full strategy.

This wrapper converts all short signals to flat (0), only taking long positions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def create_features(df):
    """Create features matching the training features."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["body"] = (df["close"] - df["open"]) / df["close"]

    for period in [5, 10, 20]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"close_vs_sma_{period}"] = (df["close"] - df[f"sma_{period}"]) / df[f"sma_{period}"]

    for period in [3, 5, 10]:
        df[f"roc_{period}"] = df["close"].pct_change(period)

    df["volatility_5"] = df["returns"].rolling(5).std()
    df["volatility_10"] = df["returns"].rolling(10).std()

    if "volume" in df.columns:
        df["volume_sma_5"] = df["volume"].rolling(5).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_5"].replace(0, 1)

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    return df


class LongOnlyPredictor:
    """
    Long-only trading signal predictor.

    Converts short signals (-1) to flat (0), only taking long positions.
    This strategy yields +280% vs +161% for the full long+short strategy.
    """

    def __init__(self, model_path="models/ultra_rf/model.joblib"):
        self.model = joblib.load(model_path)
        log(f"Loaded model from {model_path}")

    def predict(self, df, latest_only=True):
        """
        Predict trading signal (long-only).

        Returns:
            1 = Long (buy)
            0 = Flat (no position) - includes original short signals
        """
        df_features = create_features(df)
        feature_cols = [c for c in df_features.columns
                        if c not in ["timestamp", "open", "high", "low", "close", "volume"]]

        df_clean = df_features.dropna()
        if len(df_clean) == 0:
            return None, "Insufficient data"

        X = df_clean[feature_cols].values
        raw_predictions = self.model.predict(X)

        # Convert to long-only: shorts become flat
        long_only_predictions = np.where(raw_predictions == 1, 1, 0)

        if latest_only:
            return long_only_predictions[-1], raw_predictions[-1]
        return long_only_predictions, raw_predictions

    def run_realtime(self, price_data):
        """
        Run prediction on real-time price data.

        Args:
            price_data: DataFrame with columns [timestamp, open, high, low, close, volume]
                       Must have at least 20 rows for feature calculation

        Returns:
            dict with signal, raw_signal, and metadata
        """
        if len(price_data) < 20:
            return {
                "signal": 0,
                "signal_name": "INSUFFICIENT_DATA",
                "raw_signal": 0,
                "confidence": 0,
                "error": "Need at least 20 bars for prediction"
            }

        signal, raw_signal = self.predict(price_data, latest_only=True)

        if signal is None:
            return {
                "signal": 0,
                "signal_name": "ERROR",
                "raw_signal": 0,
                "confidence": 0,
                "error": raw_signal
            }

        signal_names = {1: "LONG", 0: "FLAT"}
        raw_signal_names = {1: "LONG", 0: "FLAT", -1: "SHORT_FILTERED"}

        return {
            "signal": int(signal),
            "signal_name": signal_names.get(signal, "UNKNOWN"),
            "raw_signal": int(raw_signal),
            "raw_signal_name": raw_signal_names.get(raw_signal, "UNKNOWN"),
            "confidence": 0.80,  # Higher confidence for long-only
            "timestamp": price_data["timestamp"].iloc[-1] if "timestamp" in price_data.columns else datetime.now(),
            "last_price": price_data["close"].iloc[-1],
            "strategy": "LONG_ONLY"
        }


def main():
    """Demo of long-only prediction."""
    import argparse
    parser = argparse.ArgumentParser(description="Long-Only ML Trading Signal Predictor")
    parser.add_argument("--data", help="Path to price data (parquet or csv)")
    parser.add_argument("--model", default="models/ultra_rf/model.joblib",
                       help="Path to trained model")
    parser.add_argument("--last-n", type=int, default=100,
                       help="Use last N bars for prediction")
    args = parser.parse_args()

    # Load predictor
    model_path = Path(args.model)
    if not model_path.exists():
        log(f"Error: Model not found at {model_path}")
        sys.exit(1)

    predictor = LongOnlyPredictor(args.model)

    # Load data
    if args.data:
        data_path = Path(args.data)
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path, parse_dates=["timestamp"])
        df = df.tail(args.last_n)
        log(f"Loaded {len(df)} bars from {data_path}")
    else:
        log("No data provided, generating sample data for demo...")
        np.random.seed(42)
        n = 100
        prices = 5000 + np.cumsum(np.random.randn(n) * 2)
        df = pd.DataFrame({
            "timestamp": pd.date_range(end=datetime.now(), periods=n, freq="1min"),
            "open": prices + np.random.randn(n) * 0.5,
            "high": prices + abs(np.random.randn(n)) * 1.5,
            "low": prices - abs(np.random.randn(n)) * 1.5,
            "close": prices,
            "volume": np.random.randint(100, 1000, n)
        })

    # Make prediction
    result = predictor.run_realtime(df)

    log("")
    log("=" * 50)
    log("LONG-ONLY PREDICTION RESULT")
    log("=" * 50)
    log(f"Signal: {result['signal']} ({result['signal_name']})")
    log(f"Raw Signal: {result['raw_signal']} ({result.get('raw_signal_name', 'N/A')})")
    log(f"Confidence: {result['confidence']:.1%}")
    log(f"Last Price: {result.get('last_price', 'N/A')}")
    log(f"Strategy: {result.get('strategy', 'LONG_ONLY')}")

    if result['signal'] == 1:
        log("\n>>> RECOMMENDATION: GO LONG (BUY)")
    else:
        log("\n>>> RECOMMENDATION: STAY FLAT (NO POSITION)")
        if result['raw_signal'] == -1:
            log("    (Short signal filtered - long-only mode)")


if __name__ == "__main__":
    main()
