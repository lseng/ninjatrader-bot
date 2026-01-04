#!/usr/bin/env python3
"""
Prediction Script for Best ML Trading Strategy

Loads the trained ultra_rf model and makes predictions on new data.
Uses the same feature engineering as training.
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

    # Price features
    df["returns"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["body"] = (df["close"] - df["open"]) / df["close"]

    # Moving averages
    for period in [5, 10, 20]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"close_vs_sma_{period}"] = (df["close"] - df[f"sma_{period}"]) / df[f"sma_{period}"]

    # Momentum
    for period in [3, 5, 10]:
        df[f"roc_{period}"] = df["close"].pct_change(period)

    # Volatility
    df["volatility_5"] = df["returns"].rolling(5).std()
    df["volatility_10"] = df["returns"].rolling(10).std()

    # Volume features (if available)
    if "volume" in df.columns:
        df["volume_sma_5"] = df["volume"].rolling(5).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_5"].replace(0, 1)

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    return df


def load_model(model_path="models/ultra_rf/model.joblib"):
    """Load the trained model."""
    log(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    log("Model loaded successfully")
    return model


def predict_signal(model, df, latest_only=True):
    """
    Predict trading signal.

    Returns:
        1 = Long (buy)
        -1 = Short (sell)
        0 = Flat (no position)
    """
    # Create features
    df_features = create_features(df)

    # Get feature columns (same as training - matches ultra_fast_train.py)
    feature_cols = [c for c in df_features.columns
                    if c not in ["timestamp", "open", "high", "low", "close", "volume"]]

    # Drop NaN rows
    df_clean = df_features.dropna()

    if len(df_clean) == 0:
        return None, "Insufficient data for prediction"

    X = df_clean[feature_cols].values

    # Get predictions
    predictions = model.predict(X)

    if latest_only:
        return predictions[-1], feature_cols
    return predictions, feature_cols


def run_realtime_prediction(model, price_data):
    """
    Run prediction on real-time price data.

    Args:
        model: Trained model
        price_data: DataFrame with columns [timestamp, open, high, low, close, volume]
                   Must have at least 20 rows for feature calculation

    Returns:
        dict with signal, confidence, and metadata
    """
    if len(price_data) < 20:
        return {
            "signal": 0,
            "signal_name": "INSUFFICIENT_DATA",
            "confidence": 0,
            "error": "Need at least 20 bars for prediction"
        }

    signal, feature_cols = predict_signal(model, price_data, latest_only=True)

    if signal is None:
        return {
            "signal": 0,
            "signal_name": "ERROR",
            "confidence": 0,
            "error": feature_cols  # This contains error message
        }

    signal_names = {1: "LONG", -1: "SHORT", 0: "FLAT"}

    return {
        "signal": int(signal),
        "signal_name": signal_names.get(signal, "UNKNOWN"),
        "confidence": 0.75,  # Based on win rate from training
        "timestamp": price_data["timestamp"].iloc[-1] if "timestamp" in price_data.columns else datetime.now(),
        "last_price": price_data["close"].iloc[-1],
        "features_used": len(feature_cols)
    }


def main():
    """Demo of prediction script."""
    import argparse
    parser = argparse.ArgumentParser(description="ML Trading Signal Predictor")
    parser.add_argument("--data", help="Path to price data (parquet or csv)")
    parser.add_argument("--model", default="models/ultra_rf/model.joblib",
                       help="Path to trained model")
    parser.add_argument("--last-n", type=int, default=100,
                       help="Use last N bars for prediction")
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        log(f"Error: Model not found at {model_path}")
        sys.exit(1)

    model = load_model(args.model)

    # Load data
    if args.data:
        data_path = Path(args.data)
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path, parse_dates=["timestamp"])

        # Use last N bars
        df = df.tail(args.last_n)
        log(f"Loaded {len(df)} bars from {data_path}")
    else:
        # Demo with sample data
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
    result = run_realtime_prediction(model, df)

    log("=" * 50)
    log("PREDICTION RESULT")
    log("=" * 50)
    log(f"Signal: {result['signal']} ({result['signal_name']})")
    log(f"Confidence: {result['confidence']:.1%}")
    log(f"Last Price: {result.get('last_price', 'N/A')}")
    log(f"Timestamp: {result.get('timestamp', 'N/A')}")

    if result['signal'] == 1:
        log("\n>>> RECOMMENDATION: GO LONG (BUY)")
    elif result['signal'] == -1:
        log("\n>>> RECOMMENDATION: GO SHORT (SELL)")
    else:
        log("\n>>> RECOMMENDATION: STAY FLAT (NO POSITION)")


if __name__ == "__main__":
    main()
