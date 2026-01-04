#!/usr/bin/env python3
"""
Analyze feature importance and identify which features contribute most to predictions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def create_features(df):
    """Create features matching training."""
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


def main():
    log("=" * 70)
    log("FEATURE IMPORTANCE ANALYSIS")
    log("=" * 70)

    # Load model
    model_path = "models/shallow_rf/model.joblib"
    log(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # Load and prep data at 30m timeframe (optimal)
    log("Loading data at 30m timeframe...")
    df = pd.read_parquet("data/historical/MES_1m.parquet")
    df = df.iloc[::30].reset_index(drop=True)  # 30m timeframe
    log(f"Bars: {len(df):,}")

    df = create_features(df)
    df = df.dropna()

    feature_cols = [c for c in df.columns
                    if c not in ["timestamp", "open", "high", "low", "close", "volume"]]

    log(f"Features: {len(feature_cols)}")

    # Get feature importances from model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        log("Model does not have feature_importances_ attribute")
        return

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    log("\n" + "=" * 70)
    log("FEATURE IMPORTANCE RANKING")
    log("=" * 70)

    for idx, row in importance_df.iterrows():
        bar = "█" * int(row['importance'] * 50)
        log(f"{row['feature']:>20}: {row['importance']:.4f} {bar}")

    # Identify weak features (< 1% importance)
    weak_features = importance_df[importance_df['importance'] < 0.01]['feature'].tolist()
    strong_features = importance_df[importance_df['importance'] >= 0.05]['feature'].tolist()

    log("\n" + "=" * 70)
    log("ANALYSIS SUMMARY")
    log("=" * 70)

    log(f"\nTotal features: {len(feature_cols)}")
    log(f"Strong features (>=5%): {len(strong_features)}")
    log(f"Weak features (<1%): {len(weak_features)}")

    log(f"\nSTRONG features (keep):")
    for f in strong_features:
        imp = importance_df[importance_df['feature'] == f]['importance'].values[0]
        log(f"  {f}: {imp:.4f}")

    log(f"\nWEAK features (consider removing):")
    for f in weak_features:
        imp = importance_df[importance_df['feature'] == f]['importance'].values[0]
        log(f"  {f}: {imp:.4f}")

    # Cumulative importance
    cumsum = importance_df['importance'].cumsum()
    n_80pct = (cumsum <= 0.80).sum() + 1
    n_90pct = (cumsum <= 0.90).sum() + 1

    log(f"\nCumulative importance:")
    log(f"  Top {n_80pct} features explain 80% of predictions")
    log(f"  Top {n_90pct} features explain 90% of predictions")

    # Save analysis
    output = {
        "feature_importance": importance_df.to_dict('records'),
        "strong_features": strong_features,
        "weak_features": weak_features,
        "top_80pct_features": importance_df.head(n_80pct)['feature'].tolist(),
        "analysis_time": datetime.now().isoformat()
    }

    output_path = Path("models/feature_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    log(f"\nSaved to {output_path}")

    return importance_df


if __name__ == "__main__":
    main()
