#!/usr/bin/env python3
"""Check and report training progress."""
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

def main():
    print(f"\n{'='*60}")
    print(f"Training Progress Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Check if process is running
    result = subprocess.run(
        "ps aux | grep master_train | grep -v grep",
        shell=True, capture_output=True, text=True
    )

    if "master_train" in result.stdout:
        parts = result.stdout.split()
        print(f"Status: RUNNING")
        if len(parts) > 3:
            print(f"CPU: {parts[2]}%")
            print(f"Memory: {parts[3]}%")
    else:
        print("Status: STOPPED")

    # Check log file
    log_path = Path("training_local.log")
    if log_path.exists():
        with open(log_path, 'r') as f:
            lines = f.readlines()

        # Find last progress line
        for line in reversed(lines):
            if "Best trial:" in line or "%" in line or "EXPERIMENT:" in line:
                print(f"Last progress: {line.strip()}")
                break

    # Check for completed experiments
    models_dir = Path("models")
    if models_dir.exists():
        print("\nCompleted Experiments:")
        for exp_dir in sorted(models_dir.glob("exp*")):
            results_file = exp_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)
                metrics = results.get("holdout_metrics", {})
                print(f"  {exp_dir.name}:")
                print(f"    P&L: {metrics.get('total_pnl_pct', 0):.2f}%")
                print(f"    Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
                print(f"    Sharpe: {metrics.get('sharpe', 0):.2f}")
    else:
        print("\nNo completed experiments yet.")

    # Check for master summary
    summary_file = models_dir / "master_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"\nMaster Summary:")
        print(f"  Total Experiments: {summary.get('total_experiments', 0)}")
        best = summary.get("best_experiment", {})
        if best:
            print(f"  Best Experiment: {best.get('experiment_name', 'N/A')}")
            metrics = best.get("holdout_metrics", {})
            print(f"  Best P&L: {metrics.get('total_pnl_pct', 0):.2f}%")

if __name__ == "__main__":
    main()
