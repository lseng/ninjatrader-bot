#!/usr/bin/env python3
"""Monitor training progress on RunPod."""
import subprocess
import sys
import time
from datetime import datetime

SSH_CMD = "ssh -o StrictHostKeyChecking=no root@157.157.221.30 -p 34592 -i ~/.ssh/id_ed25519"

def run_ssh(cmd):
    full_cmd = f'{SSH_CMD} "{cmd}"'
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def check_status():
    print(f"\n{'='*60}")
    print(f"Training Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Check if process is running
    ps_output = run_ssh("ps aux | grep 'master_train' | grep -v grep")
    if "master_train" in ps_output:
        print("Status: RUNNING")
        # Get CPU usage
        cpu_info = ps_output.split()
        if len(cpu_info) > 2:
            print(f"CPU Usage: {cpu_info[2]}%")
            print(f"Memory Usage: {cpu_info[3]}%")
    else:
        print("Status: STOPPED")

    # Get log tail
    print("\nRecent Log:")
    log_output = run_ssh("tail -30 /workspace/ninjatrader-bot/training.log")
    print(log_output)

    # Check for results
    print("\nExperiment Results:")
    results_output = run_ssh("ls -la /workspace/ninjatrader-bot/models/*/results.json 2>/dev/null | head -10")
    if results_output.strip():
        print(results_output)
    else:
        print("No completed experiments yet.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        while True:
            check_status()
            time.sleep(300)  # Check every 5 minutes
    else:
        check_status()
