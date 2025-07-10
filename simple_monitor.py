#!/usr/bin/env python3
"""
Simple monitor for Take 5 MuZero training progress.
Designed for early stage training and CPU environments.
"""

import os
import time
import glob
from datetime import datetime
import subprocess
import sys

def check_training_process():
    """Check if training process is running."""
    try:
        # Check for python processes running the training script
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'take5_unizero_config.py' in line and 'python' in line:
                return True
        return False
    except:
        return False

def find_latest_training_dir():
    """Find the most recent training directory."""
    base_dir = "data_muzero"
    if not os.path.exists(base_dir):
        return None

    pattern = os.path.join(base_dir, "take5_muzero_*")
    dirs = glob.glob(pattern)

    if not dirs:
        return None

    return max(dirs, key=os.path.getmtime)

def get_basic_stats(training_dir):
    """Get basic training statistics."""
    if not training_dir or not os.path.exists(training_dir):
        return None

    stats = {
        'dir': training_dir,
        'started': datetime.fromtimestamp(os.path.getctime(training_dir)),
        'last_update': datetime.fromtimestamp(os.path.getmtime(training_dir)),
        'tensorboard_files': 0,
        'log_files': 0,
        'checkpoint_files': 0
    }

    # Count files
    for root, dirs, files in os.walk(training_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                stats['tensorboard_files'] += 1
            elif file.endswith('.log'):
                stats['log_files'] += 1
            elif file.endswith('.pth.tar'):
                stats['checkpoint_files'] += 1

    return stats

def get_wandb_info():
    """Try to get wandb run info."""
    try:
        import wandb
        # Check if there's an active run
        if wandb.run is not None:
            return {
                'run_name': wandb.run.name,
                'run_id': wandb.run.id,
                'url': wandb.run.url
            }
    except:
        pass
    return None

def monitor_once():
    """Single monitoring check."""
    print(f"\n{'='*50}")
    print(f"Take 5 Training Monitor - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")

    # Check if training process is running
    is_running = check_training_process()
    if is_running:
        print("🟢 Training process: RUNNING")
    else:
        print("🔴 Training process: NOT DETECTED")

    # Find training directory
    training_dir = find_latest_training_dir()
    if not training_dir:
        print("❌ No training directory found")
        print("   Start training: python take5bot/take5_unizero_config.py")
        return

    stats = get_basic_stats(training_dir)
    if not stats:
        print("❌ Could not read training directory")
        return

    print(f"📁 Training dir: {os.path.basename(stats['dir'])}")
    print(f"🕐 Started: {stats['started'].strftime('%H:%M:%S')}")
    print(f"🕑 Last update: {stats['last_update'].strftime('%H:%M:%S')}")

    # Calculate running time
    now = datetime.now()
    runtime = now - stats['started']
    hours = int(runtime.total_seconds() // 3600)
    minutes = int((runtime.total_seconds() % 3600) // 60)
    print(f"⏱️  Runtime: {hours}h {minutes}m")

    # File counts
    print(f"📊 Files: {stats['tensorboard_files']} TB, {stats['log_files']} logs, {stats['checkpoint_files']} checkpoints")

    # Check if recently active
    time_since_update = (now - stats['last_update']).total_seconds()
    if time_since_update < 300:  # 5 minutes
        print("✅ Recently active")
    elif time_since_update < 1800:  # 30 minutes
        print("⚠️  Possibly stalled")
    else:
        print("❌ Likely stopped")

    # WandB info
    wandb_info = get_wandb_info()
    if wandb_info:
        print(f"📈 WandB: {wandb_info['run_name']}")
        print(f"🔗 URL: {wandb_info['url']}")

    # Suggestions
    print(f"\n💡 Suggestions:")
    if not is_running:
        print("   • Start training: python take5bot/take5_unizero_config.py")
    if stats['checkpoint_files'] == 0 and runtime.total_seconds() > 1800:  # 30 min
        print("   • No checkpoints after 30min - check for errors")
    if time_since_update > 1800:
        print("   • Training may have crashed - check terminal")

    print(f"{'='*50}")

def monitor_continuous(interval=60):
    """Continuous monitoring."""
    print("Starting continuous monitoring...")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop")

    try:
        while True:
            # Clear screen on Unix systems
            if os.name == 'posix':
                os.system('clear')

            monitor_once()

            print(f"\nNext update in {interval} seconds...")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        interval = 60
        if len(sys.argv) > 2:
            try:
                interval = int(sys.argv[2])
            except ValueError:
                print("Invalid interval, using 60 seconds")
        monitor_continuous(interval)
    else:
        monitor_once()
        print("\nFor continuous monitoring: python simple_monitor.py --continuous [seconds]")

if __name__ == "__main__":
    main()
