#!/usr/bin/env python3
"""
Simple training status checker for Take 5 MuZero training.
Quick way to check if training is running and get basic stats.
"""

import os
import glob
import time
from datetime import datetime


def find_training_dir():
    """Find the current training directory."""
    base_dir = "data_muzero"
    if not os.path.exists(base_dir):
        return None

    # Look for directories matching the pattern
    pattern = os.path.join(base_dir, "take5_muzero_*")
    dirs = glob.glob(pattern)

    if not dirs:
        return None

    # Return the most recently modified directory
    return max(dirs, key=os.path.getmtime)


def get_training_status(training_dir):
    """Get basic training status information."""
    if not training_dir or not os.path.exists(training_dir):
        return {"status": "not_found", "message": "No training directory found"}

    # Check if training is active (directory modified recently)
    last_modified = os.path.getmtime(training_dir)
    time_since_update = time.time() - last_modified

    # Consider training active if updated within last 5 minutes
    is_active = time_since_update < 300

    # Find checkpoint files
    checkpoint_files = glob.glob(os.path.join(training_dir, "**", "*.pth.tar"), recursive=True)

    # Find log files
    log_files = glob.glob(os.path.join(training_dir, "**", "*.log"), recursive=True)

    # Find tensorboard files
    tb_files = glob.glob(os.path.join(training_dir, "**", "events.out.tfevents.*"), recursive=True)

    status = {
        "status": "active" if is_active else "inactive",
        "directory": training_dir,
        "last_update": datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S"),
        "minutes_since_update": int(time_since_update / 60),
        "checkpoints": len(checkpoint_files),
        "log_files": len(log_files),
        "tensorboard_files": len(tb_files)
    }

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        status["latest_checkpoint"] = os.path.basename(latest_checkpoint)
        status["checkpoint_size_mb"] = round(os.path.getsize(latest_checkpoint) / (1024*1024), 1)

    return status


def estimate_training_progress(training_dir):
    """Estimate training progress based on file patterns."""
    if not training_dir:
        return None

    # Look for step information in filenames
    checkpoint_files = glob.glob(os.path.join(training_dir, "**", "*step*.pth.tar"), recursive=True)

    if not checkpoint_files:
        return None

    # Extract step numbers from filenames
    steps = []
    for file in checkpoint_files:
        filename = os.path.basename(file)
        # Try to extract step number from filename
        import re
        match = re.search(r'step_(\d+)', filename)
        if match:
            steps.append(int(match.group(1)))

    if not steps:
        return None

    max_steps = max(steps)
    target_steps = 200000  # From config

    return {
        "current_steps": max_steps,
        "target_steps": target_steps,
        "progress_percent": round((max_steps / target_steps) * 100, 1)
    }


def print_status():
    """Print training status in a readable format."""
    print("="*60)
    print("TAKE 5 MUZERO TRAINING STATUS")
    print("="*60)

    training_dir = find_training_dir()
    status = get_training_status(training_dir)

    if status["status"] == "not_found":
        print("❌ No training found")
        print("   Start training with: uv python ./take5bot/take5_unizero_config.py")
        return

    # Print basic status
    if status["status"] == "active":
        print("✅ Training is ACTIVE")
    else:
        print("⚠️  Training appears INACTIVE")

    print(f"📁 Directory: {status['directory']}")
    print(f"🕒 Last update: {status['last_update']} ({status['minutes_since_update']} minutes ago)")

    # Print file counts
    print(f"💾 Checkpoints: {status['checkpoints']}")
    if status.get('latest_checkpoint'):
        print(f"   Latest: {status['latest_checkpoint']} ({status['checkpoint_size_mb']} MB)")

    print(f"📊 Log files: {status['log_files']}")
    print(f"📈 Tensorboard files: {status['tensorboard_files']}")

    # Print progress estimate
    progress = estimate_training_progress(training_dir)
    if progress:
        print(f"🎯 Progress: {progress['current_steps']:,} / {progress['target_steps']:,} steps ({progress['progress_percent']}%)")

        # Estimate time remaining
        if progress['current_steps'] > 0 and status['minutes_since_update'] < 60:
            remaining_steps = progress['target_steps'] - progress['current_steps']
            if remaining_steps > 0:
                # Rough estimate based on typical training speed
                estimated_hours = remaining_steps / 1000  # Very rough estimate
                print(f"⏱️  Estimated time remaining: ~{estimated_hours:.1f} hours")

    print("="*60)

    # Print helpful commands
    print("\nHELPFUL COMMANDS:")
    print("  Monitor training:  uv python monitor_training.py --monitor")
    print("  View plots:        uv python monitor_training.py --plot")
    print("  Play with AI:      uv python play_take5.py")
    print("  Check status:      uv python check_training.py")

    if status["status"] == "active":
        print("\n💡 Training is running! Check back later for updates.")
    else:
        print("\n💡 Training may have finished or stopped. Check the logs for details.")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Check Take 5 MuZero training status")
    parser.add_argument("--watch", action="store_true", help="Watch status continuously")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds for watch mode")

    args = parser.parse_args()

    if args.watch:
        print("Watching training status (Press Ctrl+C to stop)")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
                print_status()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped watching.")
    else:
        print_status()


if __name__ == "__main__":
    main()
