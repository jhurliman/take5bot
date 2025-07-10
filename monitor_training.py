#!/usr/bin/env python3
"""
Training monitoring script for Take 5 MuZero training.
This script helps track training progress and provides insights into model performance.
"""

import os
import time
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def find_latest_log_dir(base_dir="data_muzero"):
    """Find the most recent training log directory."""
    if not os.path.exists(base_dir):
        return None

    # Look for directories matching the pattern
    pattern = os.path.join(base_dir, "take5_muzero_*")
    dirs = glob.glob(pattern)

    if not dirs:
        return None

    # Return the most recently modified directory
    return max(dirs, key=os.path.getmtime)

def parse_tensorboard_logs(log_dir):
    """Parse tensorboard logs to extract training metrics."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        # Find tensorboard log files
        tb_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)

        if not tb_files:
            return None

        # Use the most recent tensorboard file
        tb_file = max(tb_files, key=os.path.getmtime)

        # Load the tensorboard data
        ea = EventAccumulator(tb_file)
        ea.Reload()

        metrics = {}

        # Extract common metrics
        scalar_tags = ea.Tags()['scalars']

        print(ea.Tags())

        for tag in scalar_tags:
            if any(keyword in tag.lower() for keyword in ['reward', 'loss', 'value', 'policy', 'entropy']):
                scalar_events = ea.Scalars(tag)
                metrics[tag] = {
                    'steps': [s.step for s in scalar_events],
                    'values': [s.value for s in scalar_events]
                }

        return metrics

    except ImportError:
        print("Warning: tensorboard not available. Install with: pip install tensorboard")
        return None
    except Exception as e:
        print(f"Error parsing tensorboard logs: {e}")
        return None

def plot_training_progress(metrics, save_dir=None):
    """Plot training progress from metrics."""
    if not metrics:
        print("No metrics to plot")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Take 5 MuZero Training Progress', fontsize=16)

    # Plot reward
    reward_metrics = [k for k in metrics.keys() if 'reward' in k.lower()]
    if reward_metrics:
        ax = axes[0, 0]
        for metric in reward_metrics:
            ax.plot(metrics[metric]['steps'], metrics[metric]['values'], label=metric)
        ax.set_title('Reward Progress')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True)

    # Plot loss
    loss_metrics = [k for k in metrics.keys() if 'loss' in k.lower()]
    if loss_metrics:
        ax = axes[0, 1]
        for metric in loss_metrics:
            ax.plot(metrics[metric]['steps'], metrics[metric]['values'], label=metric)
        ax.set_title('Loss Progress')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

    # Plot value metrics
    value_metrics = [k for k in metrics.keys() if 'value' in k.lower() and 'loss' not in k.lower()]
    if value_metrics:
        ax = axes[1, 0]
        for metric in value_metrics:
            ax.plot(metrics[metric]['steps'], metrics[metric]['values'], label=metric)
        ax.set_title('Value Metrics')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    # Plot policy metrics
    policy_metrics = [k for k in metrics.keys() if 'policy' in k.lower() or 'entropy' in k.lower()]
    if policy_metrics:
        ax = axes[1, 1]
        for metric in policy_metrics:
            ax.plot(metrics[metric]['steps'], metrics[metric]['values'], label=metric)
        ax.set_title('Policy Metrics')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')

    plt.show()

def monitor_training(log_dir, refresh_interval=30):
    """Monitor training progress in real-time."""
    print(f"Monitoring training in: {log_dir}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")

    last_update = 0

    try:
        while True:
            # Check if log directory still exists and is being updated
            if not os.path.exists(log_dir):
                print(f"Log directory {log_dir} no longer exists. Stopping monitoring.")
                break

            # Get latest modification time
            try:
                current_update = os.path.getmtime(log_dir)

                if current_update > last_update:
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training activity detected")

                    # Look for checkpoint files
                    checkpoint_files = glob.glob(os.path.join(log_dir, "**", "*.pth.tar"), recursive=True)
                    if checkpoint_files:
                        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                        print(f"Latest checkpoint: {os.path.basename(latest_checkpoint)}")

                    # Look for evaluation results
                    eval_files = glob.glob(os.path.join(log_dir, "**", "*eval*"), recursive=True)
                    if eval_files:
                        print(f"Evaluation files found: {len(eval_files)}")

                    # Parse and display current metrics
                    metrics = parse_tensorboard_logs(log_dir)
                    if metrics:
                        print("\nCurrent metrics:")
                        for metric_name, data in metrics.items():
                            if data['values']:
                                latest_value = data['values'][-1]
                                latest_step = data['steps'][-1]
                                print(f"  {metric_name}: {latest_value:.4f} (step {latest_step})")

                    last_update = current_update
                else:
                    print(".", end="", flush=True)

            except OSError:
                print("Error accessing log directory")
                break

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

def print_training_summary(log_dir):
    """Print a summary of training progress."""
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY: {log_dir}")
    print(f"{'='*60}")

    # Check for checkpoint files
    checkpoint_files = glob.glob(os.path.join(log_dir, "**", "*.pth.tar"), recursive=True)
    if checkpoint_files:
        print(f"\nCheckpoints found: {len(checkpoint_files)}")
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"Latest checkpoint: {os.path.basename(latest_checkpoint)}")
        print(f"Checkpoint size: {os.path.getsize(latest_checkpoint) / (1024*1024):.1f} MB")

    # Parse metrics
    metrics = parse_tensorboard_logs(log_dir)
    if metrics:
        print("\nTraining Metrics:")
        for metric_name, data in metrics.items():
            if data['values']:
                latest_value = data['values'][-1]
                latest_step = data['steps'][-1]
                min_value = min(data['values'])
                max_value = max(data['values'])
                print(f"  {metric_name}:")
                print(f"    Latest: {latest_value:.4f} (step {latest_step})")
                print(f"    Range: [{min_value:.4f}, {max_value:.4f}]")

    print(f"\n{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Monitor Take 5 MuZero training progress")
    parser.add_argument("--log-dir", type=str, help="Path to training log directory")
    parser.add_argument("--monitor", action="store_true", help="Monitor training in real-time")
    parser.add_argument("--plot", action="store_true", help="Generate training progress plots")
    parser.add_argument("--summary", action="store_true", help="Print training summary")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval for monitoring (seconds)")

    args = parser.parse_args()

    # Find log directory
    log_dir = args.log_dir
    if not log_dir:
        log_dir = find_latest_log_dir()
        if not log_dir:
            print("No training log directory found. Make sure training has started.")
            return

    if not os.path.exists(log_dir):
        print(f"Log directory does not exist: {log_dir}")
        return

    print(f"Using log directory: {log_dir}")

    # Execute requested actions
    if args.monitor:
        monitor_training(log_dir, args.refresh)
    elif args.plot:
        metrics = parse_tensorboard_logs(log_dir)
        plot_training_progress(metrics, log_dir)
    elif args.summary:
        print_training_summary(log_dir)
    else:
        # Default: show summary and offer to monitor
        print_training_summary(log_dir)

        response = input("\nWould you like to monitor training in real-time? (y/n): ")
        if response.lower().startswith('y'):
            monitor_training(log_dir, args.refresh)

if __name__ == "__main__":
    main()
