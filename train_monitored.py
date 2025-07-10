#!/usr/bin/env python3
"""
Simple monitored training script for Take 5 MuZero with basic freeze detection.
This script wraps the original training with minimal instrumentation to identify freezes.
"""

import sys
import os
import time
import threading
import logging
from datetime import datetime

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'take5bot'))

# Simple freeze detection logger
class FreezeDetectionLogger:
    """Lightweight freeze detection for training."""

    def __init__(self, freeze_threshold=180):  # 3 minutes
        self.start_time = time.time()
        self.last_activity = time.time()
        self.freeze_threshold = freeze_threshold
        self.monitoring = True

        # Setup logging
        log_dir = "training_debug"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"freeze_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger('freeze_detector')

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.log("Freeze detection started")

    def log(self, message):
        """Log a message and update activity timestamp."""
        self.last_activity = time.time()
        elapsed = self.last_activity - self.start_time
        self.logger.info(f"[{elapsed:6.1f}s] {message}")

    def _monitor_loop(self):
        """Background monitoring for freezes."""
        while self.monitoring:
            try:
                time.sleep(30)  # Check every 30 seconds

                current_time = time.time()
                time_since_activity = current_time - self.last_activity

                if time_since_activity > self.freeze_threshold:
                    self.logger.error("=" * 60)
                    self.logger.error(f"POTENTIAL FREEZE DETECTED!")
                    self.logger.error(f"No activity for {time_since_activity:.1f} seconds")
                    self.logger.error(f"Threshold: {self.freeze_threshold} seconds")

                    try:
                        import psutil
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        cpu_percent = process.cpu_percent()
                        self.logger.error(f"Process memory: {memory_mb:.1f}MB")
                        self.logger.error(f"Process CPU: {cpu_percent}%")
                    except:
                        self.logger.error("Could not get process stats")

                    self.logger.error("=" * 60)

                    # Reset to avoid spam
                    self.last_activity = current_time

            except Exception as e:
                self.logger.error(f"Monitor error: {e}")

    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        total_time = time.time() - self.start_time
        self.log(f"Training session ended (total: {total_time:.1f}s)")


def run_monitored_training():
    """Run training with freeze monitoring."""

    monitor = FreezeDetectionLogger()

    try:
        monitor.log("Starting Take 5 MuZero training")

        # Import and load config
        monitor.log("Loading configuration")
        try:
            from take5_unizero_config import main_config, create_config, max_env_step
            monitor.log(f"Config loaded - Max steps: {max_env_step}")
            monitor.log(f"CUDA: {main_config.policy.cuda}")
            monitor.log(f"Batch size: {main_config.policy.batch_size}")
            monitor.log(f"Collectors: {main_config.env.collector_env_num}")
            monitor.log(f"Evaluators: {main_config.env.evaluator_env_num}")
        except Exception as e:
            monitor.log(f"Config loading failed: {e}")
            raise

        # Import training function
        monitor.log("Importing training modules")
        try:
            from lzero.entry import train_muzero
            monitor.log("LZero modules imported successfully")
        except Exception as e:
            monitor.log(f"Import failed: {e}")
            raise

        # Add periodic logging during training
        monitor.log("Setting up training monitoring")

        # Simple approach: just log before and after train_muzero call
        monitor.log("Entering main training function")

        # Record start of training
        training_start = time.time()

        # Call the training function
        train_muzero(
            [main_config, create_config],
            seed=0,
            model_path=main_config.policy.model_path,
            max_env_step=max_env_step,
        )

        training_duration = time.time() - training_start
        monitor.log(f"Training completed successfully in {training_duration:.1f}s")

    except KeyboardInterrupt:
        monitor.log("Training interrupted by user (Ctrl+C)")

    except Exception as e:
        monitor.log(f"Training failed with error: {e}")
        monitor.log(f"Error type: {type(e).__name__}")

        # Try to log more details
        try:
            import traceback
            tb_lines = traceback.format_exc().split('\n')
            for line in tb_lines:
                if line.strip():
                    monitor.logger.error(f"TRACEBACK: {line}")
        except:
            pass

        raise

    finally:
        monitor.stop()


# Wrapper script that provides status updates
def main():
    """Main entry point with user-friendly output."""

    print("=" * 60)
    print("TAKE 5 MUZERO TRAINING WITH FREEZE MONITORING")
    print("=" * 60)
    print("This script will:")
    print("- Monitor training for freezes (3-minute threshold)")
    print("- Log detailed progress to training_debug/")
    print("- Detect if training gets stuck")
    print()
    print("Log files will be saved in: training_debug/")
    print("Press Ctrl+C to stop training")
    print("=" * 60)
    print()

    try:
        run_monitored_training()
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("Check training_debug/ for detailed logs")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("TRAINING STOPPED BY USER")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("TRAINING FAILED!")
        print(f"Error: {e}")
        print("Check training_debug/ for detailed error logs")
        print("=" * 60)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
