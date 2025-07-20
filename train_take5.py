#!/usr/bin/env python3
"""
Simple monitored training script for Take 5 with basic freeze detection.
This script wraps the original training with minimal instrumentation to identify freezes.
"""

import sys
import os
import time
import logging
from datetime import datetime

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "take5bot"))


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

        log_file = os.path.join(
            log_dir, f"freeze_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

        self.logger = logging.getLogger("take5_training")

    def log(self, message):
        """Log a message and update activity timestamp."""
        self.last_activity = time.time()
        elapsed = self.last_activity - self.start_time
        self.logger.info(f"[{elapsed:6.1f}s] {message}")

    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        total_time = time.time() - self.start_time
        self.log(f"Training session ended (total: {total_time:.1f}s)")


def run_monitored_training():
    """Run training with freeze monitoring."""

    monitor = FreezeDetectionLogger()

    try:
        monitor.log("Starting Take 5 training")

        # Import and load config
        monitor.log("Loading configuration")
        try:
            from take5_muzero_config import main_config, create_config, max_env_step  # type: ignore

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
            from lzero.entry import train_muzero  # type: ignore

            monitor.log("LZero modules imported successfully")
        except Exception as e:
            monitor.log(f"Import failed: {e}")
            raise

        monitor.log("Entering main training function")

        # Record start of training
        training_start = time.time()

        # Call the training function
        train_muzero(
            [main_config, create_config],  # type: ignore
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

            tb_lines = traceback.format_exc().split("\n")
            for line in tb_lines:
                if line.strip():
                    monitor.logger.error(f"TRACEBACK: {line}")
        except:  # noqa: E722
            pass

        raise

    finally:
        monitor.stop()


def main():
    """Main entry point for the training script."""
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
