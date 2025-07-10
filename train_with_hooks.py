#!/usr/bin/env python3
"""
Instrumented Take 5 MuZero training script with comprehensive debugging hooks.
This script wraps the original training with detailed logging to identify freeze points.
"""

import sys
import os
import time
import threading
import psutil
import logging
from datetime import datetime
from easydict import EasyDict

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the original config directly
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'take5bot'))
from take5_unizero_config import main_config, create_config, max_env_step


class SimpleTrainingLogger:
    """Simple logger with freeze detection for training."""

    def __init__(self):
        self.start_time = time.time()
        self.last_activity = time.time()
        self.current_stage = "initialization"
        self.iteration_count = 0

        # Setup logging
        log_dir = "training_debug"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)8s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

        # Start watchdog
        self.monitoring = True
        self.watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog.start()

        self.logger.info("=" * 60)
        self.logger.info("TAKE 5 MUZERO TRAINING STARTED")
        self.logger.info("=" * 60)

    def _watchdog_loop(self):
        """Background watchdog to detect freezes."""
        freeze_threshold = 120  # 2 minutes

        while self.monitoring:
            try:
                time.sleep(30)  # Check every 30 seconds

                current_time = time.time()
                time_since_activity = current_time - self.last_activity

                if time_since_activity > freeze_threshold:
                    self.logger.error("!" * 60)
                    self.logger.error(f"POTENTIAL FREEZE DETECTED!")
                    self.logger.error(f"Time since last activity: {time_since_activity:.1f} seconds")
                    self.logger.error(f"Current stage: {self.current_stage}")
                    self.logger.error(f"Iteration: {self.iteration_count}")

                    # Log system stats
                    try:
                        memory = psutil.virtual_memory()
                        process = psutil.Process()
                        self.logger.error(f"Memory: {memory.percent}% used")
                        self.logger.error(f"Process memory: {process.memory_info().rss / 1024**2:.1f}MB")
                        self.logger.error(f"Process CPU: {process.cpu_percent()}%")
                        self.logger.error(f"Threads: {process.num_threads()}")
                    except:
                        pass

                    self.logger.error("!" * 60)

                    # Reset to avoid spam
                    self.last_activity = current_time

            except Exception as e:
                self.logger.error(f"Watchdog error: {e}")

    def log_stage(self, stage: str, details: str = ""):
        """Log current training stage."""
        self.current_stage = stage
        self.last_activity = time.time()

        elapsed = time.time() - self.start_time
        self.logger.info(f"[{elapsed:7.1f}s] {stage.upper()}: {details}")

    def log_iteration(self, iteration: int, details: str = ""):
        """Log training iteration."""
        self.iteration_count = iteration
        self.last_activity = time.time()

        if iteration % 10 == 0 or details:
            elapsed = time.time() - self.start_time
            self.logger.info(f"[{elapsed:7.1f}s] ITER {iteration:4d}: {details}")

    def log_error(self, error: str):
        """Log error message."""
        self.last_activity = time.time()
        self.logger.error(f"ERROR: {error}")

    def log_info(self, message: str):
        """Log info message."""
        self.last_activity = time.time()
        self.logger.info(message)

    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        total_time = time.time() - self.start_time
        self.logger.info("=" * 60)
        self.logger.info(f"TRAINING SESSION ENDED (total time: {total_time:.1f}s)")
        self.logger.info("=" * 60)


def instrumented_train_muzero():
    """Train MuZero with comprehensive instrumentation."""

    logger = SimpleTrainingLogger()

    try:
        # Log configuration
        logger.log_stage("configuration", "Loading training configuration")
        logger.log_info(f"Max env steps: {max_env_step}")
        logger.log_info(f"CUDA enabled: {main_config.policy.cuda}")
        logger.log_info(f"Batch size: {main_config.policy.batch_size}")
        logger.log_info(f"Collector envs: {main_config.env.collector_env_num}")
        logger.log_info(f"Evaluator envs: {main_config.env.evaluator_env_num}")
        logger.log_info(f"MCTS simulations: {main_config.policy.num_simulations}")

        # Import training modules
        logger.log_stage("imports", "Importing LZero training modules")
        from lzero.entry import train_muzero
        from lzero.worker import MuZeroCollector, MuZeroEvaluator
        from lzero.policy import MuZeroPolicy

        logger.log_stage("env_setup", "Setting up environment")

        # Monkey patch key classes to add logging
        logger.log_stage("instrumentation", "Adding instrumentation hooks")

        # Patch MuZeroCollector
        original_collect = MuZeroCollector.collect
        def logged_collect(self, *args, **kwargs):
            logger.log_iteration(getattr(self, '_iter_count', 0), "Starting data collection")
            try:
                result = original_collect(self, *args, **kwargs)
                logger.log_iteration(getattr(self, '_iter_count', 0), "Data collection completed")
                return result
            except Exception as e:
                logger.log_error(f"Collection failed: {e}")
                raise

        MuZeroCollector.collect = logged_collect

        # Patch MuZeroEvaluator
        original_eval = MuZeroEvaluator.eval
        def logged_eval(self, *args, **kwargs):
            logger.log_iteration(getattr(self, '_iter_count', 0), "Starting evaluation")
            try:
                result = original_eval(self, *args, **kwargs)
                logger.log_iteration(getattr(self, '_iter_count', 0), "Evaluation completed")
                return result
            except Exception as e:
                logger.log_error(f"Evaluation failed: {e}")
                raise

        MuZeroEvaluator.eval = logged_eval

        # Patch MuZeroPolicy
        original_learn = MuZeroPolicy.learn
        def logged_learn(self, *args, **kwargs):
            iteration = getattr(self, '_iter_count', 0)
            logger.log_iteration(iteration, "Starting model learning")
            try:
                result = original_learn(self, *args, **kwargs)
                logger.log_iteration(iteration, "Model learning completed")
                return result
            except Exception as e:
                logger.log_error(f"Learning failed: {e}")
                raise

        MuZeroPolicy.learn = logged_learn

        # Start main training
        logger.log_stage("training_start", "Entering main training loop")

        # Track iterations in training
        global_iter_count = [0]

        def increment_iter():
            global_iter_count[0] += 1
            return global_iter_count[0]

        # Override some internal methods to track progress
        import ding.framework
        original_task_step = ding.framework.task.Task.step if hasattr(ding.framework.task, 'Task') else None

        if original_task_step:
            def logged_task_step(self, *args, **kwargs):
                iter_num = increment_iter()
                logger.log_iteration(iter_num, "Task step starting")
                try:
                    result = original_task_step(self, *args, **kwargs)
                    logger.log_iteration(iter_num, "Task step completed")
                    return result
                except Exception as e:
                    logger.log_error(f"Task step failed: {e}")
                    raise

            ding.framework.task.Task.step = logged_task_step

        # Call the actual training function
        logger.log_stage("muzero_entry", "Calling train_muzero")

        train_muzero(
            [main_config, create_config],
            seed=0,
            model_path=main_config.policy.model_path,
            max_env_step=max_env_step,
        )

        logger.log_stage("training_complete", "Training completed successfully")

    except KeyboardInterrupt:
        logger.log_info("Training interrupted by user")
    except Exception as e:
        logger.log_error(f"Training failed: {str(e)}")
        logger.log_error(f"Exception type: {type(e).__name__}")

        # Log traceback
        import traceback
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                logger.log_error(f"  {line}")

        raise
    finally:
        logger.stop()


if __name__ == "__main__":
    print("Starting Take 5 MuZero training with comprehensive debugging...")
    print("Check training_debug/ directory for detailed logs")
    print("Press Ctrl+C to stop training")
    print()

    instrumented_train_muzero()
