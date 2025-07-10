#!/usr/bin/env python3
"""
Training hooks for debugging MuZero training freezes.
This module provides comprehensive logging at all major training stages.
"""

import time
import os
import threading
import psutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class TrainingLogger:
    """Comprehensive training logger with freeze detection."""

    def __init__(self, log_dir: str = "training_debug"):
        self.log_dir = log_dir
        self.start_time = time.time()
        self.last_activity = time.time()
        self.iteration_count = 0
        self.stage_times = {}
        self.current_stage = None
        self.freeze_threshold = 300  # 5 minutes without activity = freeze

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Setup detailed logging
        self.setup_logging()

        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("TrainingLogger initialized")
        self.logger.info(f"Freeze threshold: {self.freeze_threshold} seconds")

    def setup_logging(self):
        """Setup detailed logging configuration."""
        log_file = os.path.join(self.log_dir, f"training_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        # Create logger
        self.logger = logging.getLogger('training_debug')
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler with detailed format
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)8s] [%(threadName)-12s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _monitor_loop(self):
        """Background monitoring loop to detect freezes."""
        while self.monitoring:
            try:
                current_time = time.time()
                time_since_activity = current_time - self.last_activity

                if time_since_activity > self.freeze_threshold:
                    self.logger.error(f"POTENTIAL FREEZE DETECTED!")
                    self.logger.error(f"Time since last activity: {time_since_activity:.1f} seconds")
                    self.logger.error(f"Current stage: {self.current_stage}")
                    self.logger.error(f"Iteration count: {self.iteration_count}")
                    self.log_system_stats()

                    # Reset to avoid spam
                    self.last_activity = current_time

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")

    def log_system_stats(self):
        """Log current system statistics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.logger.info(f"Memory usage: {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.logger.info(f"CPU usage: {cpu_percent}%")

            # Process info
            process = psutil.Process()
            self.logger.info(f"Process memory: {process.memory_info().rss / 1024**2:.1f}MB")
            self.logger.info(f"Process CPU: {process.cpu_percent()}%")
            self.logger.info(f"Process threads: {process.num_threads()}")

            # GPU info if available
            if torch.cuda.is_available():
                self.logger.info(f"CUDA available: True")
                self.logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                self.logger.info(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
            else:
                self.logger.info("CUDA available: False")

        except Exception as e:
            self.logger.error(f"Error logging system stats: {e}")

    def stage_start(self, stage_name: str, **kwargs):
        """Log the start of a training stage."""
        self.current_stage = stage_name
        self.last_activity = time.time()

        self.logger.info(f">>> STAGE START: {stage_name}")
        for key, value in kwargs.items():
            self.logger.debug(f"    {key}: {value}")

        self.stage_times[stage_name] = time.time()

    def stage_end(self, stage_name: str, **kwargs):
        """Log the end of a training stage."""
        end_time = time.time()
        self.last_activity = end_time

        if stage_name in self.stage_times:
            duration = end_time - self.stage_times[stage_name]
            self.logger.info(f"<<< STAGE END: {stage_name} (took {duration:.2f}s)")
        else:
            self.logger.info(f"<<< STAGE END: {stage_name}")

        for key, value in kwargs.items():
            self.logger.debug(f"    {key}: {value}")

        self.current_stage = None

    def log_activity(self, message: str, level: str = "info", **kwargs):
        """Log general activity."""
        self.last_activity = time.time()

        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(f"ACTIVITY: {message}")

        for key, value in kwargs.items():
            self.logger.debug(f"    {key}: {value}")

    def stop(self):
        """Stop the monitoring thread."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("TrainingLogger stopped")


class MuZeroTrainingHooks:
    """Hook system for MuZero training with comprehensive logging."""

    def __init__(self, log_dir: str = "training_debug"):
        self.logger = TrainingLogger(log_dir)
        self.hooks = {
            'before_run': [],
            'before_iter': [],
            'after_iter': [],
            'after_run': []
        }

        # Register our logging hooks
        self.register_logging_hooks()

    def register_logging_hooks(self):
        """Register comprehensive logging hooks."""

        # Before run hooks
        self.add_hook('before_run', self._before_run_hook)

        # Before iteration hooks
        self.add_hook('before_iter', self._before_iter_hook)

        # After iteration hooks
        self.add_hook('after_iter', self._after_iter_hook)

        # After run hooks
        self.add_hook('after_run', self._after_run_hook)

    def add_hook(self, stage: str, hook_func):
        """Add a hook function to a specific stage."""
        if stage in self.hooks:
            self.hooks[stage].append(hook_func)
        else:
            raise ValueError(f"Unknown hook stage: {stage}")

    def call_hooks(self, stage: str, **kwargs):
        """Call all hooks for a specific stage."""
        self.logger.log_activity(f"Calling {stage} hooks", num_hooks=len(self.hooks[stage]))

        for i, hook in enumerate(self.hooks[stage]):
            try:
                self.logger.stage_start(f"{stage}_hook_{i}", hook_name=hook.__name__)
                hook(**kwargs)
                self.logger.stage_end(f"{stage}_hook_{i}")
            except Exception as e:
                self.logger.logger.error(f"Error in {stage} hook {i} ({hook.__name__}): {e}")
                raise

    def _before_run_hook(self, **kwargs):
        """Hook called before training run starts."""
        self.logger.stage_start("training_initialization")
        self.logger.log_activity("Training run starting")
        self.logger.log_system_stats()

        # Log configuration
        if 'config' in kwargs:
            config = kwargs['config']
            self.logger.logger.info("Training configuration:")
            for key, value in config.items():
                self.logger.logger.info(f"  {key}: {value}")

        self.logger.stage_end("training_initialization")

    def _before_iter_hook(self, **kwargs):
        """Hook called before each training iteration."""
        iteration = kwargs.get('iteration', 'unknown')
        self.logger.stage_start("iteration_start", iteration=iteration)

        # Log every 10 iterations, or if we haven't seen activity recently
        if iteration % 10 == 0 or time.time() - self.logger.last_activity > 60:
            self.logger.log_activity(f"Starting iteration {iteration}")

            # Log detailed stats every 100 iterations
            if iteration % 100 == 0:
                self.logger.log_system_stats()

    def _after_iter_hook(self, **kwargs):
        """Hook called after each training iteration."""
        iteration = kwargs.get('iteration', 'unknown')
        self.logger.iteration_count = iteration

        # Log iteration results
        results = kwargs.get('results', {})
        metrics = kwargs.get('metrics', {})

        if iteration % 10 == 0:
            self.logger.log_activity(f"Completed iteration {iteration}")

            # Log key metrics
            for key, value in metrics.items():
                if key in ['loss', 'reward', 'value_loss', 'policy_loss']:
                    self.logger.logger.debug(f"  {key}: {value}")

        self.logger.stage_end("iteration_start", iteration=iteration)

    def _after_run_hook(self, **kwargs):
        """Hook called after training run completes."""
        self.logger.stage_start("training_completion")

        total_time = time.time() - self.logger.start_time
        self.logger.log_activity(f"Training run completed in {total_time:.2f} seconds")
        self.logger.log_activity(f"Total iterations: {self.logger.iteration_count}")
        self.logger.log_system_stats()

        self.logger.stage_end("training_completion")
        self.logger.stop()


# Additional utility hooks for specific components
class ComponentHooks:
    """Additional hooks for specific training components."""

    def __init__(self, training_hooks: MuZeroTrainingHooks):
        self.training_hooks = training_hooks
        self.logger = training_hooks.logger

    def env_reset_hook(self, env_id: int = None, **kwargs):
        """Hook for environment reset operations."""
        self.logger.log_activity(f"Environment reset", env_id=env_id, level="debug")

    def env_step_hook(self, env_id: int = None, action=None, **kwargs):
        """Hook for environment step operations."""
        self.logger.log_activity(f"Environment step", env_id=env_id, action=action, level="debug")

    def model_forward_hook(self, model_name: str = None, input_shape=None, **kwargs):
        """Hook for model forward passes."""
        self.logger.log_activity(f"Model forward", model=model_name, shape=input_shape, level="debug")

    def mcts_search_hook(self, num_simulations: int = None, **kwargs):
        """Hook for MCTS search operations."""
        self.logger.log_activity(f"MCTS search", simulations=num_simulations, level="debug")

    def data_collection_hook(self, num_samples: int = None, **kwargs):
        """Hook for data collection operations."""
        self.logger.log_activity(f"Data collection", samples=num_samples)

    def model_update_hook(self, loss_value: float = None, **kwargs):
        """Hook for model update operations."""
        self.logger.log_activity(f"Model update", loss=loss_value)

    def evaluation_hook(self, eval_results: Dict = None, **kwargs):
        """Hook for evaluation operations."""
        self.logger.log_activity(f"Evaluation completed", results=eval_results)


# Convenience function to create and configure hooks
def create_training_hooks(log_dir: str = "training_debug") -> MuZeroTrainingHooks:
    """Create and configure training hooks for debugging."""
    hooks = MuZeroTrainingHooks(log_dir)
    component_hooks = ComponentHooks(hooks)

    # Add component hooks as attributes for easy access
    hooks.env_hooks = component_hooks

    return hooks


# Integration helper for existing code
def patch_training_with_hooks(training_instance, hooks: MuZeroTrainingHooks):
    """Patch an existing training instance with hooks."""

    # Store original methods
    original_methods = {}

    def create_wrapped_method(method_name, original_method, hook_stage):
        def wrapped_method(*args, **kwargs):
            # Call before hooks
            if hook_stage == 'before':
                hooks.call_hooks('before_iter', method=method_name, args=args, kwargs=kwargs)

            # Call original method
            result = original_method(*args, **kwargs)

            # Call after hooks
            if hook_stage == 'after':
                hooks.call_hooks('after_iter', method=method_name, result=result)

            return result
        return wrapped_method

    # Patch common training methods
    methods_to_patch = [
        ('step', 'before'),
        ('collect', 'before'),
        ('learn', 'before'),
        ('evaluate', 'after'),
    ]

    for method_name, hook_stage in methods_to_patch:
        if hasattr(training_instance, method_name):
            original_method = getattr(training_instance, method_name)
            wrapped_method = create_wrapped_method(method_name, original_method, hook_stage)
            setattr(training_instance, method_name, wrapped_method)
            hooks.logger.log_activity(f"Patched method: {method_name}")


if __name__ == "__main__":
    # Test the hooks system
    print("Testing training hooks...")

    hooks = create_training_hooks("test_debug")

    # Simulate training events
    hooks.call_hooks('before_run', config={'test': 'config'})

    for i in range(5):
        hooks.call_hooks('before_iter', iteration=i)
        time.sleep(0.1)  # Simulate work
        hooks.call_hooks('after_iter', iteration=i, metrics={'loss': 0.5, 'reward': -10})

    hooks.call_hooks('after_run')

    print("Test completed. Check test_debug/ for logs.")
