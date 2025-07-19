from easydict import EasyDict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from training_hooks import create_training_hooks

# ==============================================================
# begin of the most frequently changed knobs
# ==============================================================
collector_env_num = 4         # increased for better data collection
n_episode         = 8         # restored to higher value for learning
evaluator_env_num = 2         # increased for better evaluation
num_simulations   = 50        # restored to higher value for better MCTS
update_per_collect = 100      # restored to original for better learning
batch_size         = 256      # balanced for CPU performance vs learning
max_env_step       = int(2e5)  # full training length for competitive perf
reanalyze_ratio    = 0.25     # enable reanalysis for better learning
# ==============================================================
# end of the most frequently changed knobs
# ==============================================================

# -------------- MAIN CONFIG (hyper‑params, replay, etc.) ---------------------
take5_muzero_config = dict(
    exp_name=(
        f'data_muzero/take5_muzero_enhanced_'
        f'ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed0'
    ),

    # ---------- environment batch settings ----------
    env=dict(
        stop_value=1000,                 # set very high to prevent early stopping
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        is_train=True,                   # required by LightZero wrapper
        env_id='take5_openspiel',        # required by LightZero wrapper
        continuous=False,                # discrete action space
        manually_discretization=False,   # not needed for discrete actions
        each_dim_disc_size=0,           # not needed for discrete actions
        # Add timeout to prevent infinite hangs
        timeout=300,                     # 5 minutes timeout per episode
    ),

    # ---------- MuZero policy ----------
    policy=dict(
        use_wandb=True,                  # enable wandb logging for monitoring
        cuda=True,                      # CPU only for MBP

        # -------- neural network --------
        model=dict(
            observation_shape=253,       # Enhanced: 208 hand + 44 rows + 1 penalty
            action_space_size=108,       # 104 cards + 4 row‑choice actions
            model_type='mlp',
            lstm_hidden_size=256,
            latent_state_dim=256,
            self_supervised_learning_loss=False,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        model_path=None,

        # -------- training schedule --------
        env_type='not_board_games',      # LightZero uses this to pick loss heads
        action_type='varied_action_space',
        game_segment_length=50,          # back‑prop horizon
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=3e-4,

        ssl_loss_weight=0,               # off for now
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=500,                  # evaluate every 500 env steps
        replay_buffer_size=int(5e5),     # transitions
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

take5_muzero_config = EasyDict(take5_muzero_config)
main_config = take5_muzero_config

# -------------- CREATE CONFIG (env + policy factory) ------------------------
take5_muzero_create_config = dict(
    env=dict(
        type='take5_openspiel',
        import_names=['take5_env'],         # path to your wrapper module
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
take5_muzero_create_config = EasyDict(take5_muzero_create_config)
create_config = take5_muzero_create_config

# ------------------------- launch helper ------------------------------------
if __name__ == "__main__":
    # Create debugging hooks
    print("Setting up training hooks for debugging...")
    hooks = create_training_hooks("training_debug")

    # Log initial configuration
    hooks.logger.log_activity("Starting Take 5 MuZero training with debugging")
    hooks.logger.log_activity(f"Collector envs: {collector_env_num}")
    hooks.logger.log_activity(f"Evaluator envs: {evaluator_env_num}")
    hooks.logger.log_activity(f"MCTS simulations: {num_simulations}")
    hooks.logger.log_activity(f"Max env steps: {max_env_step}")
    hooks.logger.log_activity(f"Batch size: {batch_size}")
    hooks.logger.log_activity(f"CUDA enabled: {main_config.policy.cuda}")

    # Call before_run hooks
    hooks.call_hooks('before_run', config=main_config)

    try:
        from lzero.entry import train_muzero

        # Wrap the training function to add our hooks
        original_train = train_muzero

        def wrapped_train_muzero(*args, **kwargs):
            hooks.logger.log_activity("Calling train_muzero function")

            # Add iteration hooks by patching the training loop
            # This is a simple approach - we'll log at key points
            result = original_train(*args, **kwargs)

            hooks.logger.log_activity("train_muzero function completed")
            return result

        # Start training with hooks
        hooks.logger.log_activity("Entering main training loop")
        wrapped_train_muzero(
            [main_config, create_config],
            seed=0,
            model_path=main_config.policy.model_path,
            max_env_step=max_env_step,
        )

        # Call after_run hooks
        hooks.call_hooks('after_run')

    except Exception as e:
        hooks.logger.logger.error(f"Training failed with error: {e}")
        hooks.logger.log_activity(f"Exception details: {str(e)}")
        hooks.call_hooks('after_run')
        raise
    finally:
        hooks.logger.log_activity("Training session ended")
        hooks.logger.stop()
