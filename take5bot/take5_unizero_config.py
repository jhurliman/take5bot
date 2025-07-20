from easydict import EasyDict  # type: ignore
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ==============================================================
# begin of the most frequently changed knobs
# ==============================================================
collector_env_num = 4  # increased for better data collection
n_episode = 8  # restored to higher value for learning
evaluator_env_num = 2  # increased for better evaluation
num_simulations = 50  # restored to higher value for better MCTS
update_per_collect = 100  # restored to original for better learning
batch_size = 256  # balanced for CPU performance vs learning
max_env_step = int(2e5)  # full training length for competitive perf
reanalyze_ratio = 0.25  # enable reanalysis for better learning
# ==============================================================
# end of the most frequently changed knobs
# ==============================================================

# -------------- MAIN CONFIG (hyper‑params, replay, etc.) ---------------------
take5_unizero_config = dict(
    exp_name=(
        f"data_unizero/take5_unizero_enhanced_"
        f"ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed0"
    ),
    # ---------- environment batch settings ----------
    env=dict(
        stop_value=1000,  # set very high to prevent early stopping
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        is_train=True,  # required by LightZero wrapper
        env_id="take5_openspiel",  # required by LightZero wrapper
        continuous=False,  # discrete action space
        manually_discretization=False,  # not needed for discrete actions
        each_dim_disc_size=0,  # not needed for discrete actions
        # Add timeout to prevent infinite hangs
        timeout=300,  # 5 minutes timeout per episode
    ),
    # ---------- UniZero policy ----------
    policy=dict(
        use_wandb=True,  # enable wandb logging for monitoring
        cuda=True,  # CPU only for MBP
        # -------- neural network --------
        model=dict(
            observation_shape=253,  # Enhanced: 208 hand + 44 rows + 1 penalty
            action_space_size=108,  # 104 cards + 4 row‑choice actions
            model_type="mlp",
            lstm_hidden_size=256,
            latent_state_dim=256,
            self_supervised_learning_loss=False,
            discrete_action_encoding_type="one_hot",
            norm_type="BN",
            # UniZero specific world model configuration
            world_model_cfg=dict(
                obs_type="vector",  # vector observations for Take 5
                embed_dim=256,  # embedding dimension
                group_size=8,  # group size for normalization
                max_blocks=50,  # max blocks per episode
                max_tokens=100,  # max_tokens = 2 * max_blocks
                tokens_per_block=2,  # obs + action tokens per block
                context_length=100,  # context length for transformer
                num_layers=8,  # number of transformer layers
                num_heads=8,  # number of attention heads
                support_size=101,  # support size for value/reward heads
                continuous_action_space=False,  # discrete action space
                rotary_emb=False,  # use absolute position encoding
                policy_entropy_weight=0.0,  # policy entropy weight
                gamma=0.99,  # discount factor
                device="cuda" if torch.cuda.is_available() else "cpu",
                analysis_sim_norm=False,  # disable analysis hooks
                analysis_dormant_ratio=False,  # disable dormant ratio analysis
                dormant_threshold=0.025,  # dormant neuron threshold
                predict_latent_loss_type="cross_entropy",
                latent_recon_loss_weight=0.0,  # latent reconstruction loss weight
                perceptual_loss_weight=0.0,  # perceptual loss weight
                final_norm_option_in_encoder="LayerNorm",
                env_num=collector_env_num,  # number of parallel environments
                num_simulations=num_simulations,  # MCTS simulations
                max_cache_size=1000,  # cache size for KV caching
            ),
        ),
        model_path=None,
        # -------- training schedule --------
        env_type="not_board_games",  # LightZero uses this to pick loss heads
        action_type="varied_action_space",
        game_segment_length=50,  # back‑prop horizon
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type="Adam",
        piecewise_decay_lr_scheduler=False,
        learning_rate=3e-4,
        ssl_loss_weight=0,  # off for now
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=500,  # evaluate every 500 env steps
        replay_buffer_size=int(5e5),  # transitions
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

take5_unizero_config = EasyDict(take5_unizero_config)
main_config = take5_unizero_config

# -------------- CREATE CONFIG (env + policy factory) ------------------------
take5_unizero_create_config = dict(
    env=dict(
        type="take5_openspiel",
        import_names=["take5_env"],  # path to your wrapper module
    ),
    env_manager=dict(type="subprocess"),
    policy=dict(
        type="unizero",
        import_names=["lzero.policy.unizero"],
    ),
)
take5_unizero_create_config = EasyDict(take5_unizero_create_config)
create_config = take5_unizero_create_config
