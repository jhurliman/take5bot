# take5_env.py

from typing import Any, Dict, List
import numpy as np
import pyspiel
import gym
from gym import spaces
from lzero.envs.wrappers.lightzero_env_wrapper import LightZeroEnvWrapper
from ding.envs import BaseEnvTimestep
import openspiel_take5

GAME_NAME = "take5"

pyspiel.register_game(openspiel_take5.GAME_TYPE, openspiel_take5.TakeFiveGame)

class Take5OpenSpielEnv(gym.Env):
    """
    Gym-compatible wrapper for OpenSpiel Take 5 game.
    This adapts the OpenSpiel interface to work with Gym/LightZero.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__()
        self.cfg = cfg or {}
        self.game = pyspiel.load_game(GAME_NAME)
        self.state = None

        # Set up action and observation spaces
        self.action_space = spaces.Discrete(self.game.num_distinct_actions())

        # Get observation dimension from a sample state
        sample_state = self.game.new_initial_state()
        obs_tensor = sample_state.observation_tensor(0)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(obs_tensor),), dtype=np.float32
        )

        # Add reward space for DI-engine compatibility
        self.reward_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=np.float32)

        self.current_player = 0
        self.players = list(range(self.game.num_players()))

        # Track simultaneous actions for the select phase
        self.pending_actions = {}
        self.timestep = 0

    def reset(self):
        """Reset the environment and return initial observation."""
        self.state = self.game.new_initial_state()
        self.current_player = 0
        self.pending_actions = {}
        self.timestep = 0
        return self._get_observation()

    def step(self, action):
        """Execute one step in the environment."""
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping")

        if self.state.is_terminal():
            raise RuntimeError("Cannot step in terminal state")

        # Handle different phases of the game
        current_player_id = self.state.current_player()

        if current_player_id == pyspiel.PlayerId.SIMULTANEOUS:
            # In select phase - collect actions from all players
            self.pending_actions[self.current_player] = action

            # Check if we have all player actions
            if len(self.pending_actions) == self.game.num_players():
                # Apply all simultaneous actions
                joint_actions = [self.pending_actions[p] for p in range(self.game.num_players())]
                self.state.apply_actions(joint_actions)
                self.pending_actions = {}
            else:
                # Move to next player for simultaneous action collection
                self.current_player = (self.current_player + 1) % self.game.num_players()

        elif current_player_id == self.current_player:
            # Sequential action (choose_row phase)
            self.state.apply_action(action)

        else:
            # This shouldn't happen in a well-designed game
            raise RuntimeError(f"Unexpected player state: current_player={self.current_player}, "
                             f"game_current_player={current_player_id}")

        # Increment timestep
        self.timestep += 1

        # Get observation
        obs = self._get_observation()

        # Check if game is done
        done = self.state.is_terminal()

        # Calculate reward
        reward = 0.0
        if done:
            # Game is over, calculate final scores
            returns = self.state.returns()
            # For Take 5, lower score is better (penalty points)
            # Convert to reward where higher is better
            reward = -returns[self.current_player]

        # Update current player
        if not done:
            game_current_player = self.state.current_player()
            if game_current_player == pyspiel.PlayerId.SIMULTANEOUS:
                # Start collecting simultaneous actions from player 0
                self.current_player = 0
            elif game_current_player >= 0:
                # Sequential phase
                self.current_player = game_current_player

        info = {
            'legal_actions': self.state.legal_actions(self.current_player) if not done else [],
            'current_player': self.current_player,
            'is_terminal': done
        }

        return obs, reward, done, info

    def _get_observation(self):
        """Get the current observation."""
        if self.state is None:
            raise RuntimeError("State is None")

        # Get observation tensor for current player
        obs_tensor = self.state.observation_tensor(self.current_player)
        return np.array(obs_tensor, dtype=np.float32)

    def render(self, mode='human'):
        """Render the environment."""
        if self.state is not None:
            return str(self.state)
        return "No game state"

    def close(self):
        """Close the environment."""
        pass

    def seed(self, seed=None, dynamic_seed=None):
        """Set the random seed."""
        if seed is not None:
            np.random.seed(seed)
        return [seed]


class Take5LightZeroEnv(LightZeroEnvWrapper):
    """LightZero-compatible wrapper around the OpenSpiel Take 5 game."""

    def __init__(self, cfg: Dict[str, Any] | None = None):
        # Create the base OpenSpiel environment
        base_env = Take5OpenSpielEnv(cfg)

        # Initialize with required LightZero config
        if cfg is None:
            cfg = {}

        # Set required LightZero configuration defaults
        cfg.setdefault('is_train', True)
        cfg.setdefault('env_id', 'take5_openspiel')
        cfg.setdefault('continuous', False)
        cfg.setdefault('manually_discretization', False)
        cfg.setdefault('each_dim_disc_size', 0)

        super().__init__(base_env, cfg)

        # Store dimensions for reference
        self.action_dim = base_env.action_space.n
        self.observation_dim = base_env.observation_space.shape[0]

    def reset(self):
        """Reset environment and return LightZero-compatible observation."""
        obs = super().reset()
        # Add timestep to observation
        obs['timestep'] = self.env.timestep
        return obs

    def step(self, action: int):
        """Step environment and return LightZero-compatible timestep."""
        timestep = super().step(action)
        # Add timestep to observation
        if hasattr(timestep, 'obs') and isinstance(timestep.obs, dict):
            timestep.obs['timestep'] = self.env.timestep
        return timestep

    def to_play(self) -> int:
        """Return current player. -1 indicates single-player mode for LightZero."""
        return -1  # LightZero treats -1 as "shared network" mode

    def seed(self, seed=None, dynamic_seed=None):
        """Override seed to handle DI-engine's call pattern."""
        return self.env.seed(seed, dynamic_seed)

    @staticmethod
    def create_collector_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create collector environment configurations for parallel data collection."""
        import copy
        cfg_copy = copy.deepcopy(cfg)
        collector_env_num = cfg_copy.pop('collector_env_num', 1)
        return [cfg_copy for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create evaluator environment configurations for parallel evaluation."""
        import copy
        cfg_copy = copy.deepcopy(cfg)
        evaluator_env_num = cfg_copy.pop('evaluator_env_num', 1)
        return [cfg_copy for _ in range(evaluator_env_num)]

# Register the environment
from ding.utils import ENV_REGISTRY
ENV_REGISTRY.register('take5_openspiel', Take5LightZeroEnv)
