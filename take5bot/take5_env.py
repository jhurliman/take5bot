# take5_env.py

from typing import Any, Dict, List
import numpy as np
import pyspiel
import gym
from gym import spaces
from lzero.envs.wrappers.lightzero_env_wrapper import LightZeroEnvWrapper
from ding.envs import BaseEnvTimestep
from easydict import EasyDict
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
        self.reward_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(1,), dtype=np.float32
        )

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

        # Debug info
        print(f"Environment reset. Initial player: {self.current_player}")
        print(f"Game current player: {self.state.current_player()}")

        return self._get_observation()

    def step(self, action):
        """Execute one step in the environment."""
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping")

        if self.state.is_terminal():
            # Return current observation and done=True
            obs = self._get_observation()
            return (
                obs,
                0.0,
                True,
                {
                    "legal_actions": [],
                    "current_player": self.current_player,
                    "is_terminal": True,
                },
            )

        # Handle different phases of the game
        current_player_id = self.state.current_player()

        if current_player_id == pyspiel.PlayerId.SIMULTANEOUS:
            # In select phase - collect actions from all players
            legal_actions = self.state.legal_actions(self.current_player)
            if action not in legal_actions:
                print(
                    f"Warning: Action {action} not in legal actions {legal_actions} for player {self.current_player}"
                )
                print(
                    f"Player hand: {self.state._hands[self.current_player] if hasattr(self.state, '_hands') else 'N/A'}"
                )
                # Take the first legal action instead
                if legal_actions:
                    action = legal_actions[0]
                    print(f"Using first legal action: {action}")
                else:
                    print("No legal actions available!")
                    obs = self._get_observation()
                    return (
                        obs,
                        0.0,
                        True,
                        {
                            "legal_actions": [],
                            "current_player": self.current_player,
                            "is_terminal": True,
                        },
                    )

            self.pending_actions[self.current_player] = action

            # Check if we have all player actions
            if len(self.pending_actions) == self.game.num_players():
                # Apply all simultaneous actions
                joint_actions = [
                    self.pending_actions[p] for p in range(self.game.num_players())
                ]
                # print(f"Applying joint actions: {joint_actions}")
                self.state.apply_actions(joint_actions)
                self.pending_actions = {}
                # After applying simultaneous actions, update current player
                if not self.state.is_terminal():
                    new_current_player = self.state.current_player()
                    if new_current_player == pyspiel.PlayerId.SIMULTANEOUS:
                        self.current_player = 0
                    elif new_current_player >= 0:
                        self.current_player = new_current_player
            else:
                # Move to next player for simultaneous action collection
                self.current_player = (
                    self.current_player + 1
                ) % self.game.num_players()

        elif current_player_id >= 0:
            # Sequential action (choose_row phase)
            if current_player_id == self.current_player:
                legal_actions = self.state.legal_actions(self.current_player)
                # print(f"Choose_row phase: player {self.current_player}, action {action}, legal: {legal_actions}")
                if action not in legal_actions:
                    print(
                        f"Warning: Action {action} not in legal actions {legal_actions} for choose_row"
                    )
                    if legal_actions:
                        action = legal_actions[0]
                        print(f"Using first legal action: {action}")
                    else:
                        print("No legal actions in choose_row!")
                        obs = self._get_observation()
                        return (
                            obs,
                            0.0,
                            True,
                            {
                                "legal_actions": [],
                                "current_player": self.current_player,
                                "is_terminal": True,
                            },
                        )

                self.state.apply_action(action)
                # Update current player after action
                if not self.state.is_terminal():
                    new_current_player = self.state.current_player()
                    if new_current_player == pyspiel.PlayerId.SIMULTANEOUS:
                        self.current_player = 0
                    elif new_current_player >= 0:
                        self.current_player = new_current_player
            else:
                # This shouldn't happen - log and continue
                print(
                    f"Warning: Unexpected player state: current_player={self.current_player}, "
                    f"game_current_player={current_player_id}"
                )
                self.current_player = current_player_id

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
            reward = (
                -returns[self.current_player]
                if self.current_player < len(returns)
                else 0.0
            )

        info = {
            "legal_actions": (
                self.state.legal_actions(self.current_player)
                if not done and self.current_player >= 0
                else []
            ),
            "current_player": self.current_player,
            "is_terminal": done,
        }

        return obs, reward, done, info

    def _get_observation(self):
        """Get the current observation."""
        if self.state is None:
            raise RuntimeError("State is None")

        # Ensure current_player is valid
        if self.current_player < 0 or self.current_player >= self.game.num_players():
            self.current_player = 0

        # Get observation tensor for current player
        try:
            obs_tensor = self.state.observation_tensor(self.current_player)
            return np.array(obs_tensor, dtype=np.float32)
        except Exception as e:
            print(f"Error getting observation for player {self.current_player}: {e}")
            # Fall back to player 0 observation
            obs_tensor = self.state.observation_tensor(0)
            return np.array(obs_tensor, dtype=np.float32)

    def render(self, mode="human"):
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

        # Convert to EasyDict if it's a regular dict
        if isinstance(cfg, dict) and not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)

        # Set required LightZero configuration defaults
        if not hasattr(cfg, "is_train"):
            cfg.is_train = True
        if not hasattr(cfg, "env_id"):
            cfg.env_id = "take5_openspiel"
        if not hasattr(cfg, "continuous"):
            cfg.continuous = False
        if not hasattr(cfg, "manually_discretization"):
            cfg.manually_discretization = False
        if not hasattr(cfg, "each_dim_disc_size"):
            cfg.each_dim_disc_size = 0

        super().__init__(base_env, cfg)

        # Store dimensions for reference
        self.action_dim = base_env.action_space.n
        self.observation_dim = base_env.observation_space.shape[0]

    def reset(self):
        """Reset environment and return LightZero-compatible observation."""
        obs = self.env.reset()

        # Create proper action mask based on legal actions
        action_mask = np.zeros(self.env.action_space.n, dtype=np.int8)
        if self.env.state and not self.env.state.is_terminal():
            legal_actions = self.env.state.legal_actions(self.env.current_player)
            if legal_actions:
                action_mask[legal_actions] = 1

        # Create LightZero observation dict
        lightzero_obs_dict = {
            "observation": obs,
            "action_mask": action_mask,
            "to_play": self.to_play(),
            "timestep": self.env.timestep,
        }

        # Reset episode return
        self._eval_episode_return = 0.0

        return lightzero_obs_dict

    def step(self, action: int):
        """Step environment and return LightZero-compatible timestep."""
        # Get observation before stepping
        obs, rew, done, info = self.env.step(action)

        # Create proper action mask based on legal actions
        if done:
            action_mask = np.zeros(self.env.action_space.n, dtype=np.int8)
        else:
            action_mask = np.zeros(self.env.action_space.n, dtype=np.int8)
            legal_actions = info.get("legal_actions", [])
            if legal_actions:
                action_mask[legal_actions] = 1

        # Create LightZero observation dict
        lightzero_obs_dict = {
            "observation": obs,
            "action_mask": action_mask,
            "to_play": self.to_play(),
            "timestep": self.env.timestep,
        }

        # Update episode return
        self._eval_episode_return += rew
        if done:
            info["eval_episode_return"] = self._eval_episode_return

        return BaseEnvTimestep(lightzero_obs_dict, rew, done, info)

    def to_play(self) -> int:
        """Return current player."""
        if self.env.state is None or self.env.state.is_terminal():
            return 0

        # Return the actual current player from the game
        game_current_player = self.env.state.current_player()
        if game_current_player == pyspiel.PlayerId.SIMULTANEOUS:
            return self.env.current_player
        elif game_current_player >= 0:
            return game_current_player
        else:
            return 0

    def seed(self, seed=None, dynamic_seed=None):
        """Override seed to handle DI-engine's call pattern."""
        return self.env.seed(seed, dynamic_seed)

    @staticmethod
    def create_collector_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create collector environment configurations for parallel data collection."""
        import copy

        cfg_copy = copy.deepcopy(cfg)
        collector_env_num = cfg_copy.pop("collector_env_num", 1)
        return [cfg_copy for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create evaluator environment configurations for parallel evaluation."""
        import copy

        cfg_copy = copy.deepcopy(cfg)
        evaluator_env_num = cfg_copy.pop("evaluator_env_num", 1)
        return [cfg_copy for _ in range(evaluator_env_num)]


# Register the environment
from ding.utils import ENV_REGISTRY

ENV_REGISTRY.register("take5_openspiel", Take5LightZeroEnv)
