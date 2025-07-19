#!/usr/bin/env python3
"""
Interactive Take 5 gameplay script.
Human player inputs game state and receives AI recommendations.
"""

import sys
import os
import argparse
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import glob
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import take5bot.take5_env as take5_env
from lzero.policy.muzero import MuZeroPolicy
from easydict import EasyDict


class Take5AIPlayer:
    """AI player for Take 5 using trained MuZero model."""

    def __init__(self, model_path: str, config: Dict | None = None):
        """Initialize the AI player with a trained model."""
        self.model_path = model_path
        self.config = config or self._default_config()
        self.policy = None
        self.env = None
        self._load_model()

    def _default_config(self) -> Dict:
        """Default configuration for the AI player."""
        # Get base config and update with minimal required settings
        config = MuZeroPolicy.default_config()

        # Update model config
        config["model"].update(
            {
                "observation_shape": 253,  # Enhanced: 208 hand + 44 rows + 1 penalty
                "action_space_size": 108,
                "model_type": "mlp",
                "lstm_hidden_size": 256,
                "latent_state_dim": 256,
                "discrete_action_encoding_type": "one_hot",
                "norm_type": "BN",
            }
        )

        # Update other required settings
        config.update(
            {
                "num_simulations": 100,
                "cuda": torch.cuda.is_available(),
                "env_type": "not_board_games",
                "use_wandb": False,
            }
        )

        return config

    def _load_model(self):
        """Load the trained MuZero model."""
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Get the default configuration and update it
        config = MuZeroPolicy.default_config()

        # Update model config
        config["model"].update(
            {
                "observation_shape": 253,  # Enhanced: 208 hand + 44 rows + 1 penalty
                "action_space_size": 108,
                "model_type": "mlp",
                "lstm_hidden_size": 256,
                "latent_state_dim": 256,
                "discrete_action_encoding_type": "one_hot",
                "norm_type": "BN",
            }
        )

        # Update other required settings
        config.update(
            {
                "num_simulations": 100,
                "cuda": torch.cuda.is_available(),
                "env_type": "not_board_games",
                "use_wandb": False,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "action_type": "varied_action_space",  # Must match training config
            }
        )

        # Convert to EasyDict
        policy_config = EasyDict(config)

        # Create policy
        self.policy = MuZeroPolicy(policy_config)

        if self.policy is None or not hasattr(self.policy, "_model"):
            raise RuntimeError(
                "Failed to create MuZeroPolicy or policy has no _model attribute"
            )

        # Load model weights
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu")
            print(f"DEBUG: Checkpoint keys: {list(checkpoint.keys())}")

            if "model" not in checkpoint:
                raise RuntimeError(
                    f"Checkpoint missing 'model' key. Available keys: {list(checkpoint.keys())}"
                )

            # Access the underlying model and load state dict
            self.policy._model.load_state_dict(checkpoint["model"])
            self.policy._model.eval()
            print(f"Loaded model from: {self.model_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

        # Create environment for state encoding
        self.env = take5_env.Take5LightZeroEnv()

    def encode_game_state(
        self, hand: List[int], rows: List[List[int]], current_penalty: int = 0
    ) -> np.ndarray:
        """Encode the game state into the format expected by the model (253 elements).
        This MUST match exactly the observation_tensor format from openspiel_take5.py
        """

        def bullheads(card: int) -> int:
            """Return the penalty value for a card."""
            if card == 55:
                return 7
            if card % 11 == 0:
                return 5
            if card % 10 == 0:
                return 3
            if card % 5 == 0:
                return 2
            return 1

        # Hand info (208 elements): 104 presence + 104 penalties
        # This matches _hand_tensor() from openspiel_take5.py
        hand_presence = np.zeros(104, dtype=np.float32)
        hand_penalties = np.zeros(104, dtype=np.float32)

        for card in hand:
            if 1 <= card <= 104:
                hand_presence[card - 1] = 1.0
                hand_penalties[card - 1] = (
                    bullheads(card) / 7.0
                )  # normalize by max penalty

        hand_info = np.concatenate([hand_presence, hand_penalties])

        # Row info (44 elements): 20 card numbers + 20 penalties + 4 row totals
        # This matches _rows_tensor() from openspiel_take5.py
        row_cards = np.zeros((4, 5), dtype=np.float32)
        row_penalties = np.zeros((4, 5), dtype=np.float32)
        row_totals = np.zeros(4, dtype=np.float32)

        for r, row in enumerate(rows):
            if r >= 4:
                break
            row_penalty_sum = 0
            for idx, card in enumerate(row):
                if idx >= 5:
                    break
                if card > 0:
                    row_cards[r, idx] = (
                        card / 104.0
                    )  # normalize card number (NUM_CARDS = 104)
                    penalty = bullheads(card)
                    row_penalties[r, idx] = penalty / 7.0  # normalize penalty
                    row_penalty_sum += penalty
            # normalize by max possible row penalty (5 cards × 7 bulls = 35)
            row_totals[r] = row_penalty_sum / 35.0

        row_info = np.concatenate(
            [row_cards.flatten(), row_penalties.flatten(), row_totals]
        )

        # Penalty info (1 element): current penalty pile total normalized by 66
        # This matches the penalty_info calculation from openspiel_take5.py
        penalty_info = np.array([current_penalty / 66.0], dtype=np.float32)

        # Total: 208 + 44 + 1 = 253 elements
        return np.concatenate([hand_info, row_info, penalty_info])

    def get_action_recommendation(
        self, hand: List[int], rows: List[List[int]], current_penalty: int = 0
    ) -> Tuple[int, float]:
        """Get AI recommendation for the next action."""
        # Encode game state
        obs = self.encode_game_state(hand, rows, current_penalty)
        print(f"DEBUG: Observation shape: {obs.shape}")

        # Get legal actions (cards in hand)
        legal_card_actions = [card - 1 for card in hand if 1 <= card <= 104]
        print(f"DEBUG: Legal actions (0-indexed): {legal_card_actions}")
        print(f"DEBUG: Legal cards: {[a + 1 for a in legal_card_actions]}")

        if not legal_card_actions:
            raise ValueError("No legal actions found in hand")

        # Get policy recommendation using initial_inference
        with torch.no_grad():
            # Set model to eval mode
            self.policy._model.eval()

            # Convert observation to tensor and move to same device as model
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # Add batch dimension

            # Move tensor to same device as model
            device = next(self.policy._model.parameters()).device
            obs_tensor = obs_tensor.to(device)
            print(f"DEBUG: Using device: {device}")

            # Use initial_inference to get policy predictions
            result = self.policy._model.initial_inference(obs_tensor)
            print(f"DEBUG: Model result type: {type(result)}")

            # Handle different result types from MuZero model
            if hasattr(result, "policy_logits"):
                # MZNetworkOutput object with policy_logits attribute
                policy_logits = result.policy_logits
                print("DEBUG: Using policy_logits from MZNetworkOutput")
            elif isinstance(result, dict) and "policy_logits" in result:
                # Dictionary with policy_logits key
                policy_logits = result["policy_logits"]
                print("DEBUG: Using policy_logits from dict")
            else:
                available_attrs = [
                    attr for attr in dir(result) if not attr.startswith("_")
                ]
                raise RuntimeError(
                    f"Model result missing 'policy_logits'. Result type: {type(result)}, Available attributes: {available_attrs}"
                )

            print(f"DEBUG: Policy logits shape: {policy_logits.shape}")
            print(
                f"DEBUG: Policy logits range: [{policy_logits.min():.3f}, {policy_logits.max():.3f}]"
            )

            if policy_logits.shape[-1] != 108:
                raise RuntimeError(
                    f"Model output has wrong action space size: {policy_logits.shape[-1]}, expected 108"
                )

            # Apply softmax to get probabilities
            action_probs = torch.softmax(policy_logits, dim=-1)
            action_probs = action_probs.squeeze(0).cpu().numpy()
            print(f"DEBUG: Action probs shape: {action_probs.shape}")
            print(f"DEBUG: Action probs sum: {action_probs.sum():.6f}")

            # Get the best legal action
            best_action = None
            best_prob = -1

            print("DEBUG: Legal action probabilities:")
            for action in legal_card_actions:
                prob = action_probs[action]
                card = action + 1
                print(f"  Card {card}: {prob:.6f}")
                if prob > best_prob:
                    best_prob = prob
                    best_action = action

            if best_action is None:
                raise RuntimeError("No best action found among legal actions")

            print(
                f"DEBUG: Best action: {best_action} (card {best_action + 1}), prob: {best_prob:.6f}"
            )

            # Sanity check: ensure we're not getting uniform probabilities (sign of untrained model)
            prob_variance = np.var([action_probs[a] for a in legal_card_actions])
            print(
                f"DEBUG: Probability variance among legal actions: {prob_variance:.2e}"
            )

            # Also check the raw logit differences
            legal_logits = [policy_logits[0, a].item() for a in legal_card_actions]
            logit_variance = np.var(legal_logits)
            print(f"DEBUG: Legal action logits: {[f'{l:.3f}' for l in legal_logits]}")
            print(f"DEBUG: Logit variance: {logit_variance:.6f}")

            if prob_variance < 1e-8:  # Much stricter threshold
                print(
                    "WARNING: Model appears to be outputting uniform probabilities (possibly untrained)"
                )
            elif prob_variance < 1e-6:
                print(
                    "INFO: Model is making small distinctions between actions (low confidence)"
                )
            else:
                print("INFO: Model is making clear distinctions between actions")

            return best_action + 1, float(best_prob)

    def explain_recommendation(
        self,
        recommended_card: int,
        confidence: float,
        hand: List[int],
        rows: List[List[int]],
    ) -> str:
        """Provide explanation for the AI's recommendation."""
        explanations = []

        explanations.append(f"AI recommends playing card {recommended_card}")
        explanations.append(f"Confidence: {confidence:.2%}")

        # Add some basic strategic explanations
        if confidence > 0.7:
            explanations.append("High confidence - this appears to be a strong move")
        elif confidence > 0.4:
            explanations.append(
                "Moderate confidence - decent move but alternatives exist"
            )
        else:
            explanations.append("Low confidence - uncertain situation")

        # Check if card is safe (won't trigger taking a row)
        safe_plays = []
        risky_plays = []

        for card in hand:
            for row_idx, row in enumerate(rows):
                if row and card > max(row):
                    if len(row) < 5:
                        safe_plays.append(card)
                    else:
                        risky_plays.append(card)

        if recommended_card in safe_plays:
            explanations.append("This card can be played safely without taking a row")
        elif recommended_card in risky_plays:
            explanations.append("Warning: This card may force you to take a row")

        return "\n".join(explanations)


def parse_input(prompt: str) -> List[int]:
    """Parse space-separated integers from user input."""
    try:
        return [int(x) for x in input(prompt).strip().split()]
    except ValueError:
        print("Invalid input. Please enter space-separated numbers.")
        return []


def display_game_state(hand: List[int], rows: List[List[int]]):
    """Display the current game state in a readable format."""
    print("\n" + "=" * 60)
    print("CURRENT GAME STATE")
    print("=" * 60)

    print("\nTable rows:")
    for i, row in enumerate(rows):
        if row:
            print(f"Row {i+1}: {' '.join(map(str, row))}")
        else:
            print(f"Row {i+1}: [empty]")

    print(f"\nYour hand: {' '.join(map(str, sorted(hand)))}")
    print("=" * 60)


def find_best_model(base_dir: str = "data_muzero") -> Optional[str]:
    """Find the best checkpoint from the most recent training run."""
    if not os.path.exists(base_dir):
        return None

    # Look for checkpoint files
    pattern = os.path.join(base_dir, "**", "*.pth.tar")
    checkpoints = glob.glob(pattern, recursive=True)

    if not checkpoints:
        return None

    # First, try to find ckpt_best.pth.tar files (best performing models)
    best_checkpoints = [
        ckpt for ckpt in checkpoints
        if "ckpt_best.pth.tar" in ckpt
    ]

    if best_checkpoints:
        # Return the most recent best checkpoint (by modification time)
        return max(best_checkpoints, key=os.path.getmtime)

    # Fallback: if no best checkpoint exists, use the latest iteration checkpoint
    return find_latest_model(base_dir)


def find_latest_model(base_dir: str = "data_muzero") -> Optional[str]:
    """Find the latest trained model."""
    if not os.path.exists(base_dir):
        return None

    # Look for checkpoint files
    pattern = os.path.join(base_dir, "**", "*.pth.tar")
    checkpoints = glob.glob(pattern, recursive=True)

    if not checkpoints:
        return None

    # Filter for highest iteration checkpoint (not iteration_0.pth.tar)
    # and prefer iteration_X.pth.tar over ckpt_best.pth.tar for most recent training
    iteration_checkpoints = [
        ckpt
        for ckpt in checkpoints
        if "iteration_" in ckpt and not ckpt.endswith("iteration_0.pth.tar")
    ]

    if iteration_checkpoints:
        # Find the checkpoint with the highest iteration number
        def get_iteration_number(path):
            try:
                filename = os.path.basename(path)
                if "iteration_" in filename:
                    return int(filename.split("iteration_")[1].split(".")[0])
                return 0
            except (ValueError, IndexError):
                return 0

        return max(iteration_checkpoints, key=get_iteration_number)

    # Fallback to most recent checkpoint by modification time
    return max(checkpoints, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Play Take 5 with AI assistance")
    parser.add_argument("--model", type=str, help="Path to trained model checkpoint")
    parser.add_argument(
        "--simulations", type=int, default=100, help="Number of MCTS simulations"
    )

    args = parser.parse_args()

    # Find model
    model_path = args.model
    if not model_path:
        model_path = find_best_model()
        if not model_path:
            raise FileNotFoundError(
                "No trained model found. Please train a model first or specify --model"
            )

    print(f"Using model: {model_path}")

    # Initialize AI player - this will fail fast if model can't be loaded
    ai_player = Take5AIPlayer(model_path)

    print("\n" + "=" * 60)
    print("TAKE 5 AI ASSISTANT")
    print("=" * 60)
    print("Enter the current game state to get AI recommendations.")
    print("Type 'quit' to exit, 'help' for instructions.")
    print("=" * 60)

    while True:
        try:
            print("\nCommands:")
            print("  help - Show instructions")
            print("  quit - Exit the program")
            print("  play - Start a new game state input")

            command = input("\nEnter command: ").strip().lower()

            if command == "quit":
                print("Goodbye!")
                break
            elif command == "help":
                print("\nHOW TO USE:")
                print("1. Type 'play' to start entering game state")
                print(
                    "2. Enter your hand as space-separated numbers (e.g., '1 5 10 23 45')"
                )
                print("3. For each table row, enter the cards in that row")
                print("4. Enter your current penalty points (if any)")
                print("5. The AI will recommend which card to play")
                print("\nExample:")
                print("  Your hand: 15 23 67 89 104")
                print("  Row 1: 12 34")
                print("  Row 2: 7 45 67")
                print("  Row 3: 91")
                print("  Row 4: [empty, just press Enter]")
                print("  Current penalty points: 0")

            elif command == "play":
                # Get hand
                hand = []
                while not hand:
                    hand = parse_input("Enter your hand (space-separated numbers): ")
                    if not hand:
                        continue
                    # Validate hand
                    if any(card < 1 or card > 104 for card in hand):
                        print("Invalid cards. Cards must be between 1 and 104.")
                        hand = []

                # Get rows
                rows = []
                for i in range(4):
                    row_input = input(
                        f"Enter Row {i+1} (or press Enter if empty): "
                    ).strip()
                    if row_input:
                        try:
                            row = [int(x) for x in row_input.split()]
                            rows.append(row)
                        except ValueError:
                            print("Invalid input for row. Using empty row.")
                            rows.append([])
                    else:
                        rows.append([])

                # Get current penalty points
                penalty_input = input(
                    "Enter your current penalty points (or 0 if none): "
                ).strip()
                current_penalty = 0
                if penalty_input:
                    try:
                        current_penalty = int(penalty_input)
                    except ValueError:
                        print("Invalid penalty input. Using 0.")
                        current_penalty = 0

                # Display game state
                display_game_state(hand, rows)
                if current_penalty > 0:
                    print(f"Current penalty points: {current_penalty}")

                # Get AI recommendation
                print("\nThinking...")
                recommended_card, confidence = ai_player.get_action_recommendation(
                    hand, rows, current_penalty
                )

                # Display recommendation
                print("\n" + "=" * 60)
                print("AI RECOMMENDATION")
                print("=" * 60)
                explanation = ai_player.explain_recommendation(
                    recommended_card, confidence, hand, rows
                )
                print(explanation)
                print("=" * 60)

            else:
                print("Unknown command. Type 'help' for instructions.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
