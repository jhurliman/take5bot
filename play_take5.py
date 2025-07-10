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

import take5_env
from lzero.policy import MuZeroPolicy
from lzero.mcts import MCTS
from easydict import EasyDict


class Take5AIPlayer:
    """AI player for Take 5 using trained MuZero model."""

    def __init__(self, model_path: str, config: Dict = None):
        """Initialize the AI player with a trained model."""
        self.model_path = model_path
        self.config = config or self._default_config()
        self.policy = None
        self.env = None
        self._load_model()

    def _default_config(self) -> Dict:
        """Default configuration for the AI player."""
        return {
            'model': {
                'observation_shape': 124,
                'action_space_size': 108,
                'model_type': 'mlp',
                'lstm_hidden_size': 256,
                'latent_state_dim': 256,
                'discrete_action_encoding_type': 'one_hot',
                'norm_type': 'BN',
            },
            'num_simulations': 100,  # More simulations for better play
            'cuda': torch.cuda.is_available(),
        }

    def _load_model(self):
        """Load the trained MuZero model."""
        try:
            # Create policy
            policy_config = EasyDict(self.config)
            self.policy = MuZeroPolicy(policy_config)

            # Load model weights
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location='cpu')
                self.policy.load_state_dict(checkpoint['model'])
                self.policy.eval()
                print(f"Loaded model from: {self.model_path}")
            else:
                print(f"Warning: Model file not found at {self.model_path}")
                print("Using randomly initialized model")

            # Create environment for state encoding
            self.env = take5_env.Take5LightZeroEnv()

        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def encode_game_state(self, hand: List[int], rows: List[List[int]]) -> np.ndarray:
        """Encode the game state into the format expected by the model."""
        # Create a 124-dimensional observation vector
        obs = np.zeros(124, dtype=np.float32)

        # First 104 bits: hand encoding (1 if card is in hand, 0 otherwise)
        for card in hand:
            if 1 <= card <= 104:
                obs[card - 1] = 1.0

        # Next 20 bits: row state (4 rows × 5 positions)
        for row_idx, row in enumerate(rows):
            if row_idx < 4:  # Ensure we don't exceed 4 rows
                for pos_idx, card in enumerate(row):
                    if pos_idx < 5 and card > 0:  # Ensure we don't exceed 5 positions
                        bit_idx = 104 + row_idx * 5 + pos_idx
                        obs[bit_idx] = 1.0

        return obs

    def get_action_recommendation(self, hand: List[int], rows: List[List[int]]) -> Tuple[int, float]:
        """Get AI recommendation for the next action."""
        try:
            # Encode game state
            obs = self.encode_game_state(hand, rows)

            # Create observation dict as expected by LightZero
            obs_dict = {
                'observation': obs,
                'action_mask': np.ones(108, dtype=np.int8),  # All actions initially legal
                'to_play': -1,
                'timestep': 0
            }

            # Get legal actions (cards in hand)
            legal_card_actions = [card - 1 for card in hand if 1 <= card <= 104]

            # Update action mask to only allow legal actions
            obs_dict['action_mask'] = np.zeros(108, dtype=np.int8)
            for action in legal_card_actions:
                obs_dict['action_mask'][action] = 1

            # Get policy recommendation
            with torch.no_grad():
                action_probs = self.policy.forward(obs_dict, temperature=0.1)

                # Get the best legal action
                best_action = None
                best_prob = -1

                for action in legal_card_actions:
                    if action_probs[action] > best_prob:
                        best_prob = action_probs[action]
                        best_action = action

                if best_action is not None:
                    return best_action + 1, best_prob  # Convert back to card number
                else:
                    # Fallback: return first card in hand
                    return hand[0], 0.0

        except Exception as e:
            print(f"Error getting AI recommendation: {e}")
            # Fallback: return first card in hand
            return hand[0], 0.0

    def explain_recommendation(self, recommended_card: int, confidence: float, hand: List[int], rows: List[List[int]]) -> str:
        """Provide explanation for the AI's recommendation."""
        explanations = []

        explanations.append(f"AI recommends playing card {recommended_card}")
        explanations.append(f"Confidence: {confidence:.2%}")

        # Add some basic strategic explanations
        if confidence > 0.7:
            explanations.append("High confidence - this appears to be a strong move")
        elif confidence > 0.4:
            explanations.append("Moderate confidence - decent move but alternatives exist")
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
    print("\n" + "="*60)
    print("CURRENT GAME STATE")
    print("="*60)

    print("\nTable rows:")
    for i, row in enumerate(rows):
        if row:
            print(f"Row {i+1}: {' '.join(map(str, row))}")
        else:
            print(f"Row {i+1}: [empty]")

    print(f"\nYour hand: {' '.join(map(str, sorted(hand)))}")
    print("="*60)


def find_latest_model(base_dir: str = "data_muzero") -> Optional[str]:
    """Find the latest trained model."""
    if not os.path.exists(base_dir):
        return None

    # Look for checkpoint files
    pattern = os.path.join(base_dir, "**", "*.pth.tar")
    checkpoints = glob.glob(pattern, recursive=True)

    if not checkpoints:
        return None

    # Return the most recent checkpoint
    return max(checkpoints, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Play Take 5 with AI assistance")
    parser.add_argument("--model", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--simulations", type=int, default=100, help="Number of MCTS simulations")

    args = parser.parse_args()

    # Find model
    model_path = args.model
    if not model_path:
        model_path = find_latest_model()
        if not model_path:
            print("No trained model found. Please train a model first or specify --model")
            return

    print(f"Using model: {model_path}")

    # Initialize AI player
    config = {
        'model': {
            'observation_shape': 124,
            'action_space_size': 108,
            'model_type': 'mlp',
            'lstm_hidden_size': 256,
            'latent_state_dim': 256,
            'discrete_action_encoding_type': 'one_hot',
            'norm_type': 'BN',
        },
        'num_simulations': args.simulations,
        'cuda': torch.cuda.is_available(),
    }

    ai_player = Take5AIPlayer(model_path, config)

    print("\n" + "="*60)
    print("TAKE 5 AI ASSISTANT")
    print("="*60)
    print("Enter the current game state to get AI recommendations.")
    print("Type 'quit' to exit, 'help' for instructions.")
    print("="*60)

    while True:
        try:
            print("\nCommands:")
            print("  help - Show instructions")
            print("  quit - Exit the program")
            print("  play - Start a new game state input")

            command = input("\nEnter command: ").strip().lower()

            if command == 'quit':
                print("Goodbye!")
                break
            elif command == 'help':
                print("\nHOW TO USE:")
                print("1. Type 'play' to start entering game state")
                print("2. Enter your hand as space-separated numbers (e.g., '1 5 10 23 45')")
                print("3. For each table row, enter the cards in that row")
                print("4. The AI will recommend which card to play")
                print("\nExample:")
                print("  Your hand: 15 23 67 89 104")
                print("  Row 1: 12 34")
                print("  Row 2: 7 45 67")
                print("  Row 3: 91")
                print("  Row 4: [empty, just press Enter]")

            elif command == 'play':
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
                    row_input = input(f"Enter Row {i+1} (or press Enter if empty): ").strip()
                    if row_input:
                        try:
                            row = [int(x) for x in row_input.split()]
                            rows.append(row)
                        except ValueError:
                            print("Invalid input for row. Using empty row.")
                            rows.append([])
                    else:
                        rows.append([])

                # Display game state
                display_game_state(hand, rows)

                # Get AI recommendation
                print("\nThinking...")
                recommended_card, confidence = ai_player.get_action_recommendation(hand, rows)

                # Display recommendation
                print("\n" + "="*60)
                print("AI RECOMMENDATION")
                print("="*60)
                explanation = ai_player.explain_recommendation(recommended_card, confidence, hand, rows)
                print(explanation)
                print("="*60)

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
