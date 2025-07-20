# take5bot

> Agent implementation for the "Take 5" card game with CUDA support.

Uses OpenSpiel and LightZero to play Take 5 and learn via self-play with GPU acceleration.

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd take5bot
   ```

2. **Create a virtual environment:**
   ```bash
   # Install uv (https://github.com/astral-sh/uv) if not already installed
   uv venv --python=python3.10
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

### Usage

- **Train a model:**
  ```bash
  python ./train_take5.py
  ```

- **Play Take 5:**
  ```bash
  python ./play_take5.py
  ```

## 🏗️ Architecture

### Core Components

- **`openspiel_take5.py`** - OpenSpiel game implementation
- **`take5_env.py`** - LightZero environment wrapper
- **`take5_muzero_config.py`** - Training configuration

### Environment Specs
- **Observation Space**: 253-dimensional vector
  - Elements 0-103: Hand presence (binary)
  - Elements 104-207: Penalty (bull point) values for cards present in hand (normalized)
  - Elements 208-227: Row card numbers (normalized)
  - Elements 228-247: Row card penalties (normalized)
  - Elements 248-251: Row penalty totals (normalized)
  - Element 252: Player penalty pile total (normalized)
- **Action Space**: 108 actions (104 cards + 4 row choices)
- **Game Rules**: Standard Take 5 with penalty minimization

## License

MIT License - see LICENSE file for details.
