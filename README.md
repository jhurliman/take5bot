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

2. **Install dependencies:**
   ```bash
   # For CPU-only training
   pip install -e ".[train]"
   
   # For CUDA-enabled training (Linux/Windows)
   pip install -e ".[train-cuda]" --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify installation:**
   ```bash
   python verify_setup.py
   ```

### CUDA Setup (GPU Training)

For GPU-accelerated training on Linux and Windows:

#### Automated Setup
```bash
# Linux
./setup_cuda.sh

# Windows
setup_cuda.bat

# Cross-platform Python script
python setup_cuda.py
```

#### Manual Setup
```bash
# Install CUDA PyTorch (replace cu121 with your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
pip install "LightZero>=0.2.0"
```

### Prerequisites for CUDA
- NVIDIA GPU with compute capability 3.5+
- NVIDIA GPU drivers (latest recommended)
- CUDA Toolkit 11.8 or 12.1+ (optional, PyTorch includes CUDA runtime)

### Usage

```python
import torch
from take5bot import YourAgent

# Automatically use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize your agent
agent = YourAgent().to(device)

# Train with GPU acceleration
agent.train()
```

## Documentation

- [CUDA Setup Guide](CUDA_SETUP.md) - Detailed CUDA installation instructions
- [Training Guide](docs/training.md) - How to train agents
- [API Reference](docs/api.md) - Code documentation

## Project Structure

```
take5bot/
├── setup_cuda.py          # Cross-platform CUDA setup script
├── setup_cuda.sh           # Linux CUDA setup script
├── setup_cuda.bat          # Windows CUDA setup script
├── verify_setup.py         # Installation verification
├── CUDA_SETUP.md          # Detailed CUDA setup guide
├── pyproject.toml         # Project configuration
└── src/                   # Source code
```

## Features

- **OpenSpiel Integration**: Standard Take 5 game implementation
- **LightZero Training**: State-of-the-art reinforcement learning
- **CUDA Support**: GPU-accelerated training on Linux and Windows
- **Cross-Platform**: Works on Linux, Windows, and macOS
- **Easy Setup**: Automated installation scripts

## Requirements

- Python 3.10+
- PyTorch 2.0+
- OpenSpiel 1.2+
- LightZero 0.2+
- CUDA 11.8+ (for GPU support)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
