#!/usr/bin/env python3
"""
Simple verification script to check if GPU-accelerated PyTorch is available.
This script can be run non-interactively to verify the setup.
Supports CUDA (Linux/Windows) and MPS (macOS).
"""

import sys
import platform


def check_pytorch():
    """Check if PyTorch is installed and GPU acceleration is available."""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")

        system = platform.system()
        print(f"✓ System: {system}")

        # Check GPU availability based on platform
        gpu_available = False

        if system.lower() == "darwin":  # macOS
            mps_available = torch.backends.mps.is_available()
            print(f"✓ MPS available: {mps_available}")

            if mps_available:
                print("✓ Apple Silicon GPU detected - MPS acceleration enabled")
                gpu_available = True

                # Test MPS operations
                try:
                    device = torch.device('mps')
                    x = torch.randn(100, 100, device=device)
                    y = torch.randn(100, 100, device=device)
                    z = torch.mm(x, y)
                    print(f"✓ MPS computation test: PASSED (device: {z.device})")
                except Exception as e:
                    print(f"✗ MPS computation test: FAILED - {e}")
                    return False
            else:
                print("ℹ MPS not available on this macOS system")
        else:
            # Linux/Windows - check CUDA
            cuda_available = torch.cuda.is_available()
            print(f"✓ CUDA available: {cuda_available}")

            if cuda_available:
                print(f"✓ CUDA version: {torch.version.cuda}")
                print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
                gpu_available = True

                # List available GPUs
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    print(f"  GPU {i}: {gpu_name}")

                # Test basic GPU operations
                try:
                    device = torch.device('cuda:0')
                    x = torch.randn(100, 100, device=device)
                    y = torch.randn(100, 100, device=device)
                    z = torch.mm(x, y)
                    print(f"✓ CUDA computation test: PASSED (device: {z.device})")
                except Exception as e:
                    print(f"✗ CUDA computation test: FAILED - {e}")
                    return False
            else:
                print("ℹ CUDA not available on this system")

        if not gpu_available:
            print("ℹ Using CPU-only PyTorch")

        return True

    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        return False


def check_dependencies():
    """Check if training dependencies are available."""
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        print("✗ NumPy not found")
        return False

    try:
        import open_spiel
        print(f"✓ OpenSpiel available")
    except ImportError:
        print("ℹ OpenSpiel not found (install with: pip install open-spiel)")

    try:
        import ding
        print(f"✓ DI-engine (ding) available")
    except ImportError:
        print("ℹ DI-engine not found (install with training dependencies)")

    try:
        import lzero
        print(f"✓ LightZero available")
    except ImportError:
        print("ℹ LightZero not found (install with: pip install LightZero)")

    return True


def main():
    """Main verification function."""
    print("PyTorch GPU Setup Verification")
    print("=" * 40)
    print("Supports CUDA (Linux/Windows) and MPS (macOS)")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print()

    # Check PyTorch
    pytorch_ok = check_pytorch()
    print()

    # Check dependencies
    deps_ok = check_dependencies()
    print()

    if pytorch_ok and deps_ok:
        print("✓ All checks passed! Your setup is ready.")
        return 0
    else:
        print("✗ Some checks failed. Please review the setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
