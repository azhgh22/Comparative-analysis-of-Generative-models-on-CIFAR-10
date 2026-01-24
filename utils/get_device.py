import torch

def get_device() -> str:
    """
    Safely determine the best available device

    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  CUDA version: {torch.version.cuda}")

    # Check for MPS (Apple Silicon GPUs - M1/M2/M3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print(f"✓ MPS (Apple Silicon) is available")

    # Fallback to CPU
    else:
        device = 'cpu'
        print(f"✗ No GPU available, using CPU")

    return device