import time
import torch
from datetime import datetime

print("=" * 60)
print("Turmite GPU test Start")
print("=" * 60)

# Basic system info
print(f"Current time: {datetime.now()}")
print(f"PyTorch version: {torch.__version__}")

# CUDA check
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()

    print(f"Number of GPUs visible: {device_count}")
    print(f"Current CUDA device index: {current_device}")
    print(f"GPU name: {torch.cuda.get_device_name(current_device)}")

    # Allocate a tensor on GPU
    x = torch.rand((5000, 5000), device="cuda")

    print("\nAllocated tensor on GPU.")
    print(f"Tensor device: {x.device}")

else:
    print("Running on CPU only.")


print("=" * 60)
print("Turmite GPU test DONE")
print("=" * 60)

