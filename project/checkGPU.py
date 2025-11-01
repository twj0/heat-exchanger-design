# !/usr/bin/env python3
import torch

def check_gpu():
    """Check GPU availability and print details."""
    print("Checking GPU availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("nvcc --version")
        print("torch.version.cuda:", torch.version.cuda)
        print("CUDA is not available. PyTorch will use CPU.")

if __name__ == "__main__":
    check_gpu()
