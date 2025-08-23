#!/usr/bin/env python3
"""
Test script to verify GPU detection and PyTorch installation
"""

import sys
import torch

def test_gpu():
    """Test GPU detection and PyTorch installation"""
    print("GPU Detection Test")
    print("=" * 30)
    
    # Test PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
        
        # Test GPU memory
        if torch.cuda.device_count() > 0:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU memory: {gpu_memory:.1f} GB")
    else:
        print("No CUDA GPU detected")
        print("This could be because:")
        print("1. No NVIDIA GPU is installed")
        print("2. CUDA drivers are not installed")
        print("3. PyTorch was installed without CUDA support")
        print("4. Virtual environment is not activated")
    
    # Test device selection
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Selected device: {device}")
    
    # Test tensor creation on device
    try:
        x = torch.randn(3, 3, device=device)
        print(f"Successfully created tensor on {device}")
        print(f"Tensor device: {x.device}")
    except Exception as e:
        print(f"Error creating tensor on {device}: {e}")
    
    return cuda_available

if __name__ == "__main__":
    success = test_gpu()
    if success:
        print("\n✓ GPU test passed!")
    else:
        print("\n✗ GPU test failed - will use CPU")
    
    print("\nPress Enter to exit...")
    input()
