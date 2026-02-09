import torch
import sys

def check_gpu():
    print("=" * 40)
    print("GPU DIAGNOSTICS")
    print("=" * 40)
    print(f"Python version: {sys.version.split()[0]}")
    try:
        print(f"PyTorch version: {torch.__version__}")
    except Exception as e:
        print(f"Error checking PyTorch version: {e}")
        return

    is_available = torch.cuda.is_available()
    print(f"\nCUDA available: {is_available}")

    if is_available:
        try:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDNN version: {torch.backends.cudnn.version()}")
            device_count = torch.cuda.device_count()
            print(f"GPU Device Count: {device_count}")
            for i in range(device_count):
                prop = torch.cuda.get_device_properties(i)
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {prop.total_memory / 1024**3:.2f} GB")
                print(f"  Compute Capability: {prop.major}.{prop.minor}")

            # Simple Tensor Test
            try:
                x = torch.tensor([1.0, 2.0]).cuda()
                print("\n[OK] Tensor allocation on GPU successful.")
            except Exception as e:
                print(f"\n[FAIL] Tensor allocation on GPU failed: {e}")
        except Exception as e:
            print(f"Error accessing CUDA info: {e}")
    else:
        print("\n[WARNING] No NVIDIA GPU detected or PyTorch is not compiled with CUDA support.")
        print("Potential Reasons:")
        print("1. NVIDIA Driver is missing or outdated.")
        print("2. You installed the CPU-only version of PyTorch.")
        print("3. CUDA Toolkit is not installed or not in PATH.")

        print("\nTo reinstall PyTorch with CUDA support (example for CUDA 11.8):")
        print("pip uninstall torch torchvision torchaudio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_gpu()
