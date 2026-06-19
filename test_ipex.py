import sys
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch is not installed.")

try:
    import intel_extension_for_pytorch as ipex
    print(f"IPEX version: {ipex.__version__}")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print(f"XPU detected: {torch.xpu.get_device_name(0)}")
    else:
        print("IPEX installed, but XPU NOT available.")
except ImportError:
    print("IPEX is not installed.")
except Exception as e:
    print(f"IPEX error: {e}")
