#!/usr/bin/env python
"""Test if ML packages are installed correctly."""

print("Testing ML package availability...\n")

# Test PyTorch
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} - INSTALLED")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except ImportError as e:
    print(f"❌ PyTorch - NOT INSTALLED")
    print(f"   Error: {e}")

# Test scikit-learn
try:
    import sklearn
    print(f"\n✅ scikit-learn {sklearn.__version__} - INSTALLED")
except ImportError as e:
    print(f"\n❌ scikit-learn - NOT INSTALLED")
    print(f"   Error: {e}")
    print(f"\n   To install: conda install -n ddstartupenv scikit-learn")

# Test if both are available
try:
    import torch
    import sklearn
    print(f"\n✅ All ML packages ready for pairwise plots!")
except ImportError:
    print(f"\n⚠️  Some packages missing - ML pairwise plots will be disabled")
