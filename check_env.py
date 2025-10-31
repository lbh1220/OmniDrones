import sys
import os
import torch

print("--- Environment Check ---")

# 1. 检查 Python 可执行文件
print(f"\n[Python Executable Path]")
print(sys.executable)

# 2. 检查 PyTorch
print(f"\n[PyTorch Version]")
print(torch.__version__)
print(f"\n[PyTorch File Path]")
print(torch.__file__)

# 3. 检查 PyTorch 使用的 cuDNN
print(f"\n[PyTorch cuDNN Available?]")
print(torch.backends.cudnn.is_available())
print(f"\n[PyTorch cuDNN Version]")
print(torch.backends.cudnn.version())

# 4. 检查 'nvidia.cudnn' 包（如果安装了）
try:
    from nvidia.cudnn import lib
    print(f"\n[nvidia.cudnn Package Path]")
    print(os.path.dirname(lib.__file__))
except ImportError:
    print(f"\n[nvidia.cudnn Package Path]")
    print("Python package 'nvidia-cudnn' not found in this env.")

print("\n--- End of Check ---")