# PyTorch
#import torch
#print(torch.cuda.is_available())  # 输出True表示GPU可用

# TensorFlow
#import tensorflow as tf
#print(tf.config.list_physical_devices('GPU'))  # 应显示GPU设备信息

#import torch
#print(torch.__version__)      # PyTorch版本
#print(torch.version.cuda)
import sys
import torch
import numpy
import pandas
import matplotlib
import gym
import tqdm
import dgl# PyTorch
#import torch
#print(torch.cuda.is_available())  # 输出True表示GPU可用

# TensorFlow
#import tensorflow as tf
#print(tf.config.list_physical_devices('GPU'))  # 应显示GPU设备信息

#import torch
#print(torch.__version__)      # PyTorch版本
#print(torch.version.cuda)
import sys
import torch
import numpy
import pandas
import matplotlib
import gym
import tqdm
import dgl
import visdom

# 尝试导入torch_geometric，处理可能的导入错误
try:
    import torch_geometric
    torch_geometric_version = torch_geometric.__version__
except ImportError:
    torch_geometric_version = "未安装"

# 打印所有库的版本信息
print("===== 环境配置检查 =====")
print(f"Python版本: {sys.version.split()[0]}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")
print(f"torch_geometric版本: {torch_geometric_version}")
print(f"NumPy版本: {numpy.__version__}")
print(f"Pandas版本: {pandas.__version__}")
print(f"Matplotlib版本: {matplotlib.__version__}")
print(f"Gym版本: {gym.__version__}")
print(f"TQDM版本: {tqdm.__version__}")
print(f"DGL版本: {dgl.__version__}")
print(f"visdom版本: {visdom.__version__}")


# 检查PyTorch CUDA可用性
if torch.cuda.is_available():
    print("\n===== CUDA配置 =====")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")
else:
    print("\n警告: CUDA不可用，将使用CPU运行（速度较慢）")

# 检查DGL后端
try:
    import dgl.backend as F
    print(f"\nDGL后端: {F.backend_name}")
except Exception as e:
    print(f"\nDGL后端检查失败: {e}")

# 简单测试DGL功能
try:
    g = dgl.graph(([0, 1], [1, 2]))
    print("\nDGL基本功能测试: 通过")
except Exception as e:
    print(f"\nDGL功能测试失败: {e}")
import visdom

# 尝试导入torch_geometric，处理可能的导入错误
try:
    import torch_geometric
    torch_geometric_version = torch_geometric.__version__
except ImportError:
    torch_geometric_version = "未安装"

# 打印所有库的版本信息
print("===== 环境配置检查 =====")
print(f"Python版本: {sys.version.split()[0]}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")
print(f"torch_geometric版本: {torch_geometric_version}")
print(f"NumPy版本: {numpy.__version__}")
print(f"Pandas版本: {pandas.__version__}")
print(f"Matplotlib版本: {matplotlib.__version__}")
print(f"Gym版本: {gym.__version__}")
print(f"TQDM版本: {tqdm.__version__}")
print(f"DGL版本: {dgl.__version__}")
print(f"visdom版本: {visdom.__version__}")


# 检查PyTorch CUDA可用性
if torch.cuda.is_available():
    print("\n===== CUDA配置 =====")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")
else:
    print("\n警告: CUDA不可用，将使用CPU运行（速度较慢）")

# 检查DGL后端
try:
    import dgl.backend as F
    print(f"\nDGL后端: {F.backend_name}")
except Exception as e:
    print(f"\nDGL后端检查失败: {e}")

# 简单测试DGL功能
try:
    g = dgl.graph(([0, 1], [1, 2]))
    print("\nDGL基本功能测试: 通过")
except Exception as e:
    print(f"\nDGL功能测试失败: {e}")