# ushd2

usv
更新：2025.8.3 usv-scheduling-hgnn-drl/ 
├── data/ # 数据存储
├── env/ # 环境定义 
├── graph/ # 图神经网络模块 
├── model/ # 模型保存 
├── results/ # 结果存储 
└── utils/ # 工具函数

依赖 python 3.8.1及以上 
pip install torch torch_geometric gym numpy pandas matplotlib tqdm dgl

===== 环境配置检查 =====
Python版本: 3.8.10
PyTorch版本: 2.0.1+cu117
CUDA版本: 11.7
cuDNN版本: 8500
torch_geometric版本: 2.6.1
NumPy版本: 1.24.4
Pandas版本: 2.0.3
Matplotlib版本: 3.7.5
Gym版本: 0.26.2
TQDM版本: 4.67.1
DGL版本: 1.1.2+cu117

pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://download.pytorch.org/whl/cu117

===== CUDA配置 =====
GPU数量: 1
当前GPU: NVIDIA GeForce RTX 2060
GPU内存: 6144 MB

DGL后端: pytorch

DGL基本功能测试: 通过
