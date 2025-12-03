# src/utils.py
import os
import sys
import torch
import random
import logging
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_logger(log_file='training.log'):
    """
    创建一个 logger，既打印到终端，又保存到文件
    """
    logger = logging.getLogger('project_logger') 
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止 Jupyter 重复打印

    # 格式设置
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 只有在没有 handler 时才添加，防止重复添加导致一条日志打印多次
    if not logger.handlers:
        # 1. 文件输出 (FileHandler)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 2. 控制台输出 (StreamHandler) - 修复原代码无法在终端显示的问题
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_device():
    """
    自动获取当前设备（CUDA / MPS / CPU）
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():   # 适配 Mac /M1/M2/M3
        return torch.device("mps")
    else:
        return torch.device("cpu")

def seed_everything(seed=42):
    """
    固定所有随机种子，确保实验可复现 (Reproducibility)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(batch_size=64, data_root='../../data'):
    # 定义变换：转Tensor + 归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载 Dataset
    # 注意：这里的路径是相对于你运行脚本的位置
    # 我们的 Notebook 在 stage1/week3，所以 ../../data 是对的
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader