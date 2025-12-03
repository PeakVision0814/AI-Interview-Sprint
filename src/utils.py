# src/utils.py
import os
import torch
import sys
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_logger(log_file='training.log'):
    """
    创建一个 logger，既打印到终端，又保存到文件
    """
    # 1. 使用具体名字，隔离 Jupyter 默认 logger
    logger = logging.getLogger('training_logger') 
    
    # 2. 禁止向上传播，防止 Jupyter 界面重复打印
    logger.propagate = False 
    
    logger.setLevel(logging.INFO)

    # 3. 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode='a') 
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    # 4. 【关键修正】只在没有 handler 时添加，且只添加一次
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    # 删除原代码中这里多余的 logger.addHandler(file_handler)
    
    return logger


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