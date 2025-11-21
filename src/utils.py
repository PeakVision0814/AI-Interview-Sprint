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
    # 1. 【关键修改】使用一个具体的名字，不要用空名字(Root)
    # 这样就创建了一个独立的 logger，不再受 Jupyter 默认设置的干扰
    logger = logging.getLogger('training_logger') 
    
    # 2. 【关键修改】禁止日志“向上传播”给 Root Logger
    # 这行代码彻底堵死了日志泄露到屏幕的路径
    logger.propagate = False 
    
    logger.setLevel(logging.INFO)

    # 3. 创建文件处理器 (FileHandler) -> 写文件
    file_handler = logging.FileHandler(log_file, mode='a') 
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    # 4. 添加到 logger
    # 因为我们用了新的名字，所以不需要 logger.handlers = [] 清空了，它是全新的
    # 为了防止多次运行重复添加，先判断一下
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    # 注意：这里依然不要加 StreamHandler (控制台)，全权交给 tqdm 处理

    logger.addHandler(file_handler)
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