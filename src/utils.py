import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

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