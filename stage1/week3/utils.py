# utils.py (更新版)
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64, data_root='../../data'):
    """
    下载并加载 MNIST 数据集
    :param batch_size: 批次大小
    :param data_root: 数据集存放路径 (默认指向你的旧数据位置)
    """
    # 1. 定义变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. 加载 Dataset (使用传入的 data_root)
    # 增加一个容错：如果 ../../data 不存在，代码不会直接崩溃，而是可能下载到当前目录，
    # 但为了利用你的旧数据，尽量确保路径正确。
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # 3. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # 测试代码
    print(f"尝试从 ../../data 加载数据...")
    try:
        train_loader, _ = get_data_loaders()
        print(f"✅ 加载成功! Batch size: {train_loader.batch_size}")
        print(f"数据位置: {train_loader.dataset.root}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")