import torch
import torch.nn as nn
import torch.nn.functional as F # 通常F被用于调用无参数的函数式API
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(batch_size=64, root='../../data'):
    """
    一个创建并返回 MNIST 训练和测试 DataLoader 的函数。
    """
    
    # 1. 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 2. [Dataset] 加载训练集
    train_dataset = datasets.MNIST(
        root=root, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 3. [Dataset] 加载测试集 (好习惯，我们 W3 会用到)
    test_dataset = datasets.MNIST(
        root=root,
        train=False, # train=False 表示获取测试集
        download=True,
        transform=transform
    )

    # 4. [DataLoader] 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,    # 训练时打乱
        num_workers=4    # (在 WSL/Linux 上设为 4 或 8, 在 Windows 本机上设为 0)
    )
    
    # 5. [DataLoader] 创建测试数据加载器
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,   # 测试时不需要打乱
        num_workers=4
    )
    
    print(f"数据加载器创建成功：")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  测试集样本数: {len(test_dataset)}")
    
    # 6. 返回这两个加载器
    return train_loader, test_loader


# 1. 定义我们自己的模型类，它必须继承 nn.Module
class MyMLP(nn.Module):
    
    # 2. [__init__]: 声明模型将用到的所有“层”
    def __init__(self):
        super(MyMLP, self).__init__() # 必须的“开场白”
        
        # 声明第一个全连接层 (in_features=784, out_features=128)
        self.fc1 = nn.Linear(784, 128) 
        
        # 声明第二个全连接层 (in_features=128, out_features=10)
        # 注意：这一层的 in_features 必须等于上一层的 out_features
        self.fc2 = nn.Linear(128, 10)
        
        # 我们也可以在这里声明激活函数层
        # self.relu = nn.ReLU() 
        # (但我们待会儿在 forward 中将使用 F.relu() 来展示另一种方式)

    # 3. [forward]: 定义数据如何流过这些层
    def forward(self, x):
        # x 的输入形状: [batch_size, 784]
        
        # 数据流过第一个全连接层
        x = self.fc1(x) # 形状变为 [batch_size, 128]
        
        # 应用 ReLU 激活函数
        # 方式1: 使用在 __init__ 中定义的 nn.ReLU()
        # x = self.relu(x) 
        # 方式2: 使用 functional API (更常见，因为它没有参数)
        x = F.relu(x)
        
        # 数据流过第二个全连接层
        x = self.fc2(x) # 形状变为 [batch_size, 10]
        
        # 输出层通常不加激活，因为 nn.CrossEntropyLoss 会帮我们处理
        return x
    


