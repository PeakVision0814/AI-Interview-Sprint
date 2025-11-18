import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # --- 卷积块1 ---
        # 输入: [B, 1, 28, 28]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,stride=1, padding=1)
        # 经过 conv1: [B, 16, 28, 28]
        # 经过 ReLU: [B, 16, 28, 28]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 经过 pool1: [B, 16, 14, 14]

        # --- 卷积块 2 ---
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 经过 conv2: [B, 32, 14, 14]
        # 经过 ReLU: [B, 32, 14, 14]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 经过 pool2: [B, 32, 7, 7]

        # --- 全连接 (分类) 块 ---
        # 我们需要将 [B, 32, 7, 7] 压平为 [B, 32 * 7 * 7] 即 [B, 1568]
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10) # MNIST 是 10 分类

    def forward(self, x):
        # 卷积块1
        x = self.conv1(x)   
        # print(f"1. 经过 Conv1: {x.shape}") # 这里会显示 [B, 16, 28, 28]
        x = F.relu(x)
        x = self.pool1(x)
        # print(f"2. 经过 Pool1: {x.shape}") # 这里会显示 [B, 16, 14, 14]

        # 卷积块2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # 压平
        x = x.view(x.size(0), -1)  # 或者使用 nn
        # 全连接块
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)     # 输出 logits
        return x