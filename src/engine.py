import torch
import torch.nn as nn

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    """
    执行一个 Epoch 的训练
    """
    model.train()  # 1. 切换到训练模式
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 搬运数据
        data, target = data.to(device), target.to(device)
        
        # 五步法
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # 打印日志
        # batch_idx 是当前是第几批数据 (0, 1, 2...)
        # (batch_idx + 1) % 100 == 0 意思是：每当处理完 100 批数据，就打印一次
        if (batch_idx + 1) % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx+1}/{len(train_loader)} ({100. * (batch_idx+1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def evaluate(model, device, test_loader):
    """
    执行测试集评估
    """
    model.eval()   # 1. 切换到评估模式
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum') # 累加 Loss
    
    with torch.no_grad(): # 2. 阻断梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            test_loss += criterion(output, target).item() # 累加 Loss
            pred = output.argmax(dim=1, keepdim=True)     # 3. 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\n🔴 [Test set] Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy