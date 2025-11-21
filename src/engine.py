# src/engine.py
import torch
import torch.nn as nn
from tqdm.auto import tqdm 

def train(model, device, train_loader, optimizer, epoch, log_interval=100, logger=None):
    """
    执行一个 Epoch 的训练 (带进度条和文件日志)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # 1. 创建进度条
    # leave=False: 跑完一轮后进度条消失，保持屏幕清爽
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=True)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # 2. 实时更新进度条尾部的 Loss 显示 (给人类看)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 3. 阶段性日志记录
        if (batch_idx + 1) % log_interval == 0:
            msg = f'Train Epoch: {epoch} [{batch_idx+1}/{len(train_loader)} ({100. * (batch_idx+1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
            
            # 写文件 (utils.py 里配置了只写文件)
            if logger:
                logger.info(msg)
            
            # 写屏幕 (使用 pbar.write，它会在进度条上方插入一行，不会打断进度条)
            # pbar.write(msg)

def evaluate(model, device, test_loader, logger=None):
    """
    执行测试集评估 (带进度条和文件日志)
    """
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    # 1. 评估也加个进度条，体验更统一
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算指标
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # 准备结果消息
    msg = f'\n🔴 [Test set] Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n'
    
    # 2. 写文件
    if logger:
        logger.info(msg)
    
    # 3. 写屏幕
    # 因为评估循环已经结束，进度条(leave=False)已经消失了，所以直接 print 也没问题
    print(msg)
    
    return accuracy