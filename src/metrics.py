import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_pred):
    """
    Hugging Face Trainer 专用的回调函数
    :param eval_pred: 包含 (logits, labels) 的元组
    """
    logits, labels = eval_pred
    
    # 1. 获取预测类别 (Argmax)
    # logits shape: [batch_size, num_labels] -> predictions shape: [batch_size]
    predictions = np.argmax(logits, axis=-1)
    
    # 2. 计算基础指标
    acc = accuracy_score(labels, predictions)
    
    # 3. 计算 F1-Score (加权平均)
    # 'weighted': 考虑类别不平衡，样本多的类别权重高
    # 'macro': 不考虑样本数量，所有类别平等（适合关注少数类）
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }