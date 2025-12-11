# src/models/model_bert
import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, model_path, num_labels):
        """
        初始化 BERT 分类器
        :param model_path: 本地预训练模型的路径 (str)
        :param num_labels: 分类的类别数 (int)
        """
        super(BertClassifier, self).__init__()
        
        # 1. 加载预训练的 BERT Backbone
        self.bert = BertModel.from_pretrained(model_path)
        
        # 2. 定义分类头
        self.classifier = nn.Linear(768, num_labels)
        
        # 3. Dropout
        self.dropout = nn.Dropout(0.1)

        # ------------------------------------------------------------------
        # ★★★ 新增修改 A：定义加权 Loss (Weighted Loss) ★★★
        # ------------------------------------------------------------------
        # 场景假设：
        # Label 0 (MOVE): 样本很多 -> 权重 1.0 (正常)
        # Label 1 (STOP): 样本很少 -> 权重 5.0 (模型错判它惩罚会更重)
        # Label 2 (GRAB): 样本中等 -> 权重 2.0 
        
        # 注意：这里的权重数量必须等于 num_labels
        # 在实际项目中，这个 tensor 通常是通过计算训练集样本比例自动生成的
        # 这里为了演示，我们先硬编码 (Hardcode)
        
        # device 兼容性处理：模型在哪里，权重 tensor 就得去哪里，所以这里暂时不 .to(device)
        # 我们会在 forward 里面处理 device 问题
        self.class_weights = torch.tensor([1.0, 5.0, 2.0]) 
        
        # 初始化 Loss 函数 (不带 reduction，稍后解释)
        # 注意：这里不能直接传 weight，因为 init 时还没确定 device
        self.loss_fct = nn.CrossEntropyLoss() 

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooler_output = outputs.pooler_output
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        
        loss = None
        if labels is not None:
            # --------------------------------------------------------------
            # ★★★ 新增修改 B：使用带权重的 Loss ★★★
            # --------------------------------------------------------------
            
            # 1. 确保权重 Tensor 和输入数据在同一个 Device (GPU/CPU) 上
            if self.class_weights.device != input_ids.device:
                self.class_weights = self.class_weights.to(input_ids.device)
                # 更新 loss 函数的权重
                self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            
            # 2. 计算 Loss
            # view(-1) 操作依然保留，确保维度匹配
            loss = self.loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            
            return (loss, logits)
        
        return logits