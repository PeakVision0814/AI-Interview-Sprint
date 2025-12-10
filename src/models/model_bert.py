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
        
        # 1. 加载预训练的 BERT Backbone (脊柱)
        # config.json 和 model.safetensors 必须在 model_path 下
        self.bert = BertModel.from_pretrained(model_path)
        
        # 2. 定义分类头 (Classification Head)
        # BERT base 的 hidden_size 是 768
        # 这里的逻辑是: [Batch, 768] -> [Batch, num_labels]
        self.classifier = nn.Linear(768, num_labels)
        
        # (可选) Dropout 层，防止过拟合
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        前向传播
        """
        # 1. BERT 编码
        # output[0]: last_hidden_state (Batch, Seq, 768)
        # output[1]: pooler_output (Batch, 768) -> 这是 CLS 经过加工后的向量，适合做分类
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooler_output = outputs.pooler_output
        
        # 2. 经过 Dropout
        pooler_output = self.dropout(pooler_output)
        
        # 3. 经过线性层，得到 Logits
        logits = self.classifier(pooler_output)
        
        return logits