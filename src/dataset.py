import torch
from torch.utils.data import Dataset

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        """
        :param texts: 文本列表 ["打开灯", "向左转"]
        :param labels: 标签列表 [0, 1]
        :param tokenizer: 分词器实例
        :param max_len: 最大截断长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length', # 注意：这里暂时用 max_length，后面会讲动态 padding
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Trainer 要求返回字典，且 key 必须能对应上模型 forward 的参数名
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }