import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple

class RerankClient:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化重排序模型 (Cross-Encoder)
        """
        print(f"Loading Reranker model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval() # 开启推理模式
        
        # 如果有 GPU，转到 GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Reranker loaded on {self.device}")

    def rank(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        对文档列表进行重排序
        Args:
            query: 用户的问题
            documents: 向量检索回来的候选文档列表 (Strings)
            top_k: 返回前 k 个最好的
        Returns:
            List of (document_content, score), sorted by score descending
        """
        if not documents:
            return []

        # 1. 构建输入对: [[Query, Doc1], [Query, Doc2], ...]
        pairs = [[query, doc] for doc in documents]

        # 2. Tokenize (自动处理 [CLS] 和 [SEP])
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(self.device)

            # 3. 计算分数 (Forward Pass)
            # BGE Reranker 输出的是 logits，数值范围不限，越大越好
            scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()

        # 4. 排序
        # 将文档和分数打包
        results = list(zip(documents, scores.cpu().tolist()))
        
        # 按分数从高到低排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

# 简单的测试代码
if __name__ == "__main__":
    client = RerankClient()
    q = "这就去办"
    docs = ["好的，马上处理", "我不明白你的意思", "苹果是一种水果", "这就去办的相关政策"]
    print(client.rank(q, docs))
