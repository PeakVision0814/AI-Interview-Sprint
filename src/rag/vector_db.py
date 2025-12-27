import chromadb
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.abspath("../../"))


from src.rag.embedding import EmbeddingModel

# 再次定义适配器 (为了保持 vector_db.py 的独立性)
class MyCustomEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self):
        self.model = EmbeddingModel()

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.get_embeddings(input)

class VectorDB:
    def __init__(self, collection_name: str = "knowledge_base", persist_path: str = "./chroma_db"):
        """
        初始化向量数据库
        """
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedding_fn = MyCustomEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            # HNSW 参数通常默认即可，如果有性能需求可在此调整 metadata={"hnsw:space": "cosine"}
            metadata={"hnsw:space": "cosine"} 
        )

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str] = None):
        """
        批量添加文档
        """
        if ids is None:
            # 如果没给 ID，就用哈希生成，或者简单的 index
            ids = [str(hash(text)) for text in texts]
            
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added {len(texts)} documents to collection.")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        语义搜索
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # 整理 Chroma 的返回格式，使其更易读
        structured_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                item = {
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "id": results['ids'][0][i]
                }
                structured_results.append(item)
        
        return structured_results

# 测试代码
if __name__ == "__main__":
    db = VectorDB(persist_path="./test_db")
    db.add_documents(
        texts=["测试文档一", "测试文档二"],
        metadatas=[{"id": 1}, {"id": 2}],
        ids=["1", "2"]
    )
    res = db.search("测试")
    print(res)
