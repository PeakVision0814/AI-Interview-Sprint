import os
from sentence_transformers import SentenceTransformer

# --- 关键修正 1: 全局注入镜像地址 ---
# 必须在 import SentenceTransformer 之前或初始化之前生效
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class EmbeddingModel:
    """
    Embedding 模型单例封装
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            
            # 模型名称
            model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
            print(f"Initializing Embedding Model: {model_name}...")
            
            try:
                # --- 关键修正 2: 优先尝试纯离线加载 ---
                # local_files_only=True 会强制库只看本地缓存，绝对不联网
                print("Attempting to load from local cache (Offline Mode)...")
                cls._model = SentenceTransformer(model_name, local_files_only=True)
                print("✅ Successfully loaded from local cache.")
                
            except Exception as e:
                print(f"⚠️ Local cache not found. Error: {e}")
                print("🌐 Attempting to download from HF Mirror...")
                
                # --- 关键修正 3: 降级方案 ---
                # 如果本地真没有，再通过镜像下载
                try:
                    cls._model = SentenceTransformer(model_name)
                    print("✅ Successfully downloaded and loaded.")
                except Exception as e2:
                    print(f"❌ Critical Error: Failed to load model. Check your network.")
                    raise e2

        return cls._instance

    def get_embedding(self, text: str) -> list:
        """获取单条文本的向量"""
        # normalize_embeddings=True 之后，点积等同于余弦相似度
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def get_embeddings(self, texts: list) -> list:
        """批量获取文本向量"""
        return self._model.encode(texts, normalize_embeddings=True).tolist()

# 测试代码
if __name__ == "__main__":
    try:
        embedder = EmbeddingModel()
        vec = embedder.get_embedding("测试一下")
        print(f"Vector dimension: {len(vec)}")
    except Exception as e:
        print(f"\nFATAL: {e}")