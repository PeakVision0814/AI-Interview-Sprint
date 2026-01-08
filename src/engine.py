# src/rag/engine.py
import os
from typing import List, Optional
import json

# 引入配置
from src.config import config
# 引入工具 (确保这些模块你已经搬运到了 src/llm 和 src/rag 下)
from src.llm.client import LLMClient         # 需自行封装或使用 openai 原生
from src.rag.vector_db import VectorDBHandler # 需自行封装
from src.rag.embedding import EmbeddingModel  # 需自行封装

class RAGPipeline:
    def __init__(self):
        """初始化 RAG 流水线的所有组件"""
        print(f"⚙️ 初始化 RAG Pipeline (Model: {config.LLM_MODEL_NAME})...")
        
        # 1. 加载 Embedding 模型 (比如 SentenceTransformer)
        # 注意：这里需要确保 EmbeddingModel 类已实现
        self.embedding_model = EmbeddingModel(model_name=config.EMBEDDING_MODEL)
        
        # 2. 连接向量数据库 (Chroma)
        # 注意：这里需要确保 VectorDBHandler 类已实现
        self.vector_db = VectorDBHandler(
            persist_directory=str(config.CHROMA_DB_DIR), # Path转str，防止报错
            embedding_fn=self.embedding_model
        )
        
        # 3. 初始化 LLM 客户端 (DeepSeek / SiliconFlow)
        # 如果你还没有封装 LLMClient，可以直接在这里用 OpenAI(api_key=...)
        self.llm = LLMClient(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            model_name=config.LLM_MODEL_NAME
        )
    
    def ingest_documents(self, file_path: str):
        """
        [ETL] 数据入库流程: 读取 -> 切分 -> 向量化 -> 存储
        """
        print(f"📥 [ETL] 正在处理文件: {file_path}")
        # 这里应该调用 TextSplitter
        # 为了演示，暂且假设输入就是列表
        # 实际项目中：chunks = text_splitter.split_documents(load(file_path))
        chunks = [f"这是从 {file_path} 读取的测试片段..."] 
        
        # 存入向量库
        self.vector_db.add_texts(chunks)
        print(f"✅ 入库完成，共 {len(chunks)} 个片段。")

    def query(self, user_query: str) -> str:
        """
        [Inference] RAG 核心链路：检索 + 生成
        """
        print(f"🔍 [RAG] 用户提问: {user_query}")
        
        # 1. 检索 (Retrieve)
        relevant_docs = self.vector_db.search(
            query=user_query, 
            top_k=config.RETRIEVAL_TOP_K
        )
        
        # 2. 构建上下文 (Augment)
        context_str = "\n\n".join(relevant_docs)
        if not context_str:
            context_str = "暂无相关文档。"
        
        # 3. 组装 Prompt
        system_prompt = config.RAG_SYSTEM_PROMPT.format(context=context_str)
        
        # 4. 生成 (Generate)
        print("🤖 [LLM] 正在生成回答...")
        answer = self.llm.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        
        return answer

# 单例模式
try:
    rag_engine = RAGPipeline()
except Exception as e:
    print(f"⚠️ RAG Engine 初始化失败 (可能是依赖组件未完成): {e}")
    rag_engine = None