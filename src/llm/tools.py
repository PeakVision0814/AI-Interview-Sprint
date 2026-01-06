# src/llm/tools.py
import json

# 尝试导入你之前写的 RAG 组件
# 如果之前的代码路径不同，请调整 import
try:
    # 假设你在 src.rag.vector_db 里有个 query_vector_db 函数
    # from src.rag.vector_db import query_vector_db 
    pass 
except ImportError:
    pass

def search_knowledge_base(query: str) -> str:
    """
    根据用户的查询，在本地向量知识库(Vector DB)中检索相关文档。
    """
    print(f"🔍 [Tool]: 正在知识库中检索: {query} ...")
    
    # --- 这里是连接真实 RAG 的接口 ---
    # 真实场景：results = query_vector_db(query, top_k=3)
    # 真实场景：return json.dumps(results)
    
    # --- 模拟数据 (Mock) ---
    # 为了今天先跑通 Agent 逻辑，我们先返回模拟数据
    mock_db = {
        "transformer": "Transformer 是一种基于自注意力机制(Self-Attention)的深度学习模型，由 Google 在 2017 年提出。",
        "rag": "RAG (Retrieval-Augmented Generation) 是一种结合了检索和生成的架构，用于解决 LLM 的幻觉问题。",
        "resnet": "ResNet (残差网络) 通过引入 Skip Connection 解决了深层网络难以训练的问题。"
    }
    
    for key, value in mock_db.items():
        if key in query.lower():
            return json.dumps({"status": "success", "content": value})
            
    return json.dumps({"status": "empty", "content": "知识库中未找到相关内容，请尝试换个关键词。"})

# 定义工具 Schema
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "当用户询问具体的技术概念、定义或需要查阅内部文档时调用此工具。不要用于日常闲聊。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于检索的关键词或问题，例如 'Transformer原理' ",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

# 函数映射表
AVAILABLE_FUNCTIONS = {
    "search_knowledge_base": search_knowledge_base,
}