import os
from pathlib import Path
from dotenv import load_dotenv

# 0. 加载环境变量 (.env)
load_dotenv()

class Config:
    """
    全局配置类
    使用类变量管理，方便在其他模块中通过 config.XXX 调用
    """
    # ==========================
    # 1. 路径配置 (Path Configuration)
    # ==========================
    # Project Root: project/src/config.py -> parent: src -> parent: project
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Data Dirs
    DATA_DIR = PROJECT_ROOT / "data"
    # RAG 专用: 向量数据库路径
    CHROMA_DB_DIR = DATA_DIR / "chroma_db_data" 
    
    # 之前的 BERT 相关路径 (保留，以备后用)
    PRETRAINED_MODEL_DIR = DATA_DIR / "pretrained_models/bert-base-chinese"
    RAW_DATA_DIR = DATA_DIR / "raw_data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed_data"

    # Output Dirs
    OUTPUT_DIR = PROJECT_ROOT / "output"
    TRAIN_DIR = OUTPUT_DIR / "train"
    CHECKPOINT_DIR = TRAIN_DIR / "checkpoints"
    LOG_DIR = TRAIN_DIR / "logs"
    MODEL_SAVE_DIR = TRAIN_DIR / "saved_model"
    
    TEST_DIR = OUTPUT_DIR / "test"
    ANALYSIS_DIR = OUTPUT_DIR / "analysis"

    # ==========================
    # 2. LLM & RAG 参数 (New!)
    # ==========================
    # API 配置 (优先读取 DeepSeek，没有则读取 SiliconFlow)
    LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    # 注意：如果用 DeepSeek 官方，Base URL 是 https://api.deepseek.com
    LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or "https://api.siliconflow.cn/v1"
    LLM_MODEL_NAME = "deepseek-chat"  # 或者 "deepseek-ai/DeepSeek-V2.5"
    
    # Embedding & 向量库配置
    # 这是一个轻量级、支持中文的 Embedding 模型，适合本地运行
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # 切片策略
    CHUNK_SIZE = 500       # 每个文本块的字符数
    CHUNK_OVERLAP = 50     # 重叠字符数，防止上下文切断
    RETRIEVAL_TOP_K = 3    # 每次 RAG 检索回来的文档数量

    # ==========================
    # 3. 训练超参数 (Training Hyperparams)
    # ==========================
    MAX_LEN = 128
    BATCH_SIZE = 32

    # ==========================
    # 4. Prompt Templates
    # ==========================
    RAG_SYSTEM_PROMPT = """
    你是一个基于知识库的智能助手。请严格根据以下【参考文档】回答用户问题。
    如果参考文档中没有答案，请直接说"不知道"，不要编造。
    
    【参考文档】:
    {context}
    """

    @classmethod
    def init_directories(cls):
        """
        初始化目录结构的工具函数
        调用方式: Config.init_directories()
        """
        paths_to_create = [
            cls.CHROMA_DB_DIR,
            cls.CHECKPOINT_DIR, 
            cls.LOG_DIR, 
            cls.MODEL_SAVE_DIR, 
            cls.TEST_DIR, 
            cls.ANALYSIS_DIR
        ]
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory ready: {path}")

# 实例化单例，方便导出
# 以后在别的代码里：from src.config import config
config = Config()