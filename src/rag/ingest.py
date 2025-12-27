import os
import sys
# 路径处理：确保能导入 src 下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.rag.etl import TextChunker
from src.rag.vector_db import VectorDB

def load_and_process_file(file_path: str):
    """
    Step 1: Extract - 读取文件
    """
    print(f"📄 Processing file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], []
    
    # 简单的元数据提取 (实际工程中可能需要解析文件名或正则提取章节号)
    file_name = os.path.basename(file_path)
    
    """
    Step 2: Transform - 切分与清洗
    """
    # 初始化我们 Day 1 写的切分器
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.split(text)
    
    # 构造 Metadata
    metadatas = []
    for i, chunk in enumerate(chunks):
        metadatas.append({
            "source": file_name,
            "chunk_id": i,
            "length": len(chunk)
        })
    
    print(f"✂️  Split into {len(chunks)} chunks.")
    return chunks, metadatas

def main():
    # 配置
    DATA_DIR = "data"
    DB_PATH = "chroma_db_data" # 注意这里要和 Day 3 保持一致
    
    # 初始化组件
    print("🚀 Starting ETL Pipeline...")
    
    # 初始化 DB (Day 3 的组件)
    # 注意：如果目录存在，它会加载旧数据；如果不存在，会新建
    vector_db = VectorDB(persist_path=DB_PATH)
    
    # 遍历 data 目录下的所有 txt 文件
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)
            
            # 1. 处理数据
            chunks, metadatas = load_and_process_file(file_path)
            
            if chunks:
                # 2. Step 3: Load - 存入数据库
                print(f"💾 Ingesting to VectorDB...")
                vector_db.add_documents(chunks, metadatas)
                print(f"✅ Successfully ingested {filename}")

    print("🎉 ETL Pipeline Completed!")

    # --- 验证环节 ---
    print("\n🔍 Verifying with a test query...")
    results = vector_db.search("故意伤害致死怎么判？", top_k=1)
    for res in results:
        print(f"Answer found in [{res['metadata']['source']}]:")
        print(f"Content: {res['text'][:50]}...") # 只打印前50个字
        print(f"Distance: {res['distance']:.4f}")

if __name__ == "__main__":
    main()
