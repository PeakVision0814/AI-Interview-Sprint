from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """
    RAG ETL Pipeline 中的切分模块
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 针对中文优化分隔符
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )

    def split(self, text: str) -> List[str]:
        """
        输入原始长文本，输出文本块列表
        """
        if not text:
            return []
        return self.splitter.split_text(text)

# 使用示例 (便于测试)
if __name__ == "__main__":
    text = "测试文本..." * 50
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split(text)
    print(f"Generated {len(chunks)} chunks.")
