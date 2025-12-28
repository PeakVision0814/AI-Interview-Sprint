import os
# 假设你之前写好了这些模块
# from embedding_utils import get_embedding
# from vector_db_utils import search_similar_chunks
# from llm_utils import call_llm_api

# --- 模拟模块 (如果你还没有封装好，请替换为真实调用) ---
def mock_get_embedding(text):
    # 实际应用中调用 OpenAI/HuggingFace 接口
    return [0.1, 0.2, 0.3] 

def mock_vector_search(query_vec, top_k=3):
    # 实际应用中在你的向量数据库(FAISS/Chroma/List)中查找
    # 这里我们硬编码，模拟针对"盗窃罪数额巨大"的检索结果
    return [
        "《最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释》第一条：盗窃公私财物价值一千元至三千元以上、三万元至十万元以上、三十万元至五十万元以上的，应当分别认定为刑法第二百六十四条规定的“数额较大”、“数额巨大”、“数额特别巨大”。",
        "刑法第二百六十四条：盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金...",
    ]

def call_llm(system_prompt, user_prompt):
    # 这里调用你的 LLM (DeepSeek/GPT/Gemini)
    # 下面是一个模拟的打印，展示发送给 LLM 的最终 Prompt 长什么样
    print("\n" + "="*20 + " DEBUG: 发送给 LLM 的 Prompt " + "="*20)
    print(f"【System】: {system_prompt}")
    print(f"【User】: {user_prompt}")
    print("="*60 + "\n")
    
    # 模拟 LLM 的回答
    return "根据司法解释，盗窃公私财物价值三万元至十万元以上的，应当认定为“数额巨大”。"

# --- 核心 RAG 流程 ---

def main():
    print("⚖️  法律小助手 v0.2 (RAG Enabled) - 输入 'exit' 退出")
    
    # 1. 定义 System Prompt (增加严格限制)
    system_prompt = (
        "你是一个专业的法律助手。"
        "请仅根据以下提供的【参考上下文】回答问题，"
        "如果上下文中没有答案，请说不知道，严禁编造。"
    )

    while True:
        user_query = input("\n请输入法律问题: ")
        if user_query.strip().lower() == 'exit':
            break

        print("正在检索相关法律条文...")

        # 2. Embedding (User Input -> Vector)
        query_vector = mock_get_embedding(user_query)

        # 3. Vector Search (Vector -> Top-K Chunks)
        # 这一步是 RAG 的关键：只把相关的知识拿出来
        relevant_chunks = mock_vector_search(query_vector, top_k=2)
        
        # 将检索到的文本拼接成一个字符串
        context_str = "\n\n".join(relevant_chunks)

        # 4. Construct Prompt (System + Context + User Input)
        # 我们把检索到的内容塞给用户 Prompt，或者放在 System Prompt 里都可以
        # 这里采用常见的结构：
        final_user_prompt = f"【参考上下文】:\n{context_str}\n\n【用户问题】: {user_query}"

        # 5. LLM Generation
        answer = call_llm(system_prompt, final_user_prompt)
        
        print(f"🤖 回答: {answer}")

if __name__ == "__main__":
    main()