import sys
import os
import json
from typing import List

# --- 1. 环境设置 (Week 8 Day 1) ---
# 确保能找到 src 目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 导入我们自己造的轮子
from src.llm.client import LLMClient       # W8D1
from src.llm.memory import MemoryBuffer     # W8D5
from src.llm.parsers import JsonOutputParser # W8D3

# 加载环境变量
load_dotenv()

# --- 2. 定义数据结构 (Week 8 Day 3 - Structured Output) ---
class LegalVerdict(BaseModel):
    analysis: str = Field(description="案件的详细法律分析，包含思考过程")
    verdict: str = Field(description="初步判决建议，如：有期徒刑3年")
    confidence: float = Field(description="置信度，0.0到1.0之间")
    laws: List[str] = Field(description="涉及的相关法律条款列表")

# --- 3. 初始化组件 ---
def init_app():
    print("🤖 初始化法律小助手 v0.1...")
    
    # 1. 客户端
    client = LLMClient() 
    
    # 2. 记忆模块 (限制最近 2000 token)
    memory = MemoryBuffer(max_tokens=2000, system_prompt="你是一名专业的刑事辩护律师。")
    
    # 3. 解析器
    parser = JsonOutputParser(pydantic_model=LegalVerdict)
    
    return client, memory, parser

# --- 4. 核心交互循环 (Integration) ---
def main():
    client, memory, parser = init_app()
    
    print("\n⚖️  法律小助手已就绪！(输入 'exit' 或 'quit' 退出)")
    print("--------------------------------------------------")

    while True:
        # A. 获取用户输入
        user_input = input("\n👤 当事人描述: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("👋 再见！")
            break
        
        if not user_input:
            continue

        # B. 这里的逻辑核心：拼装 Prompt (Week 8 Day 2 - Prompt Engineering)
        # 我们需要把 历史记录 + 当前问题 + 格式要求 拼在一起
        
        # 1. 获取格式说明 (Schema)
        schema_json = json.dumps(LegalVerdict.model_json_schema(), ensure_ascii=False)
        
        # 2. 获取历史记录 (转换为文本)
        history_text = ""
        for msg in memory.get_context():
            role = msg['role']
            content = msg['content']
            # 跳过 system prompt，因为它通常不放在 history 文本段里重复
            if role != "system":
                history_text += f"{role}: {content}\n"

        # 3. 构造最终 Prompt (CoT + JSON Mode)
        final_prompt = f"""
你是一名专业律师。请基于以下对话历史和新的案情描述进行分析。

【对话历史】
{history_text}

【新的案情】
{user_input}

【任务要求】
1. 请一步步思考 (Let's think step by step)，分析案件的起因、经过、结果和法律适用。
2. 必须以严格的 JSON 格式输出，不要包含 Markdown 标记。
3. 输出结构必须符合以下 Schema：
{schema_json}
"""

        print("\n🤖 AI 正在思考中... (涉及 CoT 推理)")
        
        try:
            # C. 调用模型 (Week 8 Day 1)
            # 注意：如果你的 client 支持 messages 列表，直接传 messages 更好
            # 但为了兼容最基础的 client.generate(str)，我们传 string
            raw_response = client.generate(final_prompt, temperature=0.1) # 降低温度以保证 JSON 格式
            
            # D. 解析结果 (Week 8 Day 3)
            # parser 会自动处理 Markdown 清洗和 JSON 提取
            result_dict = parser.parse(raw_response)
            
            # E. 展示结果
            print("-" * 30)
            print(f"🧐 **案情分析**: {result_dict.get('analysis')}")
            print(f"⚖️  **判决建议**: {result_dict.get('verdict')}")
            print(f"📊 **置信度**: {result_dict.get('confidence')}")
            print(f"📜 **引用法条**: {', '.join(result_dict.get('laws', []))}")
            print("-" * 30)

            # F. 更新记忆 (Week 8 Day 5)
            # 存入用户的原始输入
            memory.add("user", user_input)
            # 存入 AI 的回复 (为了节省 Token，我们只存 JSON 字符串，或者存 analysis)
            # 这里选择存完整的 JSON 字符串，以便 AI 记得自己之前的判断
            memory.add("assistant", json.dumps(result_dict, ensure_ascii=False))
            
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            # 如果解析失败，把原始回复打印出来看看
            if 'raw_response' in locals():
                 print(f"原始回复: {raw_response}")

if __name__ == "__main__":
    main()