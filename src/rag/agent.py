# src/rag/agent.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from src.llm.tools import TOOLS_SCHEMA, AVAILABLE_FUNCTIONS

load_dotenv()

class RAGAgent:
    def __init__(self):
        # 初始化 DeepSeek / SiliconFlow 客户端
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), 
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        self.model_name = "deepseek-chat" # 或者 "deepseek-ai/DeepSeek-V2.5"
        self.system_prompt = """
        你是一个专业的 AI 算法面试助手。
        你的任务是帮助用户准备 AI 面试，或者回答关于深度学习的技术问题。
        
        1. 对于日常问候（如“你好”、“你是谁”），请直接热情回复，**不要**调用工具。
        2. 对于具体的技术问题（如“什么是 RAG”、“解释下 BERT”），请务必调用 search_knowledge_base 工具检索信息。
        3. 回答要简洁、专业。
        """
        # 对话记忆 (History)
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat(self, user_input: str):
        # 1. 添加用户消息
        self.messages.append({"role": "user", "content": user_input})
        
        # 2. 第一轮调用 LLM (思考)
        print("🤖 Agent is thinking...")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto" 
        )
        
        response_msg = response.choices[0].message
        tool_calls = response_msg.tool_calls
        
        # 3. 判断是否需要行动
        if tool_calls:
            print(f"🛠️ Agent decided to use tool: {tool_calls[0].function.name}")
            
            # 将 LLM 的“想调用工具”的想法加入历史
            self.messages.append(response_msg)
            
            # 执行所有工具调用
            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                fn_to_call = AVAILABLE_FUNCTIONS[fn_name]
                fn_args = json.loads(tool_call.function.arguments)
                
                # 真正执行函数
                tool_output = fn_to_call(**fn_args)
                
                # 将结果回填给 LLM
                self.messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": tool_output
                })
            
            # 4. 第二轮调用 LLM (生成最终回答)
            final_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages
            )
            reply = final_response.choices[0].message.content
        else:
            # 不需要工具，直接回复
            print("💬 Agent decided to chat directly.")
            reply = response_msg.content
        
        # 将最终回答加入历史
        self.messages.append({"role": "assistant", "content": reply})
        return reply

# --- 测试代码 ---
if __name__ == "__main__":
    agent = RAGAgent()
    
    print("\n--- Test 1: Chit-chat ---")
    print("User: 你好")
    print("Agent:", agent.chat("你好"))
    
    print("\n--- Test 2: Technical Query ---")
    print("User: 讲讲 transformer 是什么")
    print("Agent:", agent.chat("讲讲 transformer 是什么"))