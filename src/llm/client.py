import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional


load_dotenv()

key = os.getenv("LLM_API_KEY")

class LLMClient:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, 
                 model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
        
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model
        
        if not self.api_key:
            raise ValueError("API Key Missing! Please check .env file.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.7) -> str:
        """
        通用生成接口
        :param temperature: 控制生成的随机性 (0.0 - 2.0)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=512 
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Generate Error]: {e}")
            return ""

# ✅ 修正：加上这个保护罩
if __name__ == "__main__":
    llm = LLMClient()
    print("--- 测试模式 ---")
    print(llm.generate("用一句话解释什么是递归？"))