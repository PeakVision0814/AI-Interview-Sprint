from typing import List, Dict, Optional
import tiktoken

class MemoryBuffer:
    def __init__(self, max_tokens: Optional[int] = 2000, system_prompt: str = ""):
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        
        self.max_tokens = max_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if self.max_tokens:
            self._trim()

    def get_context(self) -> List[Dict]:
        return self.messages

    def _trim(self):
        # 简单粗暴的修剪策略：保留 System Prompt，修剪中间的
        while self._count_total_tokens() > self.max_tokens:
            # 确保不删掉 system prompt (index 0)
            if len(self.messages) > 1:
                # 删除 index 1 (最早的非 system 消息)
                self.messages.pop(1)
            else:
                break
                
    def _count_total_tokens(self) -> int:
        return sum([len(self.encoder.encode(m["content"])) for m in self.messages])

    def clear(self):
        # 保留 system prompt
        system = self.messages[0] if self.messages and self.messages[0]['role'] == 'system' else None
        self.messages = []
        if system:
            self.messages.append(system)