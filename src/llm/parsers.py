import re
import json
from typing import Optional, Any, Type
from pydantic import BaseModel

class BaseOutputParser:
    def parse(self, text: str) -> Any:
        raise NotImplementedError
    
class JsonOutputParser(BaseOutputParser):
    def __init__(self, pydantic_model: Optional[Type[BaseModel]] = None):
        """
        :param pydantic_model: 如果提供，会用它来验证数据结构
        """
        self.pydantic_model = pydantic_model

        
    def parse(self, text: str) -> dict:
        # 1. 尝试清洗 Markdown 标记 (```json ... ```)
        cleaned_text = text.strip()
        # 匹配 ```json ... ``` 或 ``` ... ```
        match = re.search(r"```(json)?(.*?)```", cleaned_text, re.DOTALL)
        if match:
            cleaned_text = match.group(2).strip()
        
        # 2. 如果还有多余字符，尝试 Regex 提取最外层 {}
        match_bracket = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if match_bracket:
            cleaned_text = match_bracket.group()

        # 3. 解析 JSON
        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {e}\n原文: {text}")

        # 4. Pydantic 验证
        if self.pydantic_model:
            try:
                obj = self.pydantic_model(**data)
                return obj.model_dump() # 转回 dict，或者直接返回 obj
            except Exception as e:
                raise ValueError(f"Schema 验证失败: {e}")
        
        return data