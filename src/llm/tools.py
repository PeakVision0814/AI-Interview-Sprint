# src/llm/tools.py
import json

def get_current_weather(location, unit="celsius"):
    """
    模拟查询天气的工具函数。
    在真实项目中，这里会调用 OpenWeatherMap 或高德地图 API。
    """
    # 模拟数据
    if "hangzhou" in location.lower() or "杭州" in location:
        return json.dumps({"location": "Hangzhou", "temperature": "5", "unit": unit, "condition": "Rainy"})
    elif "beijing" in location.lower() or "北京" in location:
        return json.dumps({"location": "Beijing", "temperature": "-2", "unit": unit, "condition": "Sunny"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

# 定义工具列表供 LLM 使用
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，如 Beijing, Hangzhou",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

# 建立函数名到实际函数的映射，方便后续调用
AVAILABLE_FUNCTIONS = {
    "get_current_weather": get_current_weather,
}