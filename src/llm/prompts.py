from typing import List, Dict

class BasePromptTemplate:
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        """
        将变量填入模版
        """
        # 检查是否所有变量都提供了
        for var in self.input_variables:
            if var not in kwargs:
                raise ValueError(f"Missing variable: {var}")
        return self.template.format(**kwargs)

# 预定义一些常用的模版
class LegalPromptTemplates:
    # 意图分类模版
    INTENT_CLASSIFICATION = BasePromptTemplate(
        template="""
你是一名专业的法律助手。请根据以下示例，判断新案件描述的犯罪类型。
只输出类别名称，不要输出其他内容。

示例 1:
案件：张三在网购平台发布虚假商品，骗取5000元。
类别：诈骗

示例 2:
案件：王五持刀抢走他人手机。
类别：抢劫

新案件：
案件：{case_description}
类别：
""",
        input_variables=["case_description"]
    )
