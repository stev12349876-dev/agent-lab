"""
Day 2 实战练习：「翻译润色 Agent」— 独立手写一个完整 Chain

要求：不照抄 main.py，根据下方的规格自己写。

================================================================
规格说明
================================================================
你需要构建一个 Chain，实现以下功能：

  输入：一段中文文本
  输出：英文翻译 + 语言润色后的版本
  人设：你是一个专业的中英翻译专家，翻译要准确、地道

================================================================
具体要求
================================================================
1. 用 ChatPromptTemplate 定义 system + human 两个角色
   - system: 设定翻译专家人设
   - human: 接收用户要翻译的中文文本

2. 用 LCEL 管道语法串联组件
   提示词 → LLM → 文本解析

3. 用 ainvoke 异步调用，翻译一句话
   "人工智能正在改变我们与技术互动的方式。"

4. 用 abatch 并行翻译 3 句话
   "你好，世界。"
   "机器学习是人工智能的一个分支。"
   "Python 是最流行的编程语言之一。"

5. 打印每次调用的结果

================================================================
提示
================================================================
- 参考 main.py 的结构，但不要复制粘贴——理解每行再写
- LLM 已配好（qwen-plus + 阿里云 MaaS），直接用下面这个 llm 变量
- 运行：python3 practice.py
"""

import os
import asyncio
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# 已配好的 LLM（直接用）
# ============================================================
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.3,  # 翻译要准确，温度低一点
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# ============================================================
# TODO: 在下面写你的代码
# ============================================================
from pydantic import BaseModel, Field

# 定义结构化输出
class TranslationResult(BaseModel):
    original: str = Field(description="中文")
    translated: str = Field(description="翻译后的英文")


# Prompt 明确翻译任务
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是中英翻译专家"),
    ("human", "{text}"),
])

# 结构化输出链
structured_llm = llm.with_structured_output(TranslationResult)
chain = chat_prompt | structured_llm


async def main():
    print("=" * 50)
    print("Day 2: 结构化翻译练习")
    print("=" * 50)

    # 单句
    result = await chain.ainvoke({"text": "人工智能正在改变我们与技术互动的方式。"})
    print(f"  [单句] {result}")

    # 并行 3 句
    results = await chain.abatch([
        {"text": "你好，世界。"},
        {"text": "机器学习是人工智能的一个分支。"},
        {"text": "Python 是最流行的编程语言之一。"},
    ])
    for r in results:
        print(f"  [并行] {r.original} → {r.translated}")

### 你的代码写在上面 ###


# ============================================================
# 自检脚本（运行 python3 practice.py 查看结果）
# ============================================================
    # ====== 流式输出（逐 token 打印，Agent 必备） ======
    print("\n" + "=" * 50)
    print("流式输出（逐 token）：")
    chain_text = chat_prompt | llm | StrOutputParser()
    async for chunk in chain_text.astream({"text": "AI Agent 正在改变软件开发的范式。"}):
        print(chunk, end="", flush=True)
    print()  # 换行


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except NameError:
        print("\n❌ 请先完成上面的 TODO，定义 main() 函数")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
