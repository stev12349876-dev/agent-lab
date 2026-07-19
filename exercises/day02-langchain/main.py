"""
Day 2: LangChain 核心概念 — Prompt Templates + LLM + Output Parsers + LCEL

LangChain 六大模块：
  1. Model I/O    ← 今天重点（Prompt、LLM、Output Parser）
  2. Retrieval    ← Day 3-4（RAG）
  3. Chains       ← 今天重点（LCEL 管道语法）
  4. Agents       ← Week 3（Tool Calling）
  5. Memory       ← Week 3
  6. Callbacks    ← Week 5（Observability）

LCEL = LangChain Expression Language，核心是管道操作符 `|`
类比 Unix 管道：cat file | grep keyword | sort
"""

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # 向上查找 .env  # 自动读取 .env 文件
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ============================================================
# Part 0: 初始化 LLM
# ============================================================

# 使用阿里云 MaaS（OpenAI 兼容端点）
# .env 中配置了 OPENAI_API_KEY 和 OPENAI_BASE_URL
llm = ChatOpenAI(
    model="qwen-plus",          # 阿里云通义千问
    temperature=0.7,              # 0=严谨 1=创意
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# ============================================================
# Part 1: PromptTemplate — 最简单的模板
# ============================================================

# 花括号 {} 是占位符，运行时替换
simple_prompt = PromptTemplate.from_template(
    "用一句话解释：{concept} 是什么？"
)

# 这段代码做了什么：
#   输入 {"concept": "机器学习"}
#   输出 "用一句话解释：机器学习 是什么？"
# 本质上就是 Python 的 f-string，但 LangChain 的模板多了：
#   1. 类型校验（partial variables）
#   2. 组合能力（可以嵌套）
#   3. 序列化（可以存 JSON）


# ============================================================
# Part 2: ChatPromptTemplate — 多角色对话模板
# ============================================================

# Agent 开发中 99% 用的是 ChatPromptTemplate（system + human）
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}专家，回答要{style}。"),
    ("human", "{question}"),
])

# system = 给模型设定人设和行为约束（Agent 的"身份卡"）
# human  = 用户的实际问题


# ============================================================
# Part 3: LCEL 管道 — LangChain 最核心的语法
# ============================================================

# LCEL 用一个 `|` 操作符把组件串起来，数据从左流到右
# 类比：Unix 的 cat | grep | sort

# 示例 1：最简单的 Chain
chain_simple = simple_prompt | llm | StrOutputParser()
# 解读：模板填充 → 调 LLM → 解析为纯文本

# 示例 2：多角色 Chain
chain_chat = chat_prompt | llm | StrOutputParser()

# StrOutputParser 的作用：
#   LLM 返回的是 AIMessage 对象（含 token 用量等元数据）
#   StrOutputParser 只提取 content 字段，返回纯字符串


# ============================================================
# Part 4: 复杂 Chain — 多步组合
# ============================================================

# LCEL 的真正威力：可以嵌套、分叉、合并
# 比如：先生成大纲，再根据大纲写正文

outline_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个内容策划。为以下主题生成一个 3 点大纲。"),
    ("human", "{topic}"),
])

write_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下大纲，写一段 200 字以内的内容。"),
    ("human", "大纲：\n{outline}\n\n请展开描述。"),
])

# 链式组合：topic → 生成大纲 → 用大纲写正文
chain_multi_step = (
    {"outline": outline_prompt | llm | StrOutputParser(), "topic": lambda x: x}
    | write_prompt
    | llm
    | StrOutputParser()
)

# 解读上面这个链：
#   1. {"outline": ..., "topic": ...} = RunnableParallel，两个任务同时跑
#      - outline: topic → outline_prompt → llm → 文本（生成大纲）
#      - topic: 原样传递（lambda x: x）
#   2. 结果合并后 → write_prompt → llm → 文本


# ============================================================
# Part 5: 结构化输出 — 让 LLM 返回 JSON
# ============================================================

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 定义期望的输出结构（Agent 必备：让 LLM 按约定格式返回）
class ConceptExplain(BaseModel):
    concept: str = Field(description="概念名称")
    definition: str = Field(description="一句话定义")
    example: str = Field(description="一个具体例子")
    difficulty: str = Field(description="难度：入门/进阶/专家")

json_parser = JsonOutputParser(pydantic_object=ConceptExplain)

json_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识解释器。按 JSON 格式输出。\n{format_instructions}"),
    ("human", "解释：{concept}"),
])

chain_json = json_prompt | llm | json_parser


# ============================================================
# Part 6: 批处理 — 一次处理多个输入
# ============================================================

# LCEL 原生支持 batch（内置并发控制）
# 和 FastAPI 的 asyncio.gather 思路一样


# ============================================================
# 运行示例（需要 OPENAI_API_KEY 环境变量）
# ============================================================

async def demo():
    print("=" * 50)
    print("Day 2: LangChain LCEL 演示")
    print("=" * 50)

    # 1. 简单链
    print("\n[1] 简单 Prompt 链：")
    result = await chain_simple.ainvoke({"concept": "强化学习"})
    print(f"  {result}")

    # 2. 多角色链
    print("\n[2] 多角色 Chat 链：")
    result = await chain_chat.ainvoke({
        "role": "Python",
        "style": "简洁",
        "question": "什么是装饰器？"
    })
    print(f"  {result}")

    # 3. 多步链
    print("\n[3] 多步链（大纲→正文）：")
    result = await chain_multi_step.ainvoke("Agent 开发")
    print(f"  {result}")

    # 4. 结构化输出
    print("\n[4] JSON 结构化输出：")
    result = await chain_json.ainvoke({"concept": "Transformer"})
    print(f"  {result}")

    # 5. 批处理
    print("\n[5] 批处理（并行 3 个问题）：")
    results = await chain_simple.abatch([
        {"concept": "机器学习"},
        {"concept": "深度学习"},
        {"concept": "强化学习"},
    ])
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r}")


# 运行方式：
#   python main.py --no-llm    # 本地演示，无需 API Key
#   python main.py             # 真实调用 LLM（需要 OPENAI_API_KEY）



# ============================================================
# Bonus: 无需 API Key 的本地演示（只看链结构，不调 LLM）
# ============================================================

def demo_without_api_key():
    """演示：不调 LLM，只看 LCEL 链的结构和数据流"""
    print("=" * 50)
    print("LCEL 本地演示（无需 API Key）")
    print("=" * 50)

    # 1. 看 Prompt 渲染结果
    print("\n[1] Prompt 渲染：")
    print(f"  输入: {{'concept': 'Agent'}}")
    rendered = simple_prompt.invoke({"concept": "Agent"})
    print(f"  输出: {rendered.text}")

    # 2. 看 ChatPrompt 渲染结果
    print("\n[2] ChatPrompt 渲染：")
    rendered = chat_prompt.invoke({
        "role": "Python",
        "style": "简洁",
        "question": "什么是装饰器？"
    })
    for msg in rendered.messages:
        print(f"  [{msg.type}] {msg.content}")

    # 3. 看链的输入输出 Schema
    print("\n[3] 链的 Schema：")
    print(f"  输入: {chain_simple.input_schema.model_json_schema()['properties']}")
    print(f"  输出: {chain_simple.output_schema.model_json_schema()}")

    # 4. 看复杂链的图结构
    print("\n[4] 多步链的图结构：")
    try:
        graph = chain_multi_step.get_graph()
        graph.print_ascii()
    except Exception:
        print("  (需要安装 grandalf: pip install grandalf)")

    print("\n✅ 链结构正确！加 OPENAI_API_KEY 后运行 main.py 可调真实 LLM。")


if __name__ == "__main__":
    import sys
    if "--no-llm" in sys.argv:
        demo_without_api_key()
    else:
        import asyncio
        asyncio.run(demo())
