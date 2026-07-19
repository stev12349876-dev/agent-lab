"""
LangChain 六大模块总览（Day 2 补充）

✅ Day 2 已学：Model I/O、Chains、Retrieval（了解）
📌 这里快速过：Memory、Callbacks、Agent（后续会深入）
"""

import os, asyncio
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# ============================================================
# 模块 4：Memory — 让 LLM 记住对话历史
# ============================================================
#
# 没有 Memory：每次问都是全新对话，LLM 不记得上一句
# 有了 Memory：LLM 知道"你刚才说..."，能多轮对话
#
# 核心机制：把历史消息自动拼到 prompt 前面，LLM 就能"看到"上下文

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁的助手。"),
    ("placeholder", "{history}"),   # ← 这里会自动插入历史消息
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

# 用内存存储保存对话
store = {}
def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,                     # 从哪取历史
    input_messages_key="question",   # 哪个字段是用户输入
    history_messages_key="history",  # 历史消息塞到哪个占位符
)

async def memory_demo():
    print("=" * 50)
    print("Memory 演示：多轮对话")
    print("=" * 50)

    config = {"configurable": {"session_id": "user-001"}}

    # 第 1 轮
    r1 = await chain_with_memory.ainvoke(
        {"question": "我叫小明，今年 25 岁。"}, config
    )
    print(f"  [Q1] 我叫小明，今年 25 岁。")
    print(f"  [A1] {r1}")

    # 第 2 轮：不再提名字，看它记不记得
    r2 = await chain_with_memory.ainvoke(
        {"question": "我叫什么名字？我几岁？"}, config
    )
    print(f"  [Q2] 我叫什么名字？我几岁？")
    print(f"  [A2] {r2}")

    print("\n  👆 第二轮没有重复'我叫小明25岁'，但 LLM 记住了")


# ============================================================
# 模块 5：Callbacks — 观测 LLM 的每一步
# ============================================================
#
# Agent 系统需要知道：LLM 调了吗？花了多少 token？报错了吗？
# Callbacks 就是"钩子函数"，LLM 每做一件事就触发一次

class MyCallback(BaseCallbackHandler):
    """自定义回调：记录每次 LLM 调用"""
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"  🔔 LLM 开始调用，prompt 长度: {len(prompts[0])} 字符")

    def on_llm_end(self, response, **kwargs):
        # 取 token 用量
        usage = response.llm_output.get("token_usage", {})
        print(f"  🔔 LLM 调用结束，消耗 token: {usage}")

async def callback_demo():
    print("\n" + "=" * 50)
    print("Callbacks 演示：观测 LLM 调用")
    print("=" * 50)

    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke(
        {"question": "1+1 等于几？"},
        config={"callbacks": [MyCallback()]}
    )
    print(f"  结果: {result}")


# ============================================================
# 模块 6：Agent — LLM 自主选择工具（Week 3 深入）
# ============================================================
#
# Chain 和 Agent 的区别：
#   Chain：prompt → LLM → 输出（路径固定）
#   Agent：prompt → LLM → 决定用什么工具 → 调工具 → 看结果 → 循环
#
# 下面是最简版 Agent 循环（Week 3 会完整实现）

async def agent_demo():
    print("\n" + "=" * 50)
    print("Agent 预览：LLM 选择工具")
    print("=" * 50)

    # 定义一个工具
    def calculator(expression: str) -> str:
        """计算数学表达式，如 '2+3*4'"""
        try:
            return str(eval(expression))
        except:
            return "计算失败"

    # 把工具绑到 LLM 上
    llm_with_tools = llm.bind_tools(
        [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "计算数学表达式",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，如 2+3*4"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }]
    )

    response = await llm_with_tools.ainvoke("15 乘以 32 等于多少？")
    print(f"  LLM 返回: {response}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            print(f"  → 决定调用工具: {tc['name']}({tc['args']})")
            result = calculator(**tc['args'])
            print(f"  → 工具返回: {result}")
    else:
        print(f"  → 直接回答（不用工具）")

    print("\n  👆 Agent 的本质：LLM 自己决定调不调工具、调哪个、传什么参数")


async def main():
    await memory_demo()
    await callback_demo()
    await agent_demo()


if __name__ == "__main__":
    asyncio.run(main())
