# Day 2 学习笔记：LangChain 核心与 LCEL

## 一、LangChain 是什么

LangChain 是一个**把 LLM 调用变成搭积木**的框架。它的核心思路：

```
没有 LangChain：
  手写 prompt 字符串 → 调 OpenAI API → 解析 JSON → 拼到下一轮 → ...

有了 LangChain：
  PromptTemplate | ChatOpenAI | StrOutputParser
```

一句话：**LangChain = LLM 应用的乐高**。每个组件是一个积木块，用 `|` 串起来就是一条流水线。

## 二、六大核心模块

| 模块 | 作用 | 今天学 | 什么时候学 |
|------|------|--------|-----------|
| **Model I/O** | 提示词模板、调 LLM、解析输出 | ✅ 全学 | Day 2 |
| **Retrieval** | 文档加载、分块、向量化、检索 | 了解 | Day 3-4 |
| **Chains** | 用 LCEL 串联组件 | ✅ 重点 | Day 2 |
| **Agents** | 让 LLM 自主选择工具 | 了解 | Week 3 |
| **Memory** | 记住对话历史 | 了解 | Week 3 |
| **Callbacks** | 日志、监控、流式输出 | 了解 | Week 5 |

## 三、LCEL 管道语法（最重要）

### 3.1 基础：一个 `|` 走天下

```python
chain = prompt | llm | parser
```

- `|` 不是 Python 原生的或运算，是 LangChain **重载**的管道符
- 数据从左流到右：prompt 的输出 → llm 的输入 → parser 的输入
- 类比 Unix 管道：`cat file | grep keyword | sort`

### 3.2 为什么 LCEL 比传统写法好

**传统写法（LangChain 0.1 旧 API，已废弃）**：
```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(concept="机器学习")
```
问题：`LLMChain` 内部是黑箱，你没法知道中间发生了什么。

**LCEL 写法（现在）**：
```python
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"concept": "机器学习"})
```
优势：
1. **透明**：每一步都看得见
2. **可组合**：嵌套、分叉、合并都用一个 `|`
3. **自动并行**：`RunnableParallel` 自动并发
4. **统一接口**：所有组件都有 `.invoke()` `.ainvoke()` `.batch()` `.stream()`

### 3.3 LCEL 的四种调用方式

```python
chain = prompt | llm | parser

# 同步单次
result = chain.invoke({"concept": "AI"})

# 异步单次（Agent 开发主力！和 FastAPI async def 配合）
result = await chain.ainvoke({"concept": "AI"})

# 批量并行（自动并发控制）
results = chain.batch([{"concept": "AI"}, {"concept": "ML"}])

# 流式输出（逐 token 返回，用户体验好）
async for chunk in chain.astream({"concept": "AI"}):
    print(chunk, end="")
```

## 四、代码中的核心知识点

### 4.1 PromptTemplate vs ChatPromptTemplate

```python
# PromptTemplate：纯文本模板（老式）
PromptTemplate.from_template("解释：{concept}")

# ChatPromptTemplate：多角色模板（现在主流）
ChatPromptTemplate.from_messages([
    ("system", "你是{role}专家"),   # 系统指令（人设）
    ("human", "{question}"),         # 用户输入
])
```

Agent 开发 99% 用 `ChatPromptTemplate`，因为需要 system 角色来设定 Agent 的行为边界。

### 4.2 StrOutputParser vs JsonOutputParser

```python
# 提取纯文本
chain = prompt | llm | StrOutputParser()

# 提取结构化 JSON（配合 Pydantic，和 Day 1 的 Pydantic 呼应）
chain = prompt | llm | JsonOutputParser(pydantic_object=MyModel)
```

在 Agent 开发中，结构化输出至关重要：
- Tool Calling 参数 → 必须是合法 JSON
- Agent 的思考步骤 → 结构化便于追踪
- 评测结果 → 需要标准格式

### 4.3 RunnableParallel 自动并发

```python
# 这个字典里两个任务会同时执行（不是先后！）
{
    "outline": outline_chain,   # 生成大纲
    "topic": lambda x: x,       # 原样传 topic
}
```

这和 Day 1 的 `asyncio.gather` 一个思路——Agent 需要并行调多个工具，LCEL 天然支持。

### 重要细节：RunnableParallel 的 key 必须被下游消费

```python
# ❌ 错误：topic 传了但 write_prompt 不用它
{"outline": outline_chain, "topic": lambda x: x}
| write_prompt  # write_prompt 里没有 {topic} → topic 被静默丢弃

# ✅ 正确：write_prompt 的占位符和字典的 key 一一对应
{"outline": outline_chain, "topic": lambda x: x}
| write_prompt  # write_prompt 里同时有 {outline} 和 {topic}
```

LangChain 不会因为多余 key 报错，这很危险——你以为传了数据，实际上没用上。

## 五、和 Agent 开发的关系

你在 Day 2 学的这些，到 Week 3 会变成 Agent 的核心组件：

```
Week 3 的 Agent 结构：

ChatPromptTemplate（system 设定 Agent 人设）
    |
ChatOpenAI（调用 LLM，返回 tool_call 或 final_answer）
    |
AgentExecutor（循环：调工具 → 看结果 → 决定下一步）
```

Day 2 的 `prompt | llm | parser` 就是 Agent 的一次"思考"步骤。Agent 只是在这个基础上加了循环和工具调用。

## 六、六大模块补充（extras.py）

Day 2 的 `main.py` + `practice.py` 覆盖了前三模块。`extras.py` 快速演示后三模块：

### Memory
```python
chain_with_memory = RunnableWithMessageHistory(chain, get_history, ...)
```
历史消息自动拼到 prompt 前面，LLM 能"看到"之前的对话。Week 3 会改用 LangGraph 的持久化。

### Callbacks
```python
class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, ...): ...  # LLM 开始时触发
    def on_llm_end(self, ...): ...    # LLM 结束时触发（拿 token 用量）
```
每次 LLM 调用都触发钩子，是 Agent 可观测性的基础。Week 5 深入。

### Agent
```python
llm.bind_tools([{...}])  # 把工具定义绑到 LLM
response = await llm.ainvoke("15×32=?")
response.tool_calls  # LLM 自动决定调 calculator("15*32")
```
Chain 是固定路径，Agent 是 LLM 自己选工具。Week 3 完整实现 ReAct 循环。

## 七、和 Day 1 FastAPI 的关系

```
Day 1: FastAPI  →  "怎么接收请求、返回响应"
Day 2: LangChain → "接收到请求后，怎么处理"
Day 5-6:        →  把两个拼起来
```

到 Day 5 时，你的代码会长这样：

```python
@app.post("/query")
async def handle_query(req: QueryRequest):
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({"question": req.question})
    return {"answer": result}
```

FastAPI 负责通信，LangChain 负责智能。
