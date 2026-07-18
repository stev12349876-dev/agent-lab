# Day 1 学习笔记：FastAPI 在 Agent 开发中的角色

## 一、FastAPI 是什么

FastAPI 是一个 Python Web 框架，专门用来**构建 API 服务**。它的三个核心卖点：

| 特性 | 含义 | 代码体现 |
|------|------|---------|
| **快** | 性能对标 Node.js/Go，底层是 Starlette + uvloop | `async def` 异步函数 |
| **少写代码** | 路由、校验、文档全部自动生成，代码量比 Flask 少 50%+ | 不需要手写参数校验 |
| **类型安全** | 基于 Pydantic，入参出参全自动校验，不合规直接 422 | `QueryRequest.question: min_length=1` |

与其他 Python Web 框架对比：

| | Flask | Django | FastAPI |
|---|-------|--------|---------|
| 异步支持 | 需插件 | 3.1+ 支持 | **原生 async/await** |
| 自动校验 | 需手写 | 需手写 | **Pydantic 自动** |
| API 文档 | 需插件 | 需插件 | **自动生成 Swagger** |
| 学习曲线 | 低 | 高 | **中低** |

## 二、FastAPI 在 Agent 开发中扮演什么角色

Agent 的本质是一个**接收任务 → 调用工具 → 返回结果**的循环。FastAPI 在其中扮演三个角色：

### 角色 1：Agent 的服务外壳

你写的 Agent 不能只在自己电脑的终端里跑，它需要一个**网络入口**让外部调用。

```
用户/前端 → HTTP 请求 → FastAPI（路由分发）→ Agent 核心逻辑 → 返回 JSON
```

对应代码就是：
```python
@app.post("/query")
async def handle_query(req: QueryRequest):
    # 这里将来就是你的 Agent 核心逻辑
    result = await agent.run(req.question)
    return result
```

**没有 FastAPI**：你的 Agent 只能 `python main.py` 在终端交互。
**有了 FastAPI**：可以接到网页、小程序、Slack、企业微信等等。

### 角色 2：输入输出的"类型安检"

Agent 经常遇到两类问题：
- **LLM 输出不可控**：返回的 JSON 可能缺字段、类型不对
- **用户输入不可控**：空问题、超长文本、恶意注入

FastAPI + Pydantic 在入口和出口各设一道安检：

```
用户输入 → Pydantic 校验（自动 422）→ Agent 处理 → Pydantic 校验响应 → 返回
```

代码中 `QueryRequest.question: min_length=1` 就是这道安检——用户发空字符串直接被拦下，根本不进 Agent 逻辑。

### 角色 3：异步能力的底座

Agent 最大的性能瓶颈是 **I/O 等待**——等 LLM 返回、等搜索接口、等数据库查询。这些操作不适合同步阻塞。

FastAPI 的 `async/await` 模型天然适配：

```python
# 同步写法（差）：3 个工具串行调用 = 0.6s
result1 = call_tool_1()  # 等 0.3s
result2 = call_tool_2()  # 等 0.2s
result3 = call_tool_3()  # 等 0.1s

# 异步写法（好）：3 个工具并行调用 = 0.3s
results = await asyncio.gather(
    call_tool_1(),  # 0.3s
    call_tool_2(),  # 0.2s
    call_tool_3(),  # 0.1s
)
```

这对 Agent 至关重要：Agent 经常需要同时查多个数据源，并行能大幅降低响应时间。

## 三、Day 1 代码中学到的核心概念

### 3.1 路由（Routing）

```python
@app.get("/agents/{agent_id}")          # 路径参数：/agents/1
@app.get("/agents/{agent_id}?verbose=true")  # 查询参数：?verbose=true
@app.post("/query")                      # POST：Body 传 JSON
```

FastAPI 用 Python 类型注解自动识别参数来源：
- 函数参数名在路径中 → 路径参数
- 函数参数名不在路径中，是基本类型 → 查询参数
- 函数参数是 Pydantic 模型 → 请求体

### 3.2 Pydantic 数据校验（核心）

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
```

这 5 行代码自动完成了：
- 类型校验（string / int）
- 长度限制（1-500 字符）
- 数值范围（1-4096）
- 默认值（1024）
- 必填/可选（`...` = 必填，`default=1024` = 可选）
- 返回友好的 422 错误信息

**没有 FastAPI 你需要手写至少 30 行校验代码。**

### 3.3 异步 I/O

```python
async def handle_query(req: QueryRequest):
    await asyncio.sleep(0.1)  # 模拟 I/O 等待
    return QueryResponse(...)
```

`async def` + `await` = 这个函数在等待时不阻塞其他请求。换句话说，100 个用户同时发请求，一个等 LLM 返回时，其他 99 个不会被堵住。

### 3.4 自动 API 文档

FastAPI 根据你的代码自动生成 OpenAPI 规范的文档：
- 访问 `/docs` → Swagger UI（可交互测试）
- 访问 `/redoc` → ReDoc（更美观的阅读版）

你写的 Pydantic 模型会自动变成文档里的 Schema 说明。

## 四、和后续学习的关系

```
Day 1: FastAPI ──→ 提供 Agent 的服务外壳
Day 2: LangChain ──→ 提供 Agent 的 LLM 调用能力
Day 3-4: RAG ──→ 提供 Agent 的知识获取能力
Day 5-6: Naive RAG ──→ 把上面三个拼成一个可用的系统
```

到 Day 5-6 时，你会把 FastAPI 路由、LangChain 调用、RAG 检索整合成一个完整的 API：

```
用户 POST /query {"question": "什么是 Agent？"}
    ↓
FastAPI 校验入参
    ↓
LangChain 调用 LLM 理解问题
    ↓
RAG 检索相关文档
    ↓
LLM 综合生成回答
    ↓
FastAPI 返回 JSON
```

Day 1 的 FastAPI 就是这个管道的**入口和出口**。
