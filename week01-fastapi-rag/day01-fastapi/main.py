"""
Day 1: FastAPI 路由 + 异步 I/O + Pydantic 数据校验
目标：掌握三个核心技能，为后续 Agent API 打基础
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import asyncio
import time

app = FastAPI(title="Agent Lab API", version="0.1.0")

# ============================================================
# 1. Pydantic 数据模型 —— 强类型校验，Agent 必备
# ============================================================

class QueryRequest(BaseModel):
    """用户查询请求体"""
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    max_tokens: int = Field(default=1024, ge=1, le=4096, description="最大输出 token")

class QueryResponse(BaseModel):
    """查询响应体"""
    answer: str
    tokens_used: int
    elapsed_ms: float
    status: str = "success"

class ErrorResponse(BaseModel):
    error: str
    detail: str

# ============================================================
# 2. 路由基础：GET 端点
# ============================================================

@app.get("/")
async def root():
    """最简 GET"""
    return {"message": "Agent Lab API is running"}

@app.get("/health")
async def health_check():
    """健康检查 —— 面试常问：你的服务怎么监控？"""
    return {"status": "healthy", "version": "0.1.0"}

# ============================================================
# 3. 路径参数 + 查询参数
# ============================================================

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: int, verbose: bool = False):
    """
    路径参数 agent_id 自动校验为 int
    查询参数 verbose 可选，默认 False
    FastAPI 自动生成 OpenAPI 文档：/docs
    """
    agents = {1: "planner", 2: "executor", 3: "evaluator"}
    name = agents.get(agent_id)
    if not name:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    if verbose:
        return {"id": agent_id, "name": name, "info": f"{name} agent handles specific tasks"}
    return {"id": agent_id, "name": name}

# ============================================================
# 4. POST + Pydantic 校验 —— Agent 核心交互模式
# ============================================================

@app.post("/query", response_model=QueryResponse, responses={400: {"model": ErrorResponse}})
async def handle_query(req: QueryRequest):
    """
    接收用户问题，返回结构化响应
    Pydantic 自动校验：
      - question 不能为空，不超过 500 字符
      - max_tokens 必须在 1-4096 之间
    不合法请求自动返回 422，无需手写校验代码
    """
    start = time.time()

    # 模拟异步 LLM 调用（后续替换为真实的 openai call）
    await asyncio.sleep(0.1)
    answer = f"[Re: {req.question[:30]}...] This is a simulated agent response."

    elapsed = (time.time() - start) * 1000

    return QueryResponse(
        answer=answer,
        tokens_used=len(answer) // 4,  # 粗略估算
        elapsed_ms=round(elapsed, 2),
    )

# ============================================================
# 5. 异步 I/O —— 并发调用多个工具（Agent 的核心场景）
# ============================================================

async def mock_tool_call(tool_name: str, delay: float) -> dict:
    """模拟工具调用 —— Agent 执行时的典型异步 I/O"""
    await asyncio.sleep(delay)
    return {"tool": tool_name, "result": f"{tool_name} completed in {delay}s"}

@app.post("/agent/run")
async def run_agent(req: QueryRequest):
    """
    模拟 Agent 并行调用多个工具
    这里展示 asyncio.gather —— Agent 框架并行执行工具的核心方式
    """
    start = time.time()

    # 并行调用 3 个模拟工具（如搜索、计算、读文件）
    results = await asyncio.gather(
        mock_tool_call("web_search", 0.3),
        mock_tool_call("calculator", 0.1),
        mock_tool_call("file_reader", 0.2),
    )

    elapsed = (time.time() - start) * 1000

    return {
        "question": req.question,
        "tool_results": results,
        "total_elapsed_ms": round(elapsed, 2),
        "note": "3 tools ran in parallel, total time ≈ max(delays), not sum(delays)",
    }

# ============================================================
# 启动方式：
#   pip install fastapi uvicorn
#   uvicorn main:app --reload
# 然后访问：
#   http://127.0.0.1:8000/docs   ← FastAPI 自动生成的交互式 API 文档
# ============================================================
