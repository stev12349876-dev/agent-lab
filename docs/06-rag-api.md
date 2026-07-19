# Day 6 学习笔记：FastAPI RAG 服务与索引持久化

## 今日目标

把 Day 5 的单次 CLI 程序升级为可重复调用的 HTTP API，并把文档向量保存到磁盘。

```text
服务启动
  → 检查磁盘索引
  → 索引有效：直接加载
  → 索引缺失或过期：重新 Embedding 并保存

POST /query
  → 问题 Embedding
  → 检索和阈值过滤
  → qwen-plus 生成答案
  → JSON 答案与来源
```

代码入口：[exercises/day06-rag-api/main.py](../exercises/day06-rag-api/main.py)。

---

## 1. 为什么要在启动时建立索引

Day 5 每执行一次 CLI 都会重新向量化全部文档。API 服务是长时间运行的进程，可以在启动阶段建立或加载一次索引，后续请求共享同一个 `RagService`。

FastAPI 的 `lifespan` 负责管理这个生命周期：

```text
启动 → 创建 RagService → 接收多个请求 → 服务关闭
```

`RagService` 被保存到 `app.state`，路由通过 `request.app.state.rag_service` 获取它。

## 2. 向量保存在哪里

索引保存在运行目录下：

```text
exercises/day06-rag-api/data/
├── vector-store.json
└── index-meta.json
```

`vector-store.json` 包含每个 chunk 的向量、正文和 metadata；`index-meta.json` 保存索引指纹和 Embedding 模型名。该目录被 `.gitignore` 忽略，不会推送到 GitHub。

这仍然是适合学习和小数据的本地 JSON 索引，不是支持并发写入和海量数据的生产向量数据库。

## 3. 如何判断索引是否过期

程序根据三个内容计算 SHA-256 指纹：

- 原始 Markdown 文件内容。
- Embedding 模型名称。
- 文本分割策略版本。

任意一项变化，指纹都会变化，服务下次启动会重新生成向量。否则直接加载磁盘索引，不再为文档消耗 Embedding 额度。

`GET /health` 中的 `index_status` 会显示：

- `built`：本次启动重新建立了索引。
- `loaded`：本次启动复用了磁盘索引。

## 4. API 输入输出模型

请求：

```json
{
  "question": "谁负责评估执行结果？",
  "top_k": 3
}
```

响应：

```json
{
  "answer": "Critic Agent 负责评估执行结果 [来源1]。",
  "grounded": true,
  "sources": [
    {
      "number": 1,
      "title": "多 Agent 协作",
      "score": 0.718,
      "chunk_id": 6,
      "source": ".../sample.md"
    }
  ]
}
```

Pydantic 会自动校验问题长度和 `top_k` 范围，并让 FastAPI 生成 `/docs` 交互文档。

## 5. 为什么检索放进线程

`similarity_search_with_score` 是同步调用，其中问题 Embedding 会等待网络返回。如果直接在 `async def` 路由中调用，会阻塞事件循环。

代码使用 `asyncio.to_thread` 把同步检索放到工作线程；LLM 则使用原生异步的 `ainvoke`。这把 Day 1 的异步知识真正用到了 RAG API 中。

## 6. 运行与测试

启动服务：

```bash
cd exercises/day06-rag-api
python3 -m uvicorn main:app --reload
```

浏览器打开：

```text
http://127.0.0.1:8000/docs
```

或使用 curl：

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"谁负责评估执行结果？","top_k":3}'
```

## 7. 当前边界

- JSON 索引适合单机学习，不适合多个服务实例同时写入。
- 每个问题仍需要调用一次 Embedding API。
- 有相关证据时还会调用一次 Chat API。
- 相关度阈值仍需要用评测集校准。
- 没有认证、限流、日志和超时策略。

## 8. 今天必须能回答的问题

1. 为什么文档向量可以复用，但每个新问题仍要做 Embedding？
2. `built` 和 `loaded` 分别代表什么？
3. 为什么只判断索引文件存在还不够，还要比较指纹？
4. `app.state.rag_service` 解决了什么问题？
5. 为什么同步检索要使用 `asyncio.to_thread`？

能解释这五点，并在 `/docs` 独立完成一次问答，就掌握了第一个可调用 RAG 服务的核心结构。
