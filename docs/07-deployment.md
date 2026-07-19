# Day 7 学习笔记：使用 Docker 部署 RAG API

## 今日目标

把 Day 6 的 FastAPI RAG 服务封装成容器，让它在不同电脑或服务器上都使用相同的 Python 版本、依赖和启动命令。

```text
源代码 + requirements.txt + Dockerfile
  → Docker Image
  → Docker Container
  → 容器监听 8000，宿主机通过 8001 访问
  → 外部访问 /health、/docs、/query
```

部署配置位于 `exercises/day07-deployment/`。

---

## 1. Image 和 Container 的区别

- **Image（镜像）**：只读的软件安装包，包含 Python、依赖和项目代码。
- **Container（容器）**：镜像启动后的运行实例，里面正在执行 Uvicorn 和 FastAPI。
- **Volume（数据卷）**：独立于容器保存数据，容器删除或重建后仍可保留向量索引。

可以把镜像理解为类，把容器理解为根据这个类创建的对象。

## 2. Dockerfile 做了什么

`Dockerfile` 按顺序完成：

1. 使用 `python:3.11-slim` 作为基础环境。
2. 安装 Day 6 的 Python 依赖。
3. 复制 RAG API 和知识库样本文档。
4. 创建非 root 用户运行服务。
5. 暴露容器的 8000 端口。
6. 执行 `python -m uvicorn main:app --host 0.0.0.0 --port 8000`。

容器从空白 Python 环境开始，因此程序直接使用的依赖都必须写进 `requirements.txt`。例如，内存向量库计算余弦相似度需要 `numpy`；本机可能因其他项目已经安装而不报错，但容器不会继承本机环境。

容器中必须监听 `0.0.0.0`。如果只监听 `127.0.0.1`，服务只能在容器内部访问，宿主机端口映射也无法连接。

## 3. Compose 做了什么

`docker-compose.yml` 把启动参数集中管理：

```text
build       → 使用哪个 Dockerfile 构建镜像
env_file    → 把本地 .env 注入容器
ports       → 宿主机 8001 映射到容器 8000
volumes     → 持久化向量索引
healthcheck → 定期请求 /health 判断服务是否可用
restart     → 异常退出后自动重启
```

`.env` 只在启动时注入，不会复制进镜像，也不会提交到 Git。

## 4. 为什么需要 Volume

Day 6 把索引写到：

```text
/app/exercises/day06-rag-api/data
```

Compose 将这个目录挂载到名为 `rag-index` 的数据卷：

```yaml
volumes:
  - rag-index:/app/exercises/day06-rag-api/data
```

第一次启动时服务生成向量并写入数据卷；以后重新构建镜像或创建新容器，只要数据卷还在，就能加载已有索引。

## 5. 构建并启动

先确保 Docker Desktop 已安装并启动，然后在仓库根目录执行：

```bash
docker compose -f exercises/day07-deployment/docker-compose.yml up --build -d
```

参数含义：

- `up`：创建并启动服务。
- `--build`：启动前重新构建镜像。
- `-d`：在后台运行。

查看状态：

```bash
docker compose -f exercises/day07-deployment/docker-compose.yml ps
```

查看日志：

```bash
docker compose -f exercises/day07-deployment/docker-compose.yml logs -f rag-api
```

访问：

```text
健康检查：http://127.0.0.1:8001/health
接口文档：http://127.0.0.1:8001/docs
```

## 6. 停止与清理

停止并删除容器，但保留向量数据卷：

```bash
docker compose -f exercises/day07-deployment/docker-compose.yml down
```

如果执行下面的命令，向量数据卷也会删除，下次启动必须重新 Embedding：

```bash
docker compose -f exercises/day07-deployment/docker-compose.yml down -v
```

## 7. `.dockerignore` 的作用

Docker 构建时会把构建上下文发送给 Docker Engine。`.dockerignore` 排除了：

- `.git`
- `.env`
- Python 缓存和虚拟环境
- 本地生成的向量索引

这样既减少镜像构建内容，也避免 API Key 被复制进镜像。

## 8. 第一周系统总结

```text
Day 1  FastAPI：API 入口和数据校验
Day 2  LangChain：Prompt、模型、Parser、LCEL
Day 3  文档加载与分块
Day 4  Embedding 与向量检索
Day 5  检索 + LLM 生成，完成 Naive RAG
Day 6  FastAPI 服务 + 索引持久化
Day 7  Docker 容器化部署
```

第一周结束后，你已经能解释并运行一个最小但完整的 RAG 服务。下一阶段不再只是增加组件，而是开始解决检索质量、评测、重排、异常处理和生产可用性。

## 9. 今天必须能回答的问题

1. 镜像和容器有什么区别？
2. 为什么 Uvicorn 在容器里必须监听 `0.0.0.0`？
3. 为什么 `.env` 不能通过 `COPY` 放进镜像？
4. 数据卷解决了什么问题？
5. `docker compose down` 和 `docker compose down -v` 有什么区别？
