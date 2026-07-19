"""Day 6: FastAPI RAG 服务 + 本地向量索引持久化。"""

import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, Request
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import BaseModel, Field

# 当前脚本所在目录，用它拼接路径可以避免从不同目录启动时找不到文件。
BASE_DIR = Path(__file__).resolve().parent
# 复用 Day 3 的 Markdown 样本文档，作为当前 RAG 的知识库。
SAMPLE_PATH = BASE_DIR.parent / "day03-rag-loading" / "data" / "sample.md"
# 向量索引和索引说明文件会保存在 Day 6 的 data 目录中。
INDEX_PATH = BASE_DIR / "data" / "vector-store.json"
INDEX_META_PATH = BASE_DIR / "data" / "index-meta.json"
# 分块方式的版本号。以后修改分块参数时也要修改它，让旧索引自动失效。
SPLITTER_VERSION = "markdown-headers_recursive-200-40_v1"


# POST /query 接收的请求体。
class QueryRequest(BaseModel):
    # 问题不能为空，最多 500 个字符。
    question: str = Field(min_length=1, max_length=500)
    # 最多召回几个候选文本块，默认 3 个，只允许 1～5。
    top_k: int = Field(default=3, ge=1, le=5)


# 返回给用户的单条证据信息。
class SourceResponse(BaseModel):
    number: int
    title: str
    # 问题向量和文档向量的余弦相似度，不是答案正确率。
    score: float
    chunk_id: int
    source: str


# POST /query 的完整响应结构。
class QueryResponse(BaseModel):
    answer: str
    # True 表示找到了超过相关度阈值的知识库证据。
    grounded: bool
    sources: list[SourceResponse]


# GET /health 的响应结构，用于查看服务和索引状态。
class HealthResponse(BaseModel):
    status: str
    chunks: int
    index_status: str
    embedding_model: str
    chat_model: str


def load_chunks() -> list[Document]:
    """读取知识库并分块，返回带来源信息的 LangChain Document 列表。"""
    markdown_text = SAMPLE_PATH.read_text(encoding="utf-8")

    # 第一阶段：按 Markdown 标题切分，保留 H1/H2/H3 章节信息。
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(markdown_text)

    # 第二阶段：长章节继续切成最多 200 字符的小块，相邻块重叠 40 字符。
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = chunk_splitter.split_documents(sections)

    # 给每个 chunk 增加文件来源和编号，后面返回引用时会用到。
    for index, chunk in enumerate(chunks):
        chunk.metadata.update({"source": str(SAMPLE_PATH), "chunk_id": index})
    return chunks


def get_title(document: Document) -> str:
    """从 chunk 的 metadata 中找到最具体的标题。"""
    # 优先使用三级标题；没有就逐级退回 H2、H1，最后使用默认名称。
    return (
        document.metadata.get("H3")
        or document.metadata.get("H2")
        or document.metadata.get("H1")
        or "未命名章节"
    )


class RagService:
    """把模型、Prompt、向量库和问答流程封装成一个可复用的 RAG 服务。"""

    def __init__(self) -> None:
        # 从 .env 读取模型配置；环境变量缺失时使用默认值。
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
        self.chat_model = os.getenv("CHAT_MODEL", "qwen-plus")
        self.min_relevance_score = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.4"))

        # Embedding 模型负责把文档和问题转换成向量，用于相似度检索。
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            check_embedding_ctx_length=False,
        )

        # Chat 模型负责读取检索结果并生成最终自然语言答案。
        self.llm = ChatOpenAI(
            model=self.chat_model,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        # Prompt 明确告诉模型：只能依据检索到的知识库上下文回答。
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个严谨的知识库问答助手。"
                    "只能根据提供的知识库上下文回答，不得补充外部知识。"
                    "如果上下文不足，请明确回答：根据当前知识库无法回答。"
                    "回答中的事实必须用 [来源1]、[来源2] 这样的编号标注出处。",
                ),
                ("human", "知识库上下文：\n{context}\n\n用户问题：{question}"),
            ]
        )

        # LCEL 链：填充 Prompt → 调用 qwen-plus → 提取纯文本答案。
        self.chain = self.prompt | self.llm | StrOutputParser()

        # 服务创建时只执行一次：加载已有索引，或者重新生成并保存索引。
        self.vector_store, self.index_status = self._load_or_build_index()

    def _fingerprint(self) -> str:
        """为文档、模型和分块配置生成指纹，用于判断旧索引是否还能使用。"""
        digest = hashlib.sha256()
        # 文档内容变化，指纹会变化。
        digest.update(SAMPLE_PATH.read_bytes())
        # 更换 Embedding 模型后，旧模型生成的向量不能继续使用。
        digest.update(self.embedding_model.encode("utf-8"))
        # 修改分块策略后，chunks 变化，也必须重新生成向量。
        digest.update(SPLITTER_VERSION.encode("utf-8"))
        return digest.hexdigest()

    def _load_or_build_index(self) -> tuple[InMemoryVectorStore, str]:
        """索引有效时从磁盘加载，否则调用 Embedding API 重新建立索引。"""
        fingerprint = self._fingerprint()

        # 两个索引文件都存在时，先检查保存的指纹。
        if INDEX_PATH.exists() and INDEX_META_PATH.exists():
            metadata = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint:
                # 指纹一致，说明文档、模型和分块方式都没变，直接加载旧向量。
                return InMemoryVectorStore.load(str(INDEX_PATH), self.embeddings), "loaded"

        # 没有索引或指纹已变化：加载 chunks，并调用真实 Embedding API 生成向量。
        vector_store = InMemoryVectorStore(embedding=self.embeddings)
        vector_store.add_documents(load_chunks())

        # 把向量、正文和 metadata 保存到 JSON，下次启动可以直接复用。
        vector_store.dump(str(INDEX_PATH))

        # 单独保存本次索引的指纹，供下次启动比较。
        INDEX_META_PATH.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "embedding_model": self.embedding_model},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # built 表示本次启动重新生成了索引。
        return vector_store, "built"

    @property
    def chunk_count(self) -> int:
        """返回当前向量库保存的 chunk 数量。"""
        return len(self.vector_store.store)

    def _format_context(self, results: list[tuple[Document, float]]) -> str:
        """把检索结果整理成带来源编号的文本，准备放进 Prompt。"""
        context_parts = []
        for number, (document, _) in enumerate(results, start=1):
            context_parts.append(
                f"[来源{number}]\n"
                f"标题：{get_title(document)}\n"
                f"内容：{document.page_content}"
            )
        return "\n\n".join(context_parts)

    def _format_sources(
        self, results: list[tuple[Document, float]]
    ) -> list[SourceResponse]:
        """把检索结果转换成 API 要返回的结构化来源列表。"""
        return [
            SourceResponse(
                number=number,
                title=get_title(document),
                score=round(score, 3),
                chunk_id=document.metadata["chunk_id"],
                source=document.metadata["source"],
            )
            for number, (document, score) in enumerate(results, start=1)
        ]

    async def answer(self, question: str, top_k: int) -> QueryResponse:
        """执行完整问答流程：检索、过滤、生成答案、返回来源。"""
        # 相似度检索内部会同步等待问题的 Embedding API，因此放到工作线程中，
        # 避免阻塞 FastAPI 的异步事件循环。
        candidates = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            question,
            top_k,
        )

        # 程序化过滤低相关候选，不能只依赖 Prompt 要求模型拒答。
        results = [
            result for result in candidates if result[1] >= self.min_relevance_score
        ]

        # 没有任何候选超过阈值时，不调用 qwen-plus，直接拒绝回答。
        if not results:
            return QueryResponse(
                answer="根据当前知识库无法回答。",
                grounded=False,
                sources=[],
            )

        # 把通过过滤的 chunks 放入 Prompt，再异步调用 qwen-plus。
        answer = await self.chain.ainvoke(
            {"context": self._format_context(results), "question": question}
        )

        # 返回答案，同时把真正使用的检索证据一起交给用户。
        return QueryResponse(
            answer=answer,
            grounded=True,
            sources=self._format_sources(results),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：服务启动时创建一次 RagService。"""
    # 从项目上级目录查找并加载 .env。
    load_dotenv(find_dotenv())
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，请先在 .env 中配置。")

    # RagService 初始化可能读取磁盘或调用 Embedding API，因此放到工作线程。
    # 创建完成后保存到 app.state，所有请求共享同一个实例和向量库。
    app.state.rag_service = await asyncio.to_thread(RagService)

    # yield 之前是启动阶段；执行到这里后 FastAPI 开始接收请求。
    yield


app = FastAPI(
    title="Agent Lab RAG API",
    version="0.1.0",
    lifespan=lifespan,
)


# 健康检查接口：不调用模型，只报告服务和索引状态。
@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    # 取出服务启动时保存在 app.state 中的同一个 RagService。
    service: RagService = request.app.state.rag_service
    return HealthResponse(
        status="healthy",
        chunks=service.chunk_count,
        index_status=service.index_status,
        embedding_model=service.embedding_model,
        chat_model=service.chat_model,
    )


# RAG 问答接口：FastAPI 先用 QueryRequest 校验 JSON，再调用 RagService。
@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    service: RagService = request.app.state.rag_service
    return await service.answer(payload.question, payload.top_k)
