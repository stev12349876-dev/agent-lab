"""
Day 4: RAG Part 2 — Embedding 与向量检索

运行：
  python3 main.py
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR.parent / "day03-rag-loading" / "data" / "sample.md"


def load_chunks() -> list[Document]:
    """复用 Day 3 的样本文档，并用两阶段分割保留标题 metadata。"""
    markdown_text = SAMPLE_PATH.read_text(encoding="utf-8")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(markdown_text)
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = chunk_splitter.split_documents(sections)
    for index, chunk in enumerate(chunks):
        chunk.metadata.update({"source": str(SAMPLE_PATH), "chunk_id": index})
    return chunks


def get_embeddings() -> OpenAIEmbeddings:
    load_dotenv(find_dotenv())
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，请先在 .env 中配置。")

    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        # 百炼只接受字符串输入；关闭 LangChain 的 tiktoken 预切分。
        check_embedding_ctx_length=False,
    )


def print_results(query: str, results: list[tuple[Document, float]]) -> None:
    print(f"\n问题：{query}")
    for rank, (document, score) in enumerate(results, start=1):
        title = document.metadata.get("H3") or document.metadata.get("H2") or document.metadata.get("H1")
        preview = document.page_content.replace("\n", " ")[:90]
        print(f"  #{rank} score={score:.3f} | {title} | {preview}...")


def main() -> None:
    chunks = load_chunks()
    embeddings = get_embeddings()
    vector_store = InMemoryVectorStore(embedding=embeddings)

    # 写入时：每个 chunk → 向量 + 原文 + metadata
    vector_store.add_documents(chunks)
    print(f"已写入 {len(chunks)} 个 chunks（qwen3.7-text-embedding）。")

    # 查询时：问题 → 向量 → 相似度排序 → Top-K 文档
    for query in ["规划模块负责什么？", "ReAct 是怎样工作的？", "谁负责评估多 Agent 的结果？"]:
        results = vector_store.similarity_search_with_score(query, k=2)
        print_results(query, results)


if __name__ == "__main__":
    main()
