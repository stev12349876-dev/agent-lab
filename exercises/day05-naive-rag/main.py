"""
Day 5: Naive RAG — 检索并基于证据生成答案

运行：
  python3 main.py
  python3 main.py "谁负责评估多 Agent 的执行结果？"
"""

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR.parent / "day03-rag-loading" / "data" / "sample.md"
MIN_RELEVANCE_SCORE = 0.4


def load_chunks() -> list[Document]:
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


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        check_embedding_ctx_length=False,
    )


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("CHAT_MODEL", "qwen-plus"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def get_title(document: Document) -> str:
    return (
        document.metadata.get("H3")
        or document.metadata.get("H2")
        or document.metadata.get("H1")
        or "未命名章节"
    )


def format_context(results: list[tuple[Document, float]]) -> str:
    context_parts = []
    for source_number, (document, _) in enumerate(results, start=1):
        context_parts.append(
            f"[来源{source_number}]\n"
            f"标题：{get_title(document)}\n"
            f"内容：{document.page_content}"
        )
    return "\n\n".join(context_parts)


def print_sources(results: list[tuple[Document, float]], heading: str = "检索来源") -> None:
    print(f"\n{heading}：")
    for source_number, (document, score) in enumerate(results, start=1):
        print(
            f"  [来源{source_number}] score={score:.3f} | "
            f"{get_title(document)} | chunk_id={document.metadata['chunk_id']}"
        )


async def answer_question(question: str) -> None:
    chunks = load_chunks()
    vector_store = InMemoryVectorStore(embedding=create_embeddings())
    vector_store.add_documents(chunks)

    candidates = vector_store.similarity_search_with_score(question, k=3)
    results = [result for result in candidates if result[1] >= MIN_RELEVANCE_SCORE]
    if not results:
        print(f"问题：{question}\n")
        print("回答：根据当前知识库无法回答。")
        print_sources(candidates, heading="未达到相关度阈值的候选")
        return

    context = format_context(results)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个严谨的知识库问答助手。"
                "只能根据提供的知识库上下文回答，不得补充外部知识。"
                "如果上下文不足，请明确回答：根据当前知识库无法回答。"
                "回答中的事实必须用 [来源1]、[来源2] 这样的编号标注出处。",
            ),
            (
                "human",
                "知识库上下文：\n{context}\n\n用户问题：{question}",
            ),
        ]
    )
    chain = prompt | create_llm() | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "question": question})

    print(f"问题：{question}\n")
    print(f"回答：{answer}")
    print_sources(results)


def parse_question() -> str:
    parser = argparse.ArgumentParser(description="基于本地知识库回答问题")
    parser.add_argument(
        "question",
        nargs="?",
        default="ReAct 模式是怎样工作的？",
        help="要向知识库提出的问题",
    )
    return parser.parse_args().question


def main() -> None:
    load_dotenv(find_dotenv())
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，请先在 .env 中配置。")
    asyncio.run(answer_question(parse_question()))


if __name__ == "__main__":
    main()
