"""
Day 3: RAG Part 1 — 文档加载与文本分割

目标：掌握 LangChain 的文档加载器和文本分割器
这是 RAG 系统的第一步——把原始文档变成可检索的文本块
"""

from pathlib import Path

# ============================================================
# Part 1: 文档加载（Document Loaders）
# ============================================================

from langchain_community.document_loaders import TextLoader

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "data" / "sample.md"

# 1.1 加载单个文本文件
loader = TextLoader(str(SAMPLE_PATH), encoding="utf-8")
docs = loader.load()
print(f"[加载] 共 {len(docs)} 个文档")
print(f"[加载] 第一个文档开头: {docs[0].page_content[:100]}...")
print(f"[加载] 元数据: {docs[0].metadata}\n")

# 文档对象的结构：
#   Document(
#       page_content="完整的文本内容...",
#       metadata={"source": "data/sample.md"}  # 来源信息
#   )


# ============================================================
# Part 2: 文本分割（Text Splitters）
# ============================================================

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# 2.1 最常用的分割器：RecursiveCharacterTextSplitter
# 按优先级尝试分隔符：\n\n → \n → 空格 → 空字符
# 这样能尽量在段落/句子边界断开，而不是在词中间
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # 每块最多 200 字符
    chunk_overlap=40,      # 相邻块重叠 40 字符（防止语义断裂）
    separators=["\n\n", "\n", "。", ".", " ", ""],
)

chunks = splitter.split_documents(docs)
print(f"[递归分割] 共 {len(chunks)} 个文本块")
for i, chunk in enumerate(chunks[:5]):
    print(f"  [{i}] {len(chunk.page_content)} 字符 | {chunk.page_content[:]}...")

# 2.2 Markdown 按标题分割（保留文档结构）
headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,
)

# 先按标题保留文档结构；再按长度细分超长章节
md_text = SAMPLE_PATH.read_text(encoding="utf-8")

md_sections = md_splitter.split_text(md_text)
md_chunks = splitter.split_documents(md_sections)
print(f"\n[Markdown 标题 + 递归分割] 共 {len(md_chunks)} 个块")
for chunk in md_chunks[:5]:
    title = chunk.metadata.get("H1") or chunk.metadata.get("H2") or ""
    print(f"  [{title}] {chunk.page_content[:]}...")


# ============================================================
# Part 3: 分割参数实验（理解 chunk_size 和 chunk_overlap）
# ============================================================

text = """人工智能正在深刻改变我们的世界。从自动驾驶到医疗诊断，AI 技术已经渗透到各行各业。
大语言模型的出现更是让机器具备了理解和生成自然语言的能力。
RAG（检索增强生成）技术将 LLM 与外部知识库结合，让 AI 能够引用实时、准确的信息。"""

print(f"\n[参数实验] 原文长度: {len(text)} 字符")

for size, overlap in [(100, 0), (100, 30), (50, 15)]:
    s = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["。", "，", " ", ""],
    )
    chunks = s.split_text(text)
    print(f"  chunk_size={size}, overlap={overlap}: {len(chunks)} 块")
    for i, c in enumerate(chunks):
        print(f"    [{i}] {len(c)}字: {c[:50]}...")


# ============================================================
# 核心要点总结
# ============================================================

print("""
╔══════════════════════════════════════════════╗
║  Day 3 核心要点                              ║
╠══════════════════════════════════════════════╣
║  1. Document Loader: 把文件读成 Document 对象 ║
║  2. Document = page_content + metadata       ║
║  3. chunk_size: 越大越完整，越小越精准        ║
║  4. chunk_overlap: 防语义断裂，建议 10-20%    ║
║  5. Markdown按标题分：保留文档层级结构         ║
╚══════════════════════════════════════════════╝
""")
