# Day 3 学习笔记：RAG 文档加载与文本分割

## 一、RAG 的第一步：把文档变成数据

```
原始文件（PDF/MD/TXT）
    ↓ Document Loader
Document 对象（page_content + metadata）
    ↓ Text Splitter
文本块列表（chunks）
    ↓ Day 4：向量化
Embedding 向量
    ↓ 存入向量库
可检索的知识库
```

今天完成前两步。

## 二、Document Loader

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.md")
docs = loader.load()
# docs[0] = Document(
#     page_content="完整的文本内容...",
#     metadata={"source": "data/sample.md"}
# )
```

核心概念：`Document` 是 LangChain 的数据载体，包含两个字段：
- `page_content`：文本内容
- `metadata`：来源信息（文件名、页码等）

LangChain 支持几十种加载器：`PyPDFLoader`、`CSVLoader`、`UnstructuredMarkdownLoader` 等，接口统一都用 `.load()`。

## 三、Text Splitter

### 3.1 RecursiveCharacterTextSplitter（最常用）

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,         # 每块最多 200 字符
    chunk_overlap=40,       # 相邻块重叠 40 字符
    separators=["\n\n","\n","。","."," ",""],
)
```

工作原理：按 `separators` 优先级尝试断开——先尝试段落边界，不行再句子，不行再词，最后才字符。这样切出来的块语义更完整。

### 3.2 chunk_overlap 的作用

| overlap=0 | overlap=30 |
|-----------|------------|
| 块1: "AI正在改变世界。LLM让机器" | 块1: "AI正在改变世界。LLM让机器" |
| 块2: "能够理解自然语言。" | 块2: "。LLM让机器能够理解自然语言。" |

没有 overlap → "LLM让机器" 和 "能够理解" 被切开，语义丢失。
有了 overlap → 相邻两块都包含关键连接词，检索时不会被漏掉。

### 3.3 MarkdownHeaderTextSplitter

按 Markdown 标题层级分割，保留文档结构：

```python
headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
```

每个块的 `metadata` 会包含所属的 H1/H2/H3 标题，检索时可以用来过滤和排序。

## 四、分割参数怎么选

| 场景 | chunk_size | chunk_overlap |
|------|-----------|---------------|
| 短问答 FAQ | 128-256 | 20-40 |
| 文档问答 | 512-1024 | 50-100 |
| 长文档摘要 | 1024-2048 | 100-200 |
| 代码检索 | 256-512 | 0（代码结构靠 AST） |

没有最优值，需要在你的数据上做实验。这是 RAG 最重要的调参环节。

## 五、和 Day 2 的关系

Day 2 你学了 `ChatPromptTemplate | LLM | StrOutputParser`。
Day 3 在给这个链准备"输入材料"——把文档切成 LLM 能处理的大小。

到 Day 5-6，你会这样串起来：

```python
# 检索相关文档块
relevant_chunks = vector_store.similarity_search(question)
# 拼成上下文
context = "\n".join([c.page_content for c in relevant_chunks])
# 喂给 LLM
answer = await chain.ainvoke({"context": context, "question": question})
```
