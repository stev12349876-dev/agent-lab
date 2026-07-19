# Day 3 学习笔记：RAG 的文档加载与文本分割

## 今日目标

把原始文档处理成**可检索的文本块（chunks）**，为 Day 4 的 Embedding 和向量库做输入。

这一步不调用 LLM，却会直接决定后面的 RAG 能不能检索到正确证据、回答有没有上下文。

```
原始文件
  → Document Loader
  → Document（内容 + 元数据）
  → Text Splitter
  → chunks（内容 + 可追溯元数据）
  → Embedding / Vector Store（Day 4）
```

代码入口：[exercises/day03-rag-loading/main.py](../exercises/day03-rag-loading/main.py)。

---

## 1. 为什么 RAG 不能直接把整篇文档塞给 LLM？

直接传整篇文档有三个问题：

- **上下文有限且昂贵**：文档越长，token 成本和延迟越高。
- **噪声太多**：一个问题通常只需要其中一小段；无关内容会干扰模型。
- **无法高效检索**：向量库要比较“问题”和“候选知识”，最小可检索单位应该是语义完整的小块。

因此，RAG 的第一层工作不是“让模型读文档”，而是：**先把文档切成既有语义、又足够小的候选证据块。**

---

## 2. `Document`：RAG 的统一数据格式

LangChain 加载器不会只返回字符串，而是返回 `Document`：

```python
Document(
    page_content="文档正文……",
    metadata={"source": "data/sample.md"},
)
```

| 字段 | 作用 | 后续用途 |
|---|---|---|
| `page_content` | 真正要切分、向量化、放进提示词的文本 | 检索证据 |
| `metadata` | 来源、页码、标题、作者等附加信息 | 引用来源、过滤、调试 |

**元数据不是装饰。** 如果回答“根据哪份文档、哪一节得出这个结论？”时拿不出来源，RAG 就难以被信任。

加载单个 Markdown 文件：

```python
from langchain_community.document_loaders import TextLoader

docs = TextLoader("data/sample.md").load()
```

不同格式换加载器即可：PDF 常用 `PyPDFLoader`，CSV 常用 `CSVLoader`。它们的输出仍应归一到 `Document`，这就是 LangChain 组件能拼接的原因。

---

## 3. 为什么要分块？

检索系统不是从全文中“找关键词”，而是为问题找到最相关的 chunks。一个好 chunk 要同时满足：

1. **语义完整**：不要把定义和它的限定条件切到两块里。
2. **足够聚焦**：一块不要混入太多不相关主题。
3. **带来源信息**：检索命中后能定位回原文。

分块太大时，语义混杂、检索不精准、上下文成本高；太小时，关键条件被拆散、模型拿到的证据不完整。

---

## 4. `RecursiveCharacterTextSplitter`：默认首选

练习中最常用的分割器：

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", ".", " ", ""],
)

chunks = splitter.split_documents(docs)
```

它会按分隔符优先级递归尝试：先按段落，再按换行、句子、空格，最后才退化到逐字符。目标是尽量不在一个自然语义单元的中间切断。

### `chunk_size`

这里的 `chunk_size=200` 指的是**字符数**，不是 token 数。中文、英文、代码的字符/token 比例不同，所以这个数是实验起点，不是通用真理。

### `chunk_overlap`

`chunk_overlap=40` 让相邻块共享一部分内容：

```text
Chunk 1: RAG 将 LLM 与外部知识库结合，让 AI 能引用准确的信息。
Chunk 2: …让 AI 能引用准确的信息。它通过检索找到相关上下文。
```

重叠能保留跨边界的因果、定义和指代关系；代价是重复向量、更多存储和检索冗余。常用起点是 `chunk_size` 的 **10%–20%**，然后用真实问答集调优。

---

## 5. Markdown 文档要优先保留结构

`MarkdownHeaderTextSplitter` 会按标题切分，并把标题放进 metadata：

```python
headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
chunks = md_splitter.split_text(markdown_text)
```

输出的 chunk 可能是：

```python
Document(
    page_content="RAG 的定义……",
    metadata={"H1": "RAG", "H2": "基本概念"},
)
```

这特别适合教程、产品文档、知识库：命中内容时可以显示所属章节，也能按章节过滤。

**注意：标题切分不保证块的长度。** 一个章节很长时，应继续用 `RecursiveCharacterTextSplitter` 做第二次切分；实战中常用“先按标题保留结构，再按长度细分”的两阶段策略。

---

## 6. 参数怎么起步

| 数据类型 | 建议 `chunk_size` | 建议 `chunk_overlap` | 重点 |
|---|---:|---:|---|
| FAQ、短知识条目 | 128–256 字符 | 10%–20% | 一问一答不要混切 |
| Markdown 文档 | 400–800 字符 | 10%–20% | 先保留标题层级 |
| 普通长文 | 500–1,000 字符 | 10%–20% | 以自然段和句子为边界 |
| 代码 | 不宜只按字符切 | 少量或无重叠 | 应优先按函数、类、AST 结构 |

这些是**初始假设**。正确答案来自评测：准备一组真实问题，比较“是否检索到正确 chunk、回答是否有依据、成本和延迟是否可接受”。

---

## 7. 常见失败模式

| 现象 | 常见原因 | 调整方向 |
|---|---|---|
| 命中内容太泛 | chunk 太大、混合多个主题 | 缩小块，增加结构化 metadata |
| 命中内容缺半句 | chunk 太小或 overlap 为 0 | 增大块或重叠 |
| 同一内容反复出现 | overlap 过大 | 降低重叠，后续加去重/重排 |
| 回答无法给出处 | 丢失或没有使用 metadata | 让每个 chunk 保留 `source`、标题、页码 |
| 脚本找不到文件 | 使用相对路径且从错误工作目录运行 | 从 `day03-rag-loading` 目录运行，或改用基于文件位置的绝对路径 |

---

## 8. 与 Day 2、Day 4 的连接

Day 2 的链负责“根据输入生成回答”：

```python
prompt | llm | parser
```

Day 3 负责把外部知识整理成可检索输入；Day 4 会把每个 chunk 变成向量并存进向量库。之后完整链路是：

```python
question
  → similarity_search(question)
  → 相关 chunks
  → context = "\n\n".join(chunk.page_content for chunk in chunks)
  → prompt({"context": context, "question": question})
  → llm
  → answer + sources
```

FastAPI 是服务入口，LangChain 是编排方式，分块和检索则负责让模型在回答前拿到可靠证据。

---

## 9. 今天必须能回答的四个问题

1. `Document` 为什么不只是字符串？——因为 RAG 同时需要正文和可追溯的 metadata。
2. 为什么不能只切成固定长度？——会在句子/段落中间断开，破坏语义；递归分隔优先保留自然边界。
3. overlap 解决什么问题，又带来什么代价？——保留跨块上下文；代价是重复、存储和检索成本。
4. 为什么 Markdown 先按标题切？——标题是天然语义边界，并能作为检索后的章节来源。

能不用看笔记解释这四点，并独立修改 `chunk_size`、`chunk_overlap` 后观察输出差异，就说明 Day 3 已经学懂。
