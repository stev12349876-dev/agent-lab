# Day 4 学习笔记：Embedding 与向量检索

## 今日目标

把 Day 3 得到的文本块写入向量库，并能针对一个问题检索出最相关的 Top-K 证据块。

```text
chunks
  → Embedding 模型
  → 向量 + 原文 + metadata
  → Vector Store

question
  → 同一个 Embedding 模型
  → query vector
  → 相似度排序
  → Top-K chunks
```

代码入口：[exercises/day04-rag-vectorstore/main.py](../exercises/day04-rag-vectorstore/main.py)。

---

## 1. Embedding 是什么

Embedding 把一段文本转换成固定长度的数字列表（向量）。语义相近的文本，在向量空间里的方向通常也更接近。

```text
“Agent 有哪些组件？” → [0.12, -0.08, 0.45, ...]
“规划模块负责什么？” → [0.10, -0.06, 0.49, ...]
```

它不是给文本“打标签”，也不是 LLM 的最终回答；它的任务是让系统能先从大量 chunks 中找到候选证据。

## 2. 向量库到底存什么

每个 chunk 入库时要一起存三类信息：

| 内容 | 作用 |
|---|---|
| embedding vector | 用于计算问题和 chunk 的相似度 |
| `page_content` | 被检索命中后，作为回答上下文 |
| metadata | 显示来源、标题、页码、权限等 |

所以“向量库”不是只存数字的数据库；它是一个把向量和原始证据绑在一起的索引。

## 3. 检索过程

```python
vector_store.add_documents(chunks)
results = vector_store.similarity_search_with_score(question, k=3)
```

`add_documents` 会调用 `embed_documents`；查询时，`similarity_search_with_score` 会调用 `embed_query`。两边必须使用同一个 Embedding 模型和同一版本，否则向量空间不一致，分数没有可比性。

`k` 是返回候选块数：太小可能漏掉证据；太大则把噪声和 token 成本带给 LLM。Day 5 可以先从 `k=3` 或 `k=4` 开始评测。

## 4. 相似度分数怎么看

本练习使用的内存向量库会给出相似度分数，分数用于**同一次查询内排序**：越靠前的 chunk 越值得作为候选证据。

不要把某一个固定分数当作“正确阈值”。不同模型、距离度量和向量库的数值范围都可能不同；真正要看的，是正确证据是否稳定出现在 Top-K。

## 5. 使用真实 Embedding 模型

本项目通过 `OpenAIEmbeddings` 调用你已开通的 `qwen3.7-text-embedding`。模型名可以通过环境变量换成其他已授权模型：

```bash
python3 main.py
```

脚本会读取 `.env` 中的 API 配置，模型名可通过 `EMBEDDING_MODEL` 覆盖。模型名必须与所用服务商支持的 embedding 模型一致。

百炼的 OpenAI 兼容接口只接受字符串输入，而 LangChain 默认会先用 `tiktoken` 把文本分成 token ID 列表。练习代码通过 `check_embedding_ctx_length=False` 关闭这层预处理，直接发送 Day 3 已切好的短文本块；这也意味着生产代码要自行保证单个 chunk 不超过所选模型的输入上限。

## 6. `InMemoryVectorStore` 的定位

本练习选用 `InMemoryVectorStore`，优点是零运维、方便观察；缺点是进程退出数据就消失，无法支持大规模检索和生产部署。

后续可以把接口替换为 Chroma、FAISS、Milvus、pgvector 等。对上层 RAG 链来说，核心接口仍然是“写入 documents、按 query 检索 documents”。

## 7. 常见坑

| 问题 | 原因 | 处理 |
|---|---|---|
| 检索结果不相关 | chunk 切分差、Embedding 模型不合适、问题太模糊 | 回到 Day 3 检查 chunks，准备测试问题 |
| 换模型后结果很差 | 旧向量与新模型不在同一向量空间 | 用新模型重新向量化并全量入库 |
| 命中正确但回答仍幻觉 | 只检索没有约束生成 | Day 5 在 prompt 中要求“仅基于上下文回答并引用来源” |
| 结果重复 | overlap 太大或多个 chunk 内容相似 | 调整 overlap，后续增加去重或重排 |

## 8. 今天必须能回答的四个问题

1. Embedding 和 LLM 生成回答的职责有什么不同？
2. 为什么文档和问题必须用同一个 Embedding 模型处理？
3. 向量库为什么还要保存原文和 metadata？
4. `k` 增大有什么收益和代价？

能不用看笔记说清这四点，并运行脚本解释每个检索结果为什么排在前面，就可以进入 Day 5：把检索结果交给 LLM，完成 Naive RAG 闭环。
