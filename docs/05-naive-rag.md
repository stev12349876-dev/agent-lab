# Day 5 学习笔记：完成第一个 Naive RAG

## 今日目标

把前四天的组件串成第一个完整问答闭环：先检索证据，再让 LLM 只依据证据生成答案。

```text
用户问题
  → Embedding
  → 向量检索 Top-K
  → 格式化 context
  → Prompt + qwen-plus
  → 带来源编号的答案
```

代码入口：[exercises/day05-naive-rag/main.py](../exercises/day05-naive-rag/main.py)。

---

## 1. 为什么叫 Naive RAG

Naive RAG 是最基础的 RAG：一次检索、一次生成，中间没有查询改写、混合检索、重排或评测。

它虽然简单，但已经具备完整结构：

1. **Indexing**：加载、分块、Embedding、写入向量库。
2. **Retrieval**：把问题向量化，取相似度最高的 Top-K chunks。
3. **Generation**：把 chunks 放进 Prompt，让 LLM 基于证据回答。

后面的高级 RAG 都是在这三个阶段增加优化，而不是另一套完全不同的系统。

## 2. 检索不会自动影响 LLM

向量库只负责返回 `Document`。如果不把 `page_content` 放进 Prompt，LLM 根本不知道检索到了什么。

```python
results = vector_store.similarity_search_with_score(question, k=3)
context = format_context(results)
answer = await chain.ainvoke({"context": context, "question": question})
```

RAG 中最关键的数据传递是：

```text
retrieved Documents → context 字符串 → Prompt → LLM
```

## 3. 为什么给来源编号

每个检索结果被格式化为：

```text
[来源1]
标题：ReAct 模式
内容：ReAct（Reasoning + Acting）……
```

Prompt 要求模型用 `[来源1]` 标注事实。这样用户可以把答案与实际检索结果对照，判断模型是否引用了正确证据。

注意：引用编号仍由 LLM 生成，可能标错。生产系统还需要程序化引用校验；当前练习先建立“答案必须有证据”的习惯。

## 4. 如何降低幻觉

系统提示词加入三条约束：

- 只能根据提供的上下文回答。
- 不得补充外部知识。
- 上下文不足时明确回答“根据当前知识库无法回答”。

这些规则能降低幻觉，但不能保证完全消除。真正可靠的 RAG 还要评测检索是否正确、答案是否忠实于证据。

## 5. Top-K 的作用

本练习先取 `k=3` 个候选结果，再使用 `0.4` 的相关度阈值过滤：

- K 太小：可能漏掉回答所需的证据。
- K 太大：上下文噪声增加，token 成本和延迟也会上升。

相似度分数是排序信号，不是“答案正确概率”。应通过一组真实问题观察正确证据是否进入 Top-K。

仅靠 Prompt 要求“上下文不足就拒答”并不可靠：模型仍可能使用自身常识回答。因此代码在调用 LLM 前增加程序化门槛；如果所有候选都低于阈值，就直接返回“根据当前知识库无法回答”，不让 LLM 获得自由发挥的机会。`0.4` 只是针对当前模型和小样本的起点，后续应通过评测集调整。

## 6. 当前系统的真实边界

当前系统已经真实调用：

- `qwen3.7-text-embedding` 生成 1024 维向量。
- `InMemoryVectorStore` 执行向量相似度检索。
- `qwen-plus` 根据检索上下文生成答案。

但它仍有明显限制：

- 每次启动都会重新向量化，向量没有持久化。
- 只有一个样本文档，测试数据太少。
- 相关度阈值目前是根据少量示例设置的，还没有经过系统评测。
- 没有自动评测、重排和引用校验。
- 目前是 CLI，还没有接入 FastAPI。

这些限制正是后续课程要逐步解决的问题。

## 7. 运行方式

默认问题：

```bash
python3 main.py
```

自定义问题：

```bash
python3 main.py "谁负责评估多 Agent 的执行结果？"
```

## 8. 今天必须能回答的问题

1. 检索出的 `Document` 为什么必须格式化进 Prompt？
2. Embedding 模型和 Chat 模型分别负责哪一步？
3. Top-K 太大或太小分别有什么风险？
4. Prompt 写了“只根据上下文回答”，为什么仍不能保证绝对无幻觉？
5. 当前系统为什么每次运行都会再次消耗 Embedding 额度？

能独立描述完整数据流，并能根据终端输出判断“检索错了”还是“LLM 回答错了”，就掌握了 Naive RAG 的核心。
