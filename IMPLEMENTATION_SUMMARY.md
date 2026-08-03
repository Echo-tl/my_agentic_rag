# Agentic RAG 项目实现总结 — MVP 阶段

> 项目：论文分类与论文技术说明 Agentic RAG
>
> 技术栈：LangGraph + LlamaIndex + Qwen + BGE-M3 + Qdrant

---

## 一、项目结构

```
my_agentic_rag/
├── config.py                         # 全局配置（单一真相源）
├── main.py                           # 入口
├── requirements.txt                  # （待补充）
│
├── models/                           # 模型层
│   ├── llm.py                        # Qwen 工厂函数
│   └── embedding.py                  # BGE-M3 工厂函数
│
├── database/                         # 向量数据库
│   └── qdrant.py                     # Qdrant 客户端 + VectorStore
│
├── rag/                              # LlamaIndex RAG Pipeline
│   ├── ingestion.py                  # Document → Node（IngestionPipeline）
│   ├── index.py                      # VectorStoreIndex 构建 / 加载
│   ├── retriever.py                  # Retriever + SimilarityPostprocessor
│   ├── reranker.py                   # LLMRerank
│   └── query_engine.py               # RetrieverQueryEngine 组装
│
├── tools/                            # LangGraph Tool 层
│   ├── rag/
│   │   └── llamaindex_tool.py        # search_knowledge_base Tool
│   └── search/
│       └── web_search.py             # search_web Tool
│
├── workflows/                        # LangGraph Agent 层
│   ├── state.py                      # AgentState 定义
│   └── graph.py                      # create_agent 图构建
│
├── prompts/                          # Prompt 管理
│   └── supervisor.py                 # 系统提示词
│
├── memory/                           # 对话记忆（待实现）
├── evaluation/                       # 评估模块（待实现）
├── observability/                    # 可观测性（待实现）
├── api/                              # API 服务（待实现）
├── docker/                           # 部署（待实现）
├── data/                             # 原始文档目录
│
├── DESIGN_DOC.md                     # 系统设计文档
└── IMPLEMENTATION_SUMMARY.md         # 本文件
```

---

## 二、已实现文件详解

### 2.1 `config.py` — 全局配置

**设计模式**：Pydantic Settings 嵌套配置 + 全局单例。

```python
Settings (顶层)
├── LLMConfig          # provider, model, base_url, temperature, max_tokens
├── EmbeddingConfig    # model_name, device, batch_size
├── QdrantConfig       # url, collection_name, vector_size
├── ChunkConfig        # chunk_size, chunk_overlap
├── RetrievalConfig    # similarity_top_k, reranker_top_n, similarity_cutoff
├── MemoryConfig       # max_tokens, checkpoint_db_path
└── PathConfig         # data_dir, persist_dir, checkpoint_dir
```

**关键设计**：
- `env_prefix="AGENTIC_RAG_"` + `env_nested_delimiter="__"`，环境变量可覆盖任何默认值
- 7 个子配置类各自独立，修改一个不影响其他

**模型名**：当前默认 `qwen2.5:7b`，需改为你的 `qwen3.5:4b`。

---

### 2.2 `models/llm.py` — LLM 工厂

**函数**：`create_llm(cfg: LLMConfig) -> Ollama`

**当前实现**：只支持 Ollama Provider。

```python
def create_llm(cfg: LLMConfig):
    return Ollama(
        model=cfg.model,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        request_timeout=cfg.request_timeout,
        additional_kwargs={"options": {"num_predict": cfg.max_tokens}},
    )
```

**注意事项**：
- `max_tokens` 通过 `additional_kwargs` → `options` → `num_predict` 传递给 Ollama，不能直接传
- 后续可扩展 `provider="dashscope"` 分支

---

### 2.3 `models/embedding.py` — Embedding 工厂

**函数**：`create_embedding(cfg: EmbeddingConfig) -> HuggingFaceEmbedding`

```python
def create_embedding(cfg: EmbeddingConfig):
    return HuggingFaceEmbedding(
        model_name=cfg.model_name,       # "BAAI/bge-m3"
        device=cfg.device,               # "cpu"
        embed_batch_size=cfg.batch_size, # 32
    )
```

**BGE-M3 关键特性**：1024 维，中英文，支持 dense + sparse 双向量。

---

### 2.4 `database/qdrant.py` — Qdrant 连接

```python
client = QdrantClient(url=config.qdrant.url)
# Docker 模式：url="http://localhost:6333"

def get_vector_store():
    return QdrantVectorStore(
        client=client,
        collection_name=config.qdrant.collection_name,
    )
```

**两种部署方式**：
- Docker：`client = QdrantClient(url="http://localhost:6333")`
- 本地文件：`client = QdrantClient(path="./qdrant_data")`（无需额外服务）

---

### 2.5 `rag/ingestion.py` — 数据摄取管道

**函数**：`run_ingestion(data_dir: Path, embed_model) -> List[BaseNode]`

**数据流**：

```
data/*.md → SimpleDirectoryReader → [Document, Document, ...]
  → IngestionPipeline([
        SentenceSplitter(chunk_size=1024, chunk_overlap=128),
        BGE-M3_Embedding,
    ])
  → [TextNode(embedding=...), ...]
```

**依赖**：
- `config.chunk.chunk_size` / `config.chunk.chunk_overlap` — 切分参数
- `embed_model` — 从参数传入（依赖注入），不自行创建

---

### 2.6 `rag/index.py` — 索引构建与加载

**三个函数**：

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `build_index()` | 首次构建 | nodes + vector_store + embed_model | VectorStoreIndex |
| `load_index()` | 从磁盘加载 | vector_store + embed_model | VectorStoreIndex |
| `get_index()` | 统一入口 | data_dir + vector_store + embed_model | VectorStoreIndex |

**核心逻辑**：

```python
def get_index(data_dir, vector_store, embed_model):
    if persist_dir 存在:
        return load_index(vector_store, embed_model)   # 已构建 → 直接加载
    else:
        nodes = run_ingestion(data_dir, embed_model)    # 首次 → 构建
        return build_index(nodes, vector_store, embed_model)
```

**关键设计**：
- `VectorStoreIndex` 通过 `StorageContext.from_defaults(vector_store=vector_store)` 传入 Qdrant
- 构建后 `persist()` 到 `config.paths.persist_dir`，下次可直接加载跳过 ingestion

---

### 2.7 `rag/retriever.py` — 检索器配置

**函数**：`get_retriever(index) -> (retriever, postprocessor)`

```python
def get_retriever(index):
    retriever = index.as_retriever(
        similarity_top_k=config.retrieval.similarity_top_k  # 10
    )
    postprocessor = SimilarityPostprocessor(
        similarity_cutoff=config.retrieval.similarity_cutoff  # 0.2
    )
    return retriever, postprocessor
```

**返回的 retriever 还没有执行检索**——只是配置好了 `top_k` 和分数阈值。

---

### 2.8 `rag/reranker.py` — 重排序器

**函数**：`get_reranker(llm) -> LLMRerank`

```python
def get_reranker(llm):
    return LLMRerank(
        llm=llm,                             # Qwen 实例
        top_n=config.retrieval.reranker_top_n,  # 5
        choice_batch_size=10,
    )
```

**原理**：检索返回 10 个 Node → SimilarityPostprocessor 砍掉分数 < 0.2 的 → LLMRerank 用 Qwen 重新评估相关性 → 最终保留 top_n=5 个。

---

### 2.9 `rag/query_engine.py` — 查询引擎组装

**函数**：`get_query_engine(index, llm) -> RetrieverQueryEngine`

```python
def get_query_engine(index, llm):
    retriever, postprocessor = get_retriever(index)
    reranker = get_reranker(llm)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        node_postprocessors=[postprocessor, reranker],  # 先过滤，再精选
        response_mode="compact",                         # CompactAndRefine 合成策略
    )
```

**完整数据流**：

```
query_engine.query("问题")
  → retriever.retrieve()           → 10 个 Node
  → SimilarityPostprocessor        → 砍掉 < 0.2 的
  → LLMRerank                      → Qwen 精选 5 个
  → ResponseSynthesizer(compact)   → 塞满 context window，Refine 生成答案
  → 返回 Response("答案")
```

---

### 2.10 `tools/rag/llamaindex_tool.py` — 知识库工具

```python
# 模块加载时构建 query_engine（只做一次）
index = get_index(
    data_dir=config.paths.data_dir,
    vector_store=get_vector_store(),
    embed_model=create_embedding(config.embedding),
)
llm = create_llm(config.llm)
query_engine = get_query_engine(index=index, llm=llm)

@tool
def search_knowledge_base(query: str) -> str:
    """搜索本地知识库，包含 agent相关论文，模糊测试相关论文"""
    response = query_engine.query(query)
    return str(response)
```

**关键设计**：
- `query_engine` 在模块加载时创建（全局单例），避免每次调用 `@tool` 都重建
- `@tool` 装饰器把 docstring 转成 JSON Schema，LLM 看到后能决定何时调用

**依赖链**：

```
search_knowledge_base()
  → query_engine (模块级单例)
    → get_query_engine(index, llm)
      → index.as_retriever() → VectorStoreIndex
      → LLMRerank(llm=Qwen)
      → RetrieverQueryEngine
        → Qdrant (向量检索)
        → BGE-M3 (Embedding)
        → Qwen (Rerank + 生成答案)
```

---

### 2.11 `tools/search/web_search.py` — 外部搜索工具

```python
@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息。用于查找最新的LLM论文呢、AGENT论文等。"""
    results = DDGS().text(query, max_results=3)
    return "\n\n".join([r["body"] for r in results])
```

**MVP 阶段用 DuckDuckGo**（免费，无需 API Key）。后续可替换为 Tavily / SerpAPI。

---

### 2.12 `workflows/state.py` — Agent 状态定义

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

- `add_messages` 是 LangGraph 内置的 reducer — 新消息追加到列表，不覆盖旧消息
- MVP 阶段只保留 `messages` 字段，后续可扩展 `task_plan`、`research_results` 等

---

### 2.13 `workflows/graph.py` — Agent 图构建

```python
from langgraph.prebuilt import create_agent

agent = create_agent(
    model=create_llm(config.llm),
    tools=[search_knowledge_base, search_web],
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
)
```

**`create_agent` 内部自动处理**：
- Agent 节点：LLM 推理，判断调工具还是直接回答
- Tools 节点：执行工具并返回结果
- 条件边：有 tool_calls → tools 节点；没有 → 结束
- 循环：tools 执行完 → 回到 agent 继续推理

---

### 2.14 `prompts/supervisor.py` — 系统提示词

```
SUPERVISOR_SYSTEM_PROMPT = """
你是一个专业的研究与软件工程分析助手。你能够：
  - 搜索和检索本地技术文档
  - 查询互联网获取最新信息
  - 分析代码和系统安全问题
  你的回答应该专业、准确、基于事实。

- search_knowledge_base(query): 搜索本地论文知识库。
   包含已分类的论文、技术说明、论文摘要和核心方法等。

- search_web(query): 搜索互联网获取最新信息。
   用于查找最新论文、arXiv 预印本等。

工作流程：
  1. 确定用户想了解的具体领域或论文
  2. 先用 search_knowledge_base 查找本地已有论文和技术说明
  3. 如需补充最新论文，用 search_web 搜索互联网
  4. 对找到的论文进行分类和简要技术说明
  5. 形成结构化的分析报告

输出要求：
  - 使用结构化的格式（标题、分段、编号）
  - 关键信息注明来源
"""
```

---

### 2.15 `main.py` — 入口

```python
from workflows.graph import agent

result = agent.invoke({
    "messages": [("user", "查询关于LLM Agent的论文")]
})

answer = result["messages"][-1].content
print(answer)
```

---

## 三、完整数据流

### 构建阶段（首次运行，有耗时）

```
main.py 导入 agent
  → workflows/graph.py 创建 agent
    → tools/rag/llamaindex_tool.py 模块加载
      → get_index(data_dir, vector_store, embed_model)
        → persist_dir 不存在 → build_index()
          → run_ingestion():
              SimpleDirectoryReader(data_dir) → Documents
              IngestionPipeline(SentenceSplitter, BGE-M3) → Nodes(embedding)
          → VectorStoreIndex(nodes, StorageContext(vector_store=Qdrant))
          → persist() → storage/ 目录
      → get_query_engine(index, llm)
        → get_retriever() → VectorIndexRetriever + SimilarityPostprocessor
        → get_reranker() → LLMRerank
        → RetrieverQueryEngine(retriever, llm, postprocessors, mode="compact")
```

### 查询阶段（每次调用）

```
main.py → agent.invoke({"messages": [("user", "查询关于LLM Agent的论文")]})

Agent Node:
  Qwen 收到：system_prompt + 用户消息 + 工具列表
  Qwen 判断 → 调用 search_knowledge_base("LLM Agent 论文")

Tool Node:
  query_engine.query("LLM Agent 论文")
    → retriever.retrieve()                     → Qdrant 向量检索 (10 个 Node)
    → SimilarityPostprocessor (cutoff=0.2)      → 过滤低分 (剩 N 个)
    → LLMRerank(top_n=5)                        → Qwen 精选 (5 个)
    → ResponseSynthesizer(compact)              → prompt + Qwen → 答案
  → 返回答案文本

Agent Node:
  Qwen 收到工具返回
  Qwen 判断 → 可能再调 search_web 补充最新论文
  或 → 信息够了 → 生成结构化报告

END → 打印最终报告
```

---

## 四、已完成 vs 待实现

| 模块 | 状态 | 文件 |
|------|:--:|------|
| 全局配置 | ✅ | `config.py` |
| LLM 工厂 | ✅ | `models/llm.py` |
| Embedding 工厂 | ✅ | `models/embedding.py` |
| Qdrant 连接 | ✅ | `database/qdrant.py` |
| 数据摄取 | ✅ | `rag/ingestion.py` |
| 索引构建/加载 | ✅ | `rag/index.py` |
| 检索器 | ✅ | `rag/retriever.py` |
| 重排序 | ✅ | `rag/reranker.py` |
| 查询引擎 | ✅ | `rag/query_engine.py` |
| 知识库 Tool | ✅ | `tools/rag/llamaindex_tool.py` |
| 外搜 Tool | ✅ | `tools/search/web_search.py` |
| Agent State | ✅ | `workflows/state.py` |
| Agent 图 | ✅ | `workflows/graph.py` |
| System Prompt | ✅ | `prompts/supervisor.py` |
| 入口 | ✅ | `main.py` |
| 对话记忆 | ⏳ | `memory/` — Step 4/5 实现 |
| Query Rewrite | ⏳ | Step 4 实现 |
| Document Grading | ⏳ | Step 4 实现 |
| Multi-Agent | ⏳ | Step 5 实现 |
| 评估 | ⏳ | `evaluation/` — 第三阶段 |
| API | ⏳ | `api/` — 第三阶段 |
| Docker 部署 | ⏳ | `docker/` — 第三阶段 |

---

## 五、下一步

1. 安装 Qdrant（`docker run -d -p 6333:6333 qdrant/qdrant`）
2. 确认 Ollama `qwen3.5:4b` 可用
3. 在 `data/` 放测试论文文档
4. 安装依赖包
5. 跑 `main.py` 端到端验证
6. 进入 Step 4：Query Rewrite + Document Grading + Reflection + Rerank 增强
