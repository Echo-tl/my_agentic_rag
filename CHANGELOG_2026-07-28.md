# CHANGELOG — 2026-07-28 Step 1-4 实现与 Bug 修复

> 项目：Agentic RAG (my_agentic_rag)
>
> 初始状态：15 个文件已编写，但从未运行验证
>
> 最终状态：MVP Step 1-4 全部跑通

---

## 一、Bug 修复（10 个）

### Bug #1: `sentence_transformers` 不兼容 Python 3.14

- **现象**：`import sentence_transformers` 报 `AttributeError: module 'numpy' has no attribute 'long'`
- **根因**：Python 3.14 移除了 `np.long`/`np.uintc` 等旧属性，旧版 SciPy/scikit-learn 仍引用它们
- **修复**：放弃 HuggingFace BGE-M3，改用 Ollama 内建 embedding 模型 `nomic-embed-text`
- **影响文件**：`config.py`, `models/embedding.py`

### Bug #2: QdrantVectorStore `path` 字段必填

- **现象**：`ValidationError: 1 validation error for QdrantVectorStore — Field required [path]`
- **根因**：pip 安装的 `llama-index-vector-stores-qdrant` v0.1.4 拆分了 `path`/`url` 字段（本地/远程模式），但 `__init__` 的 `super().__init__()` 不转发 `path`，导致 Pydantic 收到缺失字段
- **修复**：卸载 pip 版本，从本地 monorepo 安装 editable 版本（`llama-index-vector-stores-qdrant==0.10.2`）
- **影响文件**：无（环境变更）

### Bug #3: Qdrant 连接 502 Bad Gateway

- **现象**：`qdrant_client.http.exceptions.UnexpectedResponse: 502 (Bad Gateway)`
- **根因**：`httpx` 默认 `trust_env=True`，在 Windows 上读取系统代理设置，将 localhost 请求发送到代理服务器。`curl` 不受影响
- **修复**：在 `database/qdrant.py` 中设置 `os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"`，使 httpx 绕过代理
- **影响文件**：`database/qdrant.py`

### Bug #4: `create_agent` 不存在于新版 langgraph

- **现象**：`ImportError: cannot import name 'create_agent' from 'langgraph.prebuilt'`
- **根因**：langgraph 升级后 `create_agent` 被 `create_react_agent` 替代，且新 API 要求 LangChain `BaseChatModel`（有 `bind_tools`），不接受 LlamaIndex `Ollama`
- **修复**：
  - Agent 层改用 `langchain_ollama.ChatOllama`（支持 `bind_tools`）
  - RAG 层保持 `llama_index.llms.ollama.Ollama`
  - `create_agent(model, system_prompt=...)` → `create_react_agent(model, prompt=...)`
- **影响文件**：`workflows/graph.py`

### Bug #5: httpx ReadTimeout

- **现象**：`httpx.ReadTimeout: timed out — During task with name 'tools'`
- **根因**：qwen3.5:4b 在 CPU 上推理慢，默认 5 秒 httpx 超时不够。LLMRerank + ResponseSynthesizer 需要多次 LLM 调用
- **修复**：
  - `config.llm.request_timeout`: 120 → 300
  - `ChatOllama` 通过 `client_kwargs={"timeout": 300}` 传递超时
- **影响文件**：`config.py`, `workflows/graph.py`

### Bug #6: qwen3.5:4b 太大，CPU 推理慢

- **现象**：模型 13GB，RTX 5060 8GB 只能放 36% 层，64% 跑 CPU，推理极慢
- **修复**：切换到 `qwen3:4b` (2.5GB)，100% 放入 GPU
- **影响文件**：`config.py`

### Bug #7: GPU Out of Memory

- **现象**：`cudaMalloc failed: out of memory — failed to allocate CUDA0 buffer of size 3221225472`
- **根因**：qwen3:4b 默认 context window 很大，KV cache 需要 3GB 显存。加上模型 2.5GB + embedding 0.3GB > 8GB
- **修复**：设置 `context_window=8192`，KV cache 降至 ~1GB
- **影响文件**：`config.py`, `models/llm.py`, `workflows/graph.py`

### Bug #8: PDF 解析失败，Qdrant 存储二进制垃圾

- **现象**：检索返回乱码，`agent` 输出"本地知识库无结果"
- **根因**：未安装任何 PDF 解析库（`pypdf`/`pdfplumber`/`PyPDF2`）和 `llama-index-readers-file`。
  `SimpleDirectoryReader` 回退为读取原始字节流
- **修复**：
  - `pip install pypdf llama-index-readers-file`
  - 删除 Qdrant 旧集合 + 删除 `storage/`，重新 ingestion
- **影响文件**：无（环境变更 + 数据重建）

### Bug #9: LLMRerank 返回 0 nodes（choice_batch_size 过大）

- **现象**：`RetrieverQueryEngine` 返回 `Empty Response`，但 retriever 检索正常（10 nodes）
- **根因**：`choice_batch_size=10` 一次性把所有 10 个 node（~40000 字符）塞进一个 LLM prompt，
  qwen3:4b 无法解析过长的排序请求，返回空
- **修复**：`choice_batch_size: 10 → 5`，分两批各 5 个，每批 ~20000 字符正常处理
- **影响文件**：`rag/reranker.py`

### Bug #10: ResourceWarning 未关闭 socket

- **现象**：`ResourceWarning: unclosed <socket.socket ... raddr=('127.0.0.1', 11434)>`
- **根因**：httpx 连接池未正确关闭，Python 退出时 socket 仍处于 TIME_WAIT 状态
- **状态**：已知问题，不影响功能，后续优化

---

## 二、Step 4 新增功能（3 个模块）

### 功能 #1: Query Rewrite（查询重写）

**文件**：`rag/query_rewriter.py`（新增）

- LLM 将用户查询改写为 3 个不同角度的检索词
- 例如 "LLM Agent" → ["大语言模型智能体架构", "LLM autonomous agents survey", "multi-agent coordination LLM"]
- 失败时回退到原始查询（fail-safe）
- 集成点：`tools/rag/llamaindex_tool.py` 的 `search_knowledge_base` 函数

### 功能 #2: Document Grading（文档评分过滤）

**文件**：`rag/document_grader.py`（新增）

- 继承 `BaseNodePostprocessor`，对每个检索结果调 LLM 判断"相关？YES/NO"
- 只保留判定为 YES 的文档
- LLM 调用失败时保留文档（fail-open），不丢失信息
- 在 postprocessor 链中位于 `SimilarityPostprocessor` 之后、`LLMRerank` 之前

### 功能 #3: Reflection（回答反思）

**文件**：`workflows/graph.py`（重写）

- 从 `create_react_agent` 改造为自定义 `StateGraph`
- 新增 `reflection_node`：Agent 生成回答后，LLM 自检质量
- 评分标准：是否直接回答问题？是否引用具体来源？是否遗漏？
- 不合格则生成反馈 → 附加为 HumanMessage → 路由回 agent_node 重试
- 最多重试 1 次（`max_reflection_retries=1`）

**图结构**：
```
START → agent_node → {has tool_calls? → tools_node → agent_node}
                      {no tool_calls  → reflection_node}
                                        → {PASS → END}
                                        → {FAIL → agent_node (with feedback)}
```

---

## 三、最终文件清单

```
my_agentic_rag/
├── config.py                    # [改] 含 Step 4 配置
├── main.py                      # [改] 适配新 State
├── models/
│   ├── llm.py                   # [改] +context_window 参数
│   └── embedding.py             # [改] HuggingFace → OllamaEmbedding
├── database/
│   └── qdrant.py                # [改] +NO_PROXY 修复
├── rag/
│   ├── ingestion.py             # 未改
│   ├── index.py                 # 未改
│   ├── retriever.py             # 未改
│   ├── reranker.py              # [改] choice_batch_size: 10→5
│   ├── query_engine.py          # [改] 加入 DocumentGrader
│   ├── query_rewriter.py        # [新增] Step 4
│   └── document_grader.py       # [新增] Step 4
├── tools/
│   ├── rag/
│   │   └── llamaindex_tool.py   # [改] 集成 Query Rewrite
│   └── search/
│       └── web_search.py        # 未改
├── workflows/
│   ├── state.py                 # 未改（旧 AgentState 不再使用）
│   └── graph.py                 # [重写] 自定义 StateGraph + Reflection
├── prompts/
│   └── supervisor.py            # 未改
├── storage/                     # 索引持久化（删除后重建）
├── data/                        # 5 篇论文 PDF
├── CHANGELOG_2026-07-28.md      # 本文件
├── DESIGN_DOC.md
└── IMPLEMENTATION_SUMMARY.md
```

---

## 四、当前完整数据流

```
用户问题
  │
  ├─ Agent Node (ChatOllama) → 决定调用工具或直接回答
  │
  ├─ Tool: search_knowledge_base(query)
  │   ├─ Query Rewrite → [q1, q2, q3]           ← Step 4 新增
  │   ├─ 每个查询 → retriever.retrieve() → 10 Nodes
  │   ├─ 按 node_id 合并去重                       ← Step 4 新增
  │   ├─ SimilarityPostprocessor(0.2) → 过滤低分
  │   ├─ DocumentGrader → LLM YES/NO 过滤         ← Step 4 新增
  │   ├─ LLMRerank(top_n=5) → LLM 精选 5 个
  │   └─ ResponseSynthesizer(compact) → 生成答案
  │
  ├─ Tool: search_web(query) → DuckDuckGo 搜索
  │
  ├─ Agent 生成最终回答
  │
  ├─ Reflection Node → 自检回答质量               ← Step 4 新增
  │   ├─ PASS → 最终答案
  │   └─ FAIL → 带反馈回到 Agent Node 重试
  │
  └─ 打印最终回答
```

---

## 五、关键环境依赖

| 组件 | 版本/来源 |
|------|----------|
| Python | 3.14.0 |
| Ollama | localhost:11434 |
| LLM | qwen3:4b (2.5GB, GPU) |
| Embedding | nomic-embed-text (768维) |
| Qdrant | Docker qdrant/qdrant, localhost:6333 |
| llama-index-vector-stores-qdrant | 0.10.2 (本地 editable) |
| httpx | 0.27.2 (降级以兼容) |
| langgraph | latest |
| pypdf | 6.14.2 |
| llama-index-readers-file | 0.6.0 |

---

# CHANGELOG — 2026-07-29 Step 5 + Phase 3 全部完成

> 今日目标：Multi-Agent、Memory、Evaluation、Observability、API、Docker——全部补齐。

---

## 一、Bug 修复（6 个）

### Bug #11: ResponseSynthesizer 编造论文名

- **现象**：Agent 回答中出现 "LLM Agents: A Survey (2024)"、"Multi-Agent Systems for LLMs (2024)" 等不存在于知识库的论文
- **根因**：`search_knowledge_base` 工具内部调用 `ResponseSynthesizer(compact)` 对检索结果做 LLM 合成，qwen3:4b 从训练数据中"补充"了虚假论文名
- **修复**：移除合成步骤，工具直接返回检索到的原始文本块 + 来源文件名 + 相关度分数。答案由 Agent 层基于原文生成而非凭空编造
- **影响文件**：`tools/rag/llamaindex_tool.py`

### Bug #12: Research Agent 不调工具直接编造

- **现象**：Multi-Agent 模式下 Research Agent 收到指令后不调 `search_knowledge_base`，直接用训练数据编造 10+ 篇虚假论文
- **根因**：qwen3:4b 优先选择"直接回答"而非"调工具"，且 `search_web` 返回的 DuckDuckGo 空结果被 LLM 补充为虚假论文
- **修复**：
  - Research Agent 移除 `search_web` 工具，只保留 `search_knowledge_base`
  - 强制要求"收到指令后立即调工具，不要先解释"
  - 工具结果直接转发，不做归纳总结
- **影响文件**：`prompts/researcher.py`, `workflows/multi_agent.py`

### Bug #13: 日期硬编码导致 LLM 时间错乱

- **现象**：Agent 回答中多次出现"2024 年"
- **根因**：supervisor prompt 中硬编码了日期
- **修复**：实现动态日期注入——`_inject_current_date()` 在每次 agent_node 首次调用时，从 `datetime.now()` 取当前日期，以 `<system-reminder>` 标签注入第一条 HumanMessage，后续调用自动跳过
- **影响文件**：`workflows/graph.py`, `prompts/supervisor.py`

### Bug #14: ResourceWarning socket 泄漏

- **现象**：`ResourceWarning: unclosed <socket.socket ... raddr=('127.0.0.1', 11434)>`
- **根因**：QdrantClient、Ollama Client、ChatOllama 的 httpx 内部连接未在进程退出时关闭
- **修复**：`atexit.register()` 在所有模块级 client 创建处注册清理函数，关闭 httpx 内部连接
- **影响文件**：`database/qdrant.py`, `models/llm.py`, `models/embedding.py`, `workflows/graph.py`

### Bug #15: DuckDuckGo 包重命名警告

- **现象**：`RuntimeWarning: This package (duckduckgo_search) has been renamed to ddgs!`
- **修复**：`pip install ddgs`，代码从 `from duckduckgo_search import DDGS` 迁移到 `from ddgs import DDGS`
- **影响文件**：`tools/search/web_search.py`（手动修改）

### Bug #16: `global` 声明位置错误

- **现象**：`SyntaxError: name 'SESSION_ID' is used prior to global declaration`
- **根因**：Python 要求在函数内使用 `global` 前不能引用该变量
- **修复**：`global SESSION_ID` 移至 `interactive()` 函数第一行
- **影响文件**：`main.py`

---

## 二、Step 5: Multi-Agent（新增）

### 文件：`workflows/multi_agent.py`, `prompts/supervisor_agent.py`, `prompts/researcher.py`, `prompts/writer.py`

Supervisor + Research + Writer 三节点协作：

```
START → Supervisor Node (任务规划 + 路由)
  ├─ "ROUTE: research_agent" → Research Agent (search_knowledge_base)
  │      └─ 返回研究成果 → 回到 Supervisor
  ├─ "ROUTE: writer_agent"   → Writer Agent (撰写报告)
  │      └─ 返回报告 → 回到 Supervisor
  └─ "ROUTE: FINISH" → END
```

- Supervisor: 纯调度，无工具。分析用户请求，决定调用 Research 还是 Writer
- Research Agent: 只调用 `search_knowledge_base`，最多 3 轮工具调用，结果原样转发不加工
- Writer Agent: 无工具，纯写作。基于研究材料生成结构化中文报告

> 注：由于 Research Agent 在 qwen3:4b 上的稳定性问题，当前 `main.py` 默认使用单 Agent 模式（`workflows/graph.py`）。Multi-Agent 模式作为高级能力备用。

---

## 三、Agent 模块（新增）

### `agents/` 目录

| 文件 | 功能 |
|------|------|
| `supervisor.py` | `create_supervisor()` — Supervisor Agent 工厂 |
| `research_agent.py` | `create_research_agent()` — Research Agent 工厂（绑定工具） |
| `writer_agent.py` | `create_writer_agent()` — Writer Agent 工厂 |
| `code_agent.py` | `create_code_agent()` — Code Agent 工厂（预留） |

每个 Agent 封装了自己的模型配置、系统提示词和工具列表，可通过工厂函数独立创建和测试。

---

## 四、Memory 模块（新增）

### `memory/` 目录

| 文件 | 功能 |
|------|------|
| `checkpoint.py` | SQLite checkpointer — LangGraph 状态持久化，支持对话恢复 |
| `conversation.py` | `Conversation` 类 — 对话历史 CRUD，JSON 持久化，token 截断 |
| `session.py` | `SessionManager` 类 — 多用户/多会话隔离，过期清理 |

Memory 使 Agent 支持：
- **多轮对话**：同一会话中 Agent 记住之前的问答
- **会话恢复**：进程重启后从 SQLite checkpoint 恢复
- **多会话隔离**：不同用户/会话互不干扰（通过 `thread_id`）

---

## 五、检索优化

### Query Rewrite 提示词改进

- **旧**：生成角度不同的描述性查询（如 "LLM agent implementation papers"）
- **新**：生成关键词密集型查询（如 "multi-agent coordination framework LLM"），更匹配论文标题和摘要中的术语

### 并行检索 + 来源多样性

- `_parallel_retrieve()`: `ThreadPoolExecutor` 并发检索
- 每个来源文件最多保留 5 个 chunks → 防止单一论文占据全部结果
- `top_k`: 5 → 10（去掉合成后检索更快，取更多不会变慢）

### 评估结果

| Metric | top_k=10 |
|--------|----------|
| Recall | 88.9% |
| MRR | 0.792 |
| Hit Rate | 100% |

---

## 六、Evaluation 模块（新增）

### `evaluation/` 目录

| 文件 | 功能 |
|------|------|
| `rag_eval.py` | 检索评估 — Recall@K, MRR, Hit Rate（6 条测试查询） |
| `agent_eval.py` | 生成评估 — 编造检测、论文覆盖度、结构完整度评分 |

运行：`python evaluation/rag_eval.py`

---

## 七、Observability 模块（新增）

### `observability/tracing.py`

- `ExecutionTrace`: 单次查询追踪（耗时、工具调用链、节点流转）
- `trace` 装饰器：自动计时
- `get_recent_traces()`: 获取最近的追踪记录

---

## 八、API 服务（新增）

### `api/server.py`

FastAPI 接口：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/query` | POST | 查询知识库（支持 `session_id` 多轮对话） |
| `/eval` | GET | 运行检索评估 |
| `/health` | GET | 健康检查 |

启动：`python -m uvicorn api.server:app --port 8000`

---

## 九、Docker 部署（新增）

### `docker/` 目录

- `Dockerfile`: Python 3.12 镜像，安装依赖，暴露 8000
- `docker-compose.yml`: app + Qdrant 双容器编排，Ollama 运行在宿主机

---

## 十、交互模式（main.py 改写）

```
python main.py              # 交互式多轮对话
python main.py "问题"        # 单次查询
```

交互命令：`/new` 新会话，`/exit` 退出。

---

## 最终完整文件清单（50+ 文件）

```
my_agentic_rag/
├── main.py                       # 入口（交互 + 单次查询）
├── config.py                     # 全局配置（9 个子配置类）
│
├── agents/                       # Agent 定义 [新增]
│   ├── __init__.py
│   ├── supervisor.py             # Supervisor Agent
│   ├── research_agent.py         # Research Agent
│   ├── writer_agent.py           # Writer Agent
│   └── code_agent.py             # Code Agent（预留）
│
├── models/                       # 模型工厂
│   ├── llm.py                    # Ollama LLM + atexit cleanup
│   └── embedding.py              # OllamaEmbedding + atexit cleanup
│
├── database/                     # 向量数据库
│   └── qdrant.py                 # QdrantClient + NO_PROXY fix
│
├── rag/                          # RAG Pipeline
│   ├── ingestion.py              # PDF → Nodes
│   ├── index.py                  # 索引构建/加载
│   ├── retriever.py              # Retriever + SimilarityPostprocessor
│   ├── reranker.py               # LLMRerank (choice_batch_size=5)
│   ├── query_rewriter.py         # Query Rewrite [新增]
│   ├── document_grader.py        # 批量 Document Grader [新增]
│   └── query_engine.py           # QueryEngine 组装
│
├── tools/                        # Agent 工具
│   ├── rag/
│   │   └── llamaindex_tool.py    # search_knowledge_base（并行检索）
│   └── search/
│       └── web_search.py         # search_web (DuckDuckGo)
│
├── workflows/                    # LangGraph 图
│   ├── state.py                  # AgentState
│   ├── graph.py                  # 单 Agent + Reflection + 动态日期
│   └── multi_agent.py            # Multi-Agent（备用）[新增]
│
├── memory/                       # 记忆管理 [新增]
│   ├── __init__.py
│   ├── checkpoint.py             # SQLite checkpointer
│   ├── conversation.py           # 对话 CRUD
│   └── session.py                # 多会话管理
│
├── evaluation/                   # 评估 [新增]
│   ├── rag_eval.py               # Retrieval: Recall@K, MRR
│   └── agent_eval.py             # Generation: 编造检测
│
├── observability/                # 可观测性 [新增]
│   ├── __init__.py
│   └── tracing.py                # 执行追踪
│
├── api/                          # API 服务 [新增]
│   └── server.py                 # FastAPI: /query, /eval, /health
│
├── docker/                       # 部署 [新增]
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── prompts/                      # 提示词（6 个）
│   ├── supervisor.py             # 单 Agent 系统提示
│   ├── supervisor_agent.py       # Supervisor Agent 提示 [新增]
│   ├── researcher.py             # Research Agent 提示 [新增]
│   └── writer.py                 # Writer Agent 提示 [新增]
│
├── data/                         # 5 篇论文 PDF
├── storage/                      # 索引持久化
│
├── DESIGN_DOC.md                 # 系统设计文档
├── IMPLEMENTATION_SUMMARY.md     # MVP 实现总结
└── CHANGELOG_2026-07-28.md       # 本文件
```

---

# 补充 — `agents/` 与 `workflows/` 对齐重构

## 问题

`agents/` 和 `workflows/multi_agent.py` 各自独立创建 `ChatOllama` 实例，存在代码重复：

- `agents/` 定义了工厂函数但无人调用
- `multi_agent.py` 手动创建 `ChatOllama`，与 `agents/` 脱节

## 修复

`multi_agent.py` 改用 `agents/` 的工厂函数：

```python
# 旧（手动创建，代码重复）
supervisor_model = ChatOllama(model=..., base_url=..., temperature=..., ...)
research_model = ChatOllama(model=..., base_url=..., ...).bind_tools([...])
writer_model = ChatOllama(model=..., base_url=..., ...)

# 新（使用 agents/ 工厂函数）
from agents.supervisor import create_supervisor
from agents.research_agent import create_research_agent
from agents.writer_agent import create_writer_agent

supervisor_model, SUPERVISOR_SYSTEM_PROMPT = create_supervisor()
research_model, RESEARCHER_SYSTEM_PROMPT = create_research_agent()
writer_model, WRITER_SYSTEM_PROMPT = create_writer_agent()
```

同时移除 `agents/research_agent.py` 中的 `search_web` 工具绑定（防止编造）。

## 对齐后的架构

```
agents/（组件库）                  workflows/（编排层）
├── supervisor.py ──工厂函数──→ multi_agent.py
│   └─ create_supervisor()         ├── supervisor_node
│       → (model, prompt)          ├── research_node
├── research_agent.py ─工厂函数─→  └── writer_node
│   └─ create_research_agent()
│       → (model_with_tools, prompt)     graph.py（main.py 使用）
├── writer_agent.py ──工厂函数──→       └── 单 Agent + Reflection
│   └─ create_writer_agent()
│       → (model, prompt)
└── code_agent.py（预留）
```

**`agents/` = 组件库**（模型+提示词+工具的封装），**`workflows/` = 编排层**（用组件搭 StateGraph）。不再有重复的 `ChatOllama` 创建代码。

- **影响文件**：`workflows/multi_agent.py`, `agents/research_agent.py`

---

## 当前模型配置

| 用途 | 模型 | 规格 |
|------|------|------|
| 对话/推理 LLM | `qwen3:4b` | 2.0GB, GPU (RTX 5060 8GB) |
| Embedding | `nomic-embed-text` | 274MB, 768维, GPU |

两个模型都运行在同一个 Ollama 实例：`http://localhost:11434`

---

## 运行方式

```powershell
# 交互式对话（单 Agent，当前默认）
python main.py

# API 服务
python -m uvicorn api.server:app --port 8000

# 检索评估
python evaluation/rag_eval.py
```
