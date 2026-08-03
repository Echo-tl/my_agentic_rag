# Agentic RAG 项目设计文档

> 项目：Research / Software Engineering Agent
>
> 目标：用户输入复杂问题，Agent 自动规划、检索、分析、生成报告

---

## 一、系统架构

```
                              ┌─────────────────┐
                              │      User        │
                              └────────┬────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │   Supervisor Agent       │
                         │   (LangGraph StateGraph) │
                         │                          │
                         │  State:                  │
                         │    - messages            │
                         │    - task_plan           │
                         │    - research_results    │
                         │    - final_report        │
                         └──────────┬──────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  Planner Node   │  │  Worker Agents   │  │  Writer Node    │
    │                  │  │                  │  │                  │
    │  分析任务         │  │  执行具体工作      │  │  合成最终报告     │
    │  拆解步骤         │  │                  │  │                  │
    └────────┬─────────┘  └────────┬────────┘  └────────┬─────────┘
             │                    │                     │
             │           ┌────────┴────────┐            │
             │           │                 │            │
             │           ▼                 ▼            │
             │  ┌─────────────────┐ ┌─────────────────┐ │
             │  │  Knowledge Tool │ │   Search Tool   │ │
             │  │                  │ │                  │ │
             │  │ LlamaIndex RAG   │ │  Web Search API  │ │
             │  │  ┌───────────┐  │ │                  │ │
             │  │  │QueryEngine│  │ │                  │ │
             │  │  │  ┌──────┐ │  │ │                  │ │
             │  │  │  │Retri─│ │  │ │                  │ │
             │  │  │  │ever  │ │  │ │                  │ │
             │  │  │  └──┬───┘ │  │ │                  │ │
             │  │  └──────┼────┘  │ │                  │ │
             │  │         ▼       │ │                  │ │
             │  │  ┌───────────┐  │ │                  │ │
             │  │  │  Qdrant   │  │ │                  │ │
             │  │  │ Vector DB │  │ │                  │ │
             │  │  └───────────┘  │ │                  │ │
             │  └─────────────────┘ └─────────────────┘ │
             │                                          │
             └─────────────────────┬────────────────────┘
                                   │
                                   ▼
                         ┌─────────────────┐
                         │   Qwen (LLM)    │
                         │   推理/生成       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Final Report   │
                         └─────────────────┘
```

### 分层图设计

```
multi_agent_graph.py (顶层 Supervisor 图)
  ┌─────────────────────────────────────────────┐
  │              Supervisor Node                 │
  │  "分析任务 → 规划步骤 → 分发给子Agent"          │
  └────┬──────────┬──────────┬──────────────────┘
       │          │          │
       ▼          ▼          ▼
  ┌────────┐ ┌────────┐ ┌────────┐
  │Research│ │ Code   │ │ Writer │
  │ Agent  │ │ Agent  │ │ Agent  │
  └───┬────┘ └────────┘ └────────┘
      │
      │ research_graph.py (Research 子图)
      │  ┌──────────────────────────────────────┐
      │  │  retrieve → grade → rewrite → retrieve│
      │  │     ↑________________________↓       │
      │  └──────────────────────────────────────┘
```

---

## 二、LangGraph 和 LlamaIndex 的职责边界

| 维度 | LangGraph 负责 | LlamaIndex 负责 |
|------|:-------------:|:-------------:|
| 任务规划 | ✅ Supervisor 拆解任务、决定步骤顺序 | — |
| 状态管理 | ✅ State 定义、节点间传递 | — |
| 路由决策 | ✅ Conditional Edge | — |
| 文档加载 | — | ✅ SimpleDirectoryReader |
| 文档切分 | — | ✅ SentenceSplitter → Node |
| 向量化 | — | ✅ BGE-M3 Embedding |
| 索引与检索 | — | ✅ VectorStoreIndex → Retriever |
| 检索后处理 | — | ✅ LLMRerank |
| 答案合成 | — | ✅ ResponseSynthesizer |
| 工具定义 | ✅ `@tool` 装饰器 | ✅ 提供 QueryEngine/Reranker 给 Tool 调用 |
| Agent 循环 | ✅ StateGraph Node→Edge→Condition | — |
| LLM 调用 | ✅ `llm.bind_tools()` | ✅ QueryEngine 内部调用 |

**一句话**：LangGraph 是"指挥官"——管流程、管状态、管决策。LlamaIndex 是"图书馆+研究员"——管数据、管检索、管答案合成。两者通过 **Tool** 连接。

**为什么不用 LlamaIndex AgentWorkflow？** `AgentWorkflow` 是事件驱动的、handoff 机制是黑盒的。LangGraph 的 StateGraph 给你显式的图控制——你能看到每个 Node、每条 Edge、每个条件分支。企业项目需要可控性和可调试性。

---

## 三、项目目录结构

```
agentic_rag/
├── agents/                        # Agent 定义
│   ├── __init__.py
│   ├── supervisor.py              # Supervisor Agent（任务规划+分发）
│   ├── research_agent.py          # Research Agent（知识检索+外搜）
│   ├── code_agent.py              # Code Analysis Agent（代码分析）
│   └── writer_agent.py            # Writer Agent（报告生成）
│
├── workflows/                     # LangGraph 图定义
│   ├── __init__.py
│   ├── state.py                   # 全局 State 定义
│   ├── research_graph.py          # Research Agent 内部子图
│   ├── multi_agent_graph.py       # 顶层 Supervisor 多 Agent 图
│   └── routing.py                 # 条件路由逻辑
│
├── tools/                         # LangGraph Tool 定义
│   ├── __init__.py
│   ├── rag/
│   │   ├── __init__.py
│   │   └── llamaindex_tool.py     # LlamaIndex RAG Tool
│   ├── search/
│   │   ├── __init__.py
│   │   └── web_search.py          # Web Search Tool
│   └── code/
│       ├── __init__.py
│       └── code_analysis.py       # Code Analysis Tool
│
├── rag/                           # LlamaIndex RAG Pipeline
│   ├── __init__.py
│   ├── ingestion.py               # IngestionPipeline 封装
│   ├── index.py                   # VectorStoreIndex 构建/加载
│   ├── retriever.py               # Retriever + Postprocessor
│   ├── reranker.py                # Reranker（LLMRerank 等）
│   └── query_engine.py            # QueryEngine 封装
│
├── memory/                        # 对话记忆与上下文管理
│   ├── __init__.py
│   ├── conversation.py            # 对话历史 CRUD
│   ├── checkpoint.py              # LangGraph checkpoint 持久化
│   └── session.py                 # 会话管理（多用户）
│
├── prompts/                       # Prompt 模板集中管理
│   ├── __init__.py
│   ├── supervisor.py              # Supervisor 系统提示词
│   ├── researcher.py              # Research Agent 检索提示词
│   ├── writer.py                  # Writer Agent 报告生成提示词
│   ├── code_agent.py              # Code Agent 分析提示词
│   └── grading.py                 # 文档相关性评分提示词
│
├── models/                        # 模型配置
│   ├── __init__.py
│   ├── llm.py                     # Qwen LLM 配置
│   └── embedding.py               # BGE-M3 Embedding 配置
│
├── database/                      # 向量数据库
│   ├── __init__.py
│   └── qdrant.py                  # Qdrant 连接配置
│
├── evaluation/                    # 评估模块
│   ├── __init__.py
│   ├── rag_eval.py                # RAG 检索评估（Recall@K, MRR, Hit Rate）
│   ├── agent_eval.py              # Agent 规划/生成评估
│   └── datasets/                  # 测试数据
│
├── observability/                 # 可观测性
│   ├── __init__.py
│   └── tracing.py                 # Agent 执行追踪
│
├── api/                           # API 服务
│   ├── __init__.py
│   └── server.py                  # FastAPI 服务
│
├── data/                          # 知识库文档
│   └── .gitkeep
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── config.py                      # 全局配置
├── main.py                        # 入口
└── requirements.txt               # 依赖
```

### 各目录职责

| 目录 | 职责 | 对应学过的源码概念 |
|------|------|:--:|
| `agents/` | 每个 Agent 的 prompt、工具列表、系统定义 | AgentWorkflow 中的 BaseWorkflowAgent |
| `workflows/` | StateGraph 构建、State 定义、条件路由、子图 | LangGraph 概念 |
| `tools/` | 把 LlamaIndex QueryEngine 包装成 LangGraph tool | FunctionTool、QueryEngineTool |
| `rag/` | Document→Node→Index→Retriever→QueryEngine 管道 | IngestionPipeline、VectorStoreIndex |
| `memory/` | 对话历史、checkpoint、session 管理 | ChatMemoryBuffer |
| `prompts/` | 所有 Agent 提示词集中管理 | ReActAgent 的 DEFAULT_PROMPT 模式 |
| `models/` | Qwen + BGE-M3 初始化 | Settings.llm、Settings.embed_model |
| `database/` | Qdrant 连接 | VectorStoreIndex 中的 vector_store |
| `evaluation/` | Recall@K、MRR、Hit Rate | — |
| `observability/` | 运行时追踪和日志 | LlamaIndex 的 CallbackManager |
| `api/` | FastAPI 对外服务 | — |

---

## 四、技术栈

| 组件 | 选型 | 原因 |
|------|------|------|
| Agent 框架 | LangGraph | 显式图控制、可调试、状态持久化 |
| RAG 框架 | LlamaIndex | 全链路数据管道、丰富检索策略 |
| LLM | Qwen（本地） | 隐私、成本、你对它熟悉 |
| Embedding | BGE-M3 | 1024 维、中英文、dense+sparse |
| 向量数据库 | Qdrant | 高性能、支持 filter、Docker 部署 |
| API | FastAPI | 异步、自动文档、企业标准 |
| 部署 | Docker Compose | Qdrant + App 一体化 |

---

## 五、开发阶段

### 第二阶段：MVP（最小可运行版本）

| Step | 内容 | 产出 |
|------|------|------|
| Step 1 | LlamaIndex RAG Pipeline | ingestion → index → retriever → query_engine |
| Step 2 | QueryEngine → LangGraph Tool | `tools/rag/llamaindex_tool.py` |
| Step 3 | LangGraph Agent Workflow | State → Planner → Retriever → Generator |
| Step 4 | Agentic RAG 增强 | Query Rewrite、Document Grading、Reflection、Rerank |
| Step 5 | Multi-Agent | Supervisor + Research + Code + Writer |

### 第三阶段：生产增强

- Memory（对话历史 + checkpoint）
- Observability（tracing）
- Evaluation（RAG + Agent 评估）
- API（FastAPI 服务）
- Docker 部署

---

## 六、核心设计原则

1. **关注点分离**：Agent 逻辑（agents/）和 RAG 逻辑（rag/）不耦合。改检索策略不动 Agent 代码
2. **显式优于隐式**：LangGraph 的图结构可以可视化、打断点。企业项目需要可控性
3. **渐进式开发**：先跑通 MVP（1 个 Agent + 1 个 QueryEngine），再加能力
4. **Prompt 集中管理**：改 prompt 不动代码，非技术人员也能调优
5. **可观测性内建**：从 Day 1 就有 tracing，不是出问题了才加
