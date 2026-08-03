# Agentic RAG — 多 Agent 论文检索与分析系统

基于 **LangGraph + LlamaIndex + Qdrant + Ollama** 的 Agentic RAG 系统。支持本地论文知识库的语义检索、多 Agent 协作分析、对话记忆和 API 服务。

## 快速开始

### 1. 启动 Ollama 并下载模型

```powershell
# 启动 Ollama（默认 http://localhost:11434）
ollama serve

# 下载对话模型（2.5GB，需 GPU 8GB+）
ollama pull qwen2.5:7b

# 下载 Embedding 模型（274MB）
ollama pull nomic-embed-text
```

### 2. 启动 Qdrant

```powershell
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
```

### 3. 安装依赖

```powershell
cd my_agentic_rag
pip install -r requirements.txt
```

### 4. 放入论文 PDF

将论文 PDF 放入 `data/` 目录：

```
data/
├── AutoGen.pdf
├── ReAct.pdf
├── Reflexion.pdf
├── AMOR.pdf
└── Voyager.pdf
```

### 5. 构建索引（首次运行自动触发）

```powershell
# 首次运行时自动 ingestion + indexing，后续自动加载
# 默认使用 Single Agentic RAG；Multi-Agent Workflow 为实验性分支。
python main.py
```

首次运行会：
1. 读取 `data/` 下的 PDF → 解析为纯文本
2. 用 `nomic-embed-text` 生成 768 维向量
3. 存入 Qdrant 集合 `agentic_rag_knowledge`
4. 持久化索引到 `storage/`

## 使用方式

### 交互式对话

```powershell
python main.py
```

```
你: AutoGen 论文的核心方法是什么？
Agent: [基于知识库原文回答，含页码引用]

你: 和 ReAct 有什么区别？
Agent: [记住上文，给出对比分析]

你: /new          ← 开启新会话
你: /exit         ← 退出
```

### 单次查询

```powershell
python main.py "LLM Agent相关的论文有哪些"
```

### API 服务

```powershell
python -m uvicorn api.server:app --port 8000
```

然后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

**curl 示例**：

```bash
# 单次查询
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "AutoGen的核心方法是什么"}'

# 多轮对话（传入 session_id）
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "和ReAct有什么区别", "session_id": "abc123"}'

# 检索评估
curl http://localhost:8000/eval

# 健康检查
curl http://localhost:8000/health
```

## 运行测试
Document Grader 为可选组件，默认关闭；默认链路使用 Query Rewrite + Similarity Filter + LLM Rerank。

```powershell
# 单元测试（14 秒，不调 LLM）
python -m pytest tests/test_rag_pipeline.py -v

# Smoke test（5 分钟，需 Ollama + Qdrant）
python -m pytest tests/test_smoke.py -v -m smoke
```

## 消融实验

对比不同 RAG 组件的贡献：

```powershell
python evaluation/ablation.py
```

输出示例：(示例输出，非最终实验结果)

```
配置                      Recall      MRR   Hit Rate
-------------------------------------------------------
Baseline (纯检索)          0.750    0.650      0.900
+ Query Rewrite            0.820    0.710      0.950
+ Document Grader          0.780    0.680      0.920
+ LLMRerank                0.810    0.730      0.940
Full Pipeline              0.889    0.792      1.000
```

## 项目结构

```
my_agentic_rag/
├── main.py                       # 入口（交互 / 单次查询）
├── config.py                     # 全局配置（9 个子配置类）
├── requirements.txt              # 锁版本依赖
├── pyproject.toml                # 项目元数据 + 工具配置
├── .env.example                  # 环境变量示例
├── README.md                     # 本文件
│
├── agents/                       # Agent 工厂函数
│   ├── supervisor.py             # Supervisor Agent
│   ├── research_agent.py         # Research Agent
│   ├── writer_agent.py           # Writer Agent
│   └── code_agent.py             # Code Agent（预留）
│
├── models/                       # 模型工厂
│   ├── llm.py                    # qwen3:4b
│   └── embedding.py              # nomic-embed-text
│
├── database/qdrant.py            # Qdrant 连接
│
├── rag/                          # RAG Pipeline
│   ├── ingestion.py              # PDF → Nodes
│   ├── index.py                  # 索引构建/加载
│   ├── retriever.py              # 检索 + 相似度过滤
│   ├── reranker.py               # LLM 重排序
│   ├── query_rewriter.py         # 查询重写
│   ├── document_grader.py        # 批量文档评分
│   └── query_engine.py           # 查询引擎组装
│
├── tools/                        # Agent 工具
│   ├── rag/llamaindex_tool.py    # search_knowledge_base
│   └── search/web_search.py      # search_web
│
├── workflows/                    # LangGraph 图
│   ├── graph.py                  # 单 Agent + Reflection（默认）
│   └── multi_agent.py            # Multi-Agent（备用）
│
├── memory/                       # 对话记忆
│   ├── checkpoint.py             # SQLite 持久化
│   ├── conversation.py           # 对话 CRUD
│   └── session.py                # 多会话管理
│
├── evaluation/                   # 评估
│   ├── rag_eval.py               # Recall@K, MRR
│   ├── agent_eval.py             # 编造检测
│   ├── ablation.py               # 消融实验
│   └── datasets/
│       └── test_queries.json     # 50 条评测集
│
├── observability/tracing.py      # 执行追踪
├── api/server.py                 # FastAPI 服务
├── docker/                       # Docker 部署
├── tests/                        # 测试
│   ├── test_rag_pipeline.py      # 单元测试
│   └── test_smoke.py             # 端到端测试
├── prompts/                      # 提示词（6 个）
└── data/                         # 论文 PDF
```

## 技术栈

| 组件 | 选型 | 规格 |
|------|------|------|
| LLM | qwen2.5:7b (Ollama) | 2.5GB, GPU |
| Embedding | nomic-embed-text (Ollama) | 274MB, 768维 |
| 向量数据库 | Qdrant (Docker) | v1.18 |
| Agent 框架 | LangGraph | v1.1 |
| RAG 框架 | LlamaIndex | v0.14 |
| API | FastAPI | v0.135 |
| PDF 解析 | pypdf | v6.14 |

## License

MIT
