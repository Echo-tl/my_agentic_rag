# Agentic RAG — 多 Agent 论文检索与分析系统

基于 **LangGraph + LlamaIndex + Qdrant + Ollama** 的 Agentic RAG 系统，面向本地论文知识库，支持语义检索、意图路由、跨文献对比、多 Agent 协作、流式回答、对话记忆与执行追踪。

## 核心特性

- **Agent 编排**：LangGraph 状态机，支持单 Agent（Agent→Tools→Reflection 自反思循环）与多 Agent（Supervisor→Research→Writer）双模式
- **意图识别与工作流路由**：基于 LLM Structured Output 的 IntentRouter，抽取任务类型/论文实体/检索约束，按意图路由到检索、总结、对比、联网搜索、澄清流程；低置信度自动触发澄清
- **流式回答**：SSE 流式输出答案 token，含进度状态提示与反思重答自动重置
- **增量摄取**：基于文件 MD5 哈希，只对新/变更文档重新 embedding，不重复处理全部文档
- **执行追踪**：每次查询记录工具调用链、节点流转与耗时，通过 `/traces` API 暴露
- **防幻觉**：检索结果直接返回原文分块并附来源/页码引用，不做 LLM 合成

## 快速开始

### 1. 启动 Ollama 并下载模型

```powershell
ollama serve

# 对话模型（无思考模式，推理快）
ollama pull qwen2.5:7b

# Embedding 模型
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

### 5. 运行

**网页聊天界面（推荐）**：

```powershell
python api/server.py
```

浏览器打开 `http://localhost:8000`。

**命令行交互**：

```powershell
python main.py
```

首次运行会自动读取 `data/` 下的 PDF → 分块 → embedding → 存入 Qdrant（集合 `agentic_rag_knowledge`）→ 持久化索引到 `storage/`。之后新增论文会自动增量摄取，无需重建。

## 网页前端

- `http://localhost:8000` — 聊天界面（浅色主题，流式输出，支持检索/总结/对比/联网）
- `http://localhost:8000/docs` — Swagger API 文档
- `http://localhost:8000/traces` — 查看最近执行追踪

## API

```powershell
python api/server.py        # 或 python -m uvicorn api.server:app --port 8000
```

| 端点 | 说明 |
|------|------|
| `POST /query` | 单次/多轮查询（非流式） |
| `POST /query/stream` | SSE 流式查询（前端使用） |
| `GET /traces` | 最近执行追踪（工具调用链、节点路径、耗时） |
| `DELETE /traces` | 清空追踪 |
| `GET /eval` | 检索质量评估 |
| `GET /health` | 健康检查 |

**curl 示例**：

```bash
# 流式查询（SSE）
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "AutoGen的核心方法是什么"}'

# 多轮对话（传入 session_id）
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "和ReAct有什么区别", "session_id": "abc123"}'
```

## 命令行使用

```powershell
# 交互式多轮对话
python main.py

# 单次查询
python main.py "LLM Agent相关的论文有哪些"
```

交互模式命令：`/new` 新会话、`/exit` 退出。

## 评测结果

基于 `evaluation/datasets/test_queries.json`（55 条自建评测集，覆盖 5 篇论文、中英混合、含对比与长尾问题）：

```powershell
python -m evaluation.rag_eval
```

| 指标 | top-5 | top-10 | top-20 |
|------|-------|--------|--------|
| **论文级命中率** | **94.5%** | **96.4%** | 96.4% |
| 检索命中率 (Hit Rate) | 85.5% | 89.1% | 92.7% |
| Recall | 64.1% | 70.8% | 74.4% |
| MRR | 0.638 | 0.608 | 0.586 |

## 运行测试

```powershell
# 全部测试（约 90s，需 Ollama + Qdrant）
python -m pytest tests/ -q

# 纯单元测试（不调 LLM）
python -m pytest tests/test_rag_pipeline.py tests/test_tracing.py tests/test_intent.py -v
```

## 项目结构

```
my_agentic_rag/
├── main.py                       # CLI 入口（交互 / 单次查询）
├── config.py                     # 全局配置（含路径绝对化、检索开关）
├── api/
│   ├── server.py                 # FastAPI 服务（含 SSE 流式 /query/stream）
│   └── static/index.html         # 网页聊天前端（浅色主题）
├── rag/
│   ├── ingestion.py              # 文档摄取（支持增量指定文件）
│   ├── index.py                  # 索引构建/加载/哈希增量更新
│   ├── intent.py                 # IntentRouter（LLM 结构化意图识别）
│   ├── retriever.py              # 检索 + 相似度过滤
│   ├── reranker.py               # LLM 重排序
│   ├── query_rewriter.py         # 查询重写
│   ├── document_grader.py        # 批量文档评分（默认关闭）
│   └── query_engine.py           # 查询引擎组装
├── workflows/
│   ├── graph.py                  # 单 Agent + 意图路由 + Reflection（默认）
│   └── multi_agent.py            # Multi-Agent（备用）
├── tools/
│   ├── rag/llamaindex_tool.py    # search_knowledge_base
│   └── search/web_search.py      # search_web
├── models/
│   ├── llm.py                    # qwen2.5:7b
│   └── embedding.py              # nomic-embed-text
├── database/qdrant.py            # Qdrant 懒加载客户端
├── memory/                       # SQLite 对话记忆
├── observability/tracing.py      # 执行追踪（trace_query / record_node / record_tool）
├── evaluation/                   # 评估（rag_eval / ablation / 评测集）
├── prompts/                      # 提示词
├── docker/                       # Docker 部署
├── tests/                        # 测试（unit / tracing / intent / incremental / smoke）
└── data/                         # 论文 PDF（gitignored）
```

## 技术栈

| 组件 | 选型 |
|------|------|
| LLM | qwen2.5:7b (Ollama，无思考模式) |
| Embedding | nomic-embed-text (Ollama, 768 维) |
| 向量数据库 | Qdrant |
| Agent 框架 | LangGraph |
| RAG 框架 | LlamaIndex |
| API | FastAPI + SSE |
| 记忆 | SQLite (LangGraph checkpoint) |

## License

MIT
