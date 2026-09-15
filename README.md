# Agentic RAG — 多 Agent 论文检索与分析系统

![Agentic RAG 演示](assets/hero.gif)

基于 **LangGraph + LlamaIndex + Qdrant + Ollama** 的 Agentic RAG 系统，面向本地论文知识库，支持语义检索、意图路由、跨文献对比、多 Agent 协作、流式回答、对话记忆与执行追踪。

## 核心特性

- **Agent 编排**：LangGraph 状态机，支持单 Agent（Agent→Tools→Reflection 自反思循环）与多 Agent（Supervisor→Research→Writer）双模式
- **意图识别与工作流路由**：基于 LLM Structured Output 的 IntentRouter，抽取任务类型/论文实体/检索约束，按意图路由到检索、总结、对比、联网搜索、澄清流程；低置信度自动触发澄清
- **流式回答**：SSE 流式输出答案 token，含进度状态提示与反思重答自动重置
- **增量摄取**：基于文件 MD5 哈希，只对新/变更文档重新 embedding，不重复处理全部文档
- **执行追踪**：每次查询记录工具调用链、节点流转与耗时，通过 `/traces` API 暴露，并归档进 MySQL
- **防幻觉**：检索结果直接返回原文分块并附来源/页码引用，不做 LLM 合成
- **存储扩展层（可选）**：Redis 承担 Session / 滚动短期摘要 / 热点问答缓存 / checkpoint，MySQL 持久化用户、会话、问答记录与文档元数据。**默认关闭**，不配置时行为与引入前完全一致

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

### 6.（可选）启用 Redis + MySQL

上面五步跑起来的就是完整系统。想要 Session / 短期摘要 / 热点问答缓存 / 落库归档，见 [存储扩展层](#存储扩展层redis--mysql)——**不配置时行为与不引入这层完全一致**。

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
| `POST /query` | 单次/多轮查询（非流式），响应头带 `X-Cache: HIT\|MISS` |
| `POST /query/stream` | SSE 流式查询（前端使用），`done` 事件带 `cached` 字段 |
| `GET /traces` | 最近执行追踪（工具调用链、节点路径、耗时） |
| `DELETE /traces` | 清空追踪 |
| `GET /cache/stats` | 热点缓存状况：缓存条数、热门问题排行、当前 kb_version |
| `GET /sessions/{sid}/qa` | 某个会话的历史问答（来自 MySQL 归档） |
| `GET /users/{uid}/sessions` | 某个用户的所有会话 |
| `GET /documents` | 文档元数据目录 |
| `GET /eval` | 检索质量评估 |
| `GET /health` | 健康检查（附带 Redis / MySQL / checkpoint 后端状态） |

`POST /query` 与 `POST /query/stream` 都接受可选的 `user_id`（字符串）；不传即游客身份，**前端无需任何改动**。

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

# 带用户身份 + 观察缓存命中
curl -i -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "AutoGen的核心方法是什么", "user_id": "alice"}'
```

## 存储扩展层（Redis + MySQL）

**默认全部关闭。** 不设下面这些环境变量时，系统行为与引入本扩展之前完全一致：SQLite checkpoint、无缓存、无归档。想启用就照 `.env.example` 打开。

```bash
AGENTIC_RAG_REDIS__ENABLED=true
AGENTIC_RAG_REDIS__URL=redis://localhost:6379/0
AGENTIC_RAG_MYSQL__ENABLED=true
AGENTIC_RAG_MYSQL__URL=mysql+pymysql://agentic:agentic@localhost:3306/agentic_rag?charset=utf8mb4

# 一键起 Redis + MySQL + Qdrant
docker compose -f docker/docker-compose.yml up -d
```

### 分工

| 层 | 承担 | Key / 表 |
|---|---|---|
| **Redis** | Session 状态、滚动短期摘要、热点问答缓存、知识库版本号、LangGraph checkpoint | `agr:sess:{sid}`、`agr:sess:{sid}:msgs`、`agr:sess:{sid}:summary`、`agr:qa:v{kbver}:{fp}:{sha1}`、`agr:kbver` |
| **MySQL** | 用户、会话、问答记录、文档元数据、执行追踪 | `users`、`sessions`、`qa_records`、`documents`、`execution_traces` |

Redis 是**加速层不是真源**：多轮对话的真源是 LangGraph checkpointer，Redis 里存的是可重建的窗口与摘要。Redis 全丢不影响对话可用性，只是缓存变冷。摘要在 MySQL `sessions.summary` 留了持久副本，会话隔天再接上不会丢上下文。

表结构由 `persistence/db.py` 在**首次连上 MySQL 时** `create_all()` 自动建立（幂等，不覆盖已有表），无需手动执行任何建表脚本。正式环境要管 schema 变更时，把 `AGENTIC_RAG_MYSQL__AUTO_CREATE` 置 false 并接入 Alembic（尚未随仓库提供）。

### 短期摘要与历史裁剪

`memory_node`（意图识别之后、Agent 之前）在消息数超过 `SUMMARY_TRIGGER` 时，把旧对话压成摘要并写进 Redis，然后用 `RemoveMessage` **真正从状态里删掉**那些消息。

这同时解决两个问题：每轮喂给 LLM 的 token 线性增长，以及 checkpoint 每轮重写整个消息列表导致的**无界膨胀**（迁移前 `checkpoints.db` 会一直涨）。

> ⚠️ 裁剪是**不可逆**的：被裁掉的消息无法回放，只剩摘要里的语义。完整问答归档由 MySQL `qa_records` 承担，但那是「轮次粒度」的记录，不是原始消息流。

### 热点问答缓存

只有**自足**的问题才按问题哈希跨会话共享；指代型追问（「和 ReAct 有什么区别」「那它的缺点呢」）改用**上下文指纹**——把最近 K 轮的用户提问哈希进 key：

| 场景 | 指纹 | 行为 |
|---|---|---|
| 「AutoGen 和 ReAct 有什么区别」（点名两个实体） | `self` | 跨会话共享 |
| 「和 ReAct 有什么区别」（缺对比对象，来自上文） | `ctx:<hash>` | 只在同一上下文内命中 |
| 首轮就问「那它的缺点呢」（无上下文可依据） | `None` | **禁止缓存** |

判定是确定性的（零 LLM 成本），且判不准时一律**不缓存**——宁可少缓存，不可答错。

不缓存的答案还包括：任务类型是澄清或联网搜索、意图置信度 < 0.6、走过反思重答、答案过短或过长、出现「未找到相关文档」兜底话术。

**失效**：`rag/index.py` 增量摄取后自增 `agr:kbver`，版本号内嵌在 key 里，老缓存自然不可达。不做 `SCAN`+`DEL`（阻塞且低效）。

**防击穿**：miss 后 `SET NX PX 30000` 抢 single-flight 锁，持有者负责回填；锁在写入后主动释放（只靠 TTL 会让相邻两次同问都抢不到锁，热点门槛成死结）。

**热点门槛**：`CACHE_MIN_HITS`（默认 2）——同一个问题被观察到 2 次才写缓存，避免一次性长尾问题污染 Redis。设 1 即立刻缓存。

### 降级行为

**主流程永远不会因为存储层失败而失败。** 这是本扩展的硬约束：

| 故障 | 表现 |
|---|---|
| Redis 挂 | 无缓存；窗口读不到 → 非自足问题自动禁止缓存（正确性自动兜底）；checkpoint 自动退回 SQLite |
| MySQL 挂 | 无归档、无历史查询接口；写队列丢弃并计数，绝不阻塞请求 |
| RedisSaver 建不起来（如用了 `redis:7`，缺 RedisJSON/RediSearch） | 自动退回 SqliteSaver |
| 全都挂 | 服务照常回答，只是没有多轮记忆、缓存和归档 |

`GET /health` 会报告当前各层的实际状态。想让它报错而不是降级，设 `AGENTIC_RAG_REDIS__STRICT=true` / `AGENTIC_RAG_MYSQL__STRICT=true`（CI / 生产推荐）。

### 已知限制

- **改 `config.retrieval.*` 后老缓存仍会命中**，直到 TTL 过期。要立即生效就手动 `INCR agr:kbver` 或重启时清一次缓存。
- **裁剪后的对话无法回放**，见上文。
- **缓存只覆盖首轮自足问题 + 同上下文追问**。跨会话的多轮问题因为上下文不同，不会被复用——这是正确性换来的命中率损失。
- **MySQL 写入走后台队列**，进程被 kill -9 时队列里未落库的记录会丢（正常退出有 `atexit` 冲刷）。
- Redis checkpoint 需要 **Redis 8.0+**（内置 RedisJSON + RediSearch），`redis:7` 会在 `setup()` 阶段报 `unknown command 'JSON.SET'`。

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

# 存储层测试 —— 不需要 Redis / MySQL / Qdrant / Ollama
# Redis 走 fakeredis，MySQL 走 SQLite 内存库（模型刻意写成方言无关）
python -m pytest tests/test_qa_cache.py tests/test_session_store.py tests/test_doc_registry.py tests/test_api_wiring.py tests/test_memory_node.py -q
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
├── persistence/                  # 存储扩展层（默认关闭，不可用时全链路降级）
│   ├── redis_client.py           #   Redis 懒加载单例 + 熔断
│   ├── db.py                     #   SQLAlchemy engine / session_scope
│   ├── models.py                 #   ORM：users/sessions/qa_records/documents/execution_traces
│   ├── user_store.py             #   user_id 透传 + 游客兜底
│   ├── session_store.py          #   会话状态 + 消息窗口 + 滚动摘要
│   ├── qa_cache.py               #   热点问答缓存（归一化 / 上下文指纹 / kb 版本失效）
│   ├── doc_registry.py           #   文档元数据目录（取代 indexed_files.json 的查询角色）
│   ├── writer.py                 #   后台写队列（SSE 里绝不阻塞）
│   └── repo.py                   #   落库处理器 + 历史查询
├── memory/
│   ├── checkpoint.py             #   RedisSaver → SqliteSaver → None 降级链
│   └── summarizer.py             #   滚动短期摘要（失败返回空串，调用方据此不裁剪）
├── observability/tracing.py      # 执行追踪（trace_query / record_node / record_tool）
├── evaluation/                   # 评估（rag_eval / ablation / citation_eval / 评测集）
├── prompts/                      # 提示词
├── docker/                       # Docker 部署（qdrant + redis:8 + mysql:8.4 + app）
├── tests/                        # 测试（unit / tracing / intent / incremental / smoke / 存储层）
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
| 记忆 | LangGraph checkpoint（Redis RedisSaver，自动降级 SQLite） |
| 会话 / 缓存 | Redis（Session、滚动摘要、热点问答缓存、kb_version） |
| 持久化 | MySQL + SQLAlchemy 2.0 同步 ORM + PyMySQL |

## License

MIT
