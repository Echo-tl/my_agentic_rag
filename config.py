"""
全局配置 —— 项目所有模块的单一真相源。

使用方式：
    from config import config
    llm = create_llm(config.llm)

环境变量可以覆盖任何默认值：
    export AGENTIC_RAG_LLM__MODEL=qwen2.5:14b
    export AGENTIC_RAG_QDRANT__URL=http://localhost:6333
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 = config.py 所在目录。所有相对路径据此解析，避免从其他目录启动时找不到知识库。
_PROJECT_ROOT = Path(__file__).parent


# ============================================================
# 子配置类
# ============================================================

class LLMConfig(BaseSettings):
    """LLM 配置。支持 Ollama 本地模型和 DashScope 云端模型。"""

    provider: Literal["ollama", "dashscope"] = "ollama"
    model: str = "qwen2.5:7b"  # qwen2.5 无思考模式，qwen3 每次调用烧 2000+ 隐藏思考 token 导致极慢
    base_url: str = "http://localhost:11434"  # Ollama 默认地址
    temperature: float = 0.0
    max_tokens: int = 4096
    context_window: int = 8192  # 限制上下文窗口，减少 GPU KV cache 显存占用
    request_timeout: float = 300.0  # CPU 推理较慢，需要更长超时

    # DashScope 专用
    api_key: Optional[str] = None


class EmbeddingConfig(BaseSettings):
    """Embedding 模型配置。"""

    provider: Literal["huggingface", "ollama"] = "ollama"
    model_name: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    batch_size: int = 32


class QdrantConfig(BaseSettings):
    """Qdrant 向量数据库配置。"""

    url: str = "http://localhost:6333"
    collection_name: str = "agentic_rag_knowledge"
    vector_size: int = 768  # nomic-embed-text 的维度
    distance: str = "Cosine"


class ChunkConfig(BaseSettings):
    """文档切分配置。"""

    chunk_size: int = 1024
    chunk_overlap: int = 128
    separators: list[str] = ["\n\n", "\n", "。", ".", " ", ""]


class RetrievalConfig(BaseSettings):
    """检索配置。"""

    similarity_top_k: int = 10         # 初始检索数量（去合成后取更多不会变慢）
    reranker_top_n: int = 5            # Rerank 后保留数量（多留一些防止结果单一）
    similarity_cutoff: float = 0.2     # 相似度阈值（砍掉低于此值的）

    # Step 4: Agentic RAG 增强
    enable_query_rewrite: bool = True  # 是否启用查询重写
    query_rewrite_count: int = 2       # 重写查询数量（含原始=3个）
    enable_document_grading: bool = False  # 文档评分与 reranker 功能重叠，且 qwen2.5 下评分过严会清空结果，故默认关闭
    enable_reflection: bool = True     # 是否启用回答反思
    max_reflection_retries: int = 1    # 反思最大重试次数
    enable_intent_routing: bool = True  # 是否启用意图识别与工作流路由（IntentRouter）


class MemoryConfig(BaseSettings):
    """记忆配置。"""

    max_tokens: int = 8000           # 对话历史最大 token 数
    checkpoint_db_path: str = "./checkpoints.db"

    @model_validator(mode="after")
    def _resolve_checkpoint_db(self):
        # 相对路径基于项目根目录解析，避免从其他目录启动时在错误位置建库
        p = Path(self.checkpoint_db_path)
        if not p.is_absolute():
            self.checkpoint_db_path = str(_PROJECT_ROOT / p)
        return self


class RedisConfig(BaseSettings):
    """Redis 配置：Session、短期摘要、热点问答缓存、知识库版本号。

    默认关闭。未启用或连不上时全链路静默降级为 no-op，主流程不受影响。
    """

    enabled: bool = False                # 默认关：不配环境变量时行为与引入前完全一致
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "agr"              # 所有 key 的统一前缀，便于多环境隔离与批量清理
    max_connections: int = 20
    socket_timeout: float = 0.5          # 单命令超时，必须短——它决定降级路径的最坏耗时
    socket_connect_timeout: float = 0.5
    health_check_interval: int = 30
    strict: bool = False                 # True=连不上直接抛错（CI / 就绪探针用）

    # ── Session ──
    session_ttl: int = 1800              # 滑动过期（每次访问刷新）
    window_size: int = 10                # 保留最近 N 条消息

    # ── 短期摘要 ──
    summary_enabled: bool = True
    summary_trigger: int = 12            # 消息数达到此值触发滚动摘要
    summary_keep_recent: int = 6         # 摘要后保留最近 N 条不压缩
    summary_max_chars: int = 2000        # 摘要长度上限
    summary_ttl: int = 86400             # 比 session 长，防会话中断后重连丢摘要

    # ── 热点问答缓存 ──
    cache_enabled: bool = True
    cache_ttl: int = 3600                # 固定 TTL，不滑动——防长尾脏数据
    cache_min_hits: int = 2              # 观察到 >= N 次才写缓存（热点门槛）；设 1 = 立即缓存
    cache_max_question_chars: int = 200  # 超长问题不缓存（复用率低）
    cache_min_answer_chars: int = 20     # 过短答案不缓存
    cache_fp_ctx_turns: int = 2          # 上下文指纹取最近几轮用户问题
    cache_lock_ttl_ms: int = 30000       # single-flight 锁

    # ── Checkpoint ──
    # redis=官方 RedisSaver（需 RedisJSON+RediSearch，即 redis:8 或 redis-stack）；
    # sqlite=沿用现有 SqliteSaver；none=不持久化
    checkpoint_backend: Literal["redis", "sqlite", "none"] = "redis"
    checkpoint_ttl: int = 1440           # 分钟


class MySQLConfig(BaseSettings):
    """MySQL 配置：用户 / 会话 / 问答记录 / 文档元数据 / 执行追踪的持久化。

    默认关闭。连不上时全部写入静默丢弃并计数告警，主流程不受影响。
    """

    enabled: bool = False
    # charset=utf8mb4 必须显式写，否则中文入库变乱码
    url: str = "mysql+pymysql://agentic:agentic@localhost:3306/agentic_rag?charset=utf8mb4"
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle: int = 1800             # 防 MySQL 默认 8h 空闲断连
    pool_pre_ping: bool = True
    connect_timeout: float = 1.0
    echo: bool = False
    auto_create: bool = True             # 启动时 create_all；接 Alembic 后置 false
    strict: bool = False

    # ── 后台写入队列（不阻塞请求/SSE）──
    write_queue_size: int = 1000
    write_batch_size: int = 20
    drop_on_overflow: bool = True        # 队列满时丢弃，绝不阻塞请求


class PathConfig(BaseSettings):
    """路径配置。"""

    project_root: Path = _PROJECT_ROOT
    data_dir: Path = Path("data")            # 原始文档
    persist_dir: Path = Path("storage")      # 索引持久化目录
    checkpoint_dir: Path = Path("checkpoints")

    @model_validator(mode="after")
    def _resolve_relative_paths(self):
        # 所有子路径基于 project_root 解析成绝对路径，避免从任意目录启动时找不到知识库
        for field in ("data_dir", "persist_dir", "checkpoint_dir"):
            p = getattr(self, field)
            if not p.is_absolute():
                setattr(self, field, self.project_root / p)
        return self


# ============================================================
# 顶层配置，聚合所有子配置
# ============================================================

class Settings(BaseSettings):
    """顶层配置。所有子配置在此聚合。"""

    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_RAG_",
        env_nested_delimiter="__",  # 环境变量：AGENTIC_RAG_LLM__MODEL=qwen2.5:14b
        case_sensitive=False,
        env_file=".env",            # 让 .env.example 承诺的用法真正生效
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    # 存储扩展层（默认关闭，见各类的 enabled 字段）
    redis: RedisConfig = Field(default_factory=RedisConfig)
    mysql: MySQLConfig = Field(default_factory=MySQLConfig)


# ============================================================
# 全局单例 —— 整个项目只导入这一个 config 对象
# ============================================================

config = Settings()
