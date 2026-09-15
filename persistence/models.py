"""
MySQL ORM 模型：用户 / 会话 / 问答记录 / 文档元数据 / 执行追踪。

刻意写成**方言无关**，同一个 schema 能在 SQLite 内存库上跑起来（单测不需要真 MySQL）：
- 枚举用 sa.Enum(native_enum=False) → 各方言统一落 VARCHAR + CHECK，不用 MySQL 原生 ENUM
- JSON 用 SQLAlchemy 通用 sa.JSON → SQLite 落 TEXT
- 大文本用 with_variant(mysql.MEDIUMTEXT) → 其它方言退回 TEXT
- 自增主键用 BigInteger().with_variant(Integer, "sqlite") → SQLite 要求恰好 INTEGER PRIMARY KEY
- 引擎/字符集写在 __table_args__ 的 mysql_* 键里，SQLite 会忽略

表边界：不建 messages 表。一轮问答的 question/answer 归 qa_records；Agent 内部消息
（工具调用、reflection 反馈）单轮可达 10–30KB 且无长期价值，其元数据并入
qa_records.tool_calls，完整流水由带 TTL 的 LangGraph checkpoint 承担。
避免同一份问答记录在两张大表里各存一遍。
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeEngine

try:  # MySQL 专有类型仅用于 with_variant，缺失时不影响其它方言
    from sqlalchemy.dialects import mysql
except ImportError:  # pragma: no cover
    mysql = None


class Base(DeclarativeBase):
    pass


# ── 方言无关的类型别名 ────────────────────────────────────────
# SQLite 只把恰好是 "INTEGER PRIMARY KEY" 的列当作 rowid 别名来自增，
# 裸 BIGINT 主键在 SQLite 上不会自增 → 必须 with_variant 降级。
BigInt: TypeEngine = BigInteger().with_variant(Integer, "sqlite")

# 毫秒精度：MySQL 用 DATETIME(3)，其它方言退回普通 DATETIME
_DateTimeMS: TypeEngine = (
    DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql") if mysql else DateTime()
)
_MediumText: TypeEngine = (
    Text().with_variant(mysql.MEDIUMTEXT(), "mysql") if mysql else Text()
)

_MYSQL_TABLE_ARGS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_unicode_ci",
}


def _enum(*values: str, length: int = 16) -> TypeEngine:
    """方言无关的枚举：MySQL 落原生 ENUM，其它方言落 VARCHAR + CHECK。"""
    from sqlalchemy import Enum as SAEnum

    return SAEnum(*values, native_enum=False, length=length, validate_strings=False)


# ============================================================
# 1. 用户
# ============================================================
class User(Base):
    """用户。本期只做 user_id 透传，未传则落到 anonymous 游客用户。

    预留：password_hash 字段留空，后续加鉴权时无需改表结构。
    """

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("username", name="uk_users_username"),
        _MYSQL_TABLE_ARGS,
    )

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_guest: Mapped[bool] = mapped_column(default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(_DateTimeMS, default=datetime.now, nullable=False)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(_DateTimeMS, nullable=True)


# ============================================================
# 2. 会话
# ============================================================
class Session(Base):
    """会话。session_id 与 LangGraph 的 thread_id 完全一致。

    summary 是 Redis 摘要的持久副本——Redis 过期后从这里兜底重灌。
    """

    __tablename__ = "sessions"
    __table_args__ = (
        Index("idx_sessions_user_active", "user_id", "last_active_at"),
        Index("idx_sessions_status_active", "status", "last_active_at"),
        _MYSQL_TABLE_ARGS,
    )

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        BigInt, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(_MediumText, nullable=True)
    summary_upto: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 已摘要到第几轮
    turn_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    kb_version: Mapped[int] = mapped_column(BigInt, default=0, nullable=False)  # 审计用
    client_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)  # IPv6 需 45 字符
    status: Mapped[str] = mapped_column(_enum("active", "archived"), default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(_DateTimeMS, default=datetime.now, nullable=False)
    last_active_at: Mapped[datetime] = mapped_column(
        _DateTimeMS, default=datetime.now, onupdate=datetime.now, nullable=False
    )


# ============================================================
# 3. 问答记录（一轮 = 一行）
# ============================================================
class QARecord(Base):
    """一轮问答的完整业务记录：问题、答案、意图、引用、耗时、是否命中缓存。"""

    __tablename__ = "qa_records"
    __table_args__ = (
        Index("idx_qa_session", "session_id", "id"),
        Index("idx_qa_user_time", "user_id", "created_at"),
        Index("idx_qa_qhash", "question_hash", "created_at"),
        Index("idx_qa_task_time", "task_type", "created_at"),
        Index("idx_qa_cache_hit", "cached", "created_at"),
        _MYSQL_TABLE_ARGS,
    )

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[Optional[int]] = mapped_column(BigInt, nullable=True)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)

    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_hash: Mapped[str] = mapped_column(String(40), nullable=False)  # sha1(归一化问题)
    context_fp: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)  # self / ctx:<hex>
    answer: Mapped[Optional[str]] = mapped_column(_MediumText, nullable=True)

    task_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    intent_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    citations: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [{file_name,page,score}]
    tool_calls: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [{tool,args,duration_ms}]

    cache_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    cached: Mapped[bool] = mapped_column(default=False, nullable=False)
    kb_version: Mapped[int] = mapped_column(BigInt, default=0, nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reflection_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    trace_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    feedback: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 预留：赞/踩
    created_at: Mapped[datetime] = mapped_column(_DateTimeMS, default=datetime.now, nullable=False)


# ============================================================
# 4. 文档元数据
# ============================================================
class Document(Base):
    """文档元数据目录。

    真源关系：data/ 下的文件内容是唯一真源；Qdrant 是检索真源；
    documents 表是人类可读的元数据投影（非真源）；
    storage/indexed_files.json 降级为 MySQL 不可用时的本地兜底。
    三者用 file_name 作为对齐键（Qdrant payload 里已有该字段）。
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("file_name", name="uk_documents_file_name"),
        Index("idx_documents_status", "status", "updated_at"),
        Index("idx_documents_hash", "file_hash"),
        _MYSQL_TABLE_ARGS,
    )

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    file_hash: Mapped[str] = mapped_column(String(32), nullable=False)  # MD5，与 indexed_files.json 对齐
    file_size: Mapped[int] = mapped_column(BigInt, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        _enum("pending", "indexed", "changed", "removed", "failed", "inconsistent"),
        default="pending",
        nullable=False,
    )
    kb_version: Mapped[int] = mapped_column(BigInt, default=0, nullable=False)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(_DateTimeMS, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(_DateTimeMS, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        _DateTimeMS, default=datetime.now, onupdate=datetime.now, nullable=False
    )


# ============================================================
# 5. 执行追踪（内存 _traces 的持久化归档）
# ============================================================
class ExecutionTrace(Base):
    """observability/tracing.py 内存 trace 的落库归档。"""

    __tablename__ = "execution_traces"
    __table_args__ = (
        Index("idx_traces_created", "created_at"),
        Index("idx_traces_session", "session_id", "created_at"),
        _MYSQL_TABLE_ARGS,
    )

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    qa_id: Mapped[Optional[int]] = mapped_column(BigInt, nullable=True)
    query: Mapped[str] = mapped_column(String(1024), nullable=False)
    elapsed_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tool_calls: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    node_path: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    tool_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(_DateTimeMS, default=datetime.now, nullable=False)
