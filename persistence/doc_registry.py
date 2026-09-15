"""
文档元数据目录：`documents` 表 + `storage/indexed_files.json` 的分工。

真源关系（重要，别搞反）：
- `data/` 下的文件内容     → 唯一真源
- Qdrant 分块              → 检索真源
- **`indexed_files.json`   → 增量摄取的判定真源**（同步写，原子替换，决定谁要重新 embedding）
- `documents` 表           → 人类可查的元数据投影 + 缓存失效的触发点

为什么增量判定仍读本地 JSON 而不读 MySQL：摄取决策必须是**同步且确定**的，
而 MySQL 写入走的是「可能丢弃」的后台队列（见 writer.py）。用一份可能丢数据的
副本决定「这个文件要不要重新 embedding」是危险的——最坏情况是文件变了却被判为
未变，检索结果静默过期。所以本地 JSON 是判定真源，MySQL 是可查询的副本。

反过来，本地 JSON 为空而 MySQL 有记录时（换机器、清了 storage 但 MySQL 还在）
从 MySQL 灌回来，避免全量重 embedding。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from config import config

logger = logging.getLogger("agentic_rag")

_HASH_FILE = "indexed_files.json"


# ── 本地哈希记录（判定真源）──────────────────────────────────
def _local_path() -> Path:
    return config.paths.persist_dir / _HASH_FILE


def _load_local() -> Dict[str, str]:
    p = _local_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_local(record: Dict[str, str]):
    """先写临时文件再原子替换——写到一半崩了不能留下半截 JSON。"""
    p = _local_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


# ── 对外接口 ─────────────────────────────────────────────────
def load_hash_record() -> Dict[str, str]:
    """{file_name: md5}。本地优先；本地为空时从 MySQL 兜底灌回。"""
    local = _load_local()
    if local:
        return local

    remote = _load_from_mysql()
    if remote:
        logger.info(f"[docs] 本地哈希记录为空，从 MySQL 灌回 {len(remote)} 条")
        _save_local(remote)
    return remote


def save_hash_record(record: Dict[str, str]):
    """同步写本地；异步镜像到 MySQL。"""
    _save_local(record)
    mirror_hash_record(record)


def mirror_hash_record(record: Dict[str, str]):
    """只把本地记录镜像进 MySQL，不重写本地文件。

    用在「本地已有记录、但 MySQL 可能还是空的」场景：比如老部署首次打开
    `AGENTIC_RAG_MYSQL__ENABLED`，本地 JSON 有 5 个文件、`documents` 表却是空的，
    而增量摄取发现无变更就直接 return 了——不补这一脚，`/documents` 会一直空着
    直到某个文件真的变了。apply_snapshot 是幂等的，重复投递无副作用。
    """
    try:
        from persistence.writer import enqueue

        enqueue("doc_snapshot", {"record": dict(record)})
    except Exception:
        pass


def upsert(file_name: str, file_hash: str, file_path: str = None,
           file_size: int = None, chunk_count: int = None,
           page_count: int = None, status: str = "indexed",
           kb_version: int = None, error: str = None):
    """登记/更新一份文档的元数据（后台写，不阻塞摄取流程）。"""
    from persistence import repo

    repo.submit_document(
        file_name=file_name,
        file_hash=file_hash,
        file_path=file_path,
        file_size=file_size,
        chunk_count=chunk_count,
        page_count=page_count,
        status=status,
        kb_version=kb_version,
        error=error,
    )


def mark_removed(file_name: str, kb_version: int = None):
    """标记文档已从 data/ 移除。不物理删除，保留审计痕迹。"""
    from persistence.writer import enqueue

    enqueue("doc_removed", {"file_name": file_name, "kb_version": kb_version})


def mark_failed(file_name: str, file_hash: str, error: str):
    upsert(file_name, file_hash, status="failed", error=str(error)[:512])


def list_documents(status: Optional[str] = None, limit: int = 200) -> list:
    """文档目录（供前端/运维查询）。"""
    from persistence.db import session_scope
    from persistence.models import Document
    from sqlalchemy import select

    try:
        with session_scope() as s:
            if s is None:
                return []
            stmt = select(Document).order_by(Document.updated_at.desc()).limit(limit)
            if status:
                stmt = stmt.where(Document.status == status)
            return [{
                "file_name": r.file_name,
                "file_hash": r.file_hash,
                "file_size": r.file_size,
                "chunk_count": r.chunk_count,
                "page_count": r.page_count,
                "status": r.status,
                "kb_version": r.kb_version,
                "indexed_at": r.indexed_at.isoformat() if r.indexed_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                "error": r.error,
            } for r in s.scalars(stmt).all()]
    except Exception as e:
        logger.debug(f"[docs] 查询文档列表失败: {e}")
        return []


def _load_from_mysql() -> Dict[str, str]:
    """从 documents 表取 {file_name: file_hash}（只取未被移除的）。"""
    from persistence.db import session_scope
    from persistence.models import Document
    from sqlalchemy import select

    try:
        with session_scope() as s:
            if s is None:
                return {}
            rows = s.scalars(
                select(Document).where(Document.status != "removed")
            ).all()
            return {r.file_name: r.file_hash for r in rows if r.file_name and r.file_hash}
    except Exception:
        return {}


# ── 缓存失效 ─────────────────────────────────────────────────
def notify_kb_changed(added=None, changed=None, removed=None, total: int = 0) -> int:
    """知识库变更 → 自增 kb_version，让所有老问答缓存不可达。

    只动一个计数器，不做 SCAN+DEL：老 key 靠 TTL 自然过期，同步清理会阻塞摄取。
    Redis 不可用时 no-op——缓存本身就不可用，没有失效问题。
    """
    from persistence.qa_cache import bump_kb_version

    if not (added or changed or removed):
        return 0
    return bump_kb_version(added=added, changed=changed, removed=removed, total=total)


def touch_ingested(total: int, added=None, changed=None, removed=None):
    """摄取结束后的统一收尾：登记文档元数据 + 失效问答缓存。"""
    for name in (added or []):
        upsert(name, _hash_of(name) or "", status="indexed")
    for name in (changed or []):
        upsert(name, _hash_of(name) or "", status="indexed")
    for name in (removed or []):
        mark_removed(name)

    return notify_kb_changed(added=added, changed=changed, removed=removed, total=total)


def _hash_of(file_name: str) -> Optional[str]:
    p = Path(config.paths.data_dir) / file_name
    if not p.exists():
        return None
    import hashlib

    h = hashlib.md5()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def apply_snapshot(session, record: Dict[str, str]) -> int:
    """把整份哈希记录合并进 documents 表。返回变更行数。

    接收调用方的 session（writer 的消费端已经开着一个事务），不自己开——
    自开事务会在 SQLite 上撞锁库，在 MySQL 上白多一个连接。
    """
    from persistence.models import Document
    from sqlalchemy import select

    existing = {r.file_name: r for r in session.scalars(select(Document)).all()}
    n = 0
    for name, h in record.items():
        row = existing.get(name)
        if row is None:
            session.add(Document(file_name=name, file_hash=h, status="indexed",
                                 indexed_at=datetime.now()))
            n += 1
        elif row.file_hash != h:
            row.file_hash = h
            row.status = "indexed"
            row.updated_at = datetime.now()
            n += 1
    return n
