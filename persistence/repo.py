"""
MySQL 写入处理器：writer 队列的消费端。

所有函数签名都是 `handler(session, payload)`，由 `persistence.writer` 在后台线程里
带着一个 ORM session 调用——**它们已经不在请求路径上**，可以放心做多语句事务。

两条容易踩的约束：
1. `qa_records.session_id` 有外键指向 `sessions`，插入问答前必须确保会话行存在，
   否则 MySQL 报 1452、SQLite（未开 FK 校验）却静默通过 → 单测绿而线上红。
   `_ensure_session` 就是为消除这个方言差异存在的。
2. turn_index 以库里的 `turn_count` 为准而不是内存计数：多 worker 部署时内存计数
   会各数各的，落库的轮次序号必然错乱。
"""

import logging
from datetime import datetime

logger = logging.getLogger("agentic_rag")


# ============================================================
# 供业务层调用的投递封装（组装 payload，不做任何 IO）
# ============================================================
def submit_session(session_id: str, user_id=None, client_ip: str = None,
                   title: str = None) -> bool:
    """登记/更新会话行（Redis touch 之外的持久副本）。"""
    if not session_id:
        return False
    from persistence.writer import enqueue

    return enqueue("session", {
        "session_id": session_id,
        "user_id": user_id,
        "client_ip": client_ip,
        "title": title,
    })


def submit_turn(session_id: str, question: str, answer: str, **fields) -> bool:
    """投递一轮问答记录。**必须在整图结束后调用**，answer 用最终状态里的答案。"""
    if not session_id:
        return False
    from persistence.writer import enqueue
    from persistence.qa_cache import question_hash

    payload = {
        "session_id": session_id,
        "question": question or "",
        "question_hash": question_hash(question or ""),
        "answer": answer or "",
    }
    payload.update(fields)
    return enqueue("turn", payload)


def submit_summary(session_id: str, summary: str, summary_upto: int = 0) -> bool:
    if not session_id or not summary:
        return False
    from persistence.writer import enqueue

    return enqueue("summary", {
        "session_id": session_id,
        "summary": summary,
        "summary_upto": summary_upto,
    })


def submit_trace(trace: dict) -> bool:
    if not trace:
        return False
    from persistence.writer import enqueue

    return enqueue("trace", trace)


def submit_document(file_name: str, file_hash: str, **fields) -> bool:
    if not file_name:
        return False
    from persistence.writer import enqueue

    payload = {"file_name": file_name, "file_hash": file_hash}
    payload.update(fields)
    return enqueue("doc_upsert", payload)


def submit_document_removed(file_name: str) -> bool:
    if not file_name:
        return False
    from persistence.writer import enqueue

    return enqueue("doc_removed", {"file_name": file_name})


# ============================================================
# 处理器
# ============================================================
def _ensure_session(s, session_id: str, user_id=None, client_ip: str = None,
                    title: str = None):
    """取会话行，不存在则建。返回 ORM 对象（MySQL 不可用时调用方已提前 return）。"""
    from persistence.models import Session

    row = s.get(Session, session_id)
    if row is None:
        row = Session(session_id=session_id, user_id=user_id, client_ip=client_ip)
        if title:
            row.title = title[:255]
        s.add(row)
        s.flush()
    elif user_id is not None and row.user_id is None:
        # 游客先到、随后带上 user_id 的场景：补归属
        row.user_id = user_id
    return row


def on_session(s, payload: dict):
    _ensure_session(
        s,
        payload["session_id"],
        payload.get("user_id"),
        payload.get("client_ip"),
        payload.get("title"),
    )


def on_turn(s, payload: dict):
    from persistence.models import QARecord

    sid = payload["session_id"]
    row = _ensure_session(s, sid, payload.get("user_id"))

    # 首轮问题顺带做会话标题（比 "新会话" 之类的占位有信息量）
    if not row.title and payload.get("question"):
        row.title = payload["question"][:255]

    row.turn_count = (row.turn_count or 0) + 1
    turn_index = row.turn_count
    row.last_active_at = datetime.now()
    if payload.get("kb_version") is not None:
        row.kb_version = payload["kb_version"]

    rec = QARecord(
        session_id=sid,
        user_id=row.user_id,
        turn_index=turn_index,
        question=payload.get("question", ""),
        question_hash=payload.get("question_hash", ""),
        context_fp=payload.get("context_fp"),
        answer=payload.get("answer"),
        task_type=payload.get("task_type"),
        intent_json=payload.get("intent"),
        citations=payload.get("citations"),
        tool_calls=payload.get("tool_calls"),
        cache_key=_truncate(payload.get("cache_key"), 255),
        cached=bool(payload.get("cached", False)),
        kb_version=payload.get("kb_version") or 0,
        model=payload.get("model"),
        latency_ms=payload.get("latency_ms"),
        reflection_count=payload.get("reflection_count") or 0,
        trace_id=payload.get("trace_id"),
        error=_truncate(payload.get("error"), 512),
    )
    s.add(rec)


def on_summary(s, payload: dict):
    from persistence.models import Session

    row = s.get(Session, payload["session_id"])
    if row is None:
        return
    row.summary = payload["summary"]
    if payload.get("summary_upto"):
        row.summary_upto = payload["summary_upto"]


def on_trace(s, payload: dict):
    from persistence.models import ExecutionTrace

    trace_id = payload.get("trace_id")
    if not trace_id:
        return

    row = s.query(ExecutionTrace).filter_by(trace_id=trace_id).one_or_none()
    if row is None:
        row = ExecutionTrace(trace_id=trace_id, query=payload.get("query") or "")
        s.add(row)

    row.session_id = payload.get("session_id") or row.session_id
    row.qa_id = payload.get("qa_id") or row.qa_id
    row.query = _truncate(payload.get("query"), 1024) or row.query
    row.elapsed_ms = payload.get("elapsed_ms")
    row.tool_calls = payload.get("tool_calls")
    row.node_path = payload.get("node_path")
    row.tool_count = payload.get("tool_count") or 0
    row.error = _truncate(payload.get("error"), 1024)


def on_doc_upsert(s, payload: dict):
    from persistence.models import Document

    row = s.query(Document).filter_by(file_name=payload["file_name"]).one_or_none()
    if row is None:
        row = Document(file_name=payload["file_name"],
                       file_hash=payload["file_hash"])
        s.add(row)

    row.file_hash = payload["file_hash"]
    for key in ("file_path", "file_size", "chunk_count", "page_count",
                "kb_version", "error"):
        if payload.get(key) is not None:
            setattr(row, key, payload[key])

    status = payload.get("status")
    if status:
        row.status = status
    if payload.get("status") == "indexed":
        row.indexed_at = datetime.now()
    row.updated_at = datetime.now()


def on_doc_snapshot(s, payload: dict):
    from persistence.doc_registry import apply_snapshot

    apply_snapshot(s, payload.get("record") or {})


def on_doc_removed(s, payload: dict):
    from persistence.models import Document

    row = s.query(Document).filter_by(file_name=payload["file_name"]).one_or_none()
    if row is None:
        return
    # 不物理删除：留一行 removed 供审计「这份文档曾经在库里」
    row.status = "removed"
    row.kb_version = payload.get("kb_version", row.kb_version)
    row.updated_at = datetime.now()


def _truncate(value, limit: int):
    if value is None:
        return None
    s = str(value)
    return s[:limit] if len(s) > limit else s


_REGISTERED = False


def register_all():
    """幂等注册。由 persistence 包首次使用时调用。"""
    global _REGISTERED
    if _REGISTERED:
        return
    from persistence.writer import register

    register("session", on_session)
    register("turn", on_turn)
    register("summary", on_summary)
    register("trace", on_trace)
    register("doc_upsert", on_doc_upsert)
    register("doc_removed", on_doc_removed)
    register("doc_snapshot", on_doc_snapshot)
    _REGISTERED = True
    logger.debug("[mysql] 写入处理器已注册")


# ============================================================
# 读取接口（查询端点在请求路径上，这里都是索引上的小查询）
# ============================================================
def list_session_qa(session_id: str, limit: int = 50, offset: int = 0) -> list:
    """会话的问答历史（不含 Agent 内部消息——那些不落库）。"""
    from persistence.db import session_scope
    from persistence.models import QARecord
    from sqlalchemy import select

    try:
        with session_scope() as s:
            if s is None:
                return []
            rows = s.scalars(
                select(QARecord)
                .where(QARecord.session_id == session_id)
                .order_by(QARecord.id.desc())
                .limit(limit).offset(offset)
            ).all()
            return [_qa_to_dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"[mysql] 查询问答历史失败: {e}")
        return []


def list_user_sessions(user_id: int, limit: int = 50) -> list:
    from persistence.db import session_scope
    from persistence.models import Session
    from sqlalchemy import select

    try:
        with session_scope() as s:
            if s is None:
                return []
            rows = s.scalars(
                select(Session)
                .where(Session.user_id == user_id)
                .order_by(Session.last_active_at.desc())
                .limit(limit)
            ).all()
            return [{
                "session_id": r.session_id,
                "title": r.title,
                "turn_count": r.turn_count,
                "status": r.status,
                "created_at": _iso(r.created_at),
                "last_active_at": _iso(r.last_active_at),
            } for r in rows]
    except Exception as e:
        logger.debug(f"[mysql] 查询用户会话失败: {e}")
        return []


def _qa_to_dict(r) -> dict:
    return {
        "id": r.id,
        "session_id": r.session_id,
        "turn_index": r.turn_index,
        "question": r.question,
        "answer": r.answer,
        "task_type": r.task_type,
        "citations": r.citations or [],
        "tool_calls": r.tool_calls or [],
        "cached": bool(r.cached),
        "model": r.model,
        "latency_ms": r.latency_ms,
        "reflection_count": r.reflection_count,
        "created_at": _iso(r.created_at),
    }


def _iso(dt):
    return dt.isoformat() if dt else None
