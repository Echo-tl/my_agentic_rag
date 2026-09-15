"""
会话状态：Redis 主存（消息窗口 + 滚动摘要）+ MySQL 持久副本。

**Redis 只是加速层，不是记忆真源。** 多轮对话的真源是 LangGraph 的 checkpointer
（Redis RedisSaver 或 SQLite SqliteSaver）；这里存的是可重建的窗口与摘要。
所以 Redis 全丢不影响对话可用性——这条是设计约束，不是巧合。

摘要的持久副本落在 MySQL sessions.summary：Redis 过期后从库里兜底重灌，
会话隔天再接上时不会丢上下文。
"""

import json
import logging
import time
from typing import Optional

from config import config
from persistence.redis_client import get_redis

logger = logging.getLogger("agentic_rag")


# ── Key 布局 ─────────────────────────────────────────────────
def _k_sess(sid: str) -> str:
    return f"{config.redis.key_prefix}:sess:{sid}"


def _k_msgs(sid: str) -> str:
    return f"{config.redis.key_prefix}:sess:{sid}:msgs"


def _k_summary(sid: str) -> str:
    return f"{config.redis.key_prefix}:sess:{sid}:summary"


# ── 会话元数据 ───────────────────────────────────────────────
def touch(session_id: str, user_id: Optional[int] = None) -> Optional[dict]:
    """登记一次会话访问：刷新滑动过期，返回会话元数据（Redis 不可用 → None）。"""
    r = get_redis()
    if r is None or not session_id:
        return None

    ttl = config.redis.session_ttl
    try:
        key = _k_sess(session_id)
        now = time.time()
        pipe = r.pipeline()
        pipe.hsetnx(key, "created_at", now)
        pipe.hincrby(key, "turn_count", 1)
        pipe.hset(key, mapping={"last_seen": now})
        if user_id is not None:
            pipe.hset(key, "user_id", user_id)
        pipe.expire(key, ttl)
        pipe.expire(_k_msgs(session_id), ttl)
        pipe.execute()
        return get_session(session_id)
    except Exception as e:
        logger.debug(f"[session] touch 失败（降级）: {e}")
        return None


def get_session(session_id: str) -> Optional[dict]:
    """读会话元数据。Redis 未命中时回落到 MySQL（持久副本）。"""
    r = get_redis()
    if r is None or not session_id:
        return _load_session_from_db(session_id)

    try:
        data = r.hgetall(_k_sess(session_id))
        if data:
            return {
                "session_id": session_id,
                "user_id": int(data["user_id"]) if data.get("user_id") else None,
                "turn_count": int(data.get("turn_count", 0)),
                "created_at": float(data.get("created_at", 0)),
                "last_seen": float(data.get("last_seen", 0)),
            }
    except Exception as e:
        logger.debug(f"[session] 读会话失败（降级）: {e}")

    return _load_session_from_db(session_id)


def _load_session_from_db(session_id: str) -> Optional[dict]:
    """MySQL 兜底读。MySQL 也不可用 → None。"""
    if not session_id:
        return None
    from persistence.db import session_scope
    from persistence.models import Session

    try:
        with session_scope() as s:
            if s is None:
                return None
            row = s.get(Session, session_id)
            if row is None:
                return None
            return {
                "session_id": row.session_id,
                "user_id": row.user_id,
                "turn_count": row.turn_count,
                "created_at": row.created_at.timestamp(),
                "last_seen": row.last_active_at.timestamp(),
            }
    except Exception:
        return None


def delete(session_id: str) -> bool:
    """删除会话的 Redis 状态（MySQL 记录保留为归档）。"""
    r = get_redis()
    if r is None:
        return False
    try:
        r.delete(_k_sess(session_id), _k_msgs(session_id), _k_summary(session_id))
        return True
    except Exception:
        return False


def list_sessions() -> list:
    """列出当前活跃会话（Redis 里还有 TTL 的）。"""
    r = get_redis()
    if r is None:
        return []
    try:
        _, keys = r.scan(0, match=f"{config.redis.key_prefix}:sess:*:msgs", count=500)
        return sorted(k.rsplit(":", 2)[-2] for k in keys)
    except Exception:
        return []


# ── 消息窗口 ─────────────────────────────────────────────────
def get_window(session_id: str) -> list:
    """读最近 N 条消息（时间正序）。Redis 不可用 → 空列表。

    空列表是**安全**的：上下文指纹会因此判定非自足问题不可缓存，
    而不是拿错误的上下文去命中缓存。
    """
    r = get_redis()
    if r is None or not session_id:
        return []
    try:
        raw = r.lrange(_k_msgs(session_id), 0, -1)
        out = []
        for item in raw:
            try:
                out.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return out
    except Exception:
        return []


def append_turn(session_id: str, question: str, answer: str):
    """把一轮问答追加进窗口（超长自动裁剪）。Redis 不可用 → no-op。"""
    r = get_redis()
    if r is None or not session_id:
        return

    ttl = config.redis.session_ttl
    window = config.redis.window_size
    now = time.time()
    try:
        key = _k_msgs(session_id)
        pipe = r.pipeline()
        pipe.rpush(key,
                   json.dumps({"role": "user", "content": question, "ts": now},
                              ensure_ascii=False),
                   json.dumps({"role": "assistant", "content": answer, "ts": now},
                              ensure_ascii=False))
        pipe.ltrim(key, -window, -1)
        pipe.expire(key, ttl)
        pipe.execute()
    except Exception as e:
        logger.debug(f"[session] append_turn 失败（降级）: {e}")


def message_count(session_id: str) -> int:
    """当前窗口内的消息条数（用于判断是否触发摘要）。"""
    r = get_redis()
    if r is None or not session_id:
        return 0
    try:
        return int(r.llen(_k_msgs(session_id)))
    except Exception:
        return 0


# ── 滚动摘要 ─────────────────────────────────────────────────
def get_summary(session_id: str) -> Optional[str]:
    """读摘要。Redis 未命中回落 MySQL；都没有 → None。"""
    r = get_redis()
    if r is not None and session_id:
        try:
            s = r.get(_k_summary(session_id))
            if s:
                return s
        except Exception:
            pass
    return _load_summary_from_db(session_id)


def _load_summary_from_db(session_id: str) -> Optional[str]:
    if not session_id:
        return None
    from persistence.db import session_scope
    from persistence.models import Session

    try:
        with session_scope() as s:
            if s is None:
                return None
            row = s.get(Session, session_id)
            return row.summary if row and row.summary else None
    except Exception:
        return None


def set_summary(session_id: str, summary: str):
    """写摘要：Redis（带 TTL）+ MySQL 持久副本（后台写，不阻塞）。"""
    if not session_id or not summary:
        return
    r = get_redis()
    if r is not None:
        try:
            r.set(_k_summary(session_id), summary, ex=config.redis.summary_ttl)
        except Exception as e:
            logger.debug(f"[session] 写摘要失败（降级）: {e}")

    # 持久副本走后台写队列，失败只记日志（Redis 过期后靠它兜底重灌）
    try:
        from persistence.repo import submit_summary

        submit_summary(session_id, summary)
    except Exception:
        pass


def should_summarize(session_id: str) -> bool:
    """窗口内消息数是否已达到摘要触发阈值。"""
    if not config.redis.summary_enabled:
        return False
    return message_count(session_id) >= config.redis.summary_trigger
