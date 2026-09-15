"""
用户解析：`user_id` 透传 + 游客兜底。

本期刻意**不做登录体系**（无密码校验、无 JWT、无前端改动）：API 收一个外部
`user_id` 字符串，映射成内部自增主键；不传就落到 anonymous 游客。

进程内 memo 的意义：resolve 在请求路径上，而 user_id → 内部 id 的映射一旦建立
就**永不变更**（username 唯一）+ 用户量级远小于进程内存。没有 memo 的话每个请求
都要多一次 MySQL 往返，把刚用 Redis 省下的延迟又还回去。
"""

import logging
import threading
from typing import Optional

from config import config

logger = logging.getLogger("agentic_rag")

GUEST_USERNAME = "anonymous"

# 外部 user_id → 内部主键。只增不改，无需失效。
_MEMO: dict = {}
_MEMO_LOCK = threading.Lock()


def resolve(external_id: Optional[str] = None) -> Optional[int]:
    """把外部 user_id 解析成内部主键。MySQL 不可用 → None（会话照常落库，只是无归属）。

    未传 external_id 或传空串 → 游客用户。
    """
    if not config.mysql.enabled:
        return None

    name = _normalize_external_id(external_id)

    with _MEMO_LOCK:
        cached = _MEMO.get(name)
    if cached is not None:
        return cached

    uid = _load_or_create(name)
    if uid is not None:
        with _MEMO_LOCK:
            _MEMO[name] = uid
    return uid


def _normalize_external_id(external_id: Optional[str]) -> str:
    """外部 id 收敛为合法 username。

    做长度截断而不是拒绝：调用方传了 200 字符的 token 也不该 500，
    截到 64 位即可（users.username 是 VARCHAR(64)）。
    """
    if external_id is None:
        return GUEST_USERNAME
    s = str(external_id).strip()
    return s[:64] if s else GUEST_USERNAME


def _load_or_create(username: str) -> Optional[int]:
    """查用户；不存在则创建。游客用户全局共用一行。"""
    from persistence.db import session_scope
    from persistence.models import User
    from sqlalchemy import select
    from datetime import datetime

    try:
        with session_scope() as s:
            if s is None:
                return None

            # 唯一键冲突 + 并发创建 → 必须容忍 Duplicate entry，重查即可
            row = s.scalar(select(User).where(User.username == username))
            if row is None:
                row = User(
                    username=username,
                    display_name="游客" if username == GUEST_USERNAME else username,
                    is_guest=(username == GUEST_USERNAME),
                )
                s.add(row)
                s.flush()          # 拿自增主键

            row.last_seen_at = datetime.now()
            return row.id
    except Exception as e:
        logger.debug(f"[user] 解析用户失败（降级为无归属）: {e}")
        return None


def get_info(user_id: int) -> Optional[dict]:
    """用户详情（供查询接口）。"""
    from persistence.db import session_scope
    from persistence.models import User

    if user_id is None:
        return None
    try:
        with session_scope() as s:
            if s is None:
                return None
            row = s.get(User, user_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "username": row.username,
                "display_name": row.display_name,
                "is_guest": bool(row.is_guest),
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
            }
    except Exception:
        return None


def reset():
    """清空 memo（测试用）。"""
    with _MEMO_LOCK:
        _MEMO.clear()
