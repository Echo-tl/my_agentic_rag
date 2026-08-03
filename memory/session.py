"""
会话管理: 多用户/多会话隔离。
"""

import uuid
from typing import Dict
from memory.conversation import Conversation


class SessionManager:
    """管理多个用户会话。"""

    def __init__(self):
        self._sessions: Dict[str, Conversation] = {}

    def create(self) -> str:
        """创建新会话，返回 session_id。"""
        sid = str(uuid.uuid4())[:8]
        self._sessions[sid] = Conversation(sid)
        return sid

    def get(self, session_id: str) -> Conversation | None:
        """获取指定会话。如果不存在返回 None。"""
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> Conversation:
        """获取会话，不存在则创建。"""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        conv = Conversation(session_id or str(uuid.uuid4())[:8])
        self._sessions[conv.session_id] = conv
        return conv

    def delete(self, session_id: str) -> bool:
        """删除会话。"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[str]:
        """列出所有活跃会话 ID。"""
        return list(self._sessions.keys())

    def cleanup_expired(self, max_age_seconds: float = 3600):
        """清理过期会话（默认 1 小时）。"""
        import time
        expired = [
            sid for sid, conv in self._sessions.items()
            if time.time() - conv.created_at > max_age_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
        return expired


# 全局会话管理器
session_manager = SessionManager()
