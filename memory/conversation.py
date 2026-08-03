"""
对话历史管理: 简单的 CRUD 操作。
"""

import json
import time
from pathlib import Path
from typing import TypedDict


class Message(TypedDict, total=False):
    role: str  # "user" | "agent"
    content: str
    timestamp: float


class Conversation:
    """单次对话的管理器（非持久化，在内存中）。"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: list[Message] = []
        self.created_at = time.time()

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text, "timestamp": time.time()})

    def add_agent(self, text: str):
        self.messages.append({"role": "agent", "content": text, "timestamp": time.time()})

    def get_history(self, max_tokens: int = 8000) -> str:
        """获取对话历史文本（截断到 max_tokens 估算值）。"""
        # 简单估算: 1 token ≈ 2 chars
        max_chars = max_tokens * 2
        lines = []
        total = 0
        for msg in reversed(self.messages):
            line = f"[{msg['role']}]: {msg['content']}"
            total += len(line)
            if total > max_chars:
                break
            lines.append(line)
        return "\n".join(reversed(lines))

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        conv = cls(data["session_id"])
        conv.messages = data.get("messages", [])
        conv.created_at = data.get("created_at", time.time())
        return conv

    def save(self, directory: str = "./conversations"):
        """持久化到 JSON 文件。"""
        path = Path(directory)
        path.mkdir(exist_ok=True)
        filepath = path / f"{self.session_id}.json"
        filepath.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, session_id: str, directory: str = "./conversations") -> "Conversation | None":
        """从 JSON 文件加载对话。"""
        filepath = Path(directory) / f"{session_id}.json"
        if filepath.exists():
            data = json.loads(filepath.read_text())
            return cls.from_dict(data)
        return None
