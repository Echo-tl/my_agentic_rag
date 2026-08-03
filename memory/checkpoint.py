"""
Memory: LangGraph checkpoint 持久化管理。

使用 SQLite 存储对话状态，支持：
- 多轮对话记忆
- 会话恢复
- 多会话隔离（通过 thread_id）
"""

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from config import config


def get_checkpointer() -> SqliteSaver:
    """创建 SQLite checkpointer，用于持久化 Agent 对话状态。"""
    db_path = config.memory.checkpoint_db_path
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)
