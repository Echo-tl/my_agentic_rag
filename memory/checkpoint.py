"""
LangGraph checkpoint 持久化：Redis 优先，SQLite 兜底。

为什么要迁移：SqliteSaver 的 `checkpoints.db` **随轮次无界增长**——每一轮都把整个
消息列表重新写一份快照。实测几个月下来就是几十 MB，且 SQLite 单文件在并发写时
会锁表。RedisSaver 带 TTL，老快照自动过期。

降级链（任一步失败都往下走，绝不阻断启动）：
    RedisSaver → SqliteSaver → None（不持久化，单轮可用但无多轮记忆）

RedisSaver 强依赖 **RedisJSON + RediSearch** 两个模块：`redis:7` 镜像会直接报
「unknown command 'JSON.SET'」，必须用 `redis:8`（8.0 起两个模块内置于所有官方
二进制发行版）或 `redis/redis-stack-server`。

注意 `config.redis.enabled=False`（默认）时 checkpoint_backend 即便配成 "redis"
也会走 SQLite——不配环境变量的用户行为与引入本模块之前完全一致。
"""

import logging
import sqlite3
import threading
from typing import Optional

from config import config

logger = logging.getLogger("agentic_rag")

_saver = None
_backend = None          # "redis" | "sqlite" | "none"
_lock = threading.Lock()


def get_checkpointer():
    """按配置返回 checkpointer 单例；全链路不可用时返回 None。

    **必须是单例**：SqliteSaver 的连接跨 Starlette threadpool 的多个线程使用
    （`check_same_thread=False`），每次调用都新建连接会各自持有独立的文件句柄与
    事务状态，并发下极易 `database is locked`。
    """
    global _saver, _backend

    if _saver is not None or _backend == "none":
        return _saver

    with _lock:
        if _saver is not None or _backend == "none":
            return _saver

        backend = config.redis.checkpoint_backend

        if backend == "redis":
            _saver = _try_redis()
            if _saver is not None:
                _backend = "redis"
                return _saver
            logger.warning("[checkpoint] RedisSaver 不可用，回退 SQLite")

        if backend in ("redis", "sqlite"):
            _saver = _try_sqlite()
            if _saver is not None:
                _backend = "sqlite"
                return _saver
            logger.warning("[checkpoint] SqliteSaver 也不可用，本轮不持久化对话状态")

        _backend = "none"
        return None


def _try_redis():
    """RedisSaver 需要 RedisJSON + RediSearch，缺一即建不起来。"""
    if not config.redis.enabled:
        logger.debug("[checkpoint] 未启用 Redis（AGENTIC_RAG_REDIS__ENABLED），跳过 RedisSaver")
        return None

    try:
        from langgraph.checkpoint.redis import RedisSaver
    except ImportError:
        logger.warning("[checkpoint] 未安装 langgraph-checkpoint-redis，无法使用 RedisSaver")
        return None

    try:
        from persistence.redis_client import get_redis

        client = get_redis()
        if client is None:
            return None

        ttl = {
            "default_ttl": max(1, int(config.redis.checkpoint_ttl)),  # 分钟
            "refresh_on_read": True,   # 会话活跃期间快照不中途蒸发
        }
        saver = RedisSaver(
            redis_client=client,
            ttl=ttl,
            checkpoint_prefix=f"{config.redis.key_prefix}:ckpt",
            checkpoint_write_prefix=f"{config.redis.key_prefix}:ckpt:w",
        )
        # setup() 建索引（RediSearch）；对已建好的索引是幂等的
        saver.setup()
        logger.info(f"[checkpoint] 使用 RedisSaver（TTL {config.redis.checkpoint_ttl} 分钟）")
        return saver
    except Exception as e:
        # 最常见的失败就是模块缺失：'unknown command JSON.SET'
        logger.warning(f"[checkpoint] RedisSaver 初始化失败: {e}")
        return None


def _try_sqlite():
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = config.memory.checkpoint_db_path
        conn = sqlite3.connect(db_path, check_same_thread=False)
        logger.info(f"[checkpoint] 使用 SqliteSaver: {db_path}")
        return SqliteSaver(conn)
    except Exception as e:
        logger.warning(f"[checkpoint] SqliteSaver 初始化失败: {e}")
        return None


def prune(thread_ids) -> int:
    """压缩这些 thread 的历史快照，只保留最新一份。

    裁剪（RemoveMessage）只让**新**快照变小，已经落盘的大快照仍然占着空间——
    必须配合 prune 才能真正把文件/内存收回来。

    失败一律吞掉：这是空间回收，不是主流程。
    """
    ids = [t for t in (thread_ids or []) if t]
    if not ids or _saver is None:
        return 0
    try:
        _saver.prune(ids, strategy="keep_latest")
        return len(ids)
    except Exception as e:
        logger.debug(f"[checkpoint] prune 失败（不影响对话）: {e}")
        return 0


def delete_thread(thread_id: str) -> bool:
    """彻底删除一个会话的 checkpoint（用户主动删会话时用）。"""
    if not thread_id or _saver is None:
        return False
    try:
        _saver.delete_thread(thread_id)
        return True
    except Exception as e:
        logger.debug(f"[checkpoint] delete_thread 失败: {e}")
        return False


def backend() -> Optional[str]:
    """当前生效的 backend（"redis" / "sqlite" / "none"），未初始化则先初始化。"""
    get_checkpointer()
    return _backend


def reset():
    """清空单例（测试用）。"""
    global _saver, _backend
    with _lock:
        _saver = None
        _backend = None
