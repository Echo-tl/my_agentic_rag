"""
MySQL 连接管理：懒加载单例 + 熔断降级 + 会话上下文。

设计原则同 persistence/redis_client.py：模块 import 不建连，不可用时 yield None。

用**同步** SQLAlchemy 而非 async：
- 所有 FastAPI 端点是同步 def，由 Starlette 丢进 threadpool，同步 DB 调用不阻塞事件循环
- agent.invoke 是几十秒级的同步阻塞调用，若端点改 async 会卡死整个 event loop
- LangGraph 用 asyncio.to_thread 在子线程跑工具，AsyncSession 无法跨线程共享
"""

import logging
import threading
import time
from contextlib import contextmanager
from config import config

logger = logging.getLogger("agentic_rag")

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import sessionmaker
except ImportError:  # pragma: no cover - 仅在未装依赖时触发
    create_engine = None
    sessionmaker = None
    text = None

    class SQLAlchemyError(Exception):  # noqa: D101 - 占位，保证模块可 import
        pass

_engine = None
_SessionLocal = None
_lock = threading.Lock()
_down_until = 0.0
_COOLDOWN = 30.0
_warned = False


def is_available() -> bool:
    """MySQL 当前是否可用（不触发重连尝试）。"""
    return get_engine() is not None


def get_engine():
    """懒加载单例。未启用 / 连不上 → None（调用方降级为 no-op）。"""
    global _engine, _SessionLocal, _down_until, _warned

    if not config.mysql.enabled or create_engine is None:
        return None
    if time.monotonic() < _down_until:
        return None
    if _engine is not None:
        return _engine

    with _lock:
        if _engine is not None:
            return _engine
        try:
            engine = create_engine(
                config.mysql.url,
                pool_size=config.mysql.pool_size,
                max_overflow=config.mysql.max_overflow,
                pool_recycle=config.mysql.pool_recycle,
                pool_pre_ping=config.mysql.pool_pre_ping,
                connect_args={
                    "connect_timeout": config.mysql.connect_timeout,
                    # URL 里已带 charset，这里再传一次做双保险
                    "charset": "utf8mb4",
                },
                echo=config.mysql.echo,
                future=True,
            )
            # 真正探活一次，别等第一次查询才发现连不上
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            if config.mysql.auto_create:
                from persistence.models import Base
                Base.metadata.create_all(engine)

            _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
            _engine = engine
            logger.info(f"[mysql] 已连接 {config.mysql.url.split('@')[-1]}")
        except Exception as e:
            if not _warned:
                logger.warning(f"[mysql] 不可用，持久化降级为 no-op: {e}")
                _warned = True
            _down_until = time.monotonic() + _COOLDOWN
            if config.mysql.strict:
                raise
            return None

    return _engine


@contextmanager
def session_scope():
    """ORM 会话上下文。MySQL 不可用时 yield None，调用方需自行判断。

    退出时自动 commit；**DB 层**异常自动 rollback 并吞掉——写入失败绝不能影响主流程。

    只吞 `SQLAlchemyError` 是刻意的：`except Exception` 会把 `AssertionError`、
    `KeyError`、`TypeError` 这些**自己代码的 bug** 一起静默掉，于是「写错了」
    看起来和「MySQL 挂了」一模一样，调用方和测试都以为正常。
    非 DB 层异常先回滚再原样抛出，让 bug 保持可见。
    """
    engine = get_engine()
    if engine is None or _SessionLocal is None:
        yield None
        return

    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.warning(f"[mysql] 写入失败（已回滚）: {e}")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def set_engine_for_testing(engine, session_factory):
    """测试注入点：传 SQLite 内存库的 engine 与 sessionmaker。"""
    global _engine, _SessionLocal, _down_until
    with _lock:
        _engine = engine
        _SessionLocal = session_factory
        _down_until = 0.0


def reset():
    """清空单例与熔断状态（测试用）。"""
    global _engine, _SessionLocal, _down_until, _warned
    with _lock:
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:
                pass
        _engine = None
        _SessionLocal = None
        _down_until = 0.0
        _warned = False
