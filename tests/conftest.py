"""
共享测试夹具。

核心目标：**在不启动真实 Redis / MySQL 的机器上跑全部单测**。
- Redis → fakeredis（纯 Python，支持 SET NX PX / INCR / HINCRBY / ZINCRBY / LTRIM / EXPIRE）
- MySQL → SQLite 内存库 + StaticPool（内存库必须 StaticPool，否则每次连接都是新库）
"""

import os
import sys

import pytest

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def fake_redis(monkeypatch):
    """内存 Redis，替代真实实例。"""
    fakeredis = pytest.importorskip("fakeredis")
    import persistence.redis_client as rc
    from config import config as settings

    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(settings.redis, "enabled", True)
    monkeypatch.setattr(settings.redis, "cache_enabled", True)
    monkeypatch.setattr(rc, "_client", client)
    monkeypatch.setattr(rc, "_down_until", 0.0)

    yield client

    try:
        client.flushall()
    except Exception:
        pass
    rc.reset()


@pytest.fixture
def no_backend(monkeypatch):
    """Redis 与 MySQL 全部不可用 —— 降级路径测试用。"""
    import persistence.db as db
    import persistence.redis_client as rc
    from config import config as settings

    monkeypatch.setattr(settings.redis, "enabled", False)
    monkeypatch.setattr(settings.mysql, "enabled", False)
    monkeypatch.setattr(rc, "_client", None)
    monkeypatch.setattr(rc, "_down_until", 0.0)
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_SessionLocal", None)
    monkeypatch.setattr(db, "_down_until", 0.0)
    yield


@pytest.fixture
def sqlite_orm(tmp_path, monkeypatch):
    """SQLite 跑完整 ORM schema（前提：persistence/models.py 保持方言无关）。

    **用临时文件库而不是 `:memory:` + StaticPool**：内存库必须靠 StaticPool 才能
    「一直是同一个库」，而 StaticPool 会把**同一条 sqlite3 连接**交给所有线程。
    写库的后台线程和测试主线程于是共用一条连接——主线程的查询游标还没关，
    后台线程的 COMMIT 就报 `cannot commit transaction - SQL statements in progress`，
    整批写入被静默回滚。这是夹具人造的竞态，真实部署（连接池每线程一条连接）
    根本不会发生，却会让用例随机变红。临时文件库 + 常规连接池与线上拓扑一致。
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import persistence.db as db
    from config import config as settings
    from persistence.models import Base

    engine = create_engine(
        f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    monkeypatch.setattr(settings.mysql, "enabled", True)
    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_SessionLocal", session_factory)
    monkeypatch.setattr(db, "_down_until", 0.0)

    yield session_factory

    engine.dispose()
    db.reset()
