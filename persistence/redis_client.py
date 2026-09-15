"""
Redis 连接管理：懒加载单例 + 熔断降级。

与 database/qdrant.py 的风格保持一致：模块 import 不建连，不可用时返回 None，
调用方（各 store）自行走 no-op 分支。

额外加了熔断（_down_until）：Redis 挂掉后一段时间内直接返回 None，避免每个请求
都白等 socket_connect_timeout 秒。没有熔断的话，Redis 一挂整个服务的延迟会被
拖垮——这是「优雅降级」和「优雅地一起变慢」的区别。
"""

import logging
import threading
import time
from config import config

logger = logging.getLogger("agentic_rag")

# 依赖未安装时不能让 import 整个模块就崩——requirements 装完之前测试仍须全绿
try:
    import redis
except ImportError:  # pragma: no cover - 仅在未装依赖时触发
    redis = None

_client = None
_lock = threading.Lock()
_down_until = 0.0
_COOLDOWN = 30.0  # 熔断冷却秒数
_warned = False


def is_available() -> bool:
    """Redis 当前是否可用（不触发重连尝试）。"""
    return get_redis() is not None


def get_redis():
    """懒加载单例。未启用 / 连不上 → None（调用方降级为 no-op）。"""
    global _client, _down_until, _warned

    if not config.redis.enabled or redis is None:
        return None
    if time.monotonic() < _down_until:
        return None
    if _client is not None:
        return _client

    with _lock:
        if _client is not None:
            return _client
        try:
            pool = redis.ConnectionPool.from_url(
                config.redis.url,
                max_connections=config.redis.max_connections,
                socket_timeout=config.redis.socket_timeout,
                socket_connect_timeout=config.redis.socket_connect_timeout,
                health_check_interval=config.redis.health_check_interval,
                decode_responses=True,
                retry_on_timeout=False,  # 不重试 → 快速失败，把时间还给主流程
            )
            client = redis.Redis(connection_pool=pool)
            client.ping()
            _client = client
            logger.info(f"[redis] 已连接 {config.redis.url}")
        except Exception as e:
            if not _warned:
                logger.warning(f"[redis] 不可用，会话/摘要/缓存降级为 no-op: {e}")
                _warned = True
            _down_until = time.monotonic() + _COOLDOWN
            if config.redis.strict:
                raise
            return None

    return _client


def set_client_for_testing(client):
    """测试注入点：传 fakeredis.FakeRedis 或 None（模拟宕机）。"""
    global _client, _down_until
    with _lock:
        _client = client
        _down_until = 0.0


def reset():
    """清空单例与熔断状态（测试用）。"""
    global _client, _down_until, _warned
    with _lock:
        _client = None
        _down_until = 0.0
        _warned = False
