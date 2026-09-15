"""
Persistence 模块：Redis（热数据）+ MySQL（持久化）扩展层。

设计原则（与 database/qdrant.py 一致）：
- 模块 import 不建连，首次调用才懒加载
- Redis / MySQL 不可用时全部降级为 no-op，主流程不受影响
- 读失败当 miss，写失败静默并 WARN
"""

from persistence import redis_client

__all__ = ["redis_client"]


def ensure_ready():
    """把后台写队列的处理器装上。由 api/server.py 启动时调用一次。

    不放在模块顶层 import 里，是因为导入即注册会让任何 `import persistence.*`
    的测试都拉起注册链路；显式调用点更好排查。
    """
    from persistence.repo import register_all

    register_all()
