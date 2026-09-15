"""
后台写队列：把 MySQL 写入完全挪出请求路径。

为什么必须有这一层：`/query/stream` 的 SSE 生成器是**同步生成器**，Starlette 每次
`next()` 都在 threadpool 里单独迭代。在生成器里直接做同步 MySQL 写会占住一个
threadpool 线程，用户端表现为末段卡顿。所以请求线程只做 `put_nowait`（O(1)），
真正落库交给单后台线程。

队列满时**直接丢弃**而不是阻塞——MySQL 挂掉时堆积的写任务绝不能拖垮进程。
"""

import atexit
import logging
import queue
import threading

from config import config

logger = logging.getLogger("agentic_rag")

_queue: queue.Queue = None
_thread = None
_stop = threading.Event()
_lock = threading.Lock()
_dropped = 0

_HANDLERS = {}


def register(kind: str, handler):
    """注册某一类写入的处理函数。handler(session, payload) -> None"""
    _HANDLERS[kind] = handler


def _ensure_started():
    """启动消费线程。**已 shutdown 过也能重启**——否则测试里跑第二次 lifespan
    （或同进程起第二个 app）之后所有写入都会静静进队列无人消费。"""
    global _queue, _thread
    if _queue is not None and _thread is not None and _thread.is_alive():
        return
    with _lock:
        if _queue is not None and _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        if _queue is None:
            _queue = queue.Queue(maxsize=config.mysql.write_queue_size)
        _thread = threading.Thread(target=_run, name="db-writer", daemon=True)
        _thread.start()
        atexit.register(shutdown)


def enqueue(kind: str, payload: dict) -> bool:
    """投递一次写入。队列满或已关闭 → 丢弃并返回 False，绝不阻塞。"""
    global _dropped

    if not config.mysql.enabled:
        return False

    _ensure_started()
    try:
        _queue.put_nowait((kind, payload))
        return True
    except queue.Full:
        if config.mysql.drop_on_overflow:
            _dropped += 1
            if _dropped % 100 == 1:
                logger.warning(f"[writer] 写队列已满，累计丢弃 {_dropped} 条（不阻塞主流程）")
        return False


def _run():
    while not _stop.is_set():
        try:
            first = _queue.get(timeout=1.0)
        except queue.Empty:
            continue

        batch = [first]
        # 顺手把队列里已有的也捞出来，凑批减少事务次数
        while len(batch) < config.mysql.write_batch_size:
            try:
                batch.append(_queue.get_nowait())
            except queue.Empty:
                break

        _flush(batch)


def _flush(batch):
    from persistence.db import session_scope

    try:
        with session_scope() as session:
            if session is None:
                return  # MySQL 不可用 → 整批丢弃
            for kind, payload in batch:
                handler = _HANDLERS.get(kind)
                if handler is None:
                    logger.debug(f"[writer] 未知写入类型: {kind}")
                    continue
                try:
                    # SAVEPOINT 隔离，一条坏数据只回滚它自己。
                    # 不能只靠 except：SQLAlchemy 在一次失败后会把 session 标成
                    # 「必须 rollback」，之后每条 handler 都抛 PendingRollbackError，
                    # 末尾 commit 也失败——整批（含无关会话的记录）一起丢。
                    with session.begin_nested():
                        handler(session, payload)
                except Exception as e:
                    logger.warning(f"[writer] {kind} 写入失败: {e}")
    except Exception as e:
        logger.warning(f"[writer] 批写失败，丢弃 {len(batch)} 条: {e}")


def dropped_count() -> int:
    return _dropped


def shutdown(wait: bool = True, timeout: float = 5.0):
    """退出前尽量把队列冲干净（任务不易丢失，但也不强求）。"""
    _stop.set()
    if _thread is not None and wait:
        _thread.join(timeout=timeout)
