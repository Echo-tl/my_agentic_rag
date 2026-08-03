"""
Agent 执行追踪: 记录每次查询的耗时、工具调用链、节点流转。

接入方式：
    with trace_query("用户问题"):
        result = agent.invoke(...)

工具函数 / workflow 节点内用 record_tool() / record_node() 记录。
trace_query 退出时自动 finish() 并写回 _traces，可用 get_recent_traces() 查询。
"""

import time
import logging
import threading
from contextvars import ContextVar
from functools import wraps
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agentic_rag")

# 内存中最多保留的 trace 数量
MAX_TRACES = 200


class ExecutionTrace:
    """单次查询的执行追踪记录。"""

    def __init__(self, query: str):
        self.query = query
        self.start_time = time.time()
        self.tool_calls: list[dict] = []
        self.node_transitions: list[str] = []
        self._last_node: Optional[str] = None
        self.error: Optional[str] = None

    def add_tool_call(self, tool_name: str, args: dict, duration_ms: float):
        self.tool_calls.append({"tool": tool_name, "args": args, "duration_ms": round(duration_ms, 1)})

    def add_transition(self, from_node: str, to_node: str):
        self.node_transitions.append(f"{from_node} → {to_node}")

    def record_node(self, node_name: str):
        """记录一次节点流转，自动衔接上一个节点（首节点记为 START）。

        连续进入同一节点（LangGraph 调度的重复唤醒）会被去重。
        """
        if self._last_node == node_name:
            return
        self.add_transition(self._last_node or "START", node_name)
        self._last_node = node_name

    def finish(self):
        elapsed = time.time() - self.start_time
        result = {
            "query": self.query,
            "elapsed_s": round(elapsed, 1),
            "tool_calls": self.tool_calls,
            "path": self.node_transitions,
            "error": self.error,
        }
        with _traces_lock:
            _traces.append(result)
            if len(_traces) > MAX_TRACES:
                del _traces[:-MAX_TRACES]
        # path 显示为节点序列，而非 "from → to" 字符串的 join（后者会让边界节点重复显示）
        path_nodes = [self.node_transitions[0].split(" → ")[0]] + [
            t.split(" → ")[1] for t in self.node_transitions
        ] if self.node_transitions else []
        logger.info(
            f"Trace complete: query='{self.query[:50]}...' elapsed={elapsed:.1f}s "
            f"tool_calls={len(self.tool_calls)} path={' → '.join(path_nodes)}"
        )
        return result


# 全局追踪存储（写访问由锁保护）
_traces: list[dict] = []
_traces_lock = threading.Lock()


# ── 当前 trace 上下文（contextvars，跨同步调用传播，支持并发请求）──
_current_trace: ContextVar[Optional[ExecutionTrace]] = ContextVar("current_trace", default=None)


def get_current_trace() -> Optional[ExecutionTrace]:
    return _current_trace.get()


def trace_query(query: str):
    """上下文管理器：包裹一次完整查询。

    用法:
        with trace_query(question):
            result = agent.invoke(...)

    无论正常还是异常退出，都会 finish() 并写回 _traces。
    """
    from contextlib import contextmanager

    @contextmanager
    def _manager():
        trace = ExecutionTrace(query)
        token = _current_trace.set(trace)
        try:
            yield trace
        except Exception as e:
            trace.error = str(e)
            raise
        finally:
            _current_trace.reset(token)
            trace.finish()

    return _manager()


def record_tool(tool_name: str, args: dict, start_time: float):
    """记录一次工具调用（耗时从 start_time 起算）。在工具函数内调用，无 trace 时为 no-op。"""
    tr = get_current_trace()
    if tr is not None:
        tr.add_tool_call(tool_name, args, (time.perf_counter() - start_time) * 1000)


def record_node(node_name: str):
    """记录一次节点进入。在 workflow 节点函数内调用，无 trace 时为 no-op。"""
    tr = get_current_trace()
    if tr is not None:
        tr.record_node(node_name)


def trace(func: Callable) -> Callable:
    """装饰器: 追踪函数调用耗时（仅打日志，不写入 trace 记录）。"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise

    return wrapper


def get_recent_traces(limit: int = 20) -> list[dict]:
    """获取最近的执行追踪记录。"""
    with _traces_lock:
        return list(_traces[-limit:])


def clear_traces():
    """清空追踪记录。"""
    with _traces_lock:
        _traces.clear()
