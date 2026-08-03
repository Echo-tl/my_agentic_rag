"""
单元测试: observability/tracing 集成。

验证 trace_query 包裹查询后 _traces 会被写入，且包含工具调用与节点流转。
无需 Qdrant / Ollama，纯内存测试。
"""

import os
import sys
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTracing:
    def test_trace_query_records_trace(self):
        from observability.tracing import (
            trace_query, get_recent_traces, clear_traces,
            record_tool, record_node,
        )
        clear_traces()
        with trace_query("test question"):
            record_node("agent")
            record_node("tools")
            record_tool("search_knowledge_base", {"query": "test"}, time.perf_counter())
            record_node("agent")
        traces = get_recent_traces()
        assert len(traces) == 1
        tr = traces[0]
        assert tr["query"] == "test question"
        assert tr["elapsed_s"] >= 0
        assert tr["path"] == ["START → agent", "agent → tools", "tools → agent"]
        assert len(tr["tool_calls"]) == 1
        assert tr["tool_calls"][0]["tool"] == "search_knowledge_base"
        assert tr["error"] is None

    def test_consecutive_same_node_deduped(self):
        from observability.tracing import trace_query, get_recent_traces, clear_traces, record_node
        clear_traces()
        with trace_query("dup"):
            record_node("agent")
            record_node("agent")  # 连续重复应被去重
            record_node("tools")
        tr = get_recent_traces()[-1]
        assert tr["path"] == ["START → agent", "agent → tools"]

    def test_trace_query_records_error(self):
        from observability.tracing import trace_query, get_recent_traces, clear_traces
        clear_traces()
        try:
            with trace_query("boom"):
                raise ValueError("boom")
        except ValueError:
            pass
        tr = get_recent_traces()[-1]
        assert tr["error"] == "boom"

    def test_no_trace_is_noop(self):
        # 无 trace 上下文时 record_* 应为 no-op，不报错
        from observability.tracing import record_tool, record_node, clear_traces, get_recent_traces
        clear_traces()
        record_node("agent")
        record_tool("x", {}, time.perf_counter())
        assert get_recent_traces() == []

    def test_get_recent_traces_limit(self):
        from observability.tracing import trace_query, get_recent_traces, clear_traces
        clear_traces()
        for i in range(5):
            with trace_query(f"q{i}"):
                pass
        traces = get_recent_traces(2)
        assert len(traces) == 2
        assert traces[-1]["query"] == "q4"
