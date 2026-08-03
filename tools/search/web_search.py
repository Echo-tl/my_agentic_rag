import time
from langchain_core.tools import tool
from ddgs import DDGS

from observability.tracing import record_tool

@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息。用于查找最新的LLM论文呢、AGENT论文等。"""
    _start = time.perf_counter()
    try:
        results = DDGS().text(query, max_results=3)
        return "\n\n".join([r["body"] for r in results])
    finally:
        record_tool("search_web", {"query": query}, _start)