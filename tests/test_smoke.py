"""
Smoke Test: 端到端快速验证，确保核心链路不崩溃。
"""

import os
import sys
import pytest

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.smoke
class TestSmokeEndToEnd:
    """端到端冒烟测试。"""

    def test_agent_compiles(self):
        """Agent 图能正常编译。"""
        from workflows.graph import agent
        assert agent is not None

    def test_knowledge_base_tool(self):
        """search_knowledge_base 工具返回有效结果。"""
        from tools.rag.llamaindex_tool import search_knowledge_base
        result = search_knowledge_base.invoke({"query": "LLM agent"})
        assert result is not None
        assert len(result) > 10
        assert "来源:" in result, f"应包含来源标记，实际: {result[:200]}"
        assert "第" in result and "页" in result, f"应包含页码引用，实际: {result[:200]}"

    def test_agent_invoke_returns_answer(self):
        """Agent 能完成一次完整查询，不崩溃。"""
        from workflows.graph import agent
        result = agent.invoke(
            {"messages": [("user", "What is AutoGen?")], "reflection_count": 0},
            config={"configurable": {"thread_id": "smoke-test"}},
        )
        messages = result.get("messages", [])
        assert len(messages) > 0
        # 至少有 AI 回复
        has_ai = any(hasattr(m, "content") and m.content for m in messages)
        assert has_ai, "Agent 应该返回至少一条有内容的 AI 回复"

    def test_kb_no_hallucination(self):
        """search_knowledge_base 不编造：只返回原文分块，绝不合成。

        工具直接格式化检索到的 Node 内容（不做 LLM 合成），真实论文正文可能
        天然包含 "arXiv:" 等字样，因此不能拿这些当编造标记。正确断言：
        要么明确返回无结果，要么返回带真实来源标记的分块。
        """
        import re
        from tools.rag.llamaindex_tool import search_knowledge_base
        result = search_knowledge_base.invoke({"query": "nonexistent fake paper quantum blockchain"})
        if "No relevant documents" in result:
            return
        assert re.search(r"\[来源: .+? \| 第\d+页 \| 相关度: \d", result), \
            f"返回内容不是真实知识库分块格式: {result[:200]}"

    def test_page_citation_format(self):
        """页码引用格式正确。"""
        from tools.rag.llamaindex_tool import search_knowledge_base
        result = search_knowledge_base.invoke({"query": "AutoGen multi-agent framework"})
        assert "第" in result, "应该包含页码"
        assert "页" in result, "应该包含页号"


@pytest.mark.slow
class TestSlowIntegration:
    """较慢的集成测试（可选运行）。"""

    def test_multi_query_queries(self):
        """查询改写生成多个查询。"""
        from rag.query_rewriter import rewrite_query
        from models.llm import create_llm
        from config import config

        llm = create_llm(config.llm)
        queries = rewrite_query(llm, "multi-agent LLM framework", 2)
        assert len(queries) >= 2
        assert "multi-agent LLM framework" in queries  # 原始查询应在第一位

    def test_evaluation_runs(self):
        """评估脚本能正常运行。"""
        from evaluation.rag_eval import evaluate_retrieval
        result = evaluate_retrieval(5)
        assert "Recall@5" in result
        assert result["Recall@5"] > 0
