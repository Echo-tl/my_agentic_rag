"""
Agent 生成质量评估: 检查输出是否包含编造、是否覆盖关键论文。
"""

import os
from evaluation.rag_eval import TEST_QUERIES

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# ── 评估标准 ──────────────────────────────────────────────────
HALLUCINATION_INDICATORS = [
    "arXiv:",           # 不应有凭空 arXiv 编号
    "LLM Agents: A Survey",  # 已知编造案例
    "Multi-Agent Systems for LLMs",  # 已知编造案例
    "未提供具体标题",      # 不应有免责声明
    "未标注具体论文来源",   # 不应有免责声明
]

REQUIRED_ELEMENTS = [
    "核心方法",
    "技术要点",
    "agent",
]


def evaluate_answer(answer: str, expected_papers: list[str] | None = None) -> dict:
    """评估单个回答的质量。

    Returns:
        dict with: hallucination_score, coverage_score, issues
    """
    text_lower = answer.lower()

    # 1. 编造检测
    hallucinations = [h for h in HALLUCINATION_INDICATORS if h.lower() in text_lower]
    hallu_score = 1.0 - len(hallucinations) * 0.3  # 每个减 0.3
    hallu_score = max(0.0, hallu_score)

    # 2. 论文覆盖度
    if expected_papers:
        found_papers = sum(1 for p in expected_papers if p.lower() in text_lower)
        coverage = found_papers / len(expected_papers) if expected_papers else 1.0
    else:
        coverage = -1  # 未提供期望列表

    # 3. 结构完整度
    structure = sum(1 for e in REQUIRED_ELEMENTS if e in text_lower)
    structure_score = structure / len(REQUIRED_ELEMENTS)

    return {
        "hallucination_score": round(hallu_score, 2),
        "hallucination_indicators": hallucinations,
        "paper_coverage": round(coverage, 2) if coverage >= 0 else "N/A",
        "structure_score": round(structure_score, 2),
        "pass": hallu_score >= 0.7 and structure_score >= 0.5,
    }


def quick_eval(ask_fn) -> dict:
    """快速评估: 用测试查询跑一遍，检查输出质量。"""
    results = {}
    for item in TEST_QUERIES[:3]:  # 只跑前 3 个，避免太慢
        query = item["query"]
        expected = item["expected"]
        try:
            answer = ask_fn(query)
            results[query[:40]] = evaluate_answer(answer, expected)
        except Exception as e:
            results[query[:40]] = {"error": str(e)}
    return results
