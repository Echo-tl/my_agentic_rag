"""
RAG 检索评估：Recall@K, MRR (Mean Reciprocal Rank), Hit Rate。

数据来源：evaluation/datasets/test_queries.json（50 条评测集）
"""

import json
import os
from pathlib import Path
from typing import Optional
from rag.index import get_index
from config import config
from database.qdrant import get_vector_store
from models.embedding import create_embedding

os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def load_queries() -> list[dict]:
    """从 JSON 文件加载评测集。"""
    path = Path(__file__).parent / "datasets" / "test_queries.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"]


def _expected_papers_in_top_k(nodes: list, expected_papers: list[str]) -> bool:
    """判断 top-K 结果中是否出现全部预期论文（按 file_name 匹配）。"""
    files = {n.node.metadata.get("file_name", "") for n in nodes}
    return all(
        any(p.lower() in f.lower() for f in files) for p in expected_papers
    )


def evaluate_retrieval(top_k: int = 10, limit: Optional[int] = None) -> dict:
    """评估检索质量。

    Args:
        top_k: 检索返回数量
        limit: 只评估前 N 条查询（None = 全部）

    Returns:
        {Recall@K, MRR, Hit Rate, Paper Hit Rate, queries_tested, top_k}
    """
    index = get_index(config.paths.data_dir, get_vector_store(), create_embedding(config.embedding))
    retriever = index.as_retriever(similarity_top_k=top_k)

    all_queries = load_queries()
    queries = all_queries[:limit] if limit else all_queries

    total_recall = 0.0
    total_mrr = 0.0
    hits = 0
    paper_hits = 0

    for item in queries:
        query = item["query"]
        expected_keywords = [k.lower() for k in item["expected_keywords"]]

        nodes = retriever.retrieve(query)
        retrieved_texts = [n.node.get_content().lower() for n in nodes]

        found = 0
        first_rank = None
        for kw in expected_keywords:
            for rank, text in enumerate(retrieved_texts, 1):
                if kw in text:
                    found += 1
                    if first_rank is None:
                        first_rank = rank
                    break

        recall = found / len(expected_keywords) if expected_keywords else 0
        total_recall += recall

        if first_rank is not None:
            total_mrr += 1.0 / first_rank
            hits += 1

        if _expected_papers_in_top_k(nodes, item["expected_papers"]):
            paper_hits += 1

    n = len(queries)
    return {
        "Recall@{}".format(top_k): round(total_recall / n, 3),
        "MRR": round(total_mrr / n, 3),
        "Hit Rate": round(hits / n, 3),
        "Paper Hit Rate": round(paper_hits / n, 3),
        "queries_tested": n,
        "top_k": top_k,
    }


if __name__ == "__main__":
    print("=" * 55)
    print(f"RAG 检索评估 — {len(load_queries())} 条评测集")
    print("=" * 55)
    for k in [5, 10, 20]:
        result = evaluate_retrieval(k)  # 跑全部评测集
        print(f"top_k={k:>2}: Recall@{k}={result['Recall@{}'.format(k)]:.3f}  "
              f"MRR={result['MRR']:.3f}  Hit={result['Hit Rate']:.3f}  "
              f"PaperHit={result['Paper Hit Rate']:.3f}")
