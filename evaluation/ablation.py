"""
Ablation Study: 对比不同配置组合的检索效果。

Baseline: 纯 dense retrieval（无任何后处理）
Ablation variants:
  + Query Rewrite
  + Document Grader
  + LLMRerank
  Full: 全部开启
"""

import json
import os
from pathlib import Path
from typing import List
from llama_index.core.schema import QueryBundle

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

from rag.index import get_index
from rag.retriever import get_retriever
from rag.reranker import get_reranker
from rag.document_grader import DocumentGrader
from rag.query_rewriter import rewrite_query
from config import config
from database.qdrant import get_vector_store
from models.embedding import create_embedding
from models.llm import create_llm


def load_queries() -> list:
    path = Path(__file__).parent / "datasets" / "test_queries.json"
    return json.loads(path.read_text(encoding="utf-8"))["queries"]


def compute_metrics(retrieved_nodes: list, expected_keywords: list[str]) -> dict:
    """计算单条查询的 Recall 和 MRR。"""
    texts = [n.node.get_content().lower() for n in retrieved_nodes]
    expected = [k.lower() for k in expected_keywords]

    found = 0
    first_rank = None
    for kw in expected:
        for rank, text in enumerate(texts, 1):
            if kw in text:
                found += 1
                if first_rank is None:
                    first_rank = rank
                break

    return {
        "recall": found / len(expected) if expected else 0,
        "mrr": 1.0 / first_rank if first_rank else 0,
        "hit": 1 if first_rank else 0,
    }


def run_baseline(queries, top_k=10) -> dict:
    """Baseline: 纯 dense retrieval。"""
    index = get_index(config.paths.data_dir, get_vector_store(), create_embedding(config.embedding))
    retriever, _ = get_retriever(index)
    retriever = index.as_retriever(similarity_top_k=top_k)

    total_recall, total_mrr, total_hit = 0, 0, 0
    for item in queries:
        nodes = retriever.retrieve(item["query"])
        m = compute_metrics(nodes, item["expected_keywords"])
        total_recall += m["recall"]
        total_mrr += m["mrr"]
        total_hit += m["hit"]

    n = len(queries)
    return {"Recall": round(total_recall / n, 3), "MRR": round(total_mrr / n, 3), "Hit Rate": round(total_hit / n, 3)}


def run_with_query_rewrite(queries, top_k=10) -> dict:
    """+Query Rewrite: 多查询重写 + 并行检索合并。"""
    index = get_index(config.paths.data_dir, get_vector_store(), create_embedding(config.embedding))
    llm = create_llm(config.llm)
    retriever = index.as_retriever(similarity_top_k=top_k)

    total_recall, total_mrr, total_hit = 0, 0, 0
    for item in queries:
        queries_list = rewrite_query(llm, item["query"], 2)
        seen, all_nodes = set(), []
        for q in queries_list:
            for n in retriever.retrieve(q):
                if n.node.node_id not in seen:
                    all_nodes.append(n)
                    seen.add(n.node.node_id)
        all_nodes.sort(key=lambda n: n.score or 0, reverse=True)
        m = compute_metrics(all_nodes[:top_k], item["expected_keywords"])
        total_recall += m["recall"]
        total_mrr += m["mrr"]
        total_hit += m["hit"]

    n = len(queries)
    return {"Recall": round(total_recall / n, 3), "MRR": round(total_mrr / n, 3), "Hit Rate": round(total_hit / n, 3)}


def run_with_document_grader(queries, top_k=10) -> dict:
    """+DocumentGrader: 批量 YES/NO 过滤。"""
    index = get_index(config.paths.data_dir, get_vector_store(), create_embedding(config.embedding))
    llm = create_llm(config.llm)
    retriever = index.as_retriever(similarity_top_k=top_k)
    grader = DocumentGrader(llm=llm)

    total_recall, total_mrr, total_hit = 0, 0, 0
    for item in queries:
        nodes = retriever.retrieve(item["query"])
        qb = QueryBundle(item["query"])
        nodes = grader.postprocess_nodes(nodes, qb)
        m = compute_metrics(nodes, item["expected_keywords"])
        total_recall += m["recall"]
        total_mrr += m["mrr"]
        total_hit += m["hit"]

    n = len(queries)
    return {"Recall": round(total_recall / n, 3), "MRR": round(total_mrr / n, 3), "Hit Rate": round(total_hit / n, 3)}


def run_with_llm_rerank(queries, top_k=10) -> dict:
    """+LLMRerank: LLM 精选 Top-N。"""
    index = get_index(config.paths.data_dir, get_vector_store(), create_embedding(config.embedding))
    llm = create_llm(config.llm)
    retriever, sim_pp = get_retriever(index)
    retriever = index.as_retriever(similarity_top_k=top_k)
    reranker = get_reranker(llm)

    total_recall, total_mrr, total_hit = 0, 0, 0
    for item in queries:
        nodes = retriever.retrieve(item["query"])
        qb = QueryBundle(item["query"])
        nodes = sim_pp.postprocess_nodes(nodes, qb)
        nodes = reranker.postprocess_nodes(nodes, qb)
        m = compute_metrics(nodes, item["expected_keywords"])
        total_recall += m["recall"]
        total_mrr += m["mrr"]
        total_hit += m["hit"]

    n = len(queries)
    return {"Recall": round(total_recall / n, 3), "MRR": round(total_mrr / n, 3), "Hit Rate": round(total_hit / n, 3)}


def run_full(queries, top_k=10) -> dict:
    """Full: 全部开启（Query Rewrite + DocumentGrader + LLMRerank）。"""
    index = get_index(config.paths.data_dir, get_vector_store(), create_embedding(config.embedding))
    llm = create_llm(config.llm)
    retriever = index.as_retriever(similarity_top_k=top_k)
    _, sim_pp = get_retriever(index)
    grader = DocumentGrader(llm=llm)
    reranker = get_reranker(llm)

    total_recall, total_mrr, total_hit = 0, 0, 0
    for item in queries:
        queries_list = rewrite_query(llm, item["query"], 2)
        seen, all_nodes = set(), []
        for q in queries_list:
            for n in retriever.retrieve(q):
                if n.node.node_id not in seen:
                    all_nodes.append(n)
                    seen.add(n.node.node_id)
        all_nodes.sort(key=lambda n: n.score or 0, reverse=True)

        qb = QueryBundle(item["query"])
        nodes = sim_pp.postprocess_nodes(all_nodes, qb)
        nodes = grader.postprocess_nodes(nodes, qb)
        nodes = reranker.postprocess_nodes(nodes, qb)
        m = compute_metrics(nodes[:top_k], item["expected_keywords"])
        total_recall += m["recall"]
        total_mrr += m["mrr"]
        total_hit += m["hit"]

    n = len(queries)
    return {"Recall": round(total_recall / n, 3), "MRR": round(total_mrr / n, 3), "Hit Rate": round(total_hit / n, 3)}


if __name__ == "__main__":
    print("=" * 65)
    print("Ablation Study — RAG Pipeline 组件贡献分析")
    print("=" * 65)

    queries = load_queries()
    print(f"\n评测集: {len(queries)} 条查询\n")

    # 使用小样本快速对比（避免全量耗时过长）
    sample = queries[:20]

    results = {
        "Baseline (纯检索)": run_baseline(sample),
        "+ Query Rewrite": run_with_query_rewrite(sample),
        "+ Document Grader": run_with_document_grader(sample),
        "+ LLMRerank": run_with_llm_rerank(sample),
        "Full Pipeline": run_full(sample),
    }

    print(f"{'配置':<25} {'Recall':>8} {'MRR':>8} {'Hit Rate':>10}")
    print("-" * 55)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['Recall']:>8.3f} {metrics['MRR']:>8.3f} {metrics['Hit Rate']:>10.3f}")
