"""
引用命中率对比评测。

- Baseline: 原始检索器（index.as_retriever，无改写/重排）
- Full pipeline: search_knowledge_base（查询改写 + 并行检索 + 相似度过滤 + LLM 重排）

对同一评测集，统计"正确论文被引用/命中"的比例，得出完整管线的引用命中率提升。
"""

import os
import re
import sys
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.rag_eval import load_queries, _expected_papers_in_top_k
from rag.index import get_index
from config import config
from database.qdrant import get_vector_store
from models.embedding import create_embedding
from tools.rag.llamaindex_tool import search_knowledge_base


def extract_cited_files(result: str) -> set:
    """从工具输出解析被引用的来源文件名：`[来源: <file> | 第<N>页 | 相关度: <x>]`。"""
    return set(re.findall(r"\[来源: (.+?) \|", result))


def papers_cited(cited_files: set, expected_papers: list) -> bool:
    """预期论文是否全部出现在引用来源中（按文件名子串匹配）。"""
    return all(any(p.lower() in f.lower() for f in cited_files) for p in expected_papers)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    all_queries = load_queries()
    queries = all_queries[:limit] if limit else all_queries
    n = len(queries)
    print(f"评测集: {n} 条（{'全部' if not limit else '前 ' + str(limit) + ' 条'}）")

    # ── Baseline：原始检索器 ──
    index = get_index(config.paths.data_dir, get_vector_store(), create_embedding(config.embedding))
    for top_k in (5, 10):
        retriever = index.as_retriever(similarity_top_k=top_k)
        hits = sum(
            1 for q in queries
            if _expected_papers_in_top_k(retriever.retrieve(q["query"]), q["expected_papers"])
        )
        print(f"Baseline 论文命中率@{top_k}: {hits}/{n} = {hits / n * 100:.1f}%")

    # ── Full pipeline：search_knowledge_base 的引用命中 ──
    t0 = time.time()
    hits = 0
    no_result = 0
    for i, q in enumerate(queries, 1):
        result = search_knowledge_base.invoke({"query": q["query"]})
        if "No relevant documents" in result:
            no_result += 1
        if papers_cited(extract_cited_files(result), q["expected_papers"]):
            hits += 1
        if i % 10 == 0:
            print(f"  进度 {i}/{n} … ({time.time() - t0:.0f}s)")
    print(f"Full pipeline 引用命中率: {hits}/{n} = {hits / n * 100:.1f}%  "
          f"(无结果 {no_result} 条, 耗时 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
