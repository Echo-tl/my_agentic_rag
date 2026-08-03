from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from langchain_core.tools import tool
from llama_index.core.schema import QueryBundle

from rag.query_rewriter import rewrite_query
from rag.index import get_index
from config import config
from database.qdrant import get_vector_store
from models.embedding import create_embedding
from models.llm import create_llm
from observability.tracing import record_tool

# ── 懒初始化运行时 ─────────────────────────────────────────────
# 所有重量级资源（Qdrant 连接、Embedding、索引、LLM）都在首次调用时才构建，
# 避免模块 import 即连库/加载索引。这样单元测试、API 启动、多进程部署
# 都不会被 Qdrant 不可用或加载耗时阻塞。
_runtime = None
_runtime_lock = threading.Lock()


def build_runtime():
    """构建检索运行时：index + llm + retriever + postprocessor 链（只构建一次）。"""
    index = get_index(
        data_dir=config.paths.data_dir,
        vector_store=get_vector_store(),
        embed_model=create_embedding(config.embedding),
    )
    llm = create_llm(config.llm)

    from rag.retriever import get_retriever
    from rag.reranker import get_reranker
    from rag.document_grader import DocumentGrader

    retriever, sim_postprocessor = get_retriever(index)
    reranker = get_reranker(llm)

    # 组装 postprocessor 链（不含 retriever，re-used 每次查询）
    postprocessors = [sim_postprocessor]
    if config.retrieval.enable_document_grading:
        postprocessors.append(DocumentGrader(llm=llm))
    postprocessors.append(reranker)

    return {
        "llm": llm,
        "retriever": retriever,
        "postprocessors": postprocessors,
    }


def get_runtime():
    """线程安全的懒加载单例：首次调用时构建一次，之后复用。"""
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = build_runtime()
    return _runtime


# ── 并行检索 ─────────────────────────────────────────────────
def _parallel_retrieve(queries: list[str], top_k: int, max_workers: int = 3) -> list:
    """并行检索多个查询，合并结果，去重 + 来源多样化。"""
    retriever = get_runtime()["retriever"]
    seen_ids = set()
    seen_sources = {}  # 每个来源文件最多保留 max_per_source 个节点
    all_nodes = []

    def _retrieve_one(query: str):
        return retriever.retrieve(query)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_retrieve_one, q): q for q in queries}
        for future in as_completed(futures):
            try:
                nodes = future.result()
                for node in nodes:
                    nid = node.node.node_id
                    if nid in seen_ids:
                        continue
                    source = node.node.metadata.get("file_name", "unknown")
                    # 每个来源最多保留 5 个 chunks，保证多样性
                    if seen_sources.get(source, 0) >= 5:
                        continue
                    all_nodes.append(node)
                    seen_ids.add(nid)
                    seen_sources[source] = seen_sources.get(source, 0) + 1
            except Exception:
                pass

    # 按 similarity score 降序排列
    all_nodes.sort(key=lambda n: n.score or 0, reverse=True)
    return all_nodes[:top_k * 2]  # 多留一些给 postprocessor


# ── Tool ─────────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """搜索本地论文知识库（英文论文）。包含 AutoGen, ReAct, Reflexion, AMOR, Voyager 等 agent 相关论文全文。请使用英文关键词查询（如 'LLM agent survey', 'multi-agent framework', 'reflexion reinforcement learning'）"""

    _start = time.perf_counter()
    try:
        return _search_knowledge_base_impl(query)
    finally:
        record_tool("search_knowledge_base", {"query": query}, _start)


def _search_knowledge_base_impl(query: str) -> str:
    runtime = get_runtime()

    # Step 1: Query Rewrite（可选）
    if config.retrieval.enable_query_rewrite:
        queries = rewrite_query(runtime["llm"], query, config.retrieval.query_rewrite_count)
    else:
        queries = [query]

    # Step 2: 并行检索 + 合并去重
    nodes = _parallel_retrieve(queries, config.retrieval.similarity_top_k)
    if not nodes:
        return "No relevant documents found in the knowledge base."

    # Step 3: postprocessor 链（过滤 + 精选）— 只跑一次
    query_bundle = QueryBundle(query)
    for pp in runtime["postprocessors"]:
        nodes = pp.postprocess_nodes(nodes, query_bundle)
        if not nodes:
            break

    if not nodes:
        return "No relevant documents found after filtering."

    # Step 4: 返回原始检索文本（不做 LLM 合成，防止编造），含页码引用
    chunks = []
    for i, node in enumerate(nodes):
        text = node.node.get_content().strip()[:1500]
        score = node.score
        meta = node.node.metadata
        source = meta.get("file_name", "unknown")
        page = meta.get("page_label", "?")
        chunks.append(f"[来源: {source} | 第{page}页 | 相关度: {score:.3f}]\n{text}")

    return "\n\n---\n\n".join(chunks)
