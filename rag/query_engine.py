from llama_index.core.query_engine import RetrieverQueryEngine
from rag.retriever import get_retriever
from rag.reranker import get_reranker
from rag.document_grader import DocumentGrader
from config import config

def get_query_engine(index, llm):
    # 1. 获取配置好的 retriever + postprocessor
    retriever, postprocessor = get_retriever(index)

    # 2. 创建 reranker
    reranker = get_reranker(llm)

    # 3. 创建 document grader（可选）
    postprocessors = [postprocessor]  # SimilarityPostprocessor 先过滤低分
    if config.retrieval.enable_document_grading:
        grader = DocumentGrader(llm=llm)
        postprocessors.append(grader)  # LLM 逐条判断相关性
    postprocessors.append(reranker)    # LLMRerank 精选 top_n

    # 4. 组装 RetrieverQueryEngine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        node_postprocessors=postprocessors,
        response_mode="compact",
    )

    return query_engine