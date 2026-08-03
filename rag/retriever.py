from llama_index.core.postprocessor import SimilarityPostprocessor
from config import config

def get_retriever(index):
    # →VectorIndexRetriever(返回 top_k=10 个 Node)
    retriever = index.as_retriever(similarity_top_k=config.retrieval.similarity_top_k)

    postprocessor = SimilarityPostprocessor(
        similarity_cutoff=config.retrieval.similarity_cutoff
    )

    # 返回配置好的 retriever（还没跑检索，只是配置好了）
    return retriever, postprocessor
