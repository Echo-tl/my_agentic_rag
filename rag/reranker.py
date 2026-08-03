from llama_index.core.postprocessor import LLMRerank
from config import config

def get_reranker(llm):
    return LLMRerank(
        llm=llm,
        top_n=config.retrieval.reranker_top_n,
        choice_batch_size=5,  # 每批 5 个，避免 prompt 过长导致 LLM 解析失败
    )