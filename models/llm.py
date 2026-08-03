import atexit
from llama_index.llms.ollama import Ollama
from config import LLMConfig

def create_llm(cfg:LLMConfig):
    llm = Ollama(
        model=cfg.model,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        context_window=cfg.context_window,
        request_timeout=cfg.request_timeout,
        additional_kwargs={
            "options": {
                "num_predict": cfg.max_tokens, # max_tokens需要通过 additional_kwargs 传递给 Ollama
            }
        },
    )
    # 进程退出时关闭 httpx 连接，避免 ResourceWarning
    def _cleanup():
        try:
            if llm._client is not None and hasattr(llm._client, '_client'):
                llm._client._client.close()
        except Exception:
            pass
    atexit.register(_cleanup)
    return llm