import atexit
from llama_index.embeddings.ollama import OllamaEmbedding
from config import EmbeddingConfig

def create_embedding(cfg: EmbeddingConfig):
    emb = OllamaEmbedding(
        model_name=cfg.model_name,
        base_url=cfg.base_url,
        embed_batch_size=cfg.batch_size,
    )
    # 进程退出时关闭 httpx 连接，避免 ResourceWarning
    def _cleanup():
        try:
            if emb._client is not None and hasattr(emb._client, '_client'):
                emb._client._client.close()
        except Exception:
            pass
    atexit.register(_cleanup)
    return emb