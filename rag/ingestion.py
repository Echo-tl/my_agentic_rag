"""
把 data/ 目录下的原始文档（PDF、Markdown 等），经过 IngestionPipeline 处理，变成带 embedding 的 Node 列表。
"""

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import BaseNode
from pathlib import Path
from typing import List, Optional

from config import config

def run_ingestion(
    data_dir: Path,
    embed_model,
    input_files: Optional[List[Path]] = None,
) -> List[BaseNode]:
    # 第 1 步：加载文档（input_files 给定则只处理这些文件，用于增量摄取）
    reader_kwargs = {"required_exts": [".md", ".txt", ".pdf"]}
    if input_files:
        reader_kwargs["input_files"] = [str(p) for p in input_files]
    else:
        reader_kwargs["input_dir"] = str(data_dir)
    reader = SimpleDirectoryReader(**reader_kwargs)
    documents = reader.load_data()

    # 第 2 步：创建管道
    pipeline = IngestionPipeline(
       transformations=[
            SentenceSplitter(
                chunk_size=config.chunk.chunk_size,
                chunk_overlap=config.chunk.chunk_overlap,
            ),
            embed_model,
        ],
    )

    # 第 3 步：跑管道
    nodes = pipeline.run(documents=documents)
    return nodes



