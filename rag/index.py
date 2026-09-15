"""
索引构建 / 加载 / 增量更新。

- 首次：全量摄取 data/ 并持久化到 Qdrant + storage/
- 之后：加载已有索引，通过文件哈希做增量摄取——只对新文件 embedding 并插入、
  对变更文件删除旧分块后重插入、对已删除文件清理分块，绝不重复 embedding 全部文档。
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Set

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

from config import config
from database.qdrant import get_client, collection_name

logger = logging.getLogger("agentic_rag")

# 已索引文件的哈希记录文件名（位于 persist_dir 下）
_HASH_FILE = "indexed_files.json"
_SUPPORTED_EXTS = {".md", ".txt", ".pdf"}


# ── 文件哈希 ─────────────────────────────────────────────────
def _hash_file(path: Path) -> str:
    """文件内容 MD5，用于检测新增/变更。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_hash_record() -> Dict[str, str]:
    """{file_name: md5}：已索引文件的哈希记录。

    委托给 doc_registry：本地 JSON 仍是**判定真源**（摄取决策必须同步且确定），
    MySQL 的 documents 表是有记录时的查询副本、无本地文件时的兜底来源。
    """
    from persistence.doc_registry import load_hash_record

    return load_hash_record()


def _save_hash_record(record: Dict[str, str]):
    """同步写本地 JSON（原子替换），后台镜像进 MySQL documents 表。"""
    from persistence.doc_registry import save_hash_record

    save_hash_record(record)


def _scan_data_files(data_dir: Path) -> Dict[str, Path]:
    """扫描 data/ 下支持的文档，返回 {file_name: path}。"""
    if not data_dir.exists():
        return {}
    return {p.name: p for p in data_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS}


def _existing_file_names_from_qdrant() -> Set[str]:
    """从 Qdrant 已存 payload 取出去重后的 file_name（用于老版本升级时初始化哈希记录）。"""
    client = get_client()
    names: Set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=["file_name"],
            with_vectors=False,
        )
        for pt in points:
            fn = pt.payload.get("file_name")
            if fn:
                names.add(fn)
        if not offset:
            break
    return names


def _delete_points_by_file_name(file_name: str):
    """从 Qdrant 删除某文件的所有分块点（文件被删除/变更时清理旧分块）。"""
    get_client().delete(
        collection_name=collection_name,
        points_selector=FilterSelector(filter=Filter(
            must=[FieldCondition(key="file_name", match=MatchValue(value=file_name))]
        )),
    )


# ── 构建 / 加载 ──────────────────────────────────────────────
def build_index(nodes, vector_store, embed_model) -> VectorStoreIndex:
    """首次构建"""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        storage_context=storage_context,
    )

    # 构建完后持久化
    index.storage_context.persist(
        persist_dir=str(config.paths.persist_dir)
    )
    return index


def load_index(vector_store, embed_model) -> VectorStoreIndex:
    """从磁盘加载"""
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(config.paths.persist_dir),
    )

    return load_index_from_storage(
        storage_context=storage_context,
        embed_model=embed_model,
    )


# ── 增量摄取 ────────────────────────────────────────────────
def incremental_ingest(index: VectorStoreIndex, data_dir: Path, embed_model) -> VectorStoreIndex:
    """哈希检测增量更新：只对新/变更文件 embedding 并插入，删除的文件清理分块。

    返回更新后的 index。
    """
    files = _scan_data_files(data_dir)
    record = _load_hash_record()

    # 老版本升级：storage 已存在但无哈希记录 → 用 Qdrant 现有 file_name 初始化，不重复摄取
    if not record:
        qdrant_names = _existing_file_names_from_qdrant()
        if not qdrant_names and files:
            # storage 在但 Qdrant 空（不一致）→ 全量摄取一次
            from rag.ingestion import run_ingestion
            nodes = run_ingestion(data_dir, embed_model)
            index.insert_nodes(nodes)
            index.storage_context.persist(persist_dir=str(config.paths.persist_dir))
            logger.warning(f"[incremental] Qdrant 为空但 storage 存在，已重新全量摄取 {len(nodes)} 个分块")
        else:
            # 清理已不在 data/ 的旧点，并把当前文件全部记为已索引
            for name in qdrant_names - set(files):
                _delete_points_by_file_name(name)
            logger.info(f"[incremental] 初始化哈希记录：{len(files)} 个文件视为已索引（未重复 embedding）")
        record = {name: _hash_file(path) for name, path in files.items()}
        _save_hash_record(record)
        # 老版本升级时 Qdrant 里已有分块 → 同样要失效可能存在的旧缓存
        _notify_kb_changed([], [], qdrant_names - set(files), files)
        return index

    new_files = [n for n in files if n not in record]
    changed_files = [n for n in files if n in record and record[n] != _hash_file(files[n])]
    removed_files = [n for n in record if n not in files]

    if not (new_files or changed_files or removed_files):
        # 无变更也要把本地记录镜像一次 MySQL：老部署首次启用 MySQL 时
        # documents 表还是空的，而这里正是唯一的「启动但没动过文件」路径。
        # 后台队列投递，不阻塞也不影响摄取判定。
        try:
            from persistence.doc_registry import mirror_hash_record

            mirror_hash_record(record)
        except Exception:
            pass
        return index

    from rag.ingestion import run_ingestion

    # 1) 清理：删除已移除 / 已变更文件的旧分块
    for name in removed_files + changed_files:
        _delete_points_by_file_name(name)
        logger.info(f"[incremental] 清理旧分块：{name}")

    # 2) 摄取：只对新/变更文件做 embedding + 插入
    to_ingest = [files[n] for n in new_files + changed_files if n in files]
    if to_ingest:
        nodes = run_ingestion(data_dir, embed_model, input_files=to_ingest)
        index.insert_nodes(nodes)
        index.storage_context.persist(persist_dir=str(config.paths.persist_dir))
        logger.info(
            f"[incremental] 新增/更新 {len(nodes)} 个分块（{len(to_ingest)} 个文件）："
            f"{[p.name for p in to_ingest]}"
        )

    # 3) 更新哈希记录
    for n in removed_files:
        record.pop(n, None)
    for n in new_files + changed_files:
        record[n] = _hash_file(files[n])
    _save_hash_record(record)

    # 4) 知识库变了 → 让热点问答缓存整体失效。
    #    必须放在最后：中途 bump 会让摄取失败留下的半成品状态也"看起来是新版本"。
    _notify_kb_changed(new_files, changed_files, removed_files, files)

    return index


def _notify_kb_changed(added, changed, removed, files=None):
    """摄取后失效问答缓存。Redis 不可用时内部 no-op，不影响摄取。"""
    try:
        from persistence.doc_registry import notify_kb_changed

        notify_kb_changed(added=added, changed=changed, removed=removed,
                          total=len(files or {}))
    except Exception as e:  # 缓存失效失败绝不能影响摄取结果
        logger.debug(f"[incremental] kb_version 自增失败（缓存靠 TTL 过期）: {e}")


def get_index(data_dir, vector_store, embed_model) -> VectorStoreIndex:
    """统一入口：有持久化就加载 + 增量更新，没有就首次全量构建。"""
    if config.paths.persist_dir.exists():
        index = load_index(vector_store, embed_model)
        return incremental_ingest(index, data_dir, embed_model)
    else:
        from rag.ingestion import run_ingestion
        nodes = run_ingestion(data_dir, embed_model)
        index = build_index(nodes, vector_store, embed_model)
        # 首次构建：把当前所有文件记入哈希记录
        all_files = _scan_data_files(data_dir)
        _save_hash_record({name: _hash_file(path) for name, path in all_files.items()})
        _notify_kb_changed(list(all_files.keys()), [], [], all_files)
        return index
