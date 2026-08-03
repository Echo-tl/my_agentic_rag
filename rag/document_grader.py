"""
Document Grader: LLM 批量判断检索结果是否与查询相关，过滤不相关文档。

与旧版不同：一次 LLM 调用处理多个文档（批量打分），而非逐文档调用。
"""

from typing import List, Optional
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

BATCH_GRADER_PROMPT = """You are a document relevance grader. Given a query and several documents, determine which documents are relevant to answering the query.

Query: {query}

Documents:
{documents}

For each document, reply with its number and YES or NO:
1: YES/NO
2: YES/NO
...

Output ONLY the numbered list. Example:
1: YES
2: NO
3: YES"""


class DocumentGrader(BaseNodePostprocessor):
    """LLM-based batch relevance grader.

    Sends ALL nodes in one batch to the LLM for YES/NO grading.
    Much faster than per-document grading.

    Args:
        llm: LLM instance for grading
        top_n: Max nodes to keep after grading (None = keep all YES)
    """

    llm: LLM = Field(description="LLM for document grading")

    @classmethod
    def class_name(cls) -> str:
        return "DocumentGrader"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Batch-grade all nodes in one LLM call, filter to YES."""
        if not nodes or query_bundle is None or len(nodes) == 0:
            return nodes
        if len(nodes) == 1:
            # Single document: simple YES/NO
            return self._grade_single(nodes[0], query_bundle)

        return self._grade_batch(nodes, query_bundle)

    def _grade_single(
        self, node: NodeWithScore, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        prompt = (
            f"Query: {query_bundle.query_str}\n"
            f"Document: {node.node.get_content()[:2000]}\n"
            f"Relevant (YES/NO):"
        )
        try:
            response = self.llm.complete(prompt)
            if str(response).strip().upper().startswith("YES"):
                return [node]
        except Exception:
            return [node]  # fail-open
        return []

    def _grade_batch(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        # Build document list with truncated content
        docs_text = []
        for i, node in enumerate(nodes, 1):
            content = node.node.get_content()[:1000]  # Shorter per doc for batch
            docs_text.append(f"--- Document {i} ---\n{content}")

        prompt = BATCH_GRADER_PROMPT.format(
            query=query_bundle.query_str,
            documents="\n\n".join(docs_text),
        )

        try:
            response = self.llm.complete(prompt)
            # Parse "1: YES", "2: NO", etc.
            yes_indices = set()
            for line in str(response).strip().split("\n"):
                line = line.strip()
                for prefix in ["- ", "* ", ""]:
                    if prefix and line.startswith(prefix):
                        line = line[len(prefix):]
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0].strip()) - 1
                        verdict = parts[1].strip().upper()
                        if verdict.startswith("YES"):
                            yes_indices.add(idx)
                    except ValueError:
                        continue

            kept = [nodes[i] for i in yes_indices if 0 <= i < len(nodes)]

            # Fallback: if parsing returned nothing, keep all (fail-open)
            if not kept:
                return nodes
            return kept

        except Exception:
            return nodes  # fail-open
