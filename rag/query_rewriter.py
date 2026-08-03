"""
Query Rewrite: 用 LLM 把用户查询改写为多个不同角度的检索词，提高召回率。
"""

from typing import List
from llama_index.core.llms.llm import LLM

QUERY_REWRITE_PROMPT = """You are a query rewriting assistant. Given a user query about research papers, generate {count} keyword-focused search queries.

Rules:
- Each query should use specific technical keywords that might appear in paper titles and abstracts
- Focus on different terminology, frameworks, or methods related to the topic
- Think about synonyms, related concepts, and specific technique names
- Be short and keyword-dense (5-10 words max)

Output ONLY the queries, one per line. No numbering, no explanation.

Original query: {query}

Keyword-dense search queries:"""


def rewrite_query(llm: LLM, query: str, count: int = 3) -> List[str]:
    """Generate alternative search queries from the original query.

    Args:
        llm: The LLM instance to use for rewriting
        query: Original user query
        count: Number of alternative queries to generate

    Returns:
        List of rewritten queries (includes the original query as first element)
    """
    prompt = QUERY_REWRITE_PROMPT.format(query=query, count=count)

    try:
        response = llm.complete(prompt)
        # Parse response: one query per line, skip empty lines
        lines = [line.strip() for line in str(response).strip().split("\n") if line.strip()]
        # Remove any numbering prefix like "1. " or "- "
        cleaned = []
        for line in lines:
            # Strip common prefixes
            for prefix in ["- ", "* ", "• "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
            # Strip numbering like "1. " or "1) "
            import re
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            cleaned.append(line.strip())

        # Deduplicate while preserving order
        seen = set()
        result = []
        # Always include original query first
        result.append(query)
        seen.add(query.lower())
        for q in cleaned:
            if q.lower() not in seen and len(q) > 3:
                result.append(q)
                seen.add(q.lower())

        return result[: count + 1]  # original + N rewrites
    except Exception:
        # Fallback: just return the original query
        return [query]
