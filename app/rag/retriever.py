"""
RAG retriever — cosine-similarity search over policy_chunks in pgvector.

Usage (quick test):
    python -c "
    from app.rag.retriever import search_policy_chunks
    import json
    results = search_policy_chunks('minimum credit score for approval')
    print(json.dumps(results, indent=2))
    "
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

load_dotenv()

_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    return SentenceTransformer(_EMBED_MODEL)


def search_policy_chunks(query: str, top_k: int = 4) -> list[dict]:
    """
    Embed `query` and return the top-k most semantically similar policy chunks.

    Args:
        query:  Natural-language question or keyword string.
        top_k:  Number of results to return (default 4).

    Returns:
        List of dicts with keys: content, source, score (0-1, higher = more relevant).
    """
    from app.utils.db import engine  # local import to avoid circular at module level

    model = _get_model()
    vector = model.encode(query).tolist()

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT content, source,
                       1 - (embedding <=> :v::vector) AS score
                FROM policy_chunks
                ORDER BY embedding <=> :v::vector
                LIMIT :k
                """
            ),
            {"v": str(vector), "k": top_k},
        ).fetchall()

    return [
        {
            "content": r.content,
            "source": r.source,
            "score": round(float(r.score), 4),
        }
        for r in rows
    ]
