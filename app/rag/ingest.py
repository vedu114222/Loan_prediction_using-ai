"""
RAG ingest pipeline.

Usage:
    python -m app.rag.ingest                        # ingest all docs in policy_docs/
    python -m app.rag.ingest --dir path/to/docs     # custom directory
    python -m app.rag.ingest --reset                # wipe table before ingesting

This script:
  1. Reads every .md or .txt file in the target directory.
  2. Chunks the text into overlapping word windows.
  3. Encodes each chunk with sentence-transformers (all-MiniLM-L6-v2 → 384-dim).
  4. Inserts (content, embedding, source) rows into the policy_chunks table.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

load_dotenv()

# Lazy import so we can run this script without starting the full app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.utils.db import engine  # noqa: E402

_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_st_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        print(f"Loading embedding model: {_EMBED_MODEL}")
        _st_model = SentenceTransformer(_EMBED_MODEL)
    return _st_model


def chunk_text(text_: str, size: int = 400, overlap: int = 60) -> list[str]:
    """
    Split text into overlapping word-window chunks.

    Args:
        text_:   Raw document text.
        size:    Chunk size in words.
        overlap: Overlap between consecutive chunks in words.

    Returns:
        List of text chunks.
    """
    words = text_.split()
    if not words:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def reset_table() -> None:
    """Delete all rows from policy_chunks."""
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM policy_chunks"))
        conn.commit()
    print("✓ policy_chunks table cleared.")


def ingest_file(path: Path) -> int:
    """
    Chunk, embed, and insert a single document.

    Returns:
        Number of chunks inserted.
    """
    model = _get_model()
    content = path.read_text(encoding="utf-8")
    chunks = chunk_text(content)
    if not chunks:
        print(f"  ⚠ Skipped {path.name} (empty file)")
        return 0

    vectors = model.encode(chunks, show_progress_bar=False).tolist()

    with engine.connect() as conn:
        for chunk, vector in zip(chunks, vectors):
            conn.execute(
                text(
                    "INSERT INTO policy_chunks (content, embedding, source) "
                    "VALUES (:c, :v::vector, :s)"
                ),
                {"c": chunk, "v": str(vector), "s": path.name},
            )
        conn.commit()

    print(f"  ✓ {path.name}: {len(chunks)} chunks ingested")
    return len(chunks)


def ingest_dir(dir_path: str = "policy_docs") -> None:
    """Ingest all .md and .txt files from a directory."""
    root = Path(dir_path)
    if not root.exists():
        print(f"Directory not found: {root.resolve()}")
        sys.exit(1)

    files = list(root.glob("*.md")) + list(root.glob("*.txt"))
    if not files:
        print(f"No .md or .txt files found in {root.resolve()}")
        return

    print(f"Ingesting {len(files)} file(s) from {root.resolve()} …")
    total = 0
    for f in sorted(files):
        total += ingest_file(f)
    print(f"\n✓ Done — {total} total chunks ingested into policy_chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest policy docs into pgvector")
    parser.add_argument("--dir", default="policy_docs", help="Directory of policy files")
    parser.add_argument("--reset", action="store_true", help="Clear table before ingesting")
    args = parser.parse_args()

    if args.reset:
        reset_table()

    ingest_dir(args.dir)
