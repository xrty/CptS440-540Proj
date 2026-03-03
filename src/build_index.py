"""
Offline index builder for the RAG Academic Paper QA System.

Run this script ONCE after ingest.py to produce:
  - indexes/bm25.pkl          — BM25Okapi serialized with pickle
  - indexes/faiss_index/      — FAISS flat inner-product index + chunk-ID metadata

These artifacts are loaded at query time by retrieval.py.
Rebuilding is fast for BM25 (~seconds) and moderate for FAISS
(~2-5 min for 10 k chunks on CPU with all-MiniLM-L6-v2).

Usage:
    # Build indexes (skips if already exist)
    python src/build_index.py

    # Force rebuild even if indexes exist
    python src/build_index.py --rebuild
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BM25_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    INDEX_DIR,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Chunk loading ──────────────────────────────────────────────────────────────

def load_chunks(chunks_path: Path = CHUNKS_PATH) -> List[Dict]:
    """
    Load chunk records from the JSON file produced by ingest.py.

    Args:
        chunks_path: Path to chunks.json.

    Returns:
        List of chunk dicts. Returns empty list if the file does not exist,
        with a clear message directing the user to run ingest.py first.

    Example:
        >>> chunks = load_chunks()
        >>> print(chunks[0]["chunk_id"])
    """
    if not chunks_path.exists():
        logger.error(
            f"chunks.json not found at {chunks_path}.\n"
            f"  → Run first:  python src/ingest.py"
        )
        return []

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path.name}")
    return chunks


# ── BM25 index ─────────────────────────────────────────────────────────────────

def build_bm25_index(chunks: List[Dict], output_path: Path = BM25_INDEX_PATH) -> BM25Okapi:
    """
    Build a BM25Okapi sparse index over all chunk texts and serialize it to disk.

    Tokenization is simple whitespace splitting — sufficient for BM25 keyword
    matching. The serialized file stores both the BM25Okapi object and the
    ordered list of chunk_ids so retrieval.py can map scores back to chunks.

    Args:
        chunks:      List of chunk dicts (must have "chunk_text" and "chunk_id").
        output_path: Destination .pkl file path.

    Returns:
        The constructed BM25Okapi object.

    Example:
        >>> chunks = load_chunks()
        >>> bm25 = build_bm25_index(chunks)
    """
    logger.info(f"Building BM25 index over {len(chunks)} chunks...")

    tokenized_corpus = [c["chunk_text"].lower().split() for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

    logger.info(f"BM25 index saved → {output_path}")
    return bm25


# ── FAISS dense index ──────────────────────────────────────────────────────────

def build_faiss_index(
    chunks: List[Dict],
    model_name: str = EMBEDDING_MODEL,
    index_path: Path = FAISS_INDEX_PATH,
    meta_path: Path = FAISS_META_PATH,
) -> faiss.Index:
    """
    Encode all chunks with a sentence-transformer and store in a FAISS index.

    Uses IndexFlatIP (inner product / cosine similarity after L2 normalization).
    The metadata file stores the ordered list of chunk_ids so that FAISS
    result positions can be mapped back to chunk records.

    Args:
        chunks:     List of chunk dicts with "chunk_text" and "chunk_id".
        model_name: HuggingFace model name for embeddings.
        index_path: Destination .faiss binary file.
        meta_path:  Destination .pkl file for chunk_id ordering.

    Returns:
        The FAISS index object (already saved to disk).

    Example:
        >>> chunks = load_chunks()
        >>> index = build_faiss_index(chunks)
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["chunk_text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    logger.info(f"Encoding {len(texts)} chunks (this may take a few minutes)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via inner product
    )

    # Build a flat inner-product index (exact nearest-neighbor, no approximation)
    dim = embeddings.shape[1]
    if dim != EMBEDDING_DIMENSION:
        logger.warning(f"Embedding dim {dim} != expected {EMBEDDING_DIMENSION}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with open(meta_path, "wb") as f:
        pickle.dump(chunk_ids, f)

    logger.info(f"FAISS index ({index.ntotal} vectors) saved → {index_path}")
    logger.info(f"FAISS metadata saved → {meta_path}")
    return index


# ── Orchestration ──────────────────────────────────────────────────────────────

def build_all(rebuild: bool = False) -> None:
    """
    Build both BM25 and FAISS indexes from chunks.json.

    Skips building if the index files already exist, unless rebuild=True.
    Exits early with an error message if chunks.json is missing.

    Args:
        rebuild: If True, rebuild both indexes even if they already exist.

    Example:
        >>> build_all()          # skip if already built
        >>> build_all(rebuild=True)  # force rebuild
    """
    chunks = load_chunks()
    if not chunks:
        logger.error("No chunks to index. Aborting.")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # BM25
    if BM25_INDEX_PATH.exists() and not rebuild:
        logger.info(f"BM25 index already exists at {BM25_INDEX_PATH} — skipping. "
                    f"Use --rebuild to force.")
    else:
        build_bm25_index(chunks)

    # FAISS
    if FAISS_INDEX_PATH.exists() and not rebuild:
        logger.info(f"FAISS index already exists at {FAISS_INDEX_PATH} — skipping. "
                    f"Use --rebuild to force.")
    else:
        build_faiss_index(chunks)

    logger.info("Index build complete. Ready for retrieval.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build BM25 and FAISS indexes from indexes/chunks.json."
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild even if index files already exist."
    )
    args = parser.parse_args()

    build_all(rebuild=args.rebuild)
