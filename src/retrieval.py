"""
Retrieval module for the RAG Academic Paper QA System.

Implements three retrieval strategies over the pre-built indexes:
  - BM25 (sparse, keyword-based)
  - Dense (semantic, FAISS + sentence-transformers)
  - Hybrid (Reciprocal Rank Fusion of BM25 + Dense)

Indexes are loaded once at Retriever construction time and reused
across all queries — following the same offline-build / online-load
pattern as the SBYEC project.

Usage:
    from retrieval import Retriever

    retriever = Retriever()
    results = retriever.retrieve("What is attention mechanism?", mode="hybrid", top_k=20)
    for r in results:
        print(r["chunk_id"], r["score"], r["chunk_text"][:80])
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Literal

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BM25_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    LOG_LEVEL,
    RERANK_TOP_K,
    RETRIEVAL_MODE,
    RRF_K,
    TOP_K,
)

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

RetrievalMode = Literal["bm25", "dense", "hybrid"]


class Retriever:
    """
    Unified retrieval interface supporting BM25, Dense, and Hybrid modes.

    All indexes are loaded from disk on __init__. After that, each call
    to retrieve() is purely in-memory for low latency.

    Attributes:
        chunks:     Full list of chunk dicts loaded from chunks.json.
        chunk_map:  Dict mapping chunk_id → chunk dict for O(1) lookup.
        bm25:       BM25Okapi object.
        bm25_ids:   Ordered list of chunk_ids parallel to BM25 corpus.
        faiss_idx:  Loaded FAISS index.
        faiss_ids:  Ordered list of chunk_ids parallel to FAISS vectors.
        embed_model: SentenceTransformer for query encoding.
    """

    def __init__(self) -> None:
        """
        Load all indexes from disk. Raises FileNotFoundError with a helpful
        message if any required artifact is missing.
        """
        self.chunks = self._load_chunks()
        self.chunk_map: Dict[str, Dict] = {c["chunk_id"]: c for c in self.chunks}

        self.bm25, self.bm25_ids = self._load_bm25()
        self.faiss_idx, self.faiss_ids = self._load_faiss()
        self.embed_model = self._load_embed_model()

    # ── Loaders ────────────────────────────────────────────────────────────────

    def _load_chunks(self) -> List[Dict]:
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"chunks.json not found at {CHUNKS_PATH}.\n"
                f"  → Run first:  python src/ingest.py"
            )
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH.name}")
        return chunks

    def _load_bm25(self):
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_INDEX_PATH}.\n"
                f"  → Run first:  python src/build_index.py"
            )
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        logger.info(f"BM25 index loaded ({len(data['chunk_ids'])} docs)")
        return data["bm25"], data["chunk_ids"]

    def _load_faiss(self):
        if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}.\n"
                f"  → Run first:  python src/build_index.py"
            )
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH, "rb") as f:
            chunk_ids = pickle.load(f)
        logger.info(f"FAISS index loaded ({index.ntotal} vectors)")
        return index, chunk_ids

    def _load_embed_model(self) -> SentenceTransformer:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        return SentenceTransformer(EMBEDDING_MODEL)

    # ── Individual retrieval methods ───────────────────────────────────────────

    def retrieve_bm25(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        BM25 sparse retrieval.

        Args:
            query: Natural-language question.
            top_k: Number of chunks to return.

        Returns:
            List of chunk dicts sorted by BM25 score (descending), each
            augmented with a "score" key (float).

        Example:
            >>> results = retriever.retrieve_bm25("attention mechanism transformer", top_k=5)
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by descending score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk_id = self.bm25_ids[idx]
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                results.append({**chunk, "score": float(scores[idx])})

        return results

    def retrieve_dense(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Dense semantic retrieval using FAISS (cosine similarity via inner product
        on L2-normalized vectors).

        Args:
            query: Natural-language question.
            top_k: Number of chunks to return.

        Returns:
            List of chunk dicts sorted by cosine similarity (descending), each
            augmented with a "score" key (float).

        Example:
            >>> results = retriever.retrieve_dense("self-attention mechanism", top_k=10)
        """
        query_vec = self.embed_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        distances, indices = self.faiss_idx.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for empty slots
            chunk_id = self.faiss_ids[idx]
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                results.append({**chunk, "score": float(dist)})

        return results

    def retrieve_hybrid(
        self, query: str, top_k: int = TOP_K, rrf_k: int = RRF_K
    ) -> List[Dict]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion (RRF) of BM25 and Dense results.

        RRF score formula:  score(d) = Σ  1 / (k + rank_i(d))
        where rank_i is 1-based position in ranking list i.

        Both retrievers are called with top_k candidates each, then fused.
        The final list is trimmed to top_k results by RRF score.

        Args:
            query: Natural-language question.
            top_k: Number of final chunks to return.
            rrf_k: RRF smoothing constant (default 60, from the RRF paper).

        Returns:
            List of chunk dicts sorted by RRF score (descending), each
            augmented with a "score" key (float, the RRF fusion score).

        Example:
            >>> results = retriever.retrieve_hybrid("BERT pre-training objectives", top_k=20)
        """
        bm25_results = self.retrieve_bm25(query, top_k=top_k)
        dense_results = self.retrieve_dense(query, top_k=top_k)

        # Build chunk_id → 1-based rank for each list
        bm25_ranks = {r["chunk_id"]: rank + 1 for rank, r in enumerate(bm25_results)}
        dense_ranks = {r["chunk_id"]: rank + 1 for rank, r in enumerate(dense_results)}

        # Collect all unique chunk_ids from both lists
        all_ids = set(bm25_ranks) | set(dense_ranks)

        rrf_scores: Dict[str, float] = {}
        for chunk_id in all_ids:
            score = 0.0
            if chunk_id in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[chunk_id])
            if chunk_id in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        # Sort by RRF score descending and take top_k
        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        results = []
        for chunk_id in ranked_ids:
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                results.append({**chunk, "score": rrf_scores[chunk_id]})

        return results

    # ── Unified interface ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RETRIEVAL_MODE,
        top_k: int = TOP_K,
    ) -> List[Dict]:
        """
        Unified retrieval entry point. Dispatches to the appropriate strategy.

        Args:
            query: Natural-language question.
            mode:  Retrieval strategy — "bm25", "dense", or "hybrid".
            top_k: Number of chunks to return.

        Returns:
            List of chunk dicts sorted by relevance score (descending).

        Raises:
            ValueError: If mode is not one of the three supported values.

        Example:
            >>> results = retriever.retrieve("What is BERT?", mode="hybrid", top_k=20)
        """
        if mode == "bm25":
            return self.retrieve_bm25(query, top_k=top_k)
        elif mode == "dense":
            return self.retrieve_dense(query, top_k=top_k)
        elif mode == "hybrid":
            return self.retrieve_hybrid(query, top_k=top_k)
        else:
            raise ValueError(f"Unknown retrieval mode: '{mode}'. "
                             f"Choose from 'bm25', 'dense', 'hybrid'.")


def load_retriever() -> Retriever:
    """
    Convenience factory: construct and return a Retriever instance.

    Intended for use by pipeline.py and evaluate.py so they share
    a single import call.

    Returns:
        Initialized Retriever with all indexes loaded.

    Example:
        >>> retriever = load_retriever()
        >>> results = retriever.retrieve("transformer attention")
    """
    return Retriever()


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is the attention mechanism in transformers?"
    print(f"\nQuery: {query}\n")

    try:
        retriever = load_retriever()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    for mode in ("bm25", "dense", "hybrid"):
        print(f"--- {mode.upper()} Top-3 ---")
        results = retriever.retrieve(query, mode=mode, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['chunk_id']}] score={r['score']:.4f}")
            print(f"     {r['chunk_text'][:120].strip()}...")
        print()
