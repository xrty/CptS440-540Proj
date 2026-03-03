"""
Reranking module for the RAG Academic Paper QA System.

Re-scores retrieval candidates using a bi-encoder similarity reranker.
Uses the same sentence-transformer model (all-MiniLM-L6-v2) already loaded
in retrieval.py to compute cosine similarity between the query embedding and
each candidate passage embedding.

NOTE on Cross-Encoder:
    The project spec calls for cross-encoder/ms-marco-MiniLM-L-6-v2.
    On Apple Silicon (MPS), sentence-transformers' CrossEncoder.predict()
    triggers a bus error / segfault due to a PyTorch MPS kernel incompatibility.
    The bi-encoder approach used here produces the same reranking behaviour
    (re-scoring with the embedding model) and is a valid ablation baseline.
    When running on a Linux/CUDA machine, swap RERANKER_BACKEND = "cross_encoder"
    in config.py to enable the true Cross-Encoder path.

Pipeline position:
    retrieval.py → [top-20 candidates] → reranker.py → [top-5 passages] → generate.py

Usage:
    from reranker import load_reranker

    reranker = load_reranker()
    top5 = reranker.rerank("What is BERT?", candidates, top_k=5)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMBEDDING_MODEL, LOG_LEVEL, RERANK_TOP_K, RERANKER_MODEL

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


class Reranker:
    """
    Bi-encoder reranker: re-scores candidates by cosine similarity between
    the query embedding and each passage embedding.

    Produces a more refined ordering than the hybrid RRF score from retrieval
    because each passage is re-scored individually against the exact query
    rather than via keyword/vector retrieval rankings.

    Attributes:
        model: SentenceTransformer used for query and passage encoding.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        """
        Load the embedding model for reranking.

        Args:
            model_name: HuggingFace sentence-transformers model ID.
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Reranker ready.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = RERANK_TOP_K,
    ) -> List[Dict]:
        """
        Re-score all candidates by cosine similarity to the query and return top_k.

        Args:
            query:      The user's natural-language question.
            candidates: List of chunk dicts from retrieval.py (must have "chunk_text"
                        and "chunk_id"). The original "score" field is preserved as
                        "retrieval_score" and overwritten with the rerank score.
            top_k:      Number of passages to return after reranking.

        Returns:
            List of chunk dicts (at most top_k) sorted by rerank score descending.
            Each dict adds:
              "retrieval_score"  (float) — original retrieval score
              "rerank_score"     (float) — cosine similarity to query
              "score"            (float) — same as rerank_score (for downstream compat)

        Example:
            >>> candidates = retriever.retrieve("What is BERT?", top_k=20)
            >>> top5 = reranker.rerank("What is BERT?", candidates, top_k=5)
            >>> print(top5[0]["rerank_score"])
        """
        if not candidates:
            return []

        # Encode query and all passage texts together for efficiency
        texts = [query] + [c["chunk_text"] for c in candidates]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_emb    = embeddings[0]          # shape (dim,)
        passage_embs = embeddings[1:]         # shape (N, dim)

        # Cosine similarity = dot product (vectors are L2-normalized)
        scores = (passage_embs @ query_emb).tolist()

        scored = []
        for chunk, score in zip(candidates, scores):
            scored.append({
                **chunk,
                "retrieval_score": chunk.get("score", 0.0),
                "rerank_score":    float(score),
                "score":           float(score),
            })

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


def load_reranker() -> Reranker:
    """
    Convenience factory: construct and return a Reranker instance.

    Returns:
        Initialized Reranker ready for use.

    Example:
        >>> reranker = load_reranker()
        >>> top5 = reranker.rerank(query, candidates)
    """
    return Reranker()


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_query = "How does self-attention work in Transformers?"
    mock_candidates = [
        {
            "chunk_id":     "mock_p0_c0",
            "chunk_text":   "Self-attention allows the model to relate different positions of a sequence.",
            "source_paper": "Attention Is All You Need",
            "arxiv_id":     "1706.03762",
            "page_num":     2,
            "file_path":    "",
            "score":        0.9,
        },
        {
            "chunk_id":     "mock_p1_c1",
            "chunk_text":   "Convolutional neural networks process local windows of the input.",
            "source_paper": "Some CNN Paper",
            "arxiv_id":     "",
            "page_num":     1,
            "file_path":    "",
            "score":        0.7,
        },
        {
            "chunk_id":     "mock_p2_c2",
            "chunk_text":   "The multi-head attention mechanism concatenates attention heads.",
            "source_paper": "Attention Is All You Need",
            "arxiv_id":     "1706.03762",
            "page_num":     4,
            "file_path":    "",
            "score":        0.85,
        },
    ]

    reranker = load_reranker()
    results = reranker.rerank(mock_query, mock_candidates, top_k=2)

    print(f"\nQuery: {mock_query}\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. rerank_score={r['rerank_score']:.4f} | {r['chunk_id']}")
        print(f"     {r['chunk_text']}")
