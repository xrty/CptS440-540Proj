"""
End-to-end RAG pipeline for the Academic Paper QA System.

Wires together:
  retrieval.py → reranker.py → generate.py

Each call to RAGPipeline.query() returns a structured result dict containing
the answer, source citations, all retrieved chunks, and latency breakdown.

Usage:
    # Interactive CLI mode
    python src/pipeline.py

    # Specify retrieval mode and k values
    python src/pipeline.py --mode hybrid --top-k 20 --rerank-top-k 5

    # Programmatic use
    from pipeline import RAGPipeline
    pipe = RAGPipeline(retrieval_mode="hybrid")
    result = pipe.query("What is the key idea behind BERT?")
    print(result["answer"])
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LOG_LEVEL, RERANK_TOP_K, RETRIEVAL_MODE, TOP_K
from generate import Generator, load_generator
from reranker import Reranker, load_reranker
from retrieval import Retriever, load_retriever

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

RetrievalMode = Literal["bm25", "dense", "hybrid"]


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Initialization loads all sub-components (retriever, reranker, generator)
    into memory. After that, each query() call is fast — no further disk I/O
    except the Groq API call.

    Attributes:
        retrieval_mode: Default retrieval strategy for this pipeline instance.
        top_k:          Candidates passed from retrieval to reranking.
        rerank_top_k:   Final passages passed to the LLM.
        retriever:      Loaded Retriever instance.
        reranker:       Loaded Reranker instance.
        generator:      Loaded Generator instance.
    """

    def __init__(
        self,
        retrieval_mode: RetrievalMode = RETRIEVAL_MODE,
        top_k: int = TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
    ) -> None:
        """
        Initialize and load all pipeline components.

        Args:
            retrieval_mode: "bm25", "dense", or "hybrid" (default from config).
            top_k:          Number of candidates retrieved before reranking.
            rerank_top_k:   Number of passages kept after reranking for the LLM.

        Raises:
            FileNotFoundError: If the indexes have not been built yet.
                               Run python src/ingest.py then python src/build_index.py.
        """
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

        logger.info("Loading pipeline components...")
        self.retriever = load_retriever()
        self.reranker  = load_reranker()
        self.generator = load_generator()
        logger.info(
            f"Pipeline ready | mode={retrieval_mode} | "
            f"top_k={top_k} → rerank_top_k={rerank_top_k}"
        )

    def query(
        self,
        question: str,
        retrieval_mode: Optional[RetrievalMode] = None,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
    ) -> Dict:
        """
        Run the full RAG pipeline for a single question.

        Steps:
          1. Retrieve top_k candidate chunks (BM25 / Dense / Hybrid)
          2. Rerank with Cross-Encoder → keep rerank_top_k
          3. Generate answer with Groq LLM using reranked passages

        Args:
            question:       The user's natural-language question.
            retrieval_mode: Override the instance-level mode for this call.
            top_k:          Override the instance-level top_k for this call.
            rerank_top_k:   Override the instance-level rerank_top_k for this call.

        Returns:
            Dict with keys:
              "question"         (str)        — original question
              "answer"           (str)        — generated answer (or placeholder)
              "sources"          (List[Dict]) — reranked top-K chunks with scores
              "retrieved_chunks" (List[Dict]) — all candidates before reranking
              "latency"          (Dict)       — {"retrieval_s", "rerank_s",
                                                 "generation_s", "total_s"}
              "retrieval_mode"   (str)        — mode used for this query

        Example:
            >>> result = pipeline.query("What is self-attention?")
            >>> print(result["answer"])
            >>> for src in result["sources"]:
            ...     print(src["source_paper"], src["page_num"])
        """
        mode         = retrieval_mode or self.retrieval_mode
        k            = top_k or self.top_k
        rerank_k     = rerank_top_k or self.rerank_top_k
        total_start  = time.perf_counter()

        # Step 1 — Retrieval
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(question, mode=mode, top_k=k)
        retrieval_s = time.perf_counter() - t0
        logger.debug(f"Retrieved {len(candidates)} candidates in {retrieval_s:.3f}s")

        # Step 2 — Reranking
        t0 = time.perf_counter()
        top_passages = self.reranker.rerank(question, candidates, top_k=rerank_k)
        rerank_s = time.perf_counter() - t0
        logger.debug(f"Reranked to {len(top_passages)} passages in {rerank_s:.3f}s")

        # Step 3 — Generation
        t0 = time.perf_counter()
        answer = self.generator.generate_answer(question, top_passages)
        generation_s = time.perf_counter() - t0
        total_s = time.perf_counter() - total_start
        logger.debug(f"Generated answer in {generation_s:.3f}s (total: {total_s:.3f}s)")

        return {
            "question":         question,
            "answer":           answer,
            "sources":          top_passages,
            "retrieved_chunks": candidates,
            "retrieval_mode":   mode,
            "latency": {
                "retrieval_s":  round(retrieval_s,  3),
                "rerank_s":     round(rerank_s,     3),
                "generation_s": round(generation_s, 3),
                "total_s":      round(total_s,      3),
            },
        }

    def print_result(self, result: Dict) -> None:
        """
        Pretty-print a query result to stdout.

        Args:
            result: Return value of query().
        """
        print("\n" + "=" * 70)
        print(f"Q: {result['question']}")
        print("=" * 70)
        print(f"\nA: {result['answer']}")
        print(f"\n--- Sources ({len(result['sources'])} passages) ---")
        for i, src in enumerate(result["sources"], 1):
            print(
                f"  {i}. {src.get('source_paper', 'Unknown')}, "
                f"p.{src.get('page_num', '?')} "
                f"[rerank_score={src.get('rerank_score', src.get('score', 0)):.3f}]"
            )
        lat = result["latency"]
        print(
            f"\n--- Latency ---\n"
            f"  Retrieval: {lat['retrieval_s']}s | "
            f"Rerank: {lat['rerank_s']}s | "
            f"Generation: {lat['generation_s']}s | "
            f"Total: {lat['total_s']}s"
        )
        print()


def run_interactive(pipeline: RAGPipeline) -> None:
    """
    Run an interactive command-line question-answer loop.

    Type 'exit' or 'quit' (or press Ctrl-C) to stop.

    Args:
        pipeline: Initialized RAGPipeline instance.
    """
    print("\n" + "=" * 70)
    print("  RAG Academic Paper QA System")
    print(f"  Retrieval mode: {pipeline.retrieval_mode}")
    print("  Type your question and press Enter. Type 'exit' to quit.")
    print("=" * 70 + "\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Exiting.")
            break

        result = pipeline.query(question)
        pipeline.print_result(result)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive RAG pipeline for academic paper QA."
    )
    parser.add_argument(
        "--mode", choices=["bm25", "dense", "hybrid"], default=RETRIEVAL_MODE,
        help=f"Retrieval strategy (default: {RETRIEVAL_MODE})."
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Candidates retrieved before reranking (default: {TOP_K})."
    )
    parser.add_argument(
        "--rerank-top-k", type=int, default=RERANK_TOP_K,
        help=f"Passages kept after reranking (default: {RERANK_TOP_K})."
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Single question mode (skips interactive loop)."
    )
    args = parser.parse_args()

    try:
        pipeline = RAGPipeline(
            retrieval_mode=args.mode,
            top_k=args.top_k,
            rerank_top_k=args.rerank_top_k,
        )
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}\n")
        sys.exit(1)

    if args.question:
        result = pipeline.query(args.question)
        pipeline.print_result(result)
    else:
        run_interactive(pipeline)
