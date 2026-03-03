"""
Evaluation module for the RAG Academic Paper QA System.

Implements all metrics required by the project specification:
  - Recall@K    : fraction of relevant chunks appearing in top-K retrieved results
  - MRR         : Mean Reciprocal Rank (position of first relevant result)
  - Precision@K : fraction of top-K retrieved results that are relevant
  - NDCG@K      : Normalized Discounted Cumulative Gain
  - RAGAS       : Faithfulness and Answer Relevancy (requires live LLM + real data)

Experiment runners:
  - run_retrieval_eval()  : BM25 vs Dense vs Hybrid comparison table
  - run_reranking_ablation() : with vs without Cross-Encoder reranking
  - run_ragas_eval()      : end-to-end RAGAS evaluation
  - run_all_experiments() : runs all experiments and prints a summary

Usage:
    python src/evaluate.py

    # Or individual experiment:
    python src/evaluate.py --experiment retrieval
    python src/evaluate.py --experiment reranking
    python src/evaluate.py --experiment ragas
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EVAL_K_VALUES, LOG_LEVEL, QA_TESTSET_PATH, RAGAS_METRICS, RERANK_TOP_K, TOP_K

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Test set loading ───────────────────────────────────────────────────────────

def load_testset(path: Path = QA_TESTSET_PATH) -> List[Dict]:
    """
    Load the hand-annotated QA test set from JSON.

    Each record must have at minimum:
      "question"           (str)
      "ground_truth_answer" (str)
      "relevant_chunk_ids"  (List[str]) — chunk_ids that are ground-truth relevant

    Args:
        path: Path to qa_testset.json.

    Returns:
        List of QA dicts. Returns empty list if the file does not exist or
        has no annotated relevant_chunk_ids (with a clear warning).

    Example:
        >>> testset = load_testset()
        >>> print(f"{len(testset)} test questions loaded")
    """
    if not path.exists():
        logger.warning(f"Test set not found at {path}. Returning empty list.")
        return []

    with open(path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    # Filter out placeholder entries that have no relevant_chunk_ids annotated
    annotated = [q for q in testset if q.get("relevant_chunk_ids")]
    if not annotated:
        logger.warning(
            f"Loaded {len(testset)} QA pairs but none have 'relevant_chunk_ids' filled in.\n"
            f"  → Annotate qa_testset.json with chunk_ids after running ingest.py."
        )
    else:
        logger.info(f"Loaded {len(annotated)} annotated QA pairs from {path.name}")

    return annotated


# ── Retrieval metrics ──────────────────────────────────────────────────────────

def compute_recall_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """
    Recall@K = |relevant ∩ top-K retrieved| / |relevant|

    Args:
        retrieved_ids: Ordered list of retrieved chunk_ids (from rank 1).
        relevant_ids:  Ground-truth relevant chunk_ids for this question.
        k:             Cutoff rank.

    Returns:
        Recall score in [0, 1]. Returns 0.0 if relevant_ids is empty.

    Example:
        >>> compute_recall_at_k(["a", "b", "c"], ["b", "d"], k=3)
        0.5
    """
    if not relevant_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k_set & relevant_set) / len(relevant_set)


def compute_precision_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """
    Precision@K = |relevant ∩ top-K retrieved| / K

    Args:
        retrieved_ids: Ordered list of retrieved chunk_ids.
        relevant_ids:  Ground-truth relevant chunk_ids.
        k:             Cutoff rank.

    Returns:
        Precision score in [0, 1].

    Example:
        >>> compute_precision_at_k(["a", "b", "c"], ["b", "d"], k=3)
        0.3333...
    """
    if k == 0:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k_set & relevant_set) / k


def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Reciprocal Rank = 1 / rank_of_first_relevant_result.

    MRR is averaged across queries in run_retrieval_eval().

    Args:
        retrieved_ids: Ordered list of retrieved chunk_ids.
        relevant_ids:  Ground-truth relevant chunk_ids.

    Returns:
        Reciprocal rank in (0, 1]. Returns 0.0 if no relevant result found.

    Example:
        >>> compute_mrr(["x", "b", "y"], ["b", "d"])
        0.5  # first relevant result at rank 2
    """
    relevant_set = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """
    NDCG@K — Normalized Discounted Cumulative Gain (binary relevance).

    DCG@K  = Σ_{i=1}^{K} rel_i / log2(i + 1)
    IDCG@K = DCG of ideal ranking (all relevant items first)
    NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved_ids: Ordered list of retrieved chunk_ids.
        relevant_ids:  Ground-truth relevant chunk_ids.
        k:             Cutoff rank.

    Returns:
        NDCG score in [0, 1]. Returns 0.0 if relevant_ids is empty.

    Example:
        >>> compute_ndcg_at_k(["b", "x", "y"], ["b", "y"], k=3)
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    def dcg(ids: List[str], cutoff: int) -> float:
        return sum(
            (1.0 / math.log2(i + 2))
            for i, cid in enumerate(ids[:cutoff])
            if cid in relevant_set
        )

    dcg_val  = dcg(retrieved_ids, k)
    ideal    = sorted(retrieved_ids, key=lambda cid: cid in relevant_set, reverse=True)
    idcg_val = dcg(ideal[:k], k)

    return dcg_val / idcg_val if idcg_val > 0 else 0.0


# ── Experiment runners ─────────────────────────────────────────────────────────

def run_retrieval_eval(
    testset: Optional[List[Dict]] = None,
    modes: List[str] = ("bm25", "dense", "hybrid"),
    k_values: List[int] = EVAL_K_VALUES,
    top_k: int = TOP_K,
) -> Dict:
    """
    Compare BM25, Dense, and Hybrid retrieval across Recall@K and MRR.

    Loads the Retriever internally so this function is self-contained.

    Args:
        testset:  Pre-loaded test set. If None, loads from config path.
        modes:    Retrieval modes to evaluate.
        k_values: List of K values for Recall@K / Precision@K / NDCG@K.
        top_k:    Number of candidates retrieved per query (should be >= max(k_values)).

    Returns:
        Dict of {mode: {metric: value}} — all scores averaged over the test set.
        Returns empty dict if testset is empty.

    Example:
        >>> results = run_retrieval_eval()
        >>> print(results["hybrid"]["Recall@5"])
    """
    if testset is None:
        testset = load_testset()
    if not testset:
        logger.warning("No annotated test data — skipping retrieval evaluation.")
        return {}

    from retrieval import load_retriever
    retriever = load_retriever()

    results: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        logger.info(f"Evaluating retrieval mode: {mode}")
        mode_scores: Dict[str, List[float]] = {
            **{f"Recall@{k}":    [] for k in k_values},
            **{f"Precision@{k}": [] for k in k_values},
            **{f"NDCG@{k}":      [] for k in k_values},
            "MRR": [],
        }

        for qa in testset:
            question     = qa["question"]
            relevant_ids = qa["relevant_chunk_ids"]
            candidates   = retriever.retrieve(question, mode=mode, top_k=top_k)
            retrieved_ids = [c["chunk_id"] for c in candidates]

            mode_scores["MRR"].append(compute_mrr(retrieved_ids, relevant_ids))
            for k in k_values:
                mode_scores[f"Recall@{k}"].append(
                    compute_recall_at_k(retrieved_ids, relevant_ids, k)
                )
                mode_scores[f"Precision@{k}"].append(
                    compute_precision_at_k(retrieved_ids, relevant_ids, k)
                )
                mode_scores[f"NDCG@{k}"].append(
                    compute_ndcg_at_k(retrieved_ids, relevant_ids, k)
                )

        results[mode] = {
            metric: round(sum(vals) / len(vals), 4)
            for metric, vals in mode_scores.items()
        }

    return results


def run_reranking_ablation(
    testset: Optional[List[Dict]] = None,
    k_values: List[int] = EVAL_K_VALUES,
    top_k: int = TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
) -> Dict:
    """
    Ablation study: retrieval only vs. retrieval + Cross-Encoder reranking.

    Uses the default RETRIEVAL_MODE (hybrid) for the retrieval step.
    Evaluates Precision@K and NDCG@K on the reranked top-K list.

    Args:
        testset:      Pre-loaded test set. If None, loads from config path.
        k_values:     Evaluation cutoffs (should be <= rerank_top_k).
        top_k:        Retrieval candidates before reranking.
        rerank_top_k: Passages kept after reranking.

    Returns:
        Dict {"without_reranking": {...}, "with_reranking": {...}}

    Example:
        >>> ablation = run_reranking_ablation()
        >>> print(ablation["with_reranking"]["Precision@5"])
    """
    if testset is None:
        testset = load_testset()
    if not testset:
        logger.warning("No annotated test data — skipping reranking ablation.")
        return {}

    from reranker import load_reranker
    from retrieval import load_retriever

    retriever = load_retriever()
    reranker  = load_reranker()

    conditions = {
        "without_reranking": [],
        "with_reranking":    [],
    }

    for qa in testset:
        question     = qa["question"]
        relevant_ids = qa["relevant_chunk_ids"]
        candidates   = retriever.retrieve(question, top_k=top_k)

        # Without reranking: use retrieval order, truncated to rerank_top_k
        no_rerank_ids = [c["chunk_id"] for c in candidates[:rerank_top_k]]

        # With reranking
        reranked      = reranker.rerank(question, candidates, top_k=rerank_top_k)
        reranked_ids  = [c["chunk_id"] for c in reranked]

        for k in k_values:
            conditions["without_reranking"].append({
                f"Precision@{k}": compute_precision_at_k(no_rerank_ids, relevant_ids, k),
                f"NDCG@{k}":      compute_ndcg_at_k(no_rerank_ids, relevant_ids, k),
            })
            conditions["with_reranking"].append({
                f"Precision@{k}": compute_precision_at_k(reranked_ids, relevant_ids, k),
                f"NDCG@{k}":      compute_ndcg_at_k(reranked_ids, relevant_ids, k),
            })

    def average_dicts(list_of_dicts: List[Dict]) -> Dict:
        agg: Dict[str, List[float]] = {}
        for d in list_of_dicts:
            for key, val in d.items():
                agg.setdefault(key, []).append(val)
        return {k: round(sum(v) / len(v), 4) for k, v in agg.items()}

    return {
        "without_reranking": average_dicts(conditions["without_reranking"]),
        "with_reranking":    average_dicts(conditions["with_reranking"]),
    }


def run_ragas_eval(
    testset: Optional[List[Dict]] = None,
) -> Dict:
    """
    End-to-end evaluation using RAGAS (Faithfulness + Answer Relevancy).

    RAGAS requires:
    - A live LLM (Groq API key must be set)
    - Actual answer strings from the pipeline
    - Ground-truth answers in the test set

    Args:
        testset: Pre-loaded test set. If None, loads from config path.

    Returns:
        Dict {"faithfulness": float, "answer_relevancy": float} averaged
        over all test questions. Returns empty dict on failure or missing data.

    Example:
        >>> scores = run_ragas_eval()
        >>> print(scores.get("faithfulness", "N/A"))
    """
    if testset is None:
        testset = load_testset()
    if not testset:
        logger.warning("No annotated test data — skipping RAGAS evaluation.")
        return {}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        logger.error("RAGAS or datasets not installed. Run: pip install ragas datasets")
        return {}

    from pipeline import RAGPipeline
    pipeline = RAGPipeline()

    questions, answers, contexts, ground_truths = [], [], [], []

    for qa in testset:
        result = pipeline.query(qa["question"])
        questions.append(qa["question"])
        answers.append(result["answer"])
        contexts.append([c["chunk_text"] for c in result["sources"]])
        ground_truths.append(qa.get("ground_truth_answer", ""))

    ragas_dataset = Dataset.from_dict({
        "question":        questions,
        "answer":          answers,
        "contexts":        contexts,
        "ground_truths":   [[gt] for gt in ground_truths],
    })

    result = evaluate(ragas_dataset, metrics=[faithfulness, answer_relevancy])
    scores = {
        "faithfulness":     round(result["faithfulness"],     4),
        "answer_relevancy": round(result["answer_relevancy"], 4),
    }
    logger.info(f"RAGAS scores: {scores}")
    return scores


def print_table(title: str, data: Dict) -> None:
    """
    Pretty-print a results dict as an ASCII table to stdout.

    Args:
        title: Table header string.
        data:  Dict of {row_name: {metric: value}}.
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    if not data:
        print("  (no data)")
        return

    # Collect all metric names
    metrics = sorted({m for row in data.values() for m in row})
    col_w = max(len(m) for m in metrics) + 2
    row_w = max(len(str(k)) for k in data) + 2

    header = f"  {'Mode':<{row_w}}" + "".join(f"{m:<{col_w}}" for m in metrics)
    print(header)
    print("  " + "-" * (row_w + col_w * len(metrics)))
    for row_name, scores in sorted(data.items()):
        row = f"  {str(row_name):<{row_w}}" + "".join(
            f"{scores.get(m, 'N/A'):<{col_w}}" for m in metrics
        )
        print(row)
    print()


def run_all_experiments() -> None:
    """
    Run all evaluation experiments in sequence and print results tables.

    Experiments:
      1. Retrieval comparison: BM25 vs Dense vs Hybrid
      2. Reranking ablation: without vs with Cross-Encoder
      3. RAGAS end-to-end evaluation

    Skips experiments gracefully if test data or API key is unavailable.
    """
    print("\n" + "=" * 60)
    print("  RAG System Evaluation")
    print("=" * 60)

    # Experiment 1: Retrieval comparison
    logger.info("Running Experiment 1: Retrieval comparison")
    retrieval_results = run_retrieval_eval()
    print_table("Experiment 1 — Retrieval Comparison (BM25 / Dense / Hybrid)", retrieval_results)

    # Experiment 2: Reranking ablation
    logger.info("Running Experiment 2: Reranking ablation")
    reranking_results = run_reranking_ablation()
    print_table("Experiment 2 — Reranking Ablation (w/ vs w/o Cross-Encoder)", reranking_results)

    # Experiment 3: RAGAS
    logger.info("Running Experiment 3: RAGAS end-to-end evaluation")
    ragas_results = run_ragas_eval()
    if ragas_results:
        print_table("Experiment 3 — RAGAS End-to-End", {"pipeline": ragas_results})
    else:
        print("\n  [Experiment 3 — RAGAS skipped (no data or API key not set)]")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG system evaluation experiments.")
    parser.add_argument(
        "--experiment",
        choices=["retrieval", "reranking", "ragas", "all"],
        default="all",
        help="Which experiment to run (default: all).",
    )
    args = parser.parse_args()

    if args.experiment == "retrieval":
        results = run_retrieval_eval()
        print_table("Retrieval Comparison", results)
    elif args.experiment == "reranking":
        results = run_reranking_ablation()
        print_table("Reranking Ablation", results)
    elif args.experiment == "ragas":
        results = run_ragas_eval()
        print_table("RAGAS Evaluation", {"pipeline": results} if results else {})
    else:
        run_all_experiments()
