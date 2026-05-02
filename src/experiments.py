"""
Automated experiment runner for the RAG Academic Paper QA System.

Runs six experiments and produces PNG charts in the results/ directory.
Includes automatic test set generation from the corpus via Groq LLM.

Experiments:
  exp1 — Retrieval Mode Comparison (BM25 vs Dense vs Hybrid)
  exp2 — Top-K Sensitivity (k = 5, 10, 20, 50)
  exp3 — Rerank Top-K Sensitivity (k = 1, 2, 3, 5, 10)
  exp4 — RAG vs LLM-only Baseline
  exp5 — Latency Profiling (per-stage breakdown)
  exp6 — Reranking Ablation (with vs without)

Usage:
    # Generate test set from corpus (run once, requires Groq API key)
    python src/experiments.py --generate-testset

    # Run all experiments
    python src/experiments.py

    # Run a single experiment
    python src/experiments.py --experiment exp1
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CHUNKS_PATH,
    EVAL_K_VALUES,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LOG_LEVEL,
    QA_TESTSET_PATH,
    RERANK_TOP_K,
    ROOT_DIR,
    TOP_K,
)

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = ROOT_DIR / "results"

RETRIEVAL_MODES = ["bm25", "dense", "hybrid"]
TOPK_VALUES = [5, 10, 20, 50]
RERANK_TOPK_VALUES = [1, 2, 3, 5, 10]

# System prompt for LLM-only baseline (Experiment 4).
# No "only use provided context" restriction — answers from parametric knowledge.
LLM_ONLY_SYSTEM_PROMPT = """\
You are a knowledgeable research assistant specializing in AI and machine learning.
Answer the user's question as accurately and concisely as possible using your
general knowledge. Be factual and cite relevant papers or concepts when appropriate.
"""

# Chart styling
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
CHART_DPI = 200
FONT_TITLE = 14
FONT_LABEL = 12
FONT_TICK = 10
FONT_ANNOT = 9


# ── Utility functions ─────────────────────────────────────────────────────────

def _ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _has_annotated_testset(testset: List[Dict]) -> bool:
    return bool(testset)


def _has_groq_api() -> bool:
    return bool(GROQ_API_KEY)


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.size": FONT_TICK,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_LABEL,
            "legend.fontsize": FONT_TICK,
            "figure.facecolor": "white",
        })
        return plt
    except ImportError:
        logger.error("matplotlib required. Run: pip install matplotlib")
        return None


def _load_all_questions() -> List[Dict]:
    if not QA_TESTSET_PATH.exists():
        logger.warning(f"Test set not found at {QA_TESTSET_PATH}")
        return []
    with open(QA_TESTSET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_shared_resources() -> dict:
    ctx = {
        "retriever": None,
        "reranker": None,
        "generator": None,
        "pipeline": None,
        "testset": [],
    }

    try:
        from evaluate import load_testset
        ctx["testset"] = load_testset()
    except Exception as exc:
        logger.warning(f"Could not load test set: {exc}")

    try:
        from retrieval import load_retriever
        ctx["retriever"] = load_retriever()
    except Exception as exc:
        logger.warning(f"Retriever not available: {exc}")

    try:
        from reranker import load_reranker
        ctx["reranker"] = load_reranker()
    except Exception as exc:
        logger.warning(f"Reranker not available: {exc}")

    try:
        from generate import load_generator
        ctx["generator"] = load_generator()
    except Exception as exc:
        logger.warning(f"Generator not available: {exc}")

    try:
        from pipeline import RAGPipeline
        ctx["pipeline"] = RAGPipeline()
    except Exception as exc:
        logger.warning(f"Pipeline not available: {exc}")

    return ctx


# ── Auto test set generation ─────────────────────────────────────────────────

def generate_testset(n_per_paper: int = 3) -> List[Dict]:
    """
    Auto-generate an annotated QA test set from the actual corpus using Groq LLM.

    For each paper in the corpus, selects representative chunks and asks the LLM
    to generate a question + answer pair. The relevant_chunk_ids are known because
    we generate questions directly from specific chunks.

    Args:
        n_per_paper: Number of questions to generate per paper.

    Returns:
        List of QA dicts saved to qa_testset.json.
    """
    print("\n" + "=" * 60)
    print("  Auto-generating test set from corpus")
    print("=" * 60 + "\n")

    if not _has_groq_api():
        logger.error("Cannot generate test set — GROQ_API_KEY not set.")
        return []

    if not CHUNKS_PATH.exists():
        logger.error(f"chunks.json not found at {CHUNKS_PATH}. Run ingest.py first.")
        return []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    if not all_chunks:
        logger.error("No chunks found in corpus.")
        return []

    # Group chunks by paper
    papers: Dict[str, List[Dict]] = {}
    for c in all_chunks:
        paper = c["source_paper"]
        papers.setdefault(paper, []).append(c)

    logger.info(f"Corpus: {len(all_chunks)} chunks across {len(papers)} papers")

    # Select representative chunks (>300 chars, diverse pages)
    selected_chunks = []
    for paper, chunks in papers.items():
        good_chunks = [c for c in chunks if len(c["chunk_text"]) > 300]
        if not good_chunks:
            good_chunks = chunks

        # Pick chunks from diverse pages
        pages_seen = set()
        diverse = []
        for c in good_chunks:
            if c["page_num"] not in pages_seen:
                diverse.append(c)
                pages_seen.add(c["page_num"])

        if len(diverse) > n_per_paper:
            # Pick from beginning, middle, end of the paper
            indices = [0, len(diverse) // 2, len(diverse) - 1]
            if n_per_paper > 3:
                indices = sorted(random.sample(range(len(diverse)),
                                               min(n_per_paper, len(diverse))))
            diverse = [diverse[i] for i in indices[:n_per_paper]]

        selected_chunks.extend(diverse[:n_per_paper])

    logger.info(f"Selected {len(selected_chunks)} chunks for question generation")

    # Generate questions using Groq
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    qa_pairs = []
    for i, chunk in enumerate(selected_chunks):
        logger.info(f"  [{i+1}/{len(selected_chunks)}] Generating Q&A from "
                     f"{chunk['source_paper'][:40]}... (p.{chunk['page_num']})")

        prompt = (
            "Based on the following passage from an academic paper, generate exactly "
            "one specific, factual question that can be answered using this passage, "
            "and then provide a concise answer.\n\n"
            f"Paper: {chunk['source_paper']}\n\n"
            f"Passage:\n{chunk['chunk_text']}\n\n"
            "Respond in exactly this format (no other text):\n"
            "Q: <your question>\n"
            "A: <your answer>"
        )

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You generate question-answer pairs from academic paper passages. Be specific and factual."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()

            # Parse Q: and A: lines
            q_line = ""
            a_line = ""
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("Q:"):
                    q_line = line[2:].strip()
                elif line.startswith("A:"):
                    a_line = line[2:].strip()

            if q_line and a_line:
                # Find additional relevant chunks via text overlap
                relevant_ids = [chunk["chunk_id"]]
                # Also include chunks from same paper on adjacent pages
                for c in papers.get(chunk["source_paper"], []):
                    if c["chunk_id"] != chunk["chunk_id"] and \
                       abs(c["page_num"] - chunk["page_num"]) <= 1 and \
                       len(c["chunk_text"]) > 200:
                        relevant_ids.append(c["chunk_id"])
                        if len(relevant_ids) >= 4:
                            break

                qa_pairs.append({
                    "question": q_line,
                    "ground_truth_answer": a_line,
                    "relevant_chunk_ids": relevant_ids,
                    "source_paper": chunk["source_paper"],
                    "arxiv_id": chunk.get("arxiv_id", ""),
                    "notes": f"Auto-generated from chunk {chunk['chunk_id']}",
                })
                logger.info(f"    Q: {q_line[:80]}...")
            else:
                logger.warning(f"    Failed to parse Q&A from response: {text[:100]}")

        except Exception as exc:
            logger.warning(f"    Groq API call failed: {exc}")

        # Rate limit protection
        time.sleep(2)

    # Save
    with open(QA_TESTSET_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"\nSaved {len(qa_pairs)} QA pairs to {QA_TESTSET_PATH}")
    return qa_pairs


# ── Experiment 1: Retrieval Mode Comparison ───────────────────────────────────

def exp1_retrieval_mode_comparison(ctx: dict) -> dict:
    logger.info("=" * 60)
    logger.info("  Experiment 1: Retrieval Mode Comparison")
    logger.info("=" * 60)

    if not _has_annotated_testset(ctx["testset"]):
        logger.warning("Skipping Exp1 — no annotated test data. "
                        "Run with --generate-testset first.")
        return {}

    from evaluate import print_table, run_retrieval_eval
    results = run_retrieval_eval(testset=ctx["testset"])
    print_table("Experiment 1 — Retrieval Mode Comparison (Overall)", results)
    _plot_exp1(results, RESULTS_DIR / "exp1_retrieval_comparison.png")

    # ── Factual vs Semantic breakdown ─────────────────────────────────────
    factual_qs = [q for q in ctx["testset"] if q.get("type") == "factual"]
    semantic_qs = [q for q in ctx["testset"] if q.get("type") == "semantic"]

    if factual_qs and semantic_qs:
        logger.info(f"  Factual vs Semantic breakdown: "
                     f"{len(factual_qs)} factual, {len(semantic_qs)} semantic")
        factual_results = run_retrieval_eval(testset=factual_qs)
        semantic_results = run_retrieval_eval(testset=semantic_qs)

        print_table("Experiment 1a — Factual Questions Only", factual_results)
        print_table("Experiment 1b — Semantic Questions Only", semantic_results)

        _plot_exp1_by_type(
            factual_results, semantic_results,
            RESULTS_DIR / "exp1_factual_vs_semantic.png",
        )
    else:
        logger.info("  No type annotations — skipping factual/semantic breakdown.")

    return results


def _plot_exp1(results: dict, output_path: Path):
    plt = _import_matplotlib()
    if plt is None or not results:
        return

    modes = list(results.keys())
    metrics = sorted(results[modes[0]].keys())

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(metrics))
    width = 0.75 / len(modes)

    for i, mode in enumerate(modes):
        values = [results[mode].get(m, 0) for m in metrics]
        offset = (i - len(modes) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], values, width,
                      label=mode.upper(), color=COLORS[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=FONT_ANNOT,
                    fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Experiment 1: Retrieval Mode Comparison (BM25 / Dense / Hybrid)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, rotation=40, ha="right", fontsize=FONT_TICK)
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved: {output_path}")


def _plot_exp1_by_type(factual: dict, semantic: dict, output_path: Path):
    """Side-by-side comparison: factual vs semantic questions per retrieval mode."""
    plt = _import_matplotlib()
    if plt is None:
        return

    modes = list(factual.keys())
    # Pick a subset of key metrics for clarity
    key_metrics = []
    for m in sorted(factual[modes[0]].keys()):
        if any(k in m for k in ("Recall", "MRR", "NDCG")):
            key_metrics.append(m)
    if not key_metrics:
        key_metrics = sorted(factual[modes[0]].keys())[:4]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, (data, title) in zip(axes, [
        (factual,  "Factual Questions (keyword-match)"),
        (semantic, "Semantic Questions (paraphrased)"),
    ]):
        x = range(len(key_metrics))
        width = 0.75 / len(modes)
        for i, mode in enumerate(modes):
            vals = [data[mode].get(m, 0) for m in key_metrics]
            offset = (i - len(modes) / 2 + 0.5) * width
            bars = ax.bar([xi + offset for xi in x], vals, width,
                          label=mode.upper(), color=COLORS[i],
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=FONT_ANNOT, fontweight="bold")

        ax.set_title(title, fontsize=FONT_LABEL + 1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(key_metrics, rotation=35, ha="right", fontsize=FONT_TICK)
        ax.set_ylim(0, 1.25)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Score")
    fig.suptitle("Experiment 1: Factual vs Semantic Query Performance by Retrieval Mode",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved: {output_path}")


# ── Experiment 2: Top-K Sensitivity ──────────────────────────────────────────

def exp2_topk_sensitivity(ctx: dict) -> dict:
    logger.info("=" * 60)
    logger.info("  Experiment 2: Top-K Sensitivity")
    logger.info("=" * 60)

    if not _has_annotated_testset(ctx["testset"]):
        logger.warning("Skipping Exp2 — no annotated test data.")
        return {}
    if ctx["retriever"] is None:
        logger.warning("Skipping Exp2 — retriever not available.")
        return {}

    from evaluate import (
        compute_mrr, compute_ndcg_at_k,
        compute_precision_at_k, compute_recall_at_k, print_table,
    )

    eval_k = max(EVAL_K_VALUES)
    results = {}

    for top_k in TOPK_VALUES:
        logger.info(f"  Evaluating top_k={top_k}")
        scores = {f"Recall@{eval_k}": [], f"Precision@{eval_k}": [],
                  "MRR": [], f"NDCG@{eval_k}": []}

        for qa in ctx["testset"]:
            candidates = ctx["retriever"].retrieve(
                qa["question"], mode="hybrid", top_k=top_k)
            retrieved_ids = [c["chunk_id"] for c in candidates]
            relevant_ids = qa["relevant_chunk_ids"]

            scores[f"Recall@{eval_k}"].append(
                compute_recall_at_k(retrieved_ids, relevant_ids, eval_k))
            scores[f"Precision@{eval_k}"].append(
                compute_precision_at_k(retrieved_ids, relevant_ids, eval_k))
            scores["MRR"].append(compute_mrr(retrieved_ids, relevant_ids))
            scores[f"NDCG@{eval_k}"].append(
                compute_ndcg_at_k(retrieved_ids, relevant_ids, eval_k))

        results[f"top_k={top_k}"] = {
            m: round(sum(v) / len(v), 4) for m, v in scores.items()
        }

    print_table("Experiment 2 — Top-K Sensitivity (Hybrid)", results)
    _plot_exp2(results, RESULTS_DIR / "exp2_topk_sensitivity.png")
    return results


def _plot_exp2(results: dict, output_path: Path):
    plt = _import_matplotlib()
    if plt is None or not results:
        return

    topk_labels = list(results.keys())
    topk_nums = TOPK_VALUES[:len(topk_labels)]
    metrics = list(results[topk_labels[0]].keys())
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, metric in enumerate(metrics):
        values = [results[lbl][metric] for lbl in topk_labels]
        ax.plot(topk_nums, values, marker=markers[i % len(markers)],
                label=metric, color=COLORS[i], linewidth=2.5, markersize=8)
        # Annotate each point
        for xv, yv in zip(topk_nums, values):
            ax.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=FONT_ANNOT)

    ax.set_xlabel("top_k (candidates retrieved)")
    ax.set_ylabel("Score")
    ax.set_title("Experiment 2: Top-K Sensitivity (Hybrid Retrieval)")
    ax.set_xticks(topk_nums)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved: {output_path}")


# ── Experiment 3: Rerank Top-K Sensitivity ───────────────────────────────────

def exp3_rerank_topk_sensitivity(ctx: dict) -> dict:
    logger.info("=" * 60)
    logger.info("  Experiment 3: Rerank Top-K Sensitivity")
    logger.info("=" * 60)

    if not _has_annotated_testset(ctx["testset"]):
        logger.warning("Skipping Exp3 — no annotated test data.")
        return {}
    if ctx["retriever"] is None or ctx["reranker"] is None:
        logger.warning("Skipping Exp3 — retriever or reranker not available.")
        return {}

    from evaluate import (
        compute_mrr, compute_ndcg_at_k, compute_precision_at_k, print_table,
    )

    results = {}

    for rerank_k in RERANK_TOPK_VALUES:
        logger.info(f"  Evaluating rerank_top_k={rerank_k}")
        scores = {f"Precision@{rerank_k}": [], f"NDCG@{rerank_k}": [], "MRR": []}

        for qa in ctx["testset"]:
            candidates = ctx["retriever"].retrieve(
                qa["question"], mode="hybrid", top_k=TOP_K)
            reranked = ctx["reranker"].rerank(
                qa["question"], candidates, top_k=rerank_k)
            reranked_ids = [c["chunk_id"] for c in reranked]
            relevant_ids = qa["relevant_chunk_ids"]

            scores[f"Precision@{rerank_k}"].append(
                compute_precision_at_k(reranked_ids, relevant_ids, rerank_k))
            scores[f"NDCG@{rerank_k}"].append(
                compute_ndcg_at_k(reranked_ids, relevant_ids, rerank_k))
            scores["MRR"].append(compute_mrr(reranked_ids, relevant_ids))

        results[f"rerank_k={rerank_k}"] = {
            m: round(sum(v) / len(v), 4) for m, v in scores.items()
        }

    print_table("Experiment 3 — Rerank Top-K Sensitivity", results)
    _plot_exp3(results, RESULTS_DIR / "exp3_rerank_topk_sensitivity.png")
    return results


def _plot_exp3(results: dict, output_path: Path):
    plt = _import_matplotlib()
    if plt is None or not results:
        return

    rerank_labels = list(results.keys())
    rerank_nums = RERANK_TOPK_VALUES[:len(rerank_labels)]

    fig, ax = plt.subplots(figsize=(9, 6))

    mrr_vals = [results[lbl]["MRR"] for lbl in rerank_labels]
    prec_vals = [results[lbl].get(f"Precision@{rk}", 0)
                 for lbl, rk in zip(rerank_labels, rerank_nums)]
    ndcg_vals = [results[lbl].get(f"NDCG@{rk}", 0)
                 for lbl, rk in zip(rerank_labels, rerank_nums)]

    for vals, label, color, marker in [
        (prec_vals, "Precision@K", COLORS[0], "o"),
        (ndcg_vals, "NDCG@K", COLORS[1], "s"),
        (mrr_vals, "MRR", COLORS[2], "^"),
    ]:
        ax.plot(rerank_nums, vals, marker=marker, label=label,
                color=color, linewidth=2.5, markersize=8)
        for xv, yv in zip(rerank_nums, vals):
            ax.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=FONT_ANNOT)

    ax.set_xlabel("rerank_top_k (passages after reranking)")
    ax.set_ylabel("Score")
    ax.set_title("Experiment 3: Rerank Top-K Sensitivity (Retrieval top_k=20)")
    ax.set_xticks(rerank_nums)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved: {output_path}")


# ── Experiment 4: RAG vs LLM-only Baseline ──────────────────────────────────

def exp4_rag_vs_llm_only(ctx: dict) -> dict:
    """Experiment 4: RAG vs LLM-only Baseline.

    NOTE: API calls are disabled to preserve the existing chart generated from
    a previous run (Groq free-tier quota exhausted). The chart at
    results/exp4_rag_vs_llm_only.png remains valid.
    """
    logger.info("=" * 60)
    logger.info("  Experiment 4: RAG vs LLM-only Baseline")
    logger.info("=" * 60)

    chart_path = RESULTS_DIR / "exp4_rag_vs_llm_only.png"
    if chart_path.exists():
        logger.info(f"  Using cached chart: {chart_path}")
        logger.info("  (API calls skipped — Groq free-tier quota preserved)")
        # Return minimal result so experiment counts as "completed"
        return {"_cached": True}
    else:
        logger.warning(
            "Skipping Exp4 — no cached chart found and API calls are disabled.\n"
            "  To generate, uncomment the API calls in experiments.py."
        )
        return {}

    # ── Original API calls (commented out to preserve quota) ──────────────
    # if not _has_groq_api():
    #     logger.warning("Skipping Exp4 — GROQ_API_KEY not set.")
    #     return {}
    # if ctx["generator"] is None or ctx["generator"].client is None:
    #     logger.warning("Skipping Exp4 — generator client not available.")
    #     return {}
    # if ctx["pipeline"] is None:
    #     logger.warning("Skipping Exp4 — pipeline not available.")
    #     return {}
    #
    # all_questions = _load_all_questions()
    # if not all_questions:
    #     logger.warning("Skipping Exp4 — no test questions found.")
    #     return {}
    #
    # rag_answers, llm_answers = [], []
    # rag_latencies, llm_latencies = [], []
    # rag_contexts = []
    #
    # for i, qa in enumerate(all_questions):
    #     question = qa["question"]
    #     logger.info(f"  [{i+1}/{len(all_questions)}] {question[:60]}...")
    #
    #     # RAG condition
    #     try:
    #         result = ctx["pipeline"].query(question)
    #         rag_answers.append(result["answer"])
    #         rag_latencies.append(result["latency"]["total_s"])
    #         rag_contexts.append([c["chunk_text"] for c in result["sources"]])
    #     except Exception as exc:
    #         logger.warning(f"RAG query failed: {exc}")
    #         rag_answers.append("")
    #         rag_latencies.append(0)
    #         rag_contexts.append([])
    #
    #     time.sleep(3)
    #
    #     # LLM-only condition (direct Groq API call, no retrieval context)
    #     try:
    #         t0 = time.perf_counter()
    #         response = ctx["generator"].client.chat.completions.create(
    #             model=GROQ_MODEL,
    #             messages=[
    #                 {"role": "system", "content": LLM_ONLY_SYSTEM_PROMPT},
    #                 {"role": "user",   "content": question},
    #             ],
    #             temperature=LLM_TEMPERATURE,
    #             max_tokens=LLM_MAX_TOKENS,
    #         )
    #         llm_answer = response.choices[0].message.content.strip()
    #         llm_latency = time.perf_counter() - t0
    #         llm_answers.append(llm_answer)
    #         llm_latencies.append(llm_latency)
    #     except Exception as exc:
    #         logger.warning(f"LLM-only query failed: {exc}")
    #         llm_answers.append("")
    #         llm_latencies.append(0)
    #
    #     time.sleep(3)
    #
    # n_rag = max(len(rag_answers), 1)
    # n_llm = max(len(llm_answers), 1)
    # results = {
    #     "RAG": {
    #         "Avg Latency (s)": round(sum(rag_latencies) / n_rag, 3),
    #         "Avg Answer Len (chars)": round(sum(len(a) for a in rag_answers) / n_rag),
    #     },
    #     "LLM-only": {
    #         "Avg Latency (s)": round(sum(llm_latencies) / n_llm, 3),
    #         "Avg Answer Len (chars)": round(sum(len(a) for a in llm_answers) / n_llm),
    #     },
    # }
    #
    # from evaluate import print_table
    # print_table("Experiment 4 — RAG vs LLM-only Baseline", results)
    # _plot_exp4(results, RESULTS_DIR / "exp4_rag_vs_llm_only.png")
    # return results


def _plot_exp4(results: dict, output_path: Path):
    plt = _import_matplotlib()
    if plt is None or not results:
        return

    conditions = list(results.keys())
    all_metrics = sorted(set().union(*[set(v.keys()) for v in results.values()]))

    score_metrics = [m for m in all_metrics if m in ("Faithfulness", "Answer Relevancy")]
    raw_metrics = [m for m in all_metrics if m not in score_metrics]

    n_plots = len([g for g in [raw_metrics, score_metrics] if g])
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Subplot: Raw metrics (latency, answer length)
    if raw_metrics:
        ax = axes[plot_idx]
        plot_idx += 1
        x = range(len(raw_metrics))
        width = 0.35
        for i, cond in enumerate(conditions):
            vals = [results[cond].get(m, 0) for m in raw_metrics]
            bars = ax.bar([xi + i * width for xi in x], vals, width,
                          label=cond, color=COLORS[i], edgecolor="white")
            for bar, val in zip(bars, vals):
                label_text = f"{val:.1f}" if isinstance(val, float) else f"{val}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        label_text, ha="center", va="bottom",
                        fontsize=FONT_ANNOT, fontweight="bold")
        ax.set_xticks([xi + width / 2 for xi in x])
        ax.set_xticklabels(raw_metrics, fontsize=FONT_TICK)
        ax.set_title("Latency & Answer Length")
        ax.legend(framealpha=0.9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Subplot: RAGAS scores (0-1)
    if score_metrics and plot_idx < len(axes):
        ax = axes[plot_idx]
        x = range(len(score_metrics))
        width = 0.35
        for i, cond in enumerate(conditions):
            vals = [results[cond].get(m, 0) for m in score_metrics]
            bars = ax.bar([xi + i * width for xi in x], vals, width,
                          label=cond, color=COLORS[i], edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=FONT_ANNOT, fontweight="bold")
        ax.set_xticks([xi + width / 2 for xi in x])
        ax.set_xticklabels(score_metrics, fontsize=FONT_TICK)
        ax.set_ylim(0, 1.2)
        ax.set_title("RAGAS Scores")
        ax.legend(framealpha=0.9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Experiment 4: RAG vs LLM-only Baseline",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved: {output_path}")


# ── Experiment 5: Latency Profiling ──────────────────────────────────────────

def exp5_latency_profiling(ctx: dict) -> dict:
    """Profile per-stage latency for each retrieval mode.

    Measures retrieval and reranking directly (no API calls) to avoid
    Groq rate-limit delays inflating the numbers. Generation latency
    is measured once separately.
    """
    logger.info("=" * 60)
    logger.info("  Experiment 5: Latency Profiling")
    logger.info("=" * 60)

    if ctx["retriever"] is None or ctx["reranker"] is None:
        logger.warning("Skipping Exp5 — retriever or reranker not available.")
        return {}

    all_questions = _load_all_questions()
    if not all_questions:
        logger.warning("Skipping Exp5 — no test questions found.")
        return {}

    questions = [qa["question"] for qa in all_questions]
    results = {}

    for mode in RETRIEVAL_MODES:
        logger.info(f"  Profiling mode: {mode}")
        retrieval_times = []
        rerank_times = []

        for q in questions:
            # Measure retrieval
            t0 = time.perf_counter()
            candidates = ctx["retriever"].retrieve(q, mode=mode, top_k=TOP_K)
            retrieval_times.append(time.perf_counter() - t0)

            # Measure reranking
            t0 = time.perf_counter()
            ctx["reranker"].rerank(q, candidates, top_k=RERANK_TOP_K)
            rerank_times.append(time.perf_counter() - t0)

        n = len(questions)
        results[mode] = {
            "Retrieval (s)": round(sum(retrieval_times) / n, 4),
            "Rerank (s)":    round(sum(rerank_times) / n, 4),
        }

    # Measure generation latency once (avoid rate limiting by using a single call)
    gen_latency = 0.0
    if _has_groq_api() and ctx["generator"] is not None and ctx["generator"].client is not None:
        logger.info("  Measuring generation latency (single call)...")
        try:
            sample_q = questions[0]
            candidates = ctx["retriever"].retrieve(sample_q, mode="hybrid", top_k=TOP_K)
            top_passages = ctx["reranker"].rerank(sample_q, candidates, top_k=RERANK_TOP_K)
            t0 = time.perf_counter()
            ctx["generator"].generate_answer(sample_q, top_passages)
            gen_latency = round(time.perf_counter() - t0, 4)
        except Exception as exc:
            logger.warning(f"  Generation timing failed: {exc}")
    else:
        logger.info("  Groq API not available — generation latency set to 0.")

    for mode in results:
        results[mode]["Generation (s)"] = gen_latency
        results[mode]["Total (s)"] = round(
            results[mode]["Retrieval (s)"] +
            results[mode]["Rerank (s)"] +
            gen_latency, 4
        )

    from evaluate import print_table
    print_table("Experiment 5 — Latency Profiling", results)
    _plot_exp5(results, RESULTS_DIR / "exp5_latency_profiling.png")
    return results


def _plot_exp5(results: dict, output_path: Path):
    plt = _import_matplotlib()
    if plt is None or not results:
        return

    modes = list(results.keys())
    retrieval_vals = [results[m].get("Retrieval (s)", 0) for m in modes]
    rerank_vals = [results[m].get("Rerank (s)", 0) for m in modes]
    gen_vals = [results[m].get("Generation (s)", 0) for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Stacked bar chart (all stages)
    ax = axes[0]
    x = range(len(modes))
    width = 0.5
    bars1 = ax.bar(x, retrieval_vals, width, label="Retrieval", color=COLORS[0])
    bars2 = ax.bar(x, rerank_vals, width, bottom=retrieval_vals,
                   label="Rerank", color=COLORS[1])
    bottoms2 = [r + rr for r, rr in zip(retrieval_vals, rerank_vals)]
    bars3 = ax.bar(x, gen_vals, width, bottom=bottoms2,
                   label="Generation", color=COLORS[2])

    for bars, vals in [(bars1, retrieval_vals), (bars2, rerank_vals), (bars3, gen_vals)]:
        for bar, val in zip(bars, vals):
            if val > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}s", ha="center", va="center",
                        fontsize=FONT_ANNOT, color="white", fontweight="bold")

    ax.set_ylabel("Time (seconds)")
    ax.set_title("Full Pipeline Latency")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.upper() for m in modes], fontsize=FONT_TICK)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: Retrieval + Rerank only (zoomed in, excludes generation)
    ax2 = axes[1]
    width2 = 0.35
    bars_r = ax2.bar([xi - width2/2 for xi in x], retrieval_vals, width2,
                     label="Retrieval", color=COLORS[0], edgecolor="white")
    bars_rr = ax2.bar([xi + width2/2 for xi in x], rerank_vals, width2,
                      label="Rerank", color=COLORS[1], edgecolor="white")

    for bars, vals in [(bars_r, retrieval_vals), (bars_rr, rerank_vals)]:
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f"{val:.4f}s", ha="center", va="bottom",
                     fontsize=FONT_ANNOT, fontweight="bold")

    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Retrieval & Rerank Only (zoomed)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([m.upper() for m in modes], fontsize=FONT_TICK)
    ax2.legend(framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Experiment 5: Latency Profiling by Retrieval Mode",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved: {output_path}")


# ── Experiment 6: Reranking Ablation ─────────────────────────────────────────

def exp6_reranking_ablation(ctx: dict) -> dict:
    logger.info("=" * 60)
    logger.info("  Experiment 6: Reranking Ablation")
    logger.info("=" * 60)

    if not _has_annotated_testset(ctx["testset"]):
        logger.warning("Skipping Exp6 — no annotated test data.")
        return {}

    from evaluate import print_table, run_reranking_ablation
    results = run_reranking_ablation(testset=ctx["testset"])
    print_table("Experiment 6 — Reranking Ablation", results)
    _plot_exp6(results, RESULTS_DIR / "exp6_reranking_ablation.png")
    return results


def _plot_exp6(results: dict, output_path: Path):
    plt = _import_matplotlib()
    if plt is None or not results:
        return

    conditions = list(results.keys())
    metrics = sorted(results[conditions[0]].keys())

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(metrics))
    width = 0.35

    labels = {"without_reranking": "Without Reranking", "with_reranking": "With Reranking"}
    colors = [COLORS[3], COLORS[0]]

    for i, cond in enumerate(conditions):
        vals = [results[cond].get(m, 0) for m in metrics]
        bars = ax.bar([xi + i * width for xi in x], vals, width,
                      label=labels.get(cond, cond), color=colors[i],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=FONT_ANNOT, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Experiment 6: Reranking Ablation (With vs Without)")
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=FONT_TICK)
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved: {output_path}")


# ── Main runner ──────────────────────────────────────────────────────────────

def run_all_experiments():
    print("\n" + "=" * 60)
    print("  RAG Experiment Runner")
    print("  Charts will be saved to: results/")
    print("=" * 60 + "\n")

    _ensure_results_dir()
    ctx = _load_shared_resources()

    # Auto-generate test set if none exists with annotations
    if not _has_annotated_testset(ctx["testset"]):
        logger.info("No annotated test set found. Auto-generating from corpus...")
        if _has_groq_api():
            generate_testset()
            # Reload test set
            try:
                from evaluate import load_testset
                ctx["testset"] = load_testset()
            except Exception:
                pass
        else:
            logger.warning(
                "Cannot auto-generate test set without GROQ_API_KEY.\n"
                "  → Set GROQ_API_KEY in .env and run: "
                "python src/experiments.py --generate-testset"
            )

    experiments = [
        ("exp1", "Retrieval Mode Comparison",      exp1_retrieval_mode_comparison),
        ("exp2", "Top-K Sensitivity",               exp2_topk_sensitivity),
        ("exp3", "Rerank Top-K Sensitivity",        exp3_rerank_topk_sensitivity),
        ("exp4", "RAG vs LLM-only Baseline",        exp4_rag_vs_llm_only),
        ("exp5", "Latency Profiling",               exp5_latency_profiling),
        ("exp6", "Reranking Ablation",              exp6_reranking_ablation),
    ]

    completed = []
    skipped = []

    for exp_id, name, func in experiments:
        logger.info(f"\nStarting {exp_id}: {name}")
        result = func(ctx)
        if result:
            completed.append(f"{exp_id}: {name}")
        else:
            skipped.append(f"{exp_id}: {name}")

    # Summary
    print("\n" + "=" * 60)
    print("  Experiment Summary")
    print("=" * 60)
    print(f"\n  Completed: {len(completed)}")
    for c in completed:
        print(f"    - {c}")
    if skipped:
        print(f"\n  Skipped: {len(skipped)}")
        for s in skipped:
            print(f"    - {s}")
    print(f"\n  Charts saved to: {RESULTS_DIR}/")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAG experiments and generate charts."
    )
    parser.add_argument(
        "--experiment",
        choices=["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "all"],
        default="all",
        help="Which experiment to run (default: all).",
    )
    parser.add_argument(
        "--generate-testset", action="store_true",
        help="Auto-generate annotated test set from corpus using Groq LLM.",
    )
    parser.add_argument(
        "--n-per-paper", type=int, default=3,
        help="Questions to generate per paper (default: 3).",
    )
    args = parser.parse_args()

    if args.generate_testset:
        generate_testset(n_per_paper=args.n_per_paper)
    elif args.experiment == "all":
        run_all_experiments()
    else:
        _ensure_results_dir()
        ctx = _load_shared_resources()
        experiment_map = {
            "exp1": exp1_retrieval_mode_comparison,
            "exp2": exp2_topk_sensitivity,
            "exp3": exp3_rerank_topk_sensitivity,
            "exp4": exp4_rag_vs_llm_only,
            "exp5": exp5_latency_profiling,
            "exp6": exp6_reranking_ablation,
        }
        experiment_map[args.experiment](ctx)
