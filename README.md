# RAG Academic Paper QA System

A Retrieval-Augmented Generation system for answering questions over ArXiv papers. Combines BM25 sparse retrieval, FAISS dense retrieval, and Reciprocal Rank Fusion (RRF) hybrid search with Cross-Encoder reranking and Groq LLM generation.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env   # Edit .env and set GROQ_API_KEY (free at https://console.groq.com)

# 3. Download papers and build indexes
python src/ingest.py --arxiv --max 20
python src/build_index.py

# 4. Interactive QA
python src/pipeline.py
```

Or use the one-command setup:

```bash
chmod +x setup.sh && ./setup.sh
```

## Usage

```bash
# Interactive QA (default: hybrid retrieval)
python src/pipeline.py

# Single question
python src/pipeline.py --question "What is attention mechanism?"

# Specify retrieval mode
python src/pipeline.py --mode bm25
python src/pipeline.py --mode dense
python src/pipeline.py --mode hybrid

# Download papers on a specific topic
python src/ingest.py --arxiv --max 50 --query "retrieval augmented generation"
python src/build_index.py --rebuild
```

## Experiments

```bash
# Run all 6 experiments (generates charts to results/)
python src/experiments.py

# Run a single experiment
python src/experiments.py --experiment exp1

# Auto-generate test set from corpus (requires Groq API)
python src/experiments.py --generate-testset
```

| # | Experiment | Output |
|---|-----------|--------|
| 1 | BM25 vs Dense vs Hybrid + Factual/Semantic breakdown | `exp1_retrieval_comparison.png`, `exp1_factual_vs_semantic.png` |
| 2 | Top-K sensitivity (k=5,10,20,50) | `exp2_topk_sensitivity.png` |
| 3 | Rerank Top-K sensitivity (k=1,2,3,5,10) | `exp3_rerank_topk_sensitivity.png` |
| 4 | RAG vs LLM-only baseline | `exp4_rag_vs_llm_only.png` |
| 5 | Latency profiling (per-stage breakdown) | `exp5_latency_profiling.png` |
| 6 | Reranking ablation (with vs without) | `exp6_reranking_ablation.png` |

## Evaluation

```bash
# Run retrieval evaluation
python src/evaluate.py

# Specific experiment
python src/evaluate.py --experiment retrieval
python src/evaluate.py --experiment reranking
```

## Known Limitations

- **Developed and tested on macOS** (Apple Silicon / MPS). Cross-Encoder reranking falls back to bi-encoder due to MPS compatibility issues with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Groq free-tier API** has rate limits (30 req/min, 100K tokens/day). Experiment 4 (RAG vs LLM-only) uses cached results to avoid quota consumption. Large-scale testing may encounter 429 rate-limit errors.
- **numpy < 2.0.0** required for faiss-cpu compatibility.
