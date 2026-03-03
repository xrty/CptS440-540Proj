"""
Central configuration for the RAG Academic Paper QA System.

All paths, hyperparameters, and API keys live here.
Every other module imports from this file — never hardcode values elsewhere.

Usage:
    from config import PAPERS_DIR, GROQ_API_KEY, CHUNK_SIZE, ...
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present — importing this module is enough,
# no other module needs to call load_dotenv() independently.
load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
# Resolves to CptS440-540Proj/ regardless of where the repo is cloned.
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
PAPERS_DIR      = DATA_DIR / "papers"          # PDF corpus (populated by ingest.py)
QA_TESTSET_PATH = DATA_DIR / "qa_testset.json" # Hand-annotated QA pairs

# ── Index paths ───────────────────────────────────────────────────────────────
INDEX_DIR        = ROOT_DIR / "indexes"
CHUNKS_PATH      = INDEX_DIR / "chunks.json"        # Parsed chunks with metadata
BM25_INDEX_PATH  = INDEX_DIR / "bm25.pkl"           # Serialized BM25Okapi index
FAISS_INDEX_DIR  = INDEX_DIR / "faiss_index"        # FAISS index directory
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"  # FAISS binary index
FAISS_META_PATH  = FAISS_INDEX_DIR / "index.pkl"    # chunk_id list (position → id)

# ── ArXiv download settings ───────────────────────────────────────────────────
ARXIV_CATEGORIES  = ["cs.AI", "cs.CL"]   # Subject categories to fetch
ARXIV_MAX_RESULTS = 100                   # Papers per run (set higher for full corpus)
ARXIV_SORT_BY     = "submittedDate"       # "relevance" | "lastUpdatedDate" | "submittedDate"

# ── Chunking hyperparameters (exposed for ablation experiments) ───────────────
# Ablation plan: test CHUNK_SIZE in {256, 512, 1024} — override via CLI flags in ingest.py
CHUNK_SIZE     = 512   # approximate tokens (~4 chars/token → 2048 chars internally)
CHUNK_OVERLAP  = 64    # overlap between consecutive chunks (~256 chars internally)
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # RecursiveCharacterTextSplitter order

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Output vector dimension of all-MiniLM-L6-v2

# ── Retrieval hyperparameters ─────────────────────────────────────────────────
# Override RETRIEVAL_MODE at runtime via pipeline.py --mode flag
RETRIEVAL_MODE = "hybrid"  # "bm25" | "dense" | "hybrid"
TOP_K          = 20        # Candidates retrieved before reranking
RERANK_TOP_K   = 5         # Final passages sent to the LLM after reranking
RRF_K          = 60        # RRF constant (standard value from the RRF paper)

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM / Groq settings ───────────────────────────────────────────────────────
# GROQ_API_KEY is read from .env — copy .env.example to .env and fill in your key.
# An empty string here is intentional: generate.py detects it and returns a
# placeholder response instead of crashing.
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2   # Low temperature for factual, grounded answers
LLM_MAX_TOKENS  = 512   # Max tokens in the generated answer

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_K_VALUES  = [1, 3, 5]  # k values for Recall@k, Precision@k, NDCG@k
RAGAS_METRICS  = ["faithfulness", "answer_relevancy"]

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"  # "DEBUG" | "INFO" | "WARNING" | "ERROR"


# ── Self-check (run as script to verify paths) ────────────────────────────────
if __name__ == "__main__":
    print("=== RAG System Configuration ===")
    print(f"ROOT_DIR        : {ROOT_DIR}")
    print(f"PAPERS_DIR      : {PAPERS_DIR}  (exists: {PAPERS_DIR.exists()})")
    print(f"CHUNKS_PATH     : {CHUNKS_PATH}  (exists: {CHUNKS_PATH.exists()})")
    print(f"BM25_INDEX_PATH : {BM25_INDEX_PATH}  (exists: {BM25_INDEX_PATH.exists()})")
    print(f"FAISS_INDEX_DIR : {FAISS_INDEX_DIR}  (exists: {FAISS_INDEX_DIR.exists()})")
    print(f"GROQ_API_KEY    : {'[SET]' if GROQ_API_KEY else '[NOT SET — placeholder mode]'}")
    print(f"CHUNK_SIZE      : {CHUNK_SIZE} tokens")
    print(f"RETRIEVAL_MODE  : {RETRIEVAL_MODE}")
    print(f"TOP_K / RERANK  : {TOP_K} → {RERANK_TOP_K}")
