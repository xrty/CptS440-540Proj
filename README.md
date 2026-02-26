# RAG Project Execution Plan

## Phase 1: Environment & Data Foundation (Week 1–2)

**Goal:** Prepare the data pipeline so all downstream modules have clean, structured input.

**Acceptance Criteria:** A clean chunk dataset (JSON/Parquet) where each record contains `chunk_text`, `source_paper`, `chunk_id`, and metadata fields.

| # | Task | Owner |
|---|------|-------|
| 1.1 | Set up project environment (Python venv, dependency management, GitHub repo structure) | **All** |
| 1.2 | Crawl 500–1,000 papers from ArXiv API (cs.AI, cs.CL categories) | **Zijian** |
| 1.3 | PDF parsing via PyMuPDF — handle noise such as garbled text, tables, and formulas | **Zijian** |
| 1.4 | Text cleaning — remove headers/footers, references, and other non-body content | **Zijian** |
| 1.5 | Implement chunking module (default: 512 tokens + 64-token overlap; expose parameters for 256/1024 ablation later) | **Zijian** |
| 1.6 | (Parallel) Scaffold BM25 and FAISS retrieval code with a small mock dataset to verify interfaces | **Tong** |
| 1.7 | (Parallel) Deploy Ollama locally (LLaMA-3 / Mistral-7B); draft prompt templates | **Yuqi** |
| 1.8 | (Parallel) Design QA test-set annotation schema and format (fields, guidelines) | **Yuhang** |

---

## Phase 2: Retrieval Module (Week 3–4)

**Goal:** Build and validate three retrieval strategies over the chunk corpus.

**Acceptance Criteria:** Given a natural-language query, all three retrieval modes (BM25 / Dense / Hybrid) return relevant top-K chunks with reasonable results.

| # | Task | Owner |
|---|------|-------|
| 2.1 | Build BM25 sparse index over all chunks using `rank_bm25`; implement query → top-K retrieval | **Tong** |
| 2.2 | Encode all chunks with `sentence-transformers` (all-MiniLM-L6-v2); store embeddings in a FAISS index; implement dense retrieval | **Tong** |
| 2.3 | Implement Reciprocal Rank Fusion (RRF) to merge BM25 and Dense results into a hybrid ranking | **Tong** |
| 2.4 | Create a unified retrieval interface that supports switching among BM25 / Dense / Hybrid modes | **Tong** |
| 2.5 | Assist with retrieval integration testing; spot-check retrieval quality on sample queries | **Yuhang** |
| 2.6 | (Parallel) Continue QA test-set annotation — begin writing question-answer pairs from the corpus | **Yuhang** |

---

## Phase 3: Reranking & Generation (Week 5)

**Goal:** Complete the end-to-end RAG pipeline: query → retrieval → reranking → answer generation.

**Acceptance Criteria:** Given a question, the system outputs a grounded answer traceable to source papers.

| # | Task | Owner |
|---|------|-------|
| 3.1 | Integrate Cross-Encoder reranker (`ms-marco-MiniLM-L-6-v2`); rerank top-20 candidates → select top-5 | **Yuqi** |
| 3.2 | Finalize LLM prompt template (concatenate top-5 passages + query); generate answers via Ollama | **Yuqi** |
| 3.3 | Wire the full pipeline end-to-end: retrieval → reranking → generation | **Yuqi** |
| 3.4 | (Optional) Connect GPT-3.5 via OpenAI API as an alternative generator for comparison | **Yuqi** |
| 3.5 | Assist with pipeline integration testing and debugging | **Tong** |
| 3.6 | (Parallel) Continue and finalize QA test-set annotation (~100 QA pairs with ground-truth source chunks) | **Yuhang** |

---

## Phase 4: Evaluation & Experiments (Week 6–7)

**Goal:** Systematically evaluate the pipeline through comparative and ablation experiments.

**Acceptance Criteria:** All experiment results compiled into tables with clear numerical comparisons and conclusions.

| # | Task | Owner |
|---|------|-------|
| 4.1 | **Retrieval Comparison:** BM25 vs. Dense vs. Hybrid — measure Recall@5, MRR | **Yuhang** |
| 4.2 | **Reranking Ablation:** With vs. without Cross-Encoder — measure Precision@5, NDCG | **Yuhang** |
| 4.3 | **Chunk Size Ablation:** Re-chunk corpus at 256 / 1024 tokens → rebuild indices → re-run retrieval metrics | **Zijian** (re-chunking) + **Tong** (re-indexing) + **Yuhang** (metrics) |
| 4.4 | **End-to-End QA:** RAG vs. LLM-only baseline — evaluate with RAGAS (Faithfulness & Relevancy) | **Yuhang** + **Yuqi** |
| 4.5 | Record latency benchmarks for each retrieval method and the full pipeline | **Tong** |
| 4.6 | Generate visualizations: retrieval score distributions, latency charts, qualitative answer comparisons | **Yuhang** |

---

## Phase 5: Packaging & Delivery (Week 8)

**Goal:** Polish all deliverables for submission.

**Acceptance Criteria:** Four deliverables ready — write-up, slides, GitHub repo, and demo notebook.

| # | Task | Owner |
|---|------|-------|
| 5.1 | **Demo Notebook:** Jupyter notebook that walks through the full pipeline end-to-end and is reproducible | **Yuhang** |
| 5.2 | **Write-up:** Final report — motivate the problem; explain search algorithm theory (BM25/TF-IDF, ANN/IVF/HNSW) tied to the course "search" theme; describe methodology; report results | **All** (Zijian: data section, Tong: retrieval section, Yuqi: reranking & generation section, Yuhang: evaluation section + overall editing) |
| 5.3 | **Slides:** 10-minute presentation highlighting pipeline architecture, experiment results, and key findings | **All** |
| 5.4 | **GitHub Polish:** Code comments, README, clean project structure, dependency list | **All** |
