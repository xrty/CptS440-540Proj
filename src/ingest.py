"""
Data ingestion pipeline for the RAG Academic Paper QA System.

Steps:
  1. (Optional) Download papers from ArXiv API into data/papers/
  2. Parse PDFs with PyMuPDF (fitz) — page by page
  3. Clean extracted text (strip page numbers, hyphenation, noise)
  4. Remove references/bibliography section (heuristic)
  5. Chunk text with LangChain's RecursiveCharacterTextSplitter
  6. Save all chunks to indexes/chunks.json

Each chunk record follows this schema:
{
    "chunk_id":     "2301.07041_p3_c12",   # "{arxiv_id}_p{page}_c{idx}"
    "chunk_text":   "...",
    "source_paper": "Attention Is All You Need",
    "arxiv_id":     "2301.07041",           # empty string for local-only PDFs
    "page_num":     3,
    "file_path":    "data/papers/2301.07041.pdf"
}

Usage:
    # Process all PDFs already in data/papers/
    python src/ingest.py

    # Download 50 papers from ArXiv first, then process
    python src/ingest.py --arxiv --max 50

    # Download with a specific query
    python src/ingest.py --arxiv --query "retrieval augmented generation" --max 20

    # Ablation: re-chunk with different chunk size
    python src/ingest.py --chunk-size 256 --chunk-overlap 32
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Allow running as a script from the repo root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ARXIV_CATEGORIES,
    ARXIV_MAX_RESULTS,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    CHUNKS_PATH,
    INDEX_DIR,
    LOG_LEVEL,
    PAPERS_DIR,
)

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ── ArXiv download ─────────────────────────────────────────────────────────────

def download_arxiv_papers(
    categories: List[str] = ARXIV_CATEGORIES,
    max_results: int = ARXIV_MAX_RESULTS,
    query: Optional[str] = None,
    output_dir: Path = PAPERS_DIR,
) -> List[Path]:
    """
    Download papers from the ArXiv API and save them as PDFs.

    Uses the `arxiv` Python package. Already-downloaded papers (same arxiv_id)
    are skipped automatically.

    Args:
        categories:  ArXiv subject categories, e.g. ["cs.AI", "cs.CL"].
        max_results: Maximum number of papers to download per call.
        query:       Optional freetext search query. If None, fetches recent
                     papers matching the given categories.
        output_dir:  Directory where PDF files are saved.

    Returns:
        List of Paths to all downloaded (or previously existing) PDF files.

    Example:
        >>> paths = download_arxiv_papers(max_results=10, query="RAG survey")
    """
    try:
        import arxiv
    except ImportError:
        logger.error("'arxiv' package not installed. Run: pip install arxiv")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    if query is None:
        query = " OR ".join(f"cat:{cat}" for cat in categories)

    logger.info(f"ArXiv search | query='{query}' | max_results={max_results}")

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    downloaded: List[Path] = []
    for result in client.results(search):
        arxiv_id = result.entry_id.split("/")[-1]
        pdf_path = output_dir / f"{arxiv_id}.pdf"

        if pdf_path.exists():
            logger.debug(f"Already downloaded: {arxiv_id}")
            downloaded.append(pdf_path)
            continue

        try:
            result.download_pdf(dirpath=str(output_dir), filename=f"{arxiv_id}.pdf")
            logger.info(f"Downloaded: {arxiv_id} — {result.title[:60]}")
            downloaded.append(pdf_path)
        except Exception as exc:
            logger.warning(f"Failed to download {arxiv_id}: {exc}")

    logger.info(f"Total PDFs available in {output_dir}: {len(downloaded)}")
    return downloaded


# ── PDF parsing ────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path) -> List[Dict]:
    """
    Extract text from a PDF file, one dict per page.

    Uses PyMuPDF (fitz) plain-text extraction. Multi-column layouts are
    handled reasonably well by fitz's default reading-order sorting.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of page dicts:
        [{"page_num": int, "raw_text": str, "file_path": str}, ...]
        Returns an empty list if the file cannot be opened.

    Example:
        >>> pages = parse_pdf(Path("data/papers/1706.03762.pdf"))
        >>> print(pages[0]["raw_text"][:200])
    """
    pages: List[Dict] = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.warning(f"Cannot open {pdf_path.name}: {exc}")
        return []

    for page_num, page in enumerate(doc):
        try:
            raw_text = page.get_text("text")
            pages.append({
                "page_num":  page_num,
                "raw_text":  raw_text,
                "file_path": str(pdf_path),
            })
        except Exception as exc:
            logger.warning(f"Failed to read page {page_num} of {pdf_path.name}: {exc}")

    doc.close()
    return pages


def clean_text(text: str) -> str:
    """
    Remove common PDF extraction noise from a text string.

    Handles:
    - Lines containing only page numbers (pure digit lines)
    - Hyphenation at line endings (e.g. "pre-\\nprocessing" → "preprocessing")
    - Runs of whitespace / blank lines
    - Non-printable / non-ASCII characters (keeps Latin extended range)

    Args:
        text: Raw text from a PDF page.

    Returns:
        Cleaned text, or empty string if input is empty/None.

    Example:
        >>> clean_text("pre-\\nprocessing\\n\\n\\n   lots  of spaces")
        'preprocessing\\n\\nlots of spaces'
    """
    if not text:
        return ""

    # Remove pure page-number lines
    lines = [ln for ln in text.split("\n") if not re.match(r"^\s*\d+\s*$", ln)]
    text = "\n".join(lines)

    # Merge hyphenated line breaks: "pre-\nfix" → "prefix"
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Collapse multiple spaces and tabs to one space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than two consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Drop non-printable characters (keep newline, tab, basic Latin, Latin-Extended)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00C0-\u024F]", "", text)

    return text.strip()


def remove_references_section(text: str) -> str:
    """
    Heuristically strip the References / Bibliography section from full-document text.

    Searches for common headings in the final 40% of the document and
    truncates at the first match. This prevents reference entries from
    being retrieved as relevant passages.

    Args:
        text: Full document text (all pages concatenated).

    Returns:
        Text with the references section removed if detected; otherwise unchanged.

    Example:
        >>> doc = "Body text...\\nReferences\\n[1] Author et al."
        >>> remove_references_section(doc)
        'Body text...'
    """
    ref_patterns = [
        r"\nReferences\n",
        r"\nREFERENCES\n",
        r"\nBibliography\n",
        r"\nBIBLIOGRAPHY\n",
        r"\n\[1\]",  # Numbered list without a heading
    ]
    cutoff = int(len(text) * 0.60)
    tail = text[cutoff:]

    for pattern in ref_patterns:
        match = re.search(pattern, tail)
        if match:
            logger.debug("References section detected and stripped.")
            return text[: cutoff + match.start()]

    return text


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_document(
    pages: List[Dict],
    arxiv_id: str = "",
    paper_title: str = "",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Split a parsed document into overlapping text chunks.

    Processes pages individually to preserve page_num metadata. Applies
    LangChain's RecursiveCharacterTextSplitter with a character-count proxy
    for token count (~4 chars ≈ 1 token for English text).

    Args:
        pages:         Output of parse_pdf() — list of page dicts.
        arxiv_id:      ArXiv ID (e.g. "1706.03762"), or "" for local PDFs.
        paper_title:   Human-readable title for citation in generated answers.
        chunk_size:    Target chunk size in approximate tokens.
        chunk_overlap: Overlap between consecutive chunks in approximate tokens.

    Returns:
        List of chunk dicts following the project schema:
        [{"chunk_id", "chunk_text", "source_paper", "arxiv_id",
          "page_num", "file_path"}, ...]

    Example:
        >>> pages = parse_pdf(Path("data/papers/1706.03762.pdf"))
        >>> chunks = chunk_document(pages, "1706.03762", "Attention Is All You Need")
        >>> print(f"{len(chunks)} chunks produced")
    """
    if not pages:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,       # ~4 chars per token
        chunk_overlap=chunk_overlap * 4,
        separators=CHUNK_SEPARATORS,
    )

    chunks: List[Dict] = []
    global_idx = 0

    for page in pages:
        cleaned = clean_text(page["raw_text"])
        if not cleaned.strip():
            continue

        for chunk_text in splitter.split_text(cleaned):
            chunk_id = f"{arxiv_id or 'local'}_p{page['page_num']}_c{global_idx}"
            chunks.append({
                "chunk_id":     chunk_id,
                "chunk_text":   chunk_text,
                "source_paper": paper_title,
                "arxiv_id":     arxiv_id,
                "page_num":     page["page_num"],
                "file_path":    page["file_path"],
            })
            global_idx += 1

    return chunks


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_title_from_pdf(pdf_path: Path) -> str:
    """
    Attempt to read the paper title from PDF metadata.

    Falls back to the filename stem (e.g. "1706.03762") if metadata is absent
    or empty — a common occurrence with ArXiv PDFs.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Title string, never empty (at minimum returns the filename stem).
    """
    try:
        doc = fitz.open(str(pdf_path))
        title = (doc.metadata or {}).get("title", "").strip()
        doc.close()
        if title:
            return title
    except Exception:
        pass
    return pdf_path.stem


# ── Orchestration ──────────────────────────────────────────────────────────────

def ingest_all(
    papers_dir: Path = PAPERS_DIR,
    output_path: Path = CHUNKS_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Process all PDFs in papers_dir and save chunks to output_path (chunks.json).

    Incremental: skips PDFs whose file_path already appears in an existing
    chunks.json. Appends new chunks to the existing file.

    Args:
        papers_dir:    Directory containing PDF files.
        output_path:   Destination JSON file for all chunks.
        chunk_size:    Passed through to chunk_document() for ablation.
        chunk_overlap: Passed through to chunk_document() for ablation.

    Returns:
        Complete list of all chunks (pre-existing + newly parsed).

    Example:
        >>> all_chunks = ingest_all()
        >>> print(f"Total chunks: {len(all_chunks)}")
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing chunks to avoid reprocessing
    existing_chunks: List[Dict] = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
        logger.info(f"Loaded {len(existing_chunks)} existing chunks from {output_path.name}")

    processed_files = {c["file_path"] for c in existing_chunks}

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(
            f"No PDF files found in {papers_dir}.\n"
            f"  → Add PDFs manually, or run:  python src/ingest.py --arxiv"
        )
        return existing_chunks

    new_chunks: List[Dict] = []
    for pdf_path in pdf_files:
        if str(pdf_path) in processed_files:
            logger.debug(f"Already ingested — skipping: {pdf_path.name}")
            continue

        logger.info(f"Ingesting: {pdf_path.name}")
        arxiv_id = pdf_path.stem
        title = extract_title_from_pdf(pdf_path)
        pages = parse_pdf(pdf_path)
        chunks = chunk_document(
            pages,
            arxiv_id=arxiv_id,
            paper_title=title,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        new_chunks.extend(chunks)
        logger.info(f"  → {len(chunks)} chunks from '{title[:50]}'")

    all_chunks = existing_chunks + new_chunks
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(all_chunks)} total chunks → {output_path}")
    return all_chunks


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into chunks.json for the RAG pipeline."
    )
    parser.add_argument(
        "--arxiv", action="store_true",
        help="Download papers from ArXiv before processing."
    )
    parser.add_argument(
        "--max", type=int, default=ARXIV_MAX_RESULTS,
        help=f"Max papers to download from ArXiv (default: {ARXIV_MAX_RESULTS})."
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="ArXiv search query (default: category filter from config)."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Chunk size in approximate tokens (default: {CHUNK_SIZE})."
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=CHUNK_OVERLAP,
        help=f"Chunk overlap in approximate tokens (default: {CHUNK_OVERLAP})."
    )
    args = parser.parse_args()

    if args.arxiv:
        download_arxiv_papers(max_results=args.max, query=args.query)

    ingest_all(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
